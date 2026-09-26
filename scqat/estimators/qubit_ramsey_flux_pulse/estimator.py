"""Ramsey fringe vs flux - the local arch around a qubit's parking point.

Dataset contract (one target; the acquisition layer splits targets off):

    coords   ``flux_bias`` (V)  - the swept flux value, in ANY order (the probe's
                                  realized order; canonicalized on entry, so every
                                  artifact is ascending and order-free)
             ``idle_time`` (s)  - the Ramsey idle, in any order
    vars     ``signal`` (flux_bias, idle_time)  - a real signal (e.g. an averaged
                                  population), OR complex ``IQdata`` / ``I`` + ``Q``,
                                  reduced here by ONE global axial projection
                                  (stored ``ref_pos_*`` centers when present)

The estimator is blind to the flux FRAME: it reports positions on the
``flux_bias`` axis as given (the acquisition layer re-references a relative axis).

Model
-----
Each flux slice is one Ramsey fringe. Its frequency ``F`` comes from the shared
reduction :func:`scqat.tools.fringe_frequency.fringe_frequency`. With the SIGNED
virtual detuning ``D`` the probe applied (the ``qubit_ramsey`` convention: fringe at
``|D + (f_q - f_drive)|``), and assuming the fringe never crossed zero (the
acquisition layer chose the sign of ``D`` to guarantee it; ``fold_suspected`` checks
it afterwards)::

    delta_f(x) = f_q(x) - f_drive = sign(D) * F(x) - D

The valid ``delta_f(x)`` points are fitted with a weighted quadratic (the transmon
arch is quadratic to 1e-4 over +-12 mV of a sweet spot). Two questions:

* **apex** (``park_frequency_hz`` None): the vertex ``x0`` and its height.
* **park** (``park_frequency_hz`` given, needs ``drive_freq_hz``): the root of
  ``f(x) = park_frequency_hz`` inside the window, chosen by ``flux_side`` when both
  roots are inside. Never extrapolated. The apex is still reported when bracketed.

The result does not depend on the order of either axis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_positions, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.fringe_frequency import fringe_frequency
from scqat.tools.iq_reduce import AXIAL_KNOBS, axial, axis_angle, validate_iq_reduce_kwargs
from scqat.tools.sweep_order import ascending
from scqat.estimators.qubit_ramsey_flux_pulse.visualization import (
    plot_flux_curve,
    plot_fringe_map,
)

#: valid ``flux_side`` values
FLUX_SIDES = ("nearest", "lower", "upper")
#: minimum valid slices for the quadratic
MIN_VALID = 5

_NAN = float("nan")


def _global_signal(dataset: xr.Dataset, axial_kwargs: dict) -> tuple[np.ndarray, str, float]:
    """(flux_bias, idle_time) real signal + reduction provenance."""
    if "signal" in dataset.data_vars:
        sig = dataset["signal"].transpose("flux_bias", "idle_time").values.astype(float)
        return sig, "signal", _NAN
    iq = with_iqdata(dataset)["IQdata"].transpose("flux_bias", "idle_time").values
    kwargs = dict(axial_kwargs)
    if kwargs.get("angle") is None and kwargs.get("positions") is None:
        stored = stored_positions(dataset)
        if stored is not None:
            kwargs["positions"] = stored
    I, Q = np.real(iq).ravel(), np.imag(iq).ravel()
    sig = axial(I, Q, **kwargs).reshape(iq.shape)
    method = ("angle" if kwargs.get("angle") is not None
              else "positions" if kwargs.get("positions") is not None else "pca")
    angle = float(axis_angle(I, Q, angle=kwargs.get("angle"), positions=kwargs.get("positions")))
    return sig, method, angle


def _vertex(coef: np.ndarray, cov: np.ndarray) -> tuple[float, float, float, float]:
    """Vertex (u0, stderr_u0, h0, stderr_h0) of a*u^2 + b*u + c."""
    a, b, c = coef
    u0 = -b / (2 * a)
    h0 = c - b * b / (4 * a)
    g_u = np.array([b / (2 * a * a), -1.0 / (2 * a), 0.0])
    g_h = np.array([b * b / (4 * a * a), -b / (2 * a), 1.0])
    su = float(np.sqrt(max(g_u @ cov @ g_u, 0.0)))
    sh = float(np.sqrt(max(g_h @ cov @ g_h, 0.0)))
    return float(u0), su, float(h0), sh


def _roots(coef: np.ndarray, level: float) -> list[float]:
    a, b, c = coef
    disc = b * b - 4 * a * (c - level)
    if a == 0 or disc < 0:
        return []
    sq = float(np.sqrt(disc))
    return sorted({float((-b - sq) / (2 * a)), float((-b + sq) / (2 * a))})


class QubitRamseyFluxPulseEstimator(BaseEstimator):
    """Local arch from a Ramsey-fringe map vs flux: apex or a target frequency."""

    estimator_name = "qubit_ramsey_flux_pulse"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError("qubit_ramsey_flux_pulse requires a 'signal' variable, or "
                             "complex 'IQdata', or both 'I' and 'Q'.")
        for coord in ("flux_bias", "idle_time"):
            if coord not in dataset.coords:
                raise ValueError(f"qubit_ramsey_flux_pulse requires a '{coord}' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Per-slice fringe frequency -> delta_f(x) -> weighted quadratic.

        Kwargs - flat and fully owned; unknown names raise:
            ramp_detuning_hz (float, REQUIRED): the SIGNED virtual detuning the probe
                applied (fringe at ``|D + (f_q - f_drive)|``).
            drive_freq_hz (float, optional): the drive frequency, to report absolute
                f01 values; required for ``park_frequency_hz``.
            park_frequency_hz (float, optional): the f01 to park at; None = apex.
            flux_side (str): ``'nearest'`` (default) / ``'lower'`` / ``'upper'`` -
                which root when both are inside the window.
            nearest_to (float): the axis value ``'nearest'`` measures from
                (default 0.0 - the parking point on an idle-relative axis).
            min_snr (float): periodogram peak / median power a slice needs to
                count (default 30).
            angle, positions, pca_sign: the axial IQ-reduction knobs.
        """
        if "ramp_detuning_hz" not in kwargs:
            raise ValueError("qubit_ramsey_flux_pulse needs ramp_detuning_hz (the signed "
                             "virtual detuning the probe applied)")
        ramp = float(kwargs.pop("ramp_detuning_hz"))
        if ramp == 0 or not np.isfinite(ramp):
            raise ValueError(f"ramp_detuning_hz must be finite and nonzero, got {ramp}")
        drive = kwargs.pop("drive_freq_hz", None)
        drive = None if drive is None else float(drive)
        park = kwargs.pop("park_frequency_hz", None)
        park = None if park is None else float(park)
        side = str(kwargs.pop("flux_side", "nearest"))
        nearest_to = float(kwargs.pop("nearest_to", 0.0))
        min_snr = float(kwargs.pop("min_snr", 30.0))
        if side not in FLUX_SIDES:
            raise ValueError(f"flux_side must be one of {FLUX_SIDES}, got {side!r}")
        if park is not None and drive is None:
            raise ValueError("park_frequency_hz needs drive_freq_hz to turn the target "
                             "into a detuning from the drive")
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, "flux_bias", "idle_time")

        x = np.asarray(dataset["flux_bias"].values, dtype=float)
        t = np.asarray(dataset["idle_time"].values, dtype=float)
        sig, red_method, red_angle = _global_signal(dataset, kwargs)
        span = float(np.nanmax(t) - np.nanmin(t)) if t.size else _NAN

        n = x.size
        fringe = np.full(n, _NAN)
        fringe_err = np.full(n, _NAN)
        snr = np.full(n, _NAN)
        amp = np.full(n, _NAN)
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            r = fringe_frequency(t, sig[i])
            fringe[i], snr[i], amp[i] = r["frequency_hz"], r["snr"], r["amplitude"]
            err = r["frequency_stderr_hz"]
            if not np.isfinite(err) and np.isfinite(r["snr"]) and r["snr"] > 0 and span > 0:
                err = 1.0 / (span * np.sqrt(r["snr"]))  # periodogram-only fallback
            fringe_err[i] = err
            edge = 2.0 / span if span > 0 else _NAN
            valid[i] = bool(r["success"] and r["snr"] >= min_snr
                            and r["frequency_hz"] >= r["f_min_hz"] + edge
                            and r["frequency_hz"] <= r["f_max_hz"] - edge)

        sgn = 1.0 if ramp > 0 else -1.0
        delta = sgn * fringe - ramp
        delta_err = fringe_err.copy()

        res: Dict[str, Any] = {
            "question": "apex" if park is None else "park",
            "ramp_detuning_hz": ramp,
            "drive_freq_hz": _NAN if drive is None else drive,
            "park_frequency_hz": _NAN if park is None else park,
            "flux_side": side,
            "n_points": int(n),
            "n_valid_points": int(valid.sum()),
            "flux_bias": x.tolist(),
            "fringe_hz": fringe.tolist(),
            "fringe_stderr_hz": fringe_err.tolist(),
            "delta_f_hz": delta.tolist(),
            "delta_f_stderr_hz": delta_err.tolist(),
            "snr": snr.tolist(),
            "amplitude": amp.tolist(),
            "valid": valid.astype(int).tolist(),
            "reduction_method": red_method,
            "reduction_angle": red_angle,
            "poly_center": _NAN, "poly_coeffs": [_NAN, _NAN, _NAN],
            "curvature_hz_per_v2": _NAN, "curvature_stderr": _NAN,
            "apex_flux": _NAN, "apex_flux_stderr": _NAN,
            "apex_delta_f_hz": _NAN, "apex_delta_f_stderr_hz": _NAN, "apex_f01_hz": _NAN,
            "apex_not_bracketed": 1,
            "park_flux": _NAN, "park_flux_stderr": _NAN, "park_slope_hz_per_v": _NAN,
            "park_f01_hz": _NAN, "park_out_of_window": 1 if park is not None else 0,
            "park_roots": [],
            "fold_suspected": 0,
            "success": False,
            "signal": sig,
        }
        if valid.sum() < MIN_VALID:
            return res

        xv, dv, ev = x[valid], delta[valid], delta_err[valid]
        order = np.argsort(xv, kind="stable")  # order-agnostic fit input
        xv, dv, ev = xv[order], dv[order], ev[order]
        x_c = float(np.mean(xv))
        u = xv - x_c
        floor = np.nanmedian(ev[np.isfinite(ev)]) if np.isfinite(ev).any() else 1.0
        w = 1.0 / np.where(np.isfinite(ev) & (ev > 0), ev, floor)
        try:
            coef, cov = np.polyfit(u, dv, 2, w=w, cov=True)
        except (np.linalg.LinAlgError, ValueError):
            return res
        res["poly_center"] = x_c
        res["poly_coeffs"] = [float(v) for v in coef]
        res["curvature_hz_per_v2"] = float(coef[0])
        res["curvature_stderr"] = float(np.sqrt(max(cov[0, 0], 0.0)))

        lo, hi = float(xv[0]), float(xv[-1])
        u0, su0, h0, sh0 = _vertex(coef, cov)
        apex_ok = (coef[0] < 0 and coef[0] + 2 * res["curvature_stderr"] < 0
                   and lo <= x_c + u0 <= hi)
        if coef[0] != 0:
            res.update(apex_flux=x_c + u0, apex_flux_stderr=su0,
                       apex_delta_f_hz=h0, apex_delta_f_stderr_hz=sh0,
                       apex_f01_hz=_NAN if drive is None else drive + h0)
        res["apex_not_bracketed"] = 0 if apex_ok else 1

        # the fitted fringe must stay on one side of zero across the window
        grid = np.linspace(lo - x_c, hi - x_c, 201)
        fitted = sgn * (ramp + np.polyval(coef, grid))
        res["fold_suspected"] = int(np.min(fitted) < 2.0 / span)

        if park is None:
            res["success"] = bool(apex_ok)
            return res

        roots = [r for r in _roots(coef, park - drive) if lo - x_c <= r <= hi - x_c]
        res["park_roots"] = [x_c + r for r in roots]
        if not roots:
            return res
        if len(roots) == 1:
            r = roots[0]
        elif side == "lower":
            r = roots[0]
        elif side == "upper":
            r = roots[1]
        else:
            r = min(roots, key=lambda v: abs(x_c + v - nearest_to))
        slope = 2 * coef[0] * r + coef[1]
        grad = -np.array([r * r, r, 1.0]) / slope if slope != 0 else np.full(3, _NAN)
        sr = float(np.sqrt(max(grad @ cov @ grad, 0.0))) if slope != 0 else _NAN
        res.update(park_flux=x_c + r, park_flux_stderr=sr, park_slope_hz_per_v=float(slope),
                   park_f01_hz=park, park_out_of_window=0, success=True)
        return res

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "signal"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        dataset = ascending(dataset, "flux_bias", "idle_time")
        x = np.asarray(results["flux_bias"], dtype=float)
        t = np.asarray(dataset["idle_time"].values, dtype=float)
        coef = np.asarray(results["poly_coeffs"], dtype=float)
        x_c = float(results["poly_center"])
        if np.all(np.isfinite(coef)) and np.isfinite(x_c) and x.size:
            fine = np.linspace(float(np.min(x)), float(np.max(x)), 201)
            fit_curve = np.polyval(coef, fine - x_c)
        else:
            fine = np.linspace(float(np.min(x)), float(np.max(x)), 2) if x.size else np.zeros(2)
            fit_curve = np.full(fine.size, _NAN)
        scalar_keys = ("question", "flux_side", "ramp_detuning_hz", "drive_freq_hz",
                       "park_frequency_hz", "n_valid_points", "curvature_hz_per_v2",
                       "apex_flux", "apex_flux_stderr", "apex_delta_f_hz",
                       "apex_f01_hz", "apex_not_bracketed", "park_flux",
                       "park_flux_stderr", "park_out_of_window", "fold_suspected",
                       "reduction_method", "reduction_angle")
        attrs = {k: results[k] for k in scalar_keys}
        attrs["success"] = int(bool(results["success"]))
        return xr.Dataset(
            {
                "signal": (("flux_bias", "idle_time"), np.asarray(results["signal"], dtype=float)),
                "fringe_hz": ("flux_bias", np.asarray(results["fringe_hz"], dtype=float)),
                "delta_f_hz": ("flux_bias", np.asarray(results["delta_f_hz"], dtype=float)),
                "delta_f_stderr_hz": ("flux_bias", np.asarray(results["delta_f_stderr_hz"],
                                                             dtype=float)),
                "valid": ("flux_bias", np.asarray(results["valid"], dtype=np.int32)),
                "snr": ("flux_bias", np.asarray(results["snr"], dtype=float)),
                "fit_delta_f_hz": ("flux_fine", fit_curve),
            },
            coords={"flux_bias": x, "idle_time": t, "flux_fine": fine},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures({
            "fringe_map": lambda: plot_fringe_map(plot_data),
            "flux_curve": lambda: plot_flux_curve(plot_data),
        }, label=self.estimator_name)
