"""Flux crosstalk onto a qubit, from its flux apex at several source amplitudes.

Dataset contract (one target; the acquisition layer splits targets off):

    coords   ``source_flux`` (V)  - the amplitude on the SOURCE flux line (another
                                    qubit's or a coupler's), in ANY order
             ``flux_bias`` (V)    - the amplitude on the target's OWN flux line, in
                                    any order
             ``idle_time`` (s)    - the Ramsey idle, in any order
    vars     ``signal`` (source_flux, flux_bias, idle_time)  - a real signal (e.g.
                                    an averaged population), OR complex ``IQdata`` /
                                    ``I`` + ``Q``, reduced here by ONE global axial
                                    projection (stored ``ref_pos_*`` centers when
                                    present)

All three axes are canonicalized on entry, so every artifact is ascending and
order-free. The estimator is blind to the flux FRAME of both lines: positions are
reported on the axes as given.

Model
-----
With the target's own line at ``a`` and the source line at ``b`` during the idle,
the target sits at::

    f(a, b) = h(b) - c * (a - a0 + m * b)^2

``m`` is the crosstalk: one volt on the source line threads the target's SQUID like
``m`` volts on its own line. ``h(b)`` is the apex HEIGHT, which a source may move
without any crosstalk (a coupler's dispersive shift on its neighbour).

Each source amplitude is read exactly like ``qubit_ramsey_flux_pulse``: one fringe
per own-line value, ``delta_f = sign(D) * F - D`` with the signed virtual detuning
``D``, then a weighted quadratic whose vertex is the apex
(:mod:`scqat.tools.local_arch`). The apex POSITION then follows a straight line in
``b`` with slope ``-m`` (:mod:`scqat.tools.flux_crosstalk`). The ruler is the
target's own line, so neither the arch model nor ``h(b)`` enters ``m``.

A source point counts only when its quadratic has the apex inside the own-line
window and its fitted fringe never crosses zero. The line needs three such points.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_positions, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.flux_crosstalk import MIN_POINTS, fit_crosstalk_line
from scqat.tools.iq_reduce import AXIAL_KNOBS, axial, axis_angle, validate_iq_reduce_kwargs
from scqat.tools.local_arch import fit_local_arch, fringe_deltas
from scqat.tools.sweep_order import ascending
from scqat.estimators.qubit_ramsey_flux_crosstalk.visualization import (
    plot_apex_vs_source,
    plot_flux_curves,
    plot_fringe_maps,
)

#: the three swept dims, in the order every array here is held
DIMS = ("source_flux", "flux_bias", "idle_time")
#: an apex this far off the fitted line (V) marks the run nonlinear
LINEARITY_TOL_V = 1.0e-4

_NAN = float("nan")


def _global_signal(dataset: xr.Dataset, axial_kwargs: dict) -> tuple[np.ndarray, str, float]:
    """(source_flux, flux_bias, idle_time) real signal + reduction provenance."""
    if "signal" in dataset.data_vars:
        sig = dataset["signal"].transpose(*DIMS).values.astype(float)
        return sig, "signal", _NAN
    iq = with_iqdata(dataset)["IQdata"].transpose(*DIMS).values
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


class QubitRamseyFluxCrosstalkEstimator(BaseEstimator):
    """Signed flux crosstalk from the apex position at each source amplitude."""

    estimator_name = "qubit_ramsey_flux_crosstalk"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError("qubit_ramsey_flux_crosstalk requires a 'signal' variable, "
                             "or complex 'IQdata', or both 'I' and 'Q'.")
        for coord in DIMS:
            if coord not in dataset.coords:
                raise ValueError(f"qubit_ramsey_flux_crosstalk requires a '{coord}' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Per source amplitude: fringes -> delta_f(a) -> quadratic -> apex; then a
        weighted line through the apex positions.

        Kwargs - flat and fully owned; unknown names raise:
            ramp_detuning_hz (float, REQUIRED): the SIGNED virtual detuning the probe
                applied (fringe at ``|D + (f_q - f_drive)|``).
            drive_freq_hz (float, optional): the drive frequency, recorded only.
            min_snr (float): periodogram peak / median power a fringe needs to
                count (default 30).
            min_source_points (int): valid source amplitudes the line needs
                (default and floor 3).
            linearity_tol_v (float): an apex further than this from the line sets
                ``nonlinear_suspected`` (default 1e-4 V).
            angle, positions, pca_sign: the axial IQ-reduction knobs.
        """
        if "ramp_detuning_hz" not in kwargs:
            raise ValueError("qubit_ramsey_flux_crosstalk needs ramp_detuning_hz (the "
                             "signed virtual detuning the probe applied)")
        ramp = float(kwargs.pop("ramp_detuning_hz"))
        if ramp == 0 or not np.isfinite(ramp):
            raise ValueError(f"ramp_detuning_hz must be finite and nonzero, got {ramp}")
        drive = kwargs.pop("drive_freq_hz", None)
        drive = None if drive is None else float(drive)
        min_snr = float(kwargs.pop("min_snr", 30.0))
        min_points = int(kwargs.pop("min_source_points", MIN_POINTS))
        tol = float(kwargs.pop("linearity_tol_v", LINEARITY_TOL_V))
        if min_points < MIN_POINTS:
            raise ValueError(f"min_source_points must be >= {MIN_POINTS}, got {min_points}")
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, *DIMS)

        b = np.asarray(dataset["source_flux"].values, dtype=float)
        x = np.asarray(dataset["flux_bias"].values, dtype=float)
        t = np.asarray(dataset["idle_time"].values, dtype=float)
        sig, red_method, red_angle = _global_signal(dataset, kwargs)

        nb, nx = b.size, x.size
        delta = np.full((nb, nx), _NAN)
        delta_err = np.full((nb, nx), _NAN)
        fringe = np.full((nb, nx), _NAN)
        snr = np.full((nb, nx), _NAN)
        valid = np.zeros((nb, nx), dtype=bool)
        centers = np.full(nb, _NAN)
        coeffs = np.full((nb, 3), _NAN)
        apex = np.full(nb, _NAN)
        apex_err = np.full(nb, _NAN)
        height = np.full(nb, _NAN)
        height_err = np.full(nb, _NAN)
        curvature = np.full(nb, _NAN)
        bracketed = np.zeros(nb, dtype=bool)
        folded = np.zeros(nb, dtype=bool)
        for j in range(nb):
            d = fringe_deltas(t, sig[j], ramp, min_snr=min_snr)
            delta[j], delta_err[j] = d["delta_f_hz"], d["delta_f_stderr_hz"]
            fringe[j], snr[j], valid[j] = d["fringe_hz"], d["snr"], d["valid"]
            arch = fit_local_arch(x, d["delta_f_hz"], d["delta_f_stderr_hz"], d["valid"],
                                  ramp_detuning_hz=ramp, span_s=d["span_s"])
            if not arch["fitted"]:
                continue
            centers[j], coeffs[j] = arch["poly_center"], arch["poly_coeffs"]
            apex[j], apex_err[j] = arch["apex_flux"], arch["apex_flux_stderr"]
            height[j], height_err[j] = arch["apex_delta_f_hz"], arch["apex_delta_f_stderr_hz"]
            curvature[j] = arch["curvature_hz_per_v2"]
            bracketed[j] = arch["apex_not_bracketed"] == 0
            folded[j] = bool(arch["fold_suspected"])

        source_valid = bracketed & ~folded & np.isfinite(apex)
        line = fit_crosstalk_line(b, apex, apex_err, source_valid, min_points=min_points)
        used = source_valid

        def over_used(values: np.ndarray, reduce) -> float:
            return float(reduce(values[used])) if used.any() else _NAN

        max_residual = line["max_residual"]
        return {
            "ramp_detuning_hz": ramp,
            "drive_freq_hz": _NAN if drive is None else drive,
            "n_source_points": int(nb),
            "n_valid_source_points": int(used.sum()),
            "n_flux_points": int(nx),
            "flux_crosstalk": line["crosstalk"],
            "flux_crosstalk_stderr": line["crosstalk_stderr"],
            "apex_flux_at_zero_source": line["apex_at_zero"],
            "apex_flux_at_zero_source_stderr": line["apex_at_zero_stderr"],
            "curvature_hz_per_v2": over_used(curvature, np.median),
            "apex_height_span_hz": over_used(height, np.ptp),
            "line_residual_rms_v": line["residual_rms"],
            "line_chi2_red": line["chi2_red"],
            "line_max_residual_v": max_residual,
            "linearity_tol_v": tol,
            "nonlinear_suspected": int(np.isfinite(max_residual) and max_residual > tol),
            "fold_suspected": int(folded.any()),
            "n_apex_not_bracketed": int(nb - bracketed.sum()),
            "reduction_method": red_method,
            "reduction_angle": red_angle,
            "source_flux": b.tolist(),
            "flux_bias": x.tolist(),
            "apex_flux": apex.tolist(),
            "apex_flux_stderr": apex_err.tolist(),
            "apex_delta_f_hz": height.tolist(),
            "apex_delta_f_stderr_hz": height_err.tolist(),
            "curvature_per_source_hz_per_v2": curvature.tolist(),
            "source_valid": source_valid.astype(int).tolist(),
            "line_residual_v": line["residuals"].tolist(),
            "fringe_hz": fringe.tolist(),
            "delta_f_hz": delta.tolist(),
            "delta_f_stderr_hz": delta_err.tolist(),
            "snr": snr.tolist(),
            "valid": valid.astype(int).tolist(),
            "poly_center": centers.tolist(),
            "poly_coeffs": coeffs.tolist(),
            "success": bool(line["success"]),
            "signal": sig,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "signal"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        dataset = ascending(dataset, *DIMS)
        b = np.asarray(results["source_flux"], dtype=float)
        x = np.asarray(results["flux_bias"], dtype=float)
        t = np.asarray(dataset["idle_time"].values, dtype=float)
        centers = np.asarray(results["poly_center"], dtype=float)
        coeffs = np.asarray(results["poly_coeffs"], dtype=float)
        if x.size:
            fine = np.linspace(float(np.min(x)), float(np.max(x)), 101)
        else:
            fine = np.zeros(2)
        curves = np.full((b.size, fine.size), _NAN)
        for j in range(b.size):
            if np.all(np.isfinite(coeffs[j])) and np.isfinite(centers[j]):
                curves[j] = np.polyval(coeffs[j], fine - centers[j])
        if b.size:
            source_fine = np.linspace(float(np.min(b)), float(np.max(b)), 2)
        else:
            source_fine = np.zeros(2)
        m, a0 = results["flux_crosstalk"], results["apex_flux_at_zero_source"]
        line = a0 - m * source_fine if np.isfinite(m) and np.isfinite(a0) else np.full(2, _NAN)
        scalar_keys = ("ramp_detuning_hz", "drive_freq_hz", "n_source_points",
                       "n_valid_source_points", "flux_crosstalk", "flux_crosstalk_stderr",
                       "apex_flux_at_zero_source", "apex_flux_at_zero_source_stderr",
                       "curvature_hz_per_v2", "apex_height_span_hz", "line_residual_rms_v",
                       "line_chi2_red", "line_max_residual_v", "linearity_tol_v",
                       "nonlinear_suspected",
                       "fold_suspected", "n_apex_not_bracketed", "reduction_method",
                       "reduction_angle")
        attrs = {k: results[k] for k in scalar_keys}
        attrs["success"] = int(bool(results["success"]))
        per_point = ("source_flux", "flux_bias")
        return xr.Dataset(
            {
                "signal": (DIMS, np.asarray(results["signal"], dtype=float)),
                "delta_f_hz": (per_point, np.asarray(results["delta_f_hz"], dtype=float)),
                "delta_f_stderr_hz": (per_point, np.asarray(results["delta_f_stderr_hz"],
                                                            dtype=float)),
                "valid": (per_point, np.asarray(results["valid"], dtype=np.int32)),
                "fit_delta_f_hz": (("source_flux", "flux_fine"), curves),
                "apex_flux": ("source_flux", np.asarray(results["apex_flux"], dtype=float)),
                "apex_flux_stderr": ("source_flux", np.asarray(results["apex_flux_stderr"],
                                                               dtype=float)),
                "apex_delta_f_hz": ("source_flux", np.asarray(results["apex_delta_f_hz"],
                                                              dtype=float)),
                "apex_delta_f_stderr_hz": ("source_flux", np.asarray(
                    results["apex_delta_f_stderr_hz"], dtype=float)),
                "source_valid": ("source_flux", np.asarray(results["source_valid"],
                                                           dtype=np.int32)),
                "fit_apex_flux": ("source_fine", line),
            },
            coords={"source_flux": b, "flux_bias": x, "idle_time": t,
                    "flux_fine": fine, "source_fine": source_fine},
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
            "fringe_maps": lambda: plot_fringe_maps(plot_data),
            "flux_curves": lambda: plot_flux_curves(plot_data),
            "apex_vs_source": lambda: plot_apex_vs_source(plot_data),
        }, label=self.estimator_name)
