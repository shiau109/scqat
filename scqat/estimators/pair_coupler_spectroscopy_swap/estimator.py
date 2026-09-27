"""The coupler's 0-1 frequency, read out by swapping its excitation into a neighbour.

Dataset contract (one qubit pair; the acquisition layer splits targets off):

    coords   ``tone_freq_hz`` (Hz)  - the ABSOLUTE tone frequency, in ANY order
                                     (canonicalized on entry, so every artifact is
                                     ascending)
             ``ramp_played``        - 1 = the swap ramp was played after the tone,
                                     0 = the reference arm (same shot, no ramp)
             ``joint_state``        - ``"00"/"01"/"10"/"11"``, leftmost digit = HIGH
    vars     ``joint_population`` (joint_state, tone_freq_hz, ramp_played)

The signal
----------
A tone on one member's drive line excites the coupler when it hits a coupler
transition. In the ramp arm a slow flux ramp then carries the coupler excitation
adiabatically into a member (and a sudden return leaves it there). WHICH member
receives it differs from pair to pair (5Q4C: q1 of q1_q2, q2 of q2_q3), so the
signal is the TOTAL excitation ``T = 1 - P00`` of each arm. The reference arm is
the same shot without the ramp.

* **Lines** are the peaks of the ramp arm's ``T_r`` (``tools.peak_fit.fit_peaks``,
  merge off), each gated on its OWN height against the trace noise (``min_snr``),
  on a width of at least ``min_fwhm_steps`` sweep steps and at most a quarter of
  the window. A line only one or two points wide is under-sampled: re-measure it
  with a finer step, the estimator does not chase it.
* **Coupler lines** are the lines the ramp CHANGES: ``D = T_r - T_0`` projected on
  the line's own Lorentzian differs from zero by ``min_snr`` of its noise, either
  sign. The rest are the members' own features (``member_lines_hz``). A line the
  reference arm shows too is NOT disqualified - a neighbour's readout can see the
  coupler state directly (5Q4C q1 sees q1_q2_c).
* **f01 is the HIGHEST coupler line.** Driven hard, the coupler shows its
  multi-photon ladder below it, ``f02/2 = f01 + alpha/2`` and ``f03/3 ~ f01 +
  alpha``, and through the more strongly coupled line the f01 broadens until
  f02/2 is the strongest (5Q4C q2_q3 through q3's line), so height never picks
  f01. Every other coupler line must sit on f01's ladder for ONE alpha in
  ``alpha_range_hz`` within ``max(ladder_tol_hz, the larger FWHM)``; alpha is
  reported from f02/2. A coupler line off the ladder is ``unexplained_lines``.

The result does not depend on the order of either axis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.pair_coupler_spectroscopy_swap.visualization import plot_spectrum
from scqat.tools.coupler_ladder import find_lines, lines_curve, read_ladder
from scqat.tools.fit_lorentzian import lorentzian
from scqat.tools.peak_fit import robust_noise
from scqat.tools.sweep_order import ascending

AXIS = "tone_freq_hz"
ARM = "ramp_played"
ROLES = ("high", "low")

#: joint-state labels whose digit for the role is 1 (label order: high, low)
_EXCITED = {"high": ("10", "11"), "low": ("01", "11")}

_NAN = float("nan")


def _marginal(joint: xr.DataArray, role: str) -> xr.DataArray:
    return sum(joint.sel(joint_state=label) for label in _EXCITED[role])


def _projection(freq: np.ndarray, trace: np.ndarray, line: dict) -> tuple[float, float]:
    """Height of ``trace`` along the line's own unit Lorentzian, and its noise.

    Least squares of ``trace - median(trace)`` on the shape within three FWHM of the
    centre: the height of the same line in another trace, with every point of the
    line contributing instead of the one nearest the centre."""
    center, fwhm = float(line["full_freq"]), float(line["fwhm"])
    near = np.abs(freq - center) <= 3 * fwhm
    shape = lorentzian(freq[near] - center, 0.0, 1.0, fwhm / 2, 0.0)
    norm = float(np.sum(shape ** 2))
    if norm <= 0:
        return _NAN, _NAN
    residual = trace - float(np.median(trace))
    height = float(np.sum(residual[near] * shape) / norm)
    return height, float(robust_noise(trace) / np.sqrt(norm))


class PairCouplerSpectroscopySwapEstimator(BaseEstimator):
    """Coupler f01 (and alpha) from the multi-photon ladder of the ramp-changed lines."""

    estimator_name = "pair_coupler_spectroscopy_swap"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError("pair_coupler_spectroscopy_swap requires a 'joint_population' "
                             "variable.")
        for coord in (AXIS, ARM, "joint_state"):
            if coord not in dataset.coords:
                raise ValueError(f"pair_coupler_spectroscopy_swap requires a '{coord}' "
                                 f"coordinate.")
        arms = {int(v) for v in dataset[ARM].values}
        if arms != {0, 1}:
            raise ValueError(f"pair_coupler_spectroscopy_swap needs both arms "
                             f"ramp_played = 1 and 0, got {sorted(arms)}")
        missing = {"00", "01", "10", "11"} - {str(v) for v in dataset["joint_state"].values}
        if missing:
            raise ValueError(f"pair_coupler_spectroscopy_swap: joint_state lacks "
                             f"{sorted(missing)}")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Total excitation per arm -> lines -> coupler lines -> the ladder's top.

        Kwargs - flat and fully owned; unknown names raise:
            min_snr (float): significance of a line's height and of its ramp change
                (default 6).
            prominence (float): relative prominence gate (``fit_peaks``, default 0.1).
            min_fwhm_steps (float): narrowest line kept, in sweep steps (default 2).
            alpha_range_hz (tuple): allowed anharmonicity (default (-400e6, -50e6)).
            ladder_tol_hz (float): smallest rung tolerance (default 15e6).
            tone_on (str, optional): ``'high'``/``'low'``, the member whose line
                carried the tone - reported, not used.
            lo_hz (float, optional): the LO the tone rode on - reported, not used.
            ramp_duration_ns (float, optional): the played ramp length - reported.
            high_name, low_name (str): member names for the figure labels.
        """
        min_snr = float(kwargs.pop("min_snr", 6.0))
        prominence = float(kwargs.pop("prominence", 0.1))
        min_fwhm_steps = float(kwargs.pop("min_fwhm_steps", 2.0))
        alpha_range = tuple(float(a) for a in kwargs.pop("alpha_range_hz", (-400e6, -50e6)))
        ladder_tol = float(kwargs.pop("ladder_tol_hz", 15e6))
        tone_on = kwargs.pop("tone_on", None)
        lo_hz = kwargs.pop("lo_hz", None)
        ramp_ns = kwargs.pop("ramp_duration_ns", None)
        names = {"high": str(kwargs.pop("high_name", "high")),
                 "low": str(kwargs.pop("low_name", "low"))}
        if kwargs:
            raise ValueError(f"pair_coupler_spectroscopy_swap: unknown kwargs {sorted(kwargs)}")
        if tone_on is not None and tone_on not in ROLES:
            raise ValueError(f"tone_on must be one of {ROLES}, got {tone_on!r}")
        if len(alpha_range) != 2:
            raise ValueError(f"alpha_range_hz must be (low, high), got {alpha_range}")
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, AXIS, ARM)

        freq = np.asarray(dataset[AXIS].values, dtype=float)
        joint = dataset["joint_population"].transpose("joint_state", ARM, AXIS)
        total = 1.0 - joint.sel(joint_state="00")
        t_ramp = np.asarray(total.sel({ARM: 1}).values, dtype=float)
        t_ref = np.asarray(total.sel({ARM: 0}).values, dtype=float)
        diff = t_ramp - t_ref
        marg = {role: _marginal(joint, role) for role in ROLES}

        res: Dict[str, Any] = {
            "tone_on": "" if tone_on is None else str(tone_on),
            "min_snr": min_snr, "prominence": prominence,
            "min_fwhm_steps": min_fwhm_steps,
            "alpha_min_hz": min(alpha_range), "alpha_max_hz": max(alpha_range),
            "ladder_tol_hz": ladder_tol,
            "lo_hz": _NAN if lo_hz is None else float(lo_hz),
            "ramp_duration_ns": _NAN if ramp_ns is None else float(ramp_ns),
            "high_name": names["high"], "low_name": names["low"],
            "n_points": int(freq.size),
            "step_hz": float(np.median(np.diff(freq))) if freq.size > 1 else _NAN,
            "f_c_hz": _NAN, "f_c_stderr_hz": _NAN, "fwhm_hz": _NAN,
            "peak_height": _NAN, "snr": _NAN,
            "alpha_hz": _NAN, "alpha_stderr_hz": _NAN,
            "f02_half_hz": _NAN, "f03_third_hz": _NAN,
            "landing_high": _NAN, "landing_low": _NAN,
            "n_lines": 0, "n_coupler_lines": 0, "n_ladder_lines": 0,
            "coupler_lines_hz": [], "member_lines_hz": [], "unexplained_lines_hz": [],
            "no_line": 1, "unexplained_lines": 0, "peak_at_edge": 0,
            "success": False,
            # plot fodder (dropped from the metadata)
            "tone_freq_hz": freq,
            "joint_state": [str(v) for v in joint["joint_state"].values],
            "joint_population": np.asarray(joint.values, dtype=float),
            "marginal": np.stack([np.asarray(marg[r].values, dtype=float) for r in ROLES]),
            "total": np.stack([t_ref, t_ramp]),
            "difference": diff,
            "fit_curve": np.full(freq.size, _NAN),
        }
        if freq.size < 5:
            return res

        lines, fit = find_lines(freq, t_ramp, min_snr=min_snr, prominence=prominence,
                                min_fwhm_steps=min_fwhm_steps)
        res["n_lines"] = len(lines)
        if not lines:
            return res
        res["fit_curve"] = lines_curve(freq, lines, fit)

        coupler, member = [], []
        for p in lines:
            height, noise = _projection(freq, diff, p)
            p["ramp_change"] = height
            ok = np.isfinite(height) and noise > 0 and abs(height) >= min_snr * noise
            (coupler if ok else member).append(p)
        res["n_coupler_lines"] = len(coupler)
        res["coupler_lines_hz"] = [float(p["full_freq"]) for p in coupler]
        res["member_lines_hz"] = [float(p["full_freq"]) for p in member]
        if not coupler:
            return res

        ladder = read_ladder(coupler, alpha_range_hz=alpha_range, tol_hz=ladder_tol)
        top = ladder["f01"]
        noise = robust_noise(fit["signal_corrected"])
        f_c, fwhm = float(top["full_freq"]), float(top["fwhm"])
        res.update(
            f_c_hz=f_c, f_c_stderr_hz=float(top["detuning_err"]), fwhm_hz=fwhm,
            peak_height=float(top["amplitude"]),
            snr=float(top["amplitude"] / noise) if noise > 0 else _NAN,
            alpha_hz=ladder["alpha_hz"], alpha_stderr_hz=ladder["alpha_stderr_hz"],
            n_ladder_lines=ladder["n_ladder_lines"],
            unexplained_lines_hz=[float(p["full_freq"]) for p in ladder["unexplained"]],
            no_line=0, unexplained_lines=int(bool(ladder["unexplained"])),
            peak_at_edge=int(min(f_c - freq.min(), freq.max() - f_c) < fwhm),
        )
        for key, line in (("f02_half_hz", ladder["f02_half"]),
                          ("f03_third_hz", ladder["f03_third"])):
            if line is not None:
                res[key] = float(line["full_freq"])
        for role in ROLES:
            per_member = (np.asarray(marg[role].sel({ARM: 1}).values, dtype=float)
                          - np.asarray(marg[role].sel({ARM: 0}).values, dtype=float))
            res[f"landing_{role}"] = _projection(freq, per_member, top)[0]
        res["success"] = bool(not res["unexplained_lines"] and not res["peak_at_edge"])
        return res

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bulky = ("tone_freq_hz", "joint_population", "marginal", "total", "difference",
                 "fit_curve")
        return {k: v for k, v in results.items() if k not in bulky}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        scalar = [k for k, v in results.items()
                  if isinstance(v, (int, float, str, bool, np.floating, np.integer))
                  and k != "success"]
        attrs = {k: results[k] for k in scalar}
        attrs["success"] = int(bool(results["success"]))
        for key in ("coupler_lines_hz", "member_lines_hz", "unexplained_lines_hz"):
            attrs[key] = np.asarray(results[key], dtype=float)
        return xr.Dataset(
            {
                "joint_population": (("joint_state", ARM, AXIS),
                                     np.asarray(results["joint_population"], dtype=float)),
                "marginal": (("member", ARM, AXIS),
                             np.asarray(results["marginal"], dtype=float)),
                "total": ((ARM, AXIS), np.asarray(results["total"], dtype=float)),
                "difference": (AXIS, np.asarray(results["difference"], dtype=float)),
                "fit_curve": (AXIS, np.asarray(results["fit_curve"], dtype=float)),
            },
            coords={AXIS: np.asarray(results["tone_freq_hz"], dtype=float),
                    ARM: [0, 1], "joint_state": list(results["joint_state"]),
                    "member": list(ROLES)},
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
        return render_figures({"spectrum": lambda: plot_spectrum(plot_data)},
                              label=self.estimator_name)
