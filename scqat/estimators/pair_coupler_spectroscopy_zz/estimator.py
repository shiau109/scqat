"""The coupler's 0-1 frequency, read out through its ZZ with a neighbour.

Dataset contract (one qubit pair; the acquisition layer splits targets off):

    coords   ``tone_freq_hz`` (Hz)  - the ABSOLUTE tone frequency, in ANY order
                                     (canonicalized on entry, so every artifact is
                                     ascending)
             ``pi_played``          - 1 = the pi member got its pi after the tone,
                                     0 = the reference arm (same shot, a wait instead)
             ``joint_state``        - ``"00"/"01"/"10"/"11"``, leftmost digit = HIGH
    vars     ``joint_population`` (joint_state, tone_freq_hz, pi_played)

The signal
----------
A tone on one member's drive line excites the coupler when it hits a coupler
transition; after the tone the OTHER member (the pi member) gets a pi. With the
coupler excited, the qubit-coupler ZZ pulls the pi member off its drive frequency and
a pi narrower than the ZZ misses, so the pi member's excited population DIPS at the
coupler's lines. The reference arm is the same shot without the pi; its difference
``D = P_pi - P_reference`` cancels what the pi member's readout sees of the coupler
directly (a neighbour's readout can, 5Q4C q1 sees q1_q2_c) and slow drift.

* **Lines** are the dips of ``D`` (``tools.coupler_ladder.find_lines`` on ``-D``),
  each at least ``min_fwhm_steps`` sweep steps wide. The tone is off before the pi,
  so only what the tone LEFT BEHIND can spoil it - every dip is the coupler's (or,
  rarely, a TLS's).
* **f01 is the highest line** and every other must sit on its multi-photon ladder
  (``tools.coupler_ladder.read_ladder``); a line off it is ``unexplained_lines``.
* ``pi_contrast`` = the median of ``D`` - the pi's efficiency times the readout
  contrast (about 0.85 for a good pi). Reported only: a poor pi makes the dips
  shallower, not displaced.

The result does not depend on the order of either axis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.pair_coupler_spectroscopy_zz.visualization import plot_spectrum
from scqat.tools.coupler_ladder import (
    ALPHA_RANGE_HZ, LADDER_TOL_HZ, find_lines, lines_curve, read_ladder,
)
from scqat.tools.peak_fit import robust_noise
from scqat.tools.sweep_order import ascending

AXIS = "tone_freq_hz"
ARM = "pi_played"
ROLES = ("high", "low")

#: joint-state labels whose digit for the role is 1 (label order: high, low)
_EXCITED = {"high": ("10", "11"), "low": ("01", "11")}

_NAN = float("nan")


def _marginal(joint: xr.DataArray, role: str) -> xr.DataArray:
    return sum(joint.sel(joint_state=label) for label in _EXCITED[role])


class PairCouplerSpectroscopyZZEstimator(BaseEstimator):
    """Coupler f01 (and alpha) from the dips a ZZ-spoiled pi leaves on a neighbour."""

    estimator_name = "pair_coupler_spectroscopy_zz"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError("pair_coupler_spectroscopy_zz requires a 'joint_population' "
                             "variable.")
        for coord in (AXIS, ARM, "joint_state"):
            if coord not in dataset.coords:
                raise ValueError(f"pair_coupler_spectroscopy_zz requires a '{coord}' "
                                 f"coordinate.")
        arms = {int(v) for v in dataset[ARM].values}
        if arms != {0, 1}:
            raise ValueError(f"pair_coupler_spectroscopy_zz needs both arms "
                             f"pi_played = 1 and 0, got {sorted(arms)}")
        missing = {"00", "01", "10", "11"} - {str(v) for v in dataset["joint_state"].values}
        if missing:
            raise ValueError(f"pair_coupler_spectroscopy_zz: joint_state lacks "
                             f"{sorted(missing)}")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Pi member's pi-minus-reference population -> dips -> the ladder's top.

        Kwargs - flat and fully owned; unknown names raise:
            pi_member (str): ``'high'`` (default) / ``'low'`` - the member that got
                the pi (the tone rode on the other one's line).
            min_snr (float): significance of a dip's depth (default 6).
            prominence (float): relative prominence gate (``fit_peaks``, default 0.1).
            min_fwhm_steps (float): narrowest line kept, in sweep steps (default 2).
            alpha_range_hz (tuple): allowed anharmonicity (default (-400e6, -50e6)).
            ladder_tol_hz (float): smallest rung tolerance (default 15e6).
            lo_hz (float, optional): the LO the tone rode on - reported, not used.
            high_name, low_name (str): member names for the figure labels.
        """
        pi_member = str(kwargs.pop("pi_member", "high"))
        min_snr = float(kwargs.pop("min_snr", 6.0))
        prominence = float(kwargs.pop("prominence", 0.1))
        min_fwhm_steps = float(kwargs.pop("min_fwhm_steps", 2.0))
        alpha_range = tuple(float(a) for a in kwargs.pop("alpha_range_hz", ALPHA_RANGE_HZ))
        ladder_tol = float(kwargs.pop("ladder_tol_hz", LADDER_TOL_HZ))
        lo_hz = kwargs.pop("lo_hz", None)
        names = {"high": str(kwargs.pop("high_name", "high")),
                 "low": str(kwargs.pop("low_name", "low"))}
        if kwargs:
            raise ValueError(f"pair_coupler_spectroscopy_zz: unknown kwargs {sorted(kwargs)}")
        if pi_member not in ROLES:
            raise ValueError(f"pi_member must be one of {ROLES}, got {pi_member!r}")
        if len(alpha_range) != 2:
            raise ValueError(f"alpha_range_hz must be (low, high), got {alpha_range}")
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, AXIS, ARM)

        freq = np.asarray(dataset[AXIS].values, dtype=float)
        joint = dataset["joint_population"].transpose("joint_state", ARM, AXIS)
        marg = {role: _marginal(joint, role) for role in ROLES}
        p_pi = np.asarray(marg[pi_member].sel({ARM: 1}).values, dtype=float)
        p_ref = np.asarray(marg[pi_member].sel({ARM: 0}).values, dtype=float)
        diff = p_pi - p_ref

        res: Dict[str, Any] = {
            "pi_member": pi_member,
            "min_snr": min_snr, "prominence": prominence,
            "min_fwhm_steps": min_fwhm_steps,
            "alpha_min_hz": min(alpha_range), "alpha_max_hz": max(alpha_range),
            "ladder_tol_hz": ladder_tol,
            "lo_hz": _NAN if lo_hz is None else float(lo_hz),
            "high_name": names["high"], "low_name": names["low"],
            "n_points": int(freq.size),
            "step_hz": float(np.median(np.diff(freq))) if freq.size > 1 else _NAN,
            "pi_contrast": float(np.median(diff)) if freq.size else _NAN,
            "f_c_hz": _NAN, "f_c_stderr_hz": _NAN, "fwhm_hz": _NAN,
            "dip_depth": _NAN, "snr": _NAN,
            "alpha_hz": _NAN, "alpha_stderr_hz": _NAN,
            "f02_half_hz": _NAN, "f03_third_hz": _NAN,
            "n_lines": 0, "n_ladder_lines": 0,
            "lines_hz": [], "unexplained_lines_hz": [],
            "no_line": 1, "unexplained_lines": 0, "peak_at_edge": 0,
            "success": False,
            # plot fodder (dropped from the metadata)
            "tone_freq_hz": freq,
            "joint_state": [str(v) for v in joint["joint_state"].values],
            "joint_population": np.asarray(joint.values, dtype=float),
            "marginal": np.stack([np.asarray(marg[r].values, dtype=float) for r in ROLES]),
            "difference": diff,
            "fit_curve": np.full(freq.size, _NAN),
        }
        if freq.size < 5:
            return res

        # the dips of D are the peaks of -D (tools.coupler_ladder.find_lines)
        lines, fit = find_lines(freq, -diff, min_snr=min_snr, prominence=prominence,
                                min_fwhm_steps=min_fwhm_steps)
        res["n_lines"] = len(lines)
        res["lines_hz"] = [float(p["full_freq"]) for p in lines]
        if not lines:
            return res
        res["fit_curve"] = -lines_curve(freq, lines, fit)

        ladder = read_ladder(lines, alpha_range_hz=alpha_range, tol_hz=ladder_tol)
        top = ladder["f01"]
        noise = robust_noise(fit["signal_corrected"])
        f_c, fwhm = float(top["full_freq"]), float(top["fwhm"])
        res.update(
            f_c_hz=f_c, f_c_stderr_hz=float(top["detuning_err"]), fwhm_hz=fwhm,
            dip_depth=float(top["amplitude"]),
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
        res["success"] = bool(not res["unexplained_lines"] and not res["peak_at_edge"])
        return res

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bulky = ("tone_freq_hz", "joint_population", "marginal", "difference", "fit_curve")
        return {k: v for k, v in results.items() if k not in bulky}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        scalar = [k for k, v in results.items()
                  if isinstance(v, (int, float, str, bool, np.floating, np.integer))
                  and k != "success"]
        attrs = {k: results[k] for k in scalar}
        attrs["success"] = int(bool(results["success"]))
        for key in ("lines_hz", "unexplained_lines_hz"):
            attrs[key] = np.asarray(results[key], dtype=float)
        return xr.Dataset(
            {
                "joint_population": (("joint_state", ARM, AXIS),
                                     np.asarray(results["joint_population"], dtype=float)),
                "marginal": (("member", ARM, AXIS),
                             np.asarray(results["marginal"], dtype=float)),
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
