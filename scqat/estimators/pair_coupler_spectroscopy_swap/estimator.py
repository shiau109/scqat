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
A tone on the PROBE member's drive line excites the coupler when it hits the
coupler's frequency. In the ramp arm a slow flux ramp then carries the coupler
excitation adiabatically into the probe (and a sudden return leaves it there), so
the probe's excited population peaks at f_c. The reference arm is the same shot
without the ramp: the coupler keeps its excitation and the probe sees only what the
tone does to it directly (its own transitions, the readout, a TLS). So:

* the coupler line is the strongest peak of ``D = P_ramp - P_reference`` of the
  probe's marginal (``tools.peak_fit.fit_peaks`` on a real signal);
* a peak of the reference arm alone within a linewidth of it
  (``reference_feature``) means the line is not the coupler's - or not only;
* further peaks of ``D`` are listed (at high power the coupler's two-photon
  0-2 transition sits about half its anharmonicity below f_c).

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
from scqat.tools.fit_lorentzian import lorentzian
from scqat.tools.peak_fit import fit_peaks, robust_noise
from scqat.tools.sweep_order import ascending

AXIS = "tone_freq_hz"
ARM = "ramp_played"
ROLES = ("high", "low")

#: joint-state labels whose digit for the role is 1 (label order: high, low)
_EXCITED = {"high": ("10", "11"), "low": ("01", "11")}

_NAN = float("nan")


def _marginal(joint: xr.DataArray, role: str) -> xr.DataArray:
    return sum(joint.sel(joint_state=label) for label in _EXCITED[role])


def _peaks(freq: np.ndarray, signal: np.ndarray, *, min_snr: float,
           prominence: float, positive_only: bool) -> tuple[list[dict], dict]:
    """``fit_peaks`` on a real trace, on a detuning axis centred in the window,
    keeping only fits that are lines in their own right.

    ``fit_peaks``' own merge keeps the larger-AREA fit of two overlapping ones, so a
    broad sub-noise fit of a detected bump swallows a real narrow line beside it
    (0.015 x 178 MHz beat 0.26 x 2.6 MHz on the decoy fixture). So the merge is off
    here and every fit is gated on ITS amplitude against the trace noise
    (``min_snr``) and on a width below a quarter of the window, then duplicates are
    dropped strongest-first. ``positive_only`` keeps peaks only: a swap can only ADD
    excitation to the probe."""
    mid = 0.5 * (float(np.min(freq)) + float(np.max(freq)))
    res = fit_peaks(freq - mid, signal, full_freq=freq, min_snr=min_snr,
                    prominence=prominence, merge_factor=0.0)
    if positive_only and res["inverted"]:
        return [], res
    sign = -1.0 if res["inverted"] else 1.0
    noise = robust_noise(res["signal_corrected"])
    span = float(np.max(freq) - np.min(freq))
    good = [p for p in res["peaks"]
            if np.isfinite(p["amplitude"]) and sign * p["amplitude"] > 0
            and (noise <= 0 or abs(p["amplitude"]) >= min_snr * noise)
            and 0 < p["fwhm"] <= span / 4]
    kept: list[dict] = []
    for p in sorted(good, key=lambda p: abs(p["amplitude"]), reverse=True):
        if all(abs(p["full_freq"] - q["full_freq"]) >= max(p["fwhm"], q["fwhm"]) for q in kept):
            kept.append(p)
    return kept, res


class PairCouplerSpectroscopySwapEstimator(BaseEstimator):
    """Coupler frequency from the swap-arm minus reference-arm probe population."""

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
        """Probe marginal per arm -> difference -> the strongest peak.

        Kwargs - flat and fully owned; unknown names raise:
            probe (str): ``'high'`` (default) / ``'low'`` - the member the ramp
                swapped the coupler into.
            min_snr (float): significance gate of a peak (``fit_peaks``, default 6).
            prominence (float): relative prominence gate (``fit_peaks``, default 0.1).
            lo_hz (float, optional): the LO the tone rode on - reported, not used.
            ramp_duration_ns (float, optional): the played ramp length - reported.
            high_name, low_name (str): member names for the figure labels.
        """
        probe = str(kwargs.pop("probe", "high"))
        min_snr = float(kwargs.pop("min_snr", 6.0))
        prominence = float(kwargs.pop("prominence", 0.1))
        lo_hz = kwargs.pop("lo_hz", None)
        ramp_ns = kwargs.pop("ramp_duration_ns", None)
        names = {"high": str(kwargs.pop("high_name", "high")),
                 "low": str(kwargs.pop("low_name", "low"))}
        if kwargs:
            raise ValueError(f"pair_coupler_spectroscopy_swap: unknown kwargs {sorted(kwargs)}")
        if probe not in ROLES:
            raise ValueError(f"probe must be one of {ROLES}, got {probe!r}")
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, AXIS, ARM)

        freq = np.asarray(dataset[AXIS].values, dtype=float)
        joint = dataset["joint_population"].transpose("joint_state", ARM, AXIS)
        marg = {role: _marginal(joint, role) for role in ROLES}
        p_ramp = np.asarray(marg[probe].sel({ARM: 1}).values, dtype=float)
        p_ref = np.asarray(marg[probe].sel({ARM: 0}).values, dtype=float)
        diff = p_ramp - p_ref

        res: Dict[str, Any] = {
            "probe": probe, "min_snr": min_snr, "prominence": prominence,
            "lo_hz": _NAN if lo_hz is None else float(lo_hz),
            "ramp_duration_ns": _NAN if ramp_ns is None else float(ramp_ns),
            "high_name": names["high"], "low_name": names["low"],
            "n_points": int(freq.size),
            "step_hz": float(np.median(np.diff(freq))) if freq.size > 1 else _NAN,
            "f_c_hz": _NAN, "f_c_stderr_hz": _NAN, "fwhm_hz": _NAN,
            "peak_height": _NAN, "snr": _NAN,
            "other_peaks_hz": [], "reference_peaks_hz": [], "n_peaks": 0,
            "no_peak": 1, "multiple_peaks": 0, "reference_feature": 0, "peak_at_edge": 0,
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

        peaks, fit = _peaks(freq, diff, min_snr=min_snr, prominence=prominence,
                            positive_only=True)
        res["n_peaks"] = len(peaks)
        if peaks:
            main = max(peaks, key=lambda p: p["amplitude"])
            noise = robust_noise(fit["signal_corrected"])
            f_c = float(main["full_freq"])
            fwhm = float(main["fwhm"])
            res.update(
                f_c_hz=f_c, f_c_stderr_hz=float(main["detuning_err"]), fwhm_hz=fwhm,
                peak_height=float(main["amplitude"]),
                snr=float(main["amplitude"] / noise) if noise > 0 else _NAN,
                other_peaks_hz=[float(p["full_freq"]) for p in peaks if p is not main],
                no_peak=0, multiple_peaks=int(len(peaks) > 1),
                peak_at_edge=int(min(f_c - freq.min(), freq.max() - f_c) < fwhm),
            )
            mid = 0.5 * (freq.min() + freq.max())
            res["fit_curve"] = fit["baseline"] + lorentzian(
                freq - mid, main["detuning"], main["amplitude"], fwhm / 2, main["offset"])
            ref_peaks, _ = _peaks(freq, p_ref, min_snr=min_snr, prominence=prominence,
                                  positive_only=False)
            res["reference_peaks_hz"] = [float(p["full_freq"]) for p in ref_peaks]
            res["reference_feature"] = int(any(
                abs(p["full_freq"] - f_c) < max(fwhm, p["fwhm"]) for p in ref_peaks))
        res["success"] = bool(not res["no_peak"] and not res["reference_feature"]
                              and not res["peak_at_edge"])
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
        for key in ("other_peaks_hz", "reference_peaks_hz"):
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
