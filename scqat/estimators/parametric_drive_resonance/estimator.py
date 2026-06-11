"""
Parametric-Drive Resonance Estimator
=====================================
Extract the parametric-resonance line(s) from a 2-D
``amplitude_ratio`` × ``driving_frequency`` map measured at a **fixed** drive
time (the ``LCH_qubit_parametric_drive_fixed_time`` node). The signal is
:math:`P(|1\\rangle)` after the parametric drive; a resonance shows up as a
peak/dip in frequency whose position drifts with the drive amplitude.

This mirrors
:class:`~scqat.estimators.qubit_spectroscopy_flux.QubitSpectroscopyFluxEstimator`
with the axis roles ``flux_bias → amplitude_ratio`` and
``detuning → driving_frequency``: for every ``amplitude_ratio`` slice the
estimator delegates to
:class:`~scqat.estimators.qubit_spectroscopy.QubitSpectroscopyEstimator` (peak
detection + single-Lorentzian fit per peak) and collects every detected peak as
a **point-cloud** ``(amplitude_ratio, frequency, fwhm, amplitude)``.

Cleaning: a peak is kept (``good``) when its centre lies strictly inside the
swept frequency window and its ``fwhm`` / ``|amplitude|`` are not robust
(median/MAD) outliers across the pooled set of detected peaks.

Expected ``xarray.Dataset`` contract
-------------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).

Coordinates:
    - amplitude_ratio   : 1-D float array — parametric drive amplitude scale.
    - driving_frequency : 1-D float array — parametric drive frequency (Hz).
Data variables:
    - state / signal : (amplitude_ratio, driving_frequency) real P(|1⟩), **or**
    - IQdata / I, Q  : the complex demodulated signal.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_spectroscopy import QubitSpectroscopyEstimator
from scqat.tools.robust import mad_outliers
from scqat.estimators.parametric_drive_resonance.visualization import plot_parametric_map


class ParametricDriveResonanceEstimator(BaseEstimator):
    """Fit the parametric-resonance peak(s) at every drive amplitude and report
    them as a point-cloud over (``amplitude_ratio``, ``driving_frequency``)."""

    estimator_name = "parametric_drive_resonance"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("amplitude_ratio", "driving_frequency"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"ParametricDriveResonanceEstimator requires a '{coord}' coordinate."
                )

    @staticmethod
    def _signal_map(ds: xr.Dataset, signal_var: Optional[str]) -> np.ndarray:
        """2-D signal magnitude oriented (amplitude_ratio, driving_frequency)."""
        var = signal_var if signal_var is not None else "IQdata"
        return np.abs(ds[var].transpose("amplitude_ratio", "driving_frequency").values)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit and collect every parametric-resonance peak in every
        ``amplitude_ratio`` slice.

        Each slice (with ``driving_frequency`` renamed to ``detuning``) is handed
        to ``QubitSpectroscopyEstimator.extract_parameters``; ``kwargs`` such as
        ``prominence`` and ``max_peaks`` are forwarded to it. By default each
        slice is capped to its 4 most prominent peaks (``max_peaks=4``); pass
        ``max_peaks=None`` to keep every peak above the prominence threshold.

        Keyword arguments
        -----------------
        n_sigma : float, optional
            Robust-sigma threshold for the width / amplitude outlier test
            (default 3.0). Not forwarded to the per-slice estimator.
        signal_var : str, optional
            Data variable holding the real signal (auto-detected as ``signal``
            or ``state`` when omitted; otherwise the IQ path is used).
        max_peaks : int or None, optional
            Per-amplitude cap forwarded to the single-slice estimator. Default 4.

        Returns
        -------
        dict
            ``{amplitude_ratio, driving_frequency, peak_amp_ratio,
            peak_amp_index, peak_frequency, peak_fwhm, peak_amplitude,
            in_window, outlier, good, fwhm_median, fwhm_mad,
            peak_amplitude_median, peak_amplitude_mad, amplitude_map,
            n_amp, n_peaks, n_in_window, n_good, n_outlier}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))
        # Default cap: keep each amplitude slice's 4 most-prominent peaks.
        kwargs.setdefault("max_peaks", 4)

        # Resolve the real signal variable; state-discriminated runs store P(|1>).
        signal_var = kwargs.get("signal_var", None)
        if signal_var is None:
            for cand in ("signal", "state"):
                if cand in dataset.data_vars:
                    signal_var = cand
                    kwargs["signal_var"] = signal_var
                    break

        ds = dataset
        if signal_var is None and "IQdata" not in ds:
            if "I" in ds and "Q" in ds:
                ds = ds.assign(IQdata=ds["I"] + 1j * ds["Q"])
            else:
                raise ValueError(
                    "ParametricDriveResonanceEstimator needs a real 'signal'/'state' "
                    "variable, an 'IQdata' variable, or both 'I' and 'Q'."
                )

        amp = ds.coords["amplitude_ratio"].values.astype(float)
        freq = ds.coords["driving_frequency"].values.astype(float)

        single = QubitSpectroscopyEstimator()
        # Flat point-cloud of all peaks found across all amplitude slices.
        pk_amp_idx, pk_amp = [], []
        pk_freq, pk_fwhm, pk_amplitude = [], [], []
        for k in range(len(amp)):
            # The single-slice estimator expects a 'detuning' coordinate.
            sl = ds.isel(amplitude_ratio=k).rename({"driving_frequency": "detuning"})
            try:
                r = single.extract_parameters(sl, **kwargs)
            except Exception:
                continue
            for pk in r.get("peaks", []):
                pk_amp_idx.append(k)
                pk_amp.append(float(amp[k]))
                pk_freq.append(float(pk["detuning"]))
                pk_fwhm.append(float(pk["fwhm"]))
                pk_amplitude.append(float(pk["amplitude"]))

        peak_amp_index = np.asarray(pk_amp_idx, dtype=int)
        peak_amp_ratio = np.asarray(pk_amp, dtype=float)
        peak_frequency = np.asarray(pk_freq, dtype=float)
        peak_fwhm = np.asarray(pk_fwhm, dtype=float)
        peak_amplitude = np.asarray(pk_amplitude, dtype=float)
        n_peaks = peak_frequency.size

        # (1) Strict window enforcement: the fitted centre must lie inside the
        # swept frequency window.
        f_lo, f_hi = float(freq.min()), float(freq.max())
        in_window = (
            np.isfinite(peak_frequency) & (peak_frequency > f_lo) & (peak_frequency < f_hi)
            if n_peaks else np.zeros(0, dtype=bool)
        )

        # (2) Robust outlier rejection on the pooled peak width and amplitude.
        outlier_fwhm, fwhm_med, fwhm_mad = mad_outliers(peak_fwhm, in_window, n_sigma)
        outlier_amp, amp_med, amp_mad = mad_outliers(np.abs(peak_amplitude), in_window, n_sigma)
        outlier = in_window & (outlier_fwhm | outlier_amp)
        good = in_window & ~outlier

        amplitude_map = self._signal_map(ds, signal_var)

        return {
            "amplitude_ratio": amp,
            "driving_frequency": freq,
            "peak_amp_ratio": peak_amp_ratio,
            "peak_amp_index": peak_amp_index,
            "peak_frequency": peak_frequency,
            "peak_fwhm": peak_fwhm,
            "peak_amplitude": peak_amplitude,
            "in_window": in_window,
            "outlier": outlier,
            "good": good,
            "fwhm_median": fwhm_med,
            "fwhm_mad": fwhm_mad,
            "peak_amplitude_median": amp_med,
            "peak_amplitude_mad": amp_mad,
            "amplitude_map": amplitude_map,
            "n_amp": int(len(amp)),
            "n_peaks": int(n_peaks),
            "n_in_window": int(in_window.sum()),
            "n_good": int(good.sum()),
            "n_outlier": int(outlier.sum()),
        }

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the peak point-cloud + stats; drop the bulky 2-D map and axes."""
        drop = {"amplitude_map", "driving_frequency"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 2-D signal map and the peak point-cloud (with good/outlier
        masks) into one self-sufficient Dataset."""
        amp = np.asarray(results["amplitude_ratio"], dtype=float)
        freq = np.asarray(results["driving_frequency"], dtype=float)
        amplitude = np.asarray(results["amplitude_map"], dtype=float)
        n_peaks = int(results["n_peaks"])

        data_vars: Dict[str, Any] = {
            "amplitude": (("amplitude_ratio", "driving_frequency"), amplitude),
            "peak_amp_ratio": ("peak", np.asarray(results["peak_amp_ratio"], float)),
            "peak_frequency": ("peak", np.asarray(results["peak_frequency"], float)),
            "peak_fwhm": ("peak", np.asarray(results["peak_fwhm"], float)),
            "peak_amplitude": ("peak", np.asarray(results["peak_amplitude"], float)),
            "good": ("peak", np.asarray(results["good"], bool)),
            "outlier": ("peak", np.asarray(results["outlier"], bool)),
        }
        coords: Dict[str, Any] = {
            "amplitude_ratio": amp, "driving_frequency": freq, "peak": np.arange(n_peaks),
        }
        attrs: Dict[str, Any] = {
            "n_amp": int(results["n_amp"]),
            "n_peaks": n_peaks,
            "n_good": int(results["n_good"]),
            "n_outlier": int(results["n_outlier"]),
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Single figure: the 2-D signal map over (amplitude_ratio,
        driving_frequency) with every kept resonance peak overlaid and outliers
        marked."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"parametric_drive_resonance": plot_parametric_map(plot_data)}
