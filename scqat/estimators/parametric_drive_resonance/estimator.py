"""
Parametric-Drive Resonance Estimator
=====================================
Extract the parametric-resonance line(s) from a 2-D
``drive_amp`` × ``driving_frequency`` map measured at a **fixed** drive
time (the ``LCH_qubit_parametric_drive_fixed_time`` node). The signal is
:math:`P(|1\\rangle)` after the parametric drive; a resonance shows up as a
peak/dip in frequency whose position drifts with the drive amplitude.

This mirrors
:class:`~scqat.estimators.qubit_spectroscopy_flux.QubitSpectroscopyFluxEstimator`
with the axis roles ``flux_bias → drive_amp`` and
``detuning → driving_frequency``: every ``drive_amp`` slice is fitted by the
family-shared per-trace reduction :func:`scqat.tools.peak_fit.fit_peaks`,
pooled by the generic map tracker :func:`scqat.tools.peak_map.track_peaks`
into a **point-cloud** ``(drive_amp, frequency, fwhm, amplitude)``.

Cleaning: a peak is kept (``good``) when its centre lies strictly inside the
swept frequency window and its ``fwhm`` / ``|amplitude|`` are not robust
(median/MAD) outliers across the pooled set of detected peaks.

Expected ``xarray.Dataset`` contract
-------------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).

Coordinates:
    - drive_amp   : 1-D float array — parametric drive amplitude scale.
    - driving_frequency : 1-D float array — parametric drive frequency (Hz).
Data variables:
    - state / signal : (drive_amp, driving_frequency) real P(|1⟩), **or**
    - IQdata / I, Q  : the complex demodulated signal.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.tools.peak_fit import validate_peak_kwargs
from scqat.tools.peak_map import track_peaks
from scqat.estimators.parametric_drive_resonance.visualization import plot_parametric_map


class ParametricDriveResonanceEstimator(BaseEstimator):
    """Fit the parametric-resonance peak(s) at every drive amplitude and report
    them as a point-cloud over (``drive_amp``, ``driving_frequency``)."""

    estimator_name = "parametric_drive_resonance"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("drive_amp", "driving_frequency"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"ParametricDriveResonanceEstimator requires a '{coord}' coordinate."
                )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit and collect every parametric-resonance peak in every
        ``drive_amp`` slice.

        Every slice is fitted by :func:`scqat.tools.peak_fit.fit_peaks` and
        pooled by :func:`scqat.tools.peak_map.track_peaks`; unknown keyword
        names raise before any per-slice fit. By default each slice is capped
        to its 4 most prominent peaks (``max_peaks=4``); pass
        ``max_peaks=None`` to keep every peak above the prominence threshold.

        Keyword arguments
        -----------------
        n_sigma : float, optional
            Robust-sigma threshold for the width / amplitude outlier test
            (default 3.0).
        signal_var : str, optional
            Data variable holding the real signal (auto-detected as ``signal``
            or ``state`` when omitted; otherwise the IQ path is used).
        prominence, max_peaks, ... :
            Knobs of :func:`scqat.tools.peak_fit.fit_peaks`.

        Returns
        -------
        dict
            ``{drive_amp, driving_frequency, peak_drive_amp,
            peak_amp_index, peak_frequency, peak_fwhm, peak_amplitude,
            in_window, outlier, good, fwhm_median, fwhm_mad,
            peak_amplitude_median, peak_amplitude_mad, amplitude_map,
            n_amp, n_peaks, n_in_window, n_good, n_outlier}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))
        signal_var = kwargs.pop("signal_var", None)
        try:
            validate_peak_kwargs(kwargs)
        except ValueError as err:
            raise ValueError(
                f"ParametricDriveResonanceEstimator: {err} "
                f"(own keyword-only tunables: n_sigma, signal_var)"
            ) from None
        # Default cap: keep each amplitude slice's 4 most-prominent peaks.
        kwargs.setdefault("max_peaks", 4)

        # Resolve the real signal variable; state-discriminated runs store P(|1>).
        if signal_var is None:
            for cand in ("signal", "state"):
                if cand in dataset.data_vars:
                    signal_var = cand
                    break

        if signal_var is not None:
            ds = dataset
            signal_map = ds[signal_var].transpose("drive_amp", "driving_frequency").values
        else:
            if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
                raise ValueError(
                    "ParametricDriveResonanceEstimator needs a real 'signal'/'state' "
                    "variable, an 'IQdata' variable, or both 'I' and 'Q'."
                )
            ds = with_iqdata(dataset)
            signal_map = ds["IQdata"].transpose("drive_amp", "driving_frequency").values

        amp = ds.coords["drive_amp"].values.astype(float)
        freq = ds.coords["driving_frequency"].values.astype(float)

        cloud = track_peaks(amp, freq, signal_map, n_sigma=n_sigma, **kwargs)

        # Relabel the generic tracker keys into the parametric vocabulary.
        return {
            "drive_amp": cloud["x"],
            "driving_frequency": cloud["y"],
            "peak_drive_amp": cloud["peak_x"],
            "peak_amp_index": cloud["peak_x_index"],
            "peak_frequency": cloud["peak_y"],
            "peak_fwhm": cloud["peak_fwhm"],
            "peak_amplitude": cloud["peak_amplitude"],
            "in_window": cloud["in_window"],
            "outlier": cloud["outlier"],
            "good": cloud["good"],
            "fwhm_median": cloud["fwhm_median"],
            "fwhm_mad": cloud["fwhm_mad"],
            "peak_amplitude_median": cloud["peak_amplitude_median"],
            "peak_amplitude_mad": cloud["peak_amplitude_mad"],
            "amplitude_map": np.abs(signal_map),
            "n_amp": cloud["n_x"],
            "n_peaks": cloud["n_peaks"],
            "n_in_window": cloud["n_in_window"],
            "n_good": cloud["n_good"],
            "n_outlier": cloud["n_outlier"],
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
        amp = np.asarray(results["drive_amp"], dtype=float)
        freq = np.asarray(results["driving_frequency"], dtype=float)
        amplitude = np.asarray(results["amplitude_map"], dtype=float)
        n_peaks = int(results["n_peaks"])

        data_vars: Dict[str, Any] = {
            "amplitude": (("drive_amp", "driving_frequency"), amplitude),
            "peak_drive_amp": ("peak", np.asarray(results["peak_drive_amp"], float)),
            "peak_frequency": ("peak", np.asarray(results["peak_frequency"], float)),
            "peak_fwhm": ("peak", np.asarray(results["peak_fwhm"], float)),
            "peak_amplitude": ("peak", np.asarray(results["peak_amplitude"], float)),
            "good": ("peak", np.asarray(results["good"], bool)),
            "outlier": ("peak", np.asarray(results["outlier"], bool)),
        }
        coords: Dict[str, Any] = {
            "drive_amp": amp, "driving_frequency": freq, "peak": np.arange(n_peaks),
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
        """Single figure: the 2-D signal map over (drive_amp,
        driving_frequency) with every kept resonance peak overlaid and outliers
        marked."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"parametric_drive_resonance": plot_parametric_map(plot_data)}
