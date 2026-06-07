"""
Qubit Spectroscopy vs Flux Estimator
===================================
Extract the qubit transition peaks from a 2-D qubit-spectroscopy-vs-flux map by
fitting **flux-by-flux**, keeping **all** peaks found at each flux (there can be
two or more transitions at the same flux — e.g. the 0->1 line and the two-photon
0->2/2 line, or, with a single xy drive source, another qubit's line showing up).

For every flux-bias slice the estimator delegates to
:class:`~scqat.estimators.qubit_spectroscopy.QubitSpectroscopyEstimator` (peak
detection + single-Lorentzian fit per peak) and collects every detected peak.
The result is therefore a **point-cloud** of peaks ``(flux, frequency, fwhm,
amplitude)`` rather than a single ``frequency(flux)`` value — assigning points to
individual transition branches belongs to the downstream flux-dependence fit.

Cleaning: a peak is kept (``good``) when its centre lies strictly inside the
swept detuning window and its ``fwhm`` / ``|amplitude|`` are not robust
(median/MAD) outliers across the pooled set of detected peaks.

Expected xarray.Dataset contract
---------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – drive-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute drive frequency (Hz). Optional; when
                  present peak centres are also reported in absolute frequency.
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata, **or**
    - a real signal named by the ``signal_var`` kwarg (e.g. ``"state"``).
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_spectroscopy import QubitSpectroscopyEstimator
from scqat.tools.robust import mad_outliers


class QubitSpectroscopyFluxEstimator(BaseEstimator):
    """Fit the qubit peak(s) at every flux bias and report them as a point-cloud.

    The result dict reports, per detected peak, the ``peak_flux`` it was found at,
    its ``peak_detuning`` (and absolute ``peak_full_freq`` when available),
    ``peak_fwhm``, ``peak_amplitude``, the strict ``in_window`` mask, the
    ``outlier`` mask (robust width/amplitude rejection) and the surviving ``good``
    mask, alongside the 2-D signal ``amplitude_map`` kept for plotting.
    """

    estimator_name = "qubit_spectroscopy_flux"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("flux_bias", "detuning"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"QubitSpectroscopyFluxEstimator requires a '{coord}' coordinate."
                )

    @staticmethod
    def _signal_map(ds: xr.Dataset, signal_var: Optional[str]) -> np.ndarray:
        """2-D signal magnitude oriented (flux_bias, detuning) for plotting."""
        if signal_var is not None:
            return np.abs(ds[signal_var].transpose("flux_bias", "detuning").values)
        return np.abs(ds["IQdata"].transpose("flux_bias", "detuning").values)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit and collect every qubit peak in every ``flux_bias`` slice.

        Each slice is handed to ``QubitSpectroscopyEstimator.extract_parameters``
        (peak detection + single-Lorentzian fit per peak); ``kwargs`` such as
        ``signal_var`` (e.g. ``"state"``), ``prominence`` and ``max_peaks`` are
        forwarded to it. By default no peak cap is applied (all peaks above the
        prominence threshold are kept).

        Keyword arguments
        -----------------
        n_sigma : float, optional
            Robust-sigma threshold for the width / amplitude outlier test
            (default 3.0). Not forwarded to the per-slice estimator.

        Returns
        -------
        dict
            ``{flux_bias, detuning, full_freq?, peak_flux, peak_flux_index,
            peak_detuning, peak_full_freq?, peak_fwhm, peak_amplitude,
            in_window, outlier, good, fwhm_median, fwhm_mad,
            peak_amplitude_median, peak_amplitude_mad, amplitude_map,
            n_flux, n_peaks, n_in_window, n_good, n_outlier}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))
        signal_var = kwargs.get("signal_var", None)

        ds = dataset
        if signal_var is None and "IQdata" not in ds:
            if "I" in ds and "Q" in ds:
                ds = ds.assign(IQdata=ds["I"] + 1j * ds["Q"])
            else:
                raise ValueError(
                    "QubitSpectroscopyFluxEstimator needs an 'IQdata' variable, both 'I' and "
                    "'Q', or a real 'signal_var'."
                )

        flux = ds.coords["flux_bias"].values.astype(float)
        detuning = ds.coords["detuning"].values.astype(float)
        has_full_freq = "full_freq" in ds.coords

        single = QubitSpectroscopyEstimator()
        # Flat point-cloud of all peaks found across all flux slices.
        pk_flux_idx, pk_flux = [], []
        pk_detuning, pk_full_freq, pk_fwhm, pk_amplitude = [], [], [], []
        for k in range(len(flux)):
            sl = ds.isel(flux_bias=k)
            try:
                r = single.extract_parameters(sl, **kwargs)
            except Exception:
                continue
            for pk in r.get("peaks", []):
                pk_flux_idx.append(k)
                pk_flux.append(float(flux[k]))
                pk_detuning.append(float(pk["detuning"]))
                pk_fwhm.append(float(pk["fwhm"]))
                pk_amplitude.append(float(pk["amplitude"]))
                pk_full_freq.append(
                    float(pk["full_freq"]) if (has_full_freq and "full_freq" in pk) else np.nan
                )

        peak_flux_index = np.asarray(pk_flux_idx, dtype=int)
        peak_flux = np.asarray(pk_flux, dtype=float)
        peak_detuning = np.asarray(pk_detuning, dtype=float)
        peak_fwhm = np.asarray(pk_fwhm, dtype=float)
        peak_amplitude = np.asarray(pk_amplitude, dtype=float)
        peak_full_freq = np.asarray(pk_full_freq, dtype=float)
        n_peaks = peak_detuning.size

        # (1) Strict window enforcement: the fitted peak centre must lie inside the
        # swept detuning window.
        det_lo, det_hi = float(detuning.min()), float(detuning.max())
        in_window = (
            np.isfinite(peak_detuning) & (peak_detuning > det_lo) & (peak_detuning < det_hi)
            if n_peaks else np.zeros(0, dtype=bool)
        )

        # (2) Robust outlier rejection on the pooled peak width and amplitude.
        outlier_fwhm, fwhm_med, fwhm_mad = mad_outliers(peak_fwhm, in_window, n_sigma)
        outlier_amp, amp_med, amp_mad = mad_outliers(np.abs(peak_amplitude), in_window, n_sigma)
        outlier = in_window & (outlier_fwhm | outlier_amp)
        good = in_window & ~outlier

        amplitude_map = self._signal_map(ds, signal_var)

        results: Dict[str, Any] = {
            "flux_bias": flux,
            "detuning": detuning,
            "peak_flux": peak_flux,
            "peak_flux_index": peak_flux_index,
            "peak_detuning": peak_detuning,
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
            "n_flux": int(len(flux)),
            "n_peaks": int(n_peaks),
            "n_in_window": int(in_window.sum()),
            "n_good": int(good.sum()),
            "n_outlier": int(outlier.sum()),
        }
        if has_full_freq:
            results["full_freq"] = ds.coords["full_freq"].values.ravel().astype(float)
            results["peak_full_freq"] = peak_full_freq

        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the peak point-cloud + stats; drop the bulky 2-D map and axes."""
        drop = {"amplitude_map", "detuning", "full_freq"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 2-D signal map and the peak point-cloud (with good/outlier
        masks) into one self-sufficient Dataset."""
        flux = np.asarray(results["flux_bias"], dtype=float)
        detuning = np.asarray(results["detuning"], dtype=float)
        amplitude = np.asarray(results["amplitude_map"], dtype=float)
        n_peaks = int(results["n_peaks"])

        data_vars: Dict[str, Any] = {
            "amplitude": (("flux_bias", "detuning"), amplitude),
            "peak_flux": ("peak", np.asarray(results["peak_flux"], float)),
            "peak_detuning": ("peak", np.asarray(results["peak_detuning"], float)),
            "peak_fwhm": ("peak", np.asarray(results["peak_fwhm"], float)),
            "peak_amplitude": ("peak", np.asarray(results["peak_amplitude"], float)),
            "good": ("peak", np.asarray(results["good"], bool)),
            "outlier": ("peak", np.asarray(results["outlier"], bool)),
        }
        coords: Dict[str, Any] = {
            "flux_bias": flux, "detuning": detuning, "peak": np.arange(n_peaks),
        }
        attrs: Dict[str, Any] = {
            "n_flux": int(results["n_flux"]),
            "n_peaks": n_peaks,
            "n_good": int(results["n_good"]),
            "n_outlier": int(results["n_outlier"]),
        }

        if "full_freq" in results:
            coords["full_freq"] = ("detuning", np.asarray(results["full_freq"], float))
            data_vars["peak_full_freq"] = ("peak", np.asarray(results["peak_full_freq"], float))
            attrs["has_full_freq"] = 1
        else:
            attrs["has_full_freq"] = 0

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
        """Single figure: the 2-D signal map over (flux, frequency) with every
        kept qubit peak overlaid and outliers marked."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        flux = plot_data.coords["flux_bias"].values.astype(float)
        amplitude = plot_data["amplitude"].values  # (flux, detuning)
        peak_flux = plot_data["peak_flux"].values.astype(float)
        good = plot_data["good"].values.astype(bool)
        outlier = plot_data["outlier"].values.astype(bool)

        use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
        if use_full:
            yvals = plot_data["full_freq"].values.astype(float) / 1e9
            peak_y = plot_data["peak_full_freq"].values / 1e9
            ylabel = "Qubit RF frequency (GHz)"
        else:
            yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
            peak_y = plot_data["peak_detuning"].values / 1e6
            ylabel = "Detuning (MHz)"

        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        pcm = ax.pcolormesh(flux, yvals, amplitude.T, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="Signal (arb. u.)")
        if good.any():
            ax.plot(peak_flux[good], peak_y[good], "o", color="white", ms=4, mec="black",
                    mew=0.4, label="peaks (kept)")
        if outlier.any():
            ax.plot(peak_flux[outlier], peak_y[outlier], "x", color="red", ms=7, mew=1.5,
                    label="rejected (outlier)")
        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel(ylabel)
        n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
        n_peaks = int(plot_data.attrs.get("n_peaks", peak_flux.size))
        ax.set_title(f"Qubit spectroscopy vs flux (kept {n_good}/{n_peaks} peaks)")
        ax.legend(fontsize=8)

        fig.tight_layout()
        return {"qubit_spectroscopy_flux": fig}
