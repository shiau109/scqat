"""Broadband Qubit Spectroscopy Estimator.

Processes wideband two-tone spectroscopy sweeps across stepped drive LOs.
Peak detection and Lorentzian fitting delegate entirely to
:func:`scqat.tools.peak_fit.fit_peaks` — the same shared reduction used by
:class:`scqat.estimators.qubit_spectroscopy.QubitSpectroscopyEstimator` — so
both experiments share identical baseline, peak polarity, FWHM guard, and
Lorentzian fitting behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_ground, with_iqdata
from scqat.tools.peak_fit import PEAK_KNOBS, fit_peaks
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from .visualization import plot_broadband_qubit_spectrum


class BroadbandQubitSpectroscopyEstimator(BaseEstimator):
    """Estimate candidate qubit transition frequencies from wideband two-tone spectroscopy.

    Peak detection and fitting are identical to QubitSpectroscopyEstimator:
    both delegate to :func:`scqat.tools.peak_fit.fit_peaks`.

    Accepted keyword arguments (forwarded verbatim to fit_peaks)
    ------------------------------------------------------------
    prominence : float
        Minimum peak prominence as a fraction of the baseline-corrected
        signal span.  Default 0.1 (10 %).
    min_snr : float
        Peak must also exceed ``min_snr * robust_sigma``.  Default 6.0.
    max_peaks : int or None
        Cap on the number of returned peaks (largest-area first).
        Default None (keep all).
    merge_factor : float
        De-duplication strength (see fit_peaks docs).  Default 1.0.
    min_fwhm_factor : float
        Sub-resolution spike guard.  Default 0.5.
    fit_window_factor : float
        Lorentzian fit window width in units of the estimated peak width.
        Default 5.0.
    ref : complex, optional
        IQ reference point override.
    """

    estimator_name = "broadband_qubit_spectroscopy"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_freq = (
            "frequency" in dataset.coords
            or "frequency_hz" in dataset.coords
            or "detuning_hz" in dataset.coords
        )
        if not has_freq:
            raise ValueError(
                "BroadbandQubitSpectroscopyEstimator requires a 'frequency', "
                "'frequency_hz', or 'detuning_hz' coordinate."
            )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "BroadbandQubitSpectroscopyEstimator requires an 'IQdata' "
                "variable, or both 'I' and 'Q'."
            )

    @classmethod
    def _arrays(cls, dataset: xr.Dataset):
        """Return (freq_array, iq_complex_array) from the dataset."""
        ds = with_iqdata(dataset)
        for key in ("frequency", "frequency_hz", "detuning_hz"):
            if key in ds.coords:
                freq = ds.coords[key].values.astype(float).ravel()
                break
        else:
            freq = np.arange(ds["IQdata"].size, dtype=float)
        iq = ds["IQdata"].values.ravel()
        return freq, iq

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Detect and fit qubit transition peaks using fit_peaks().

        All keyword arguments are forwarded to
        :func:`scqat.tools.peak_fit.fit_peaks`.  Unknown names raise ValueError
        immediately (before any fitting) so typos surface loudly.
        """
        self._check_data(dataset)
        freq, iq = self._arrays(dataset)

        # Reject unknown knobs the same way QubitSpectroscopyEstimator does
        unknown = set(kwargs) - PEAK_KNOBS
        if unknown:
            raise ValueError(
                f"Unknown keyword argument(s) {sorted(unknown)} for "
                f"BroadbandQubitSpectroscopyEstimator; valid: "
                f"{sorted(PEAK_KNOBS)}"
            )

        # Resolve the radial reference — stored ground blob > auto median
        ref_source = "supplied" if kwargs.get("ref") is not None else "median"
        if kwargs.get("ref") is None:
            stored = stored_ground(dataset)
            if stored is not None:
                kwargs["ref"] = stored
                ref_source = "stored"

        # fit_peaks expects a "detuning" axis; the frequency array is
        # semantically equivalent here (fit_peaks only uses it for interpolation
        # and sorting, not for any LO-relative arithmetic).
        results = fit_peaks(freq, iq, **kwargs)
        results["ref_source"] = ref_source

        # Alias: broadband estimator reports peaks under "frequency_hz" so
        # downstream SCQO code can read candidate_qubit_frequencies_hz directly.
        for pk in results["peaks"]:
            pk.setdefault("frequency_hz", pk["detuning"])

        candidate_qubit_freqs = [float(pk["frequency_hz"]) for pk in results["peaks"]]
        results["candidate_qubit_frequencies_hz"] = candidate_qubit_freqs
        results["num_peaks_found"] = len(results["peaks"])
        results["success"] = len(results["peaks"]) > 0

        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Produce JSON-serializable metadata summary (mirrors QubitSpectroscopyEstimator)."""
        clean_peaks = []
        for i, p in enumerate(results.get("peaks", [])):
            clean_peaks.append({
                "rank": i + 1,
                "frequency_hz": float(p.get("frequency_hz", p.get("detuning", 0.0))),
                "fwhm_hz": float(p.get("fwhm", 0.0)),
                "amplitude": float(p.get("amplitude", 0.0)),
                "success": True,
            })
        return {
            "estimator_name": self.estimator_name,
            "candidate_qubit_frequencies_hz": results.get("candidate_qubit_frequencies_hz", []),
            "num_peaks_found": results.get("num_peaks_found", 0),
            "num_peaks_requested": results.get("num_peaks_requested"),
            "peaks": clean_peaks,
            "success": bool(results.get("success", False)),
            "inverted": bool(results.get("inverted", False)),
            "ref_source": results.get("ref_source", "median"),
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle signal, baseline, corrected trace, and Lorentzian fits for plotting."""
        freq, _ = self._arrays(dataset)
        n = len(freq)

        data_vars: Dict[str, Any] = {
            "signal": ("frequency", np.asarray(results["signal"], dtype=float)),
            "baseline": ("frequency", np.asarray(results["baseline"], dtype=float)),
            "signal_corrected": ("frequency", np.asarray(results["signal_corrected"], dtype=float)),
        }

        if "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars):
            iq = with_iqdata(dataset)["IQdata"].values.ravel()
            data_vars["iq_i"] = ("frequency", np.real(iq).astype(float))
            data_vars["iq_q"] = ("frequency", np.imag(iq).astype(float))

        coords: Dict[str, Any] = {"frequency": freq}
        attrs: Dict[str, Any] = {
            "estimator_name": self.estimator_name,
            "success": int(results.get("success", False)),
            "inverted": int(bool(results.get("inverted", False))),
        }

        ref_iq = results.get("ref_iq")
        if ref_iq is not None:
            attrs["ref_iq_real"] = float(np.real(ref_iq))
            attrs["ref_iq_imag"] = float(np.imag(ref_iq))
            attrs["ref_source"] = str(results.get("ref_source", "median"))

        peaks = results.get("peaks", [])
        attrs["n_peaks"] = len(peaks)
        if peaks:
            peak_fit = np.full((len(peaks), n), np.nan)
            peak_freq = np.empty(len(peaks))
            peak_fwhm = np.empty(len(peaks))
            peak_amp = np.empty(len(peaks))

            for i, pk in enumerate(peaks):
                fit_x = np.asarray(pk["fit_x"], dtype=float)
                fit_y = np.asarray(pk["fit_y"], dtype=float)
                lo = int(np.argmin(np.abs(freq - fit_x[0])))
                hi = lo + len(fit_y)
                peak_fit[i, lo:min(hi, n)] = fit_y[:min(len(fit_y), n - lo)]
                peak_freq[i] = pk.get("frequency_hz", pk.get("detuning", 0.0))
                peak_fwhm[i] = pk["fwhm"]
                peak_amp[i] = pk["amplitude"]

            coords["peak"] = np.arange(len(peaks))
            data_vars["peak_fit"] = (["peak", "frequency"], peak_fit)
            data_vars["peak_freq"] = ("peak", peak_freq)
            data_vars["peak_fwhm"] = ("peak", peak_fwhm)
            data_vars["peak_amp"] = ("peak", peak_amp)

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Render the broadband qubit spectrum and IQ plane figures."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        if plot_data is None:
            return {}
        figs: Dict[str, plt.Figure] = {
            "broadband_qubit_spectroscopy": plot_broadband_qubit_spectrum(plot_data)
        }
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs

    def plot(self, plot_data: xr.Dataset) -> Dict[str, plt.Figure]:
        """Render figures directly from plot_data."""
        figs: Dict[str, plt.Figure] = {
            "broadband_qubit_spectroscopy": plot_broadband_qubit_spectrum(plot_data)
        }
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
