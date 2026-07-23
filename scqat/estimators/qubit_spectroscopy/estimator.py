"""
Qubit Spectroscopy Estimator
============================
Peak-finding and Lorentzian fitting for qubit spectroscopy data.

The heavy lifting is the family-shared per-trace reduction
:func:`scqat.tools.peak_fit.fit_peaks` (distance signal, baseline subtraction,
``find_peaks`` + windowed Lorentzian per peak, merge, ``max_peaks`` cap) — this
estimator only resolves the dataset into arrays, forwards its flat kwarg
surface, and owns the artifacts (metadata / plot data / figure).

When a complex ``ref`` is provided, the signal is ``|IQdata - ref|``.
Otherwise, if the dataset carries the stored ground-blob center (the
``ref_pos_*`` variables an acquisition layer attached — see
:func:`scqat.core.base_estimator.stored_ground`), that measured point is the
reference; else it is auto-estimated as the median of the complex IQ data
(robust for spectroscopy sweeps where most points are off-resonance). The
resolved choice is stamped as ``ref_source`` (supplied / stored / median).

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - detuning : 1-D float array – drive-frequency detuning from the LO (Hz).
    - full_freq: (detuning,) absolute drive frequency (Hz). Optional;
                 if present, peak positions are also reported in absolute freq.
Data variables:
    - IQdata   : (detuning,) – complex demodulated signal (I + iQ).

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_ground, with_iqdata
from scqat.tools.peak_fit import PEAK_KNOBS, fit_peaks
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators.qubit_spectroscopy.visualization import plot_spectrum


class QubitSpectroscopyEstimator(BaseEstimator):
    """
    Detect and fit peaks in qubit spectroscopy data.

    Results are returned per-qubit.  Each qubit entry contains a list of
    peak dictionaries with keys: ``detuning``, ``full_freq`` (if available),
    ``amplitude``, ``fwhm``, ``offset``, and the best-fit arrays.
    """

    estimator_name = "qubit_spectroscopy"

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if "detuning" not in dataset.coords:
            raise ValueError(
                "QubitSpectroscopyEstimator requires a 'detuning' coordinate."
            )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "QubitSpectroscopyEstimator requires an 'IQdata' (complex) variable, "
                "or both 'I' and 'Q'."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Detect peaks and fit Lorentzians in a single-qubit dataset.

        The dataset is expected to have the ``qubit`` dimension already
        removed (e.g. via ``repetition_data``).  The 1-D ``detuning``
        coordinate is the only required dimension.

        Keyword arguments — flat and fully owned; unknown names raise
        -------------------------------------------------------------
        signal_var : str, optional
            Force using a specific real data variable instead of IQdata.
            Default: use IQdata (as ``|IQdata - ref|``).
        ref, prominence, min_snr, max_peaks, merge_factor, min_fwhm_factor,
        fit_window_factor
            Knobs of :func:`scqat.tools.peak_fit.fit_peaks` — see there for
            semantics and defaults.

        Returns
        -------
        dict
            ``{signal, baseline, signal_corrected, ref_iq,
            inverted, peaks: [...]}``
        """
        signal_var = kwargs.pop("signal_var", None)
        unknown = set(kwargs) - PEAK_KNOBS
        if unknown:
            raise ValueError(
                f"Unknown keyword argument(s) {sorted(unknown)} for "
                f"QubitSpectroscopyEstimator; valid: "
                f"{sorted(PEAK_KNOBS | {'signal_var'})}"
            )

        detuning = dataset.coords["detuning"].values.astype(float)
        full_freq = (
            dataset.coords["full_freq"].values.ravel().astype(float)
            if "full_freq" in dataset.coords else None
        )

        # --- Choose / build the 1-D signal ---
        if signal_var is not None:
            signal = dataset[signal_var].values.astype(float).ravel()
        else:
            signal = with_iqdata(dataset)["IQdata"].values.ravel()

        # provenance of the radial reference — priority: a supplied IQ point ->
        # the stored ground-blob center riding the dataset -> the complex median
        # (meaningful only for complex input)
        ref_source = "supplied" if kwargs.get("ref") is not None else "median"
        if signal_var is None and kwargs.get("ref") is None:
            stored = stored_ground(dataset)
            if stored is not None:
                kwargs["ref"] = stored
                ref_source = "stored"

        results = fit_peaks(detuning, signal, full_freq=full_freq, **kwargs)
        results["ref_source"] = ref_source
        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the reference, polarity, and per-peak fit parameters; drop the
        full spectrum arrays and per-peak fit curves."""
        drop_peak = {"fit_x", "fit_y"}
        return {
            "ref_iq": results.get("ref_iq"),
            "ref_source": results.get("ref_source"),
            "inverted": results["inverted"],
            "peaks": [
                {k: v for k, v in pk.items() if k not in drop_peak}
                for pk in results["peaks"]
            ],
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the spectrum / baseline / corrected traces (over ``detuning``)
        and each Lorentzian fit, pre-evaluated on the full detuning grid and
        NaN-padded outside its window, as ``peak_fit`` (peak, detuning).  The
        optional ``full_freq`` axis and peak centres/FWHMs are included so both
        sub-plots redraw with no refitting.
        """
        detuning = dataset.coords["detuning"].values.astype(float)

        data_vars: Dict[str, Any] = {
            "signal": ("detuning", np.asarray(results["signal"], dtype=float)),
            "baseline": ("detuning", np.asarray(results["baseline"], dtype=float)),
            "signal_corrected": ("detuning", np.asarray(results["signal_corrected"], dtype=float)),
        }
        coords: Dict[str, Any] = {"detuning": detuning}
        attrs: Dict[str, Any] = {"inverted": int(bool(results["inverted"]))}

        ref_iq = results.get("ref_iq")
        if ref_iq is not None:
            attrs["ref_iq_real"] = float(np.real(ref_iq))
            attrs["ref_iq_imag"] = float(np.imag(ref_iq))
            attrs["ref_source"] = str(results.get("ref_source", "median"))

        # the raw IQ cloud for the shared IQ-plane panel (complex input only)
        if "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars):
            iq = with_iqdata(dataset)["IQdata"].values.ravel()
            data_vars["iq_i"] = ("detuning", np.real(iq).astype(float))
            data_vars["iq_q"] = ("detuning", np.imag(iq).astype(float))

        if "full_freq" in dataset.coords:
            data_vars["full_freq"] = (
                "detuning", dataset.coords["full_freq"].values.ravel().astype(float)
            )
            attrs["has_full_freq"] = 1
        else:
            attrs["has_full_freq"] = 0

        peaks = results.get("peaks", [])
        attrs["n_peaks"] = len(peaks)
        if peaks:
            n = len(detuning)
            peak_fit = np.full((len(peaks), n), np.nan)
            peak_det = np.empty(len(peaks))
            peak_fwhm = np.empty(len(peaks))
            for i, pk in enumerate(peaks):
                fit_x = np.asarray(pk["fit_x"], dtype=float)
                fit_y = np.asarray(pk["fit_y"], dtype=float)
                lo = int(np.argmin(np.abs(detuning - fit_x[0])))
                peak_fit[i, lo:lo + len(fit_y)] = fit_y
                peak_det[i] = pk["detuning"]
                peak_fwhm[i] = pk["fwhm"]
            coords["peak"] = np.arange(len(peaks))
            data_vars["peak_fit"] = (["peak", "detuning"], peak_fit)
            data_vars["peak_detuning"] = ("peak", peak_det)
            data_vars["peak_fwhm"] = ("peak", peak_fwhm)

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
        """
        Single figure showing the spectrum, baseline, and Lorentzian fits
        overlaid at each detected peak, drawn entirely from plot_data.
        """
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        figs = {"spectrum": plot_spectrum(plot_data)}
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
