"""
Qubit Spectroscopy Estimator
============================
Peak-finding and Lorentzian fitting for qubit spectroscopy data.

The estimator computes the distance of each IQ point from a reference
point in the complex plane, subtracts a polynomial baseline, detects
peaks with ``scipy.signal.find_peaks``, and fits a Lorentzian to each peak.

When a complex ``ref`` is provided, the signal is ``|IQdata - ref|``.
Otherwise the reference is auto-estimated as the median of the complex
IQ data (robust for spectroscopy sweeps where most points are off-resonance).

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

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_lorentzian import FitLorentzian, lorentzian
from scqat.estimators.qubit_spectroscopy.visualization import plot_spectrum


# ------------------------------------------------------------------
# Helper: baseline estimation
# ------------------------------------------------------------------

def _estimate_baseline(x: np.ndarray, y: np.ndarray, order: int = 1,
                       quantile: float = 0.25):
    """
    Fit a polynomial baseline through the *quietest* portion of the
    spectrum (bottom ``quantile`` fraction sorted by absolute deviation
    from the median).
    """
    med = np.median(y)
    dev = np.abs(y - med)
    threshold = np.quantile(dev, quantile)
    mask = dev <= threshold
    coeffs = np.polyfit(x[mask], y[mask], deg=order)
    return np.polyval(coeffs, x)


# ------------------------------------------------------------------
# Estimator
# ------------------------------------------------------------------

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
    @staticmethod
    def _with_iqdata(dataset: xr.Dataset) -> xr.Dataset:
        """Return a dataset that has an ``IQdata`` variable, building it from
        ``I``/``Q`` when only the quadratures are present (so a saved ds_raw with
        netCDF-safe float ``I``/``Q`` is read natively)."""
        if "IQdata" in dataset:
            return dataset
        if "I" in dataset and "Q" in dataset:
            return dataset.assign(IQdata=dataset["I"] + 1j * dataset["Q"])
        raise ValueError(
            "QubitSpectroscopyEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
        )

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

        Keyword arguments
        -----------------
        ref : complex, optional
            Reference point in the IQ plane.  When provided, the signal
            for peak detection is ``|IQdata - ref|``.  When omitted, the
            reference is auto-estimated as the median of the complex IQ
            data (works well when most sweep points are off-resonance).
        prominence : float, optional
            Minimum prominence for ``find_peaks`` relative to the baseline-
            subtracted signal span.  Default 0.1 (10 %).
        min_snr : float, optional
            Significance gate: a peak's prominence must also exceed
            ``min_snr * robust_sigma`` (robust_sigma = 1.4826 * MAD of the
            baseline-corrected signal). Rejects noise-only sweeps (returns no
            peaks) and keeps all genuine lines regardless of count. Default 6.0.
        max_peaks : int or None, optional
            Maximum number of peaks to return.  When set, the *max_peaks*
            most prominent peaks are kept (sorted by amplitude) and the
            rest are discarded.  Default ``None`` (keep all).
        fit_window_factor : float, optional
            Each peak is fitted inside a window of
            ``fit_window_factor * estimated_width`` around the peak centre.
            Default 5.
        signal_var : str, optional
            Force using a specific data variable instead of IQdata.
            Default: use IQdata.

        Returns
        -------
        dict
            ``{signal, baseline, signal_corrected, ref_iq,
            inverted, peaks: [...]}``
        """
        ref = kwargs.get("ref", None)
        prominence_rel = kwargs.get("prominence", 0.1)
        min_snr = kwargs.get("min_snr", 6.0)
        max_peaks = kwargs.get("max_peaks", None)
        fit_window_factor = kwargs.get("fit_window_factor", 5.0)
        signal_var = kwargs.get("signal_var", None)

        detuning = dataset.coords["detuning"].values.astype(float)

        # --- Choose / build the 1-D signal ---
        if signal_var is not None:
            signal = dataset[signal_var].values.astype(float).ravel()
            ref_iq = None
        else:
            dataset = self._with_iqdata(dataset)
            iq_complex = dataset["IQdata"].values.ravel()
            if ref is not None:
                ref_iq = complex(ref)
            else:
                # Auto-estimate: median of the complex IQ cloud
                ref_iq = complex(np.median(iq_complex.real),
                                 np.median(iq_complex.imag))
            signal = np.abs(iq_complex - ref_iq)

        # --- Baseline subtraction ---
        baseline = _estimate_baseline(detuning, signal)
        signal_corrected = signal - baseline

        # --- Peak detection ---
        # Try both polarities; keep the one whose most prominent
        # peak is larger (handles both absorption dips and emission peaks).
        span = signal_corrected.max() - signal_corrected.min()
        # Significance gate: a peak must rise above the noise, not merely be the most
        # prominent bump within the span. robust sigma = 1.4826 * MAD (median-based, so a
        # few strong peaks don't inflate it). This rejects noise-only sweeps (no peak found)
        # while keeping every genuine line, e.g. both peaks of a two-transition sweep.
        robust_sigma = 1.4826 * np.median(np.abs(signal_corrected - np.median(signal_corrected)))
        abs_prom = max(prominence_rel * span, min_snr * robust_sigma)

        idx_pos, props_pos = find_peaks(
            signal_corrected, prominence=abs_prom, width=1,
        )
        idx_neg, props_neg = find_peaks(
            -signal_corrected, prominence=abs_prom, width=1,
        )

        best_pos = props_pos["prominences"].max() if len(idx_pos) else 0
        best_neg = props_neg["prominences"].max() if len(idx_neg) else 0

        if best_neg > best_pos:
            peak_indices, properties = idx_neg, props_neg
            signal_corrected = -signal_corrected
            inverted = True
        else:
            peak_indices, properties = idx_pos, props_pos
            inverted = False

        # --- Keep only the most prominent peaks if max_peaks is set ---
        if max_peaks is not None and len(peak_indices) > max_peaks:
            top_idx = np.argsort(properties["prominences"])[::-1][:max_peaks]
            top_idx = np.sort(top_idx)  # preserve original order
            peak_indices = peak_indices[top_idx]
            for key in properties:
                properties[key] = properties[key][top_idx]

        # --- Fit Lorentzians to each peak ---
        peaks_info: List[Dict[str, Any]] = []
        for i, idx in enumerate(peak_indices):
            est_width_pts = properties["widths"][i] if "widths" in properties else 10
            hw = int(fit_window_factor * max(est_width_pts, 3))
            lo = max(idx - hw, 0)
            hi = min(idx + hw + 1, len(detuning))

            x_win = detuning[lo:hi]
            y_win = signal_corrected[lo:hi]

            da_win = xr.DataArray(y_win, coords={'x': x_win}, dims='x')
            x_lo_b = float(detuning[lo])
            x_hi_b = float(detuning[hi - 1])
            gamma_max = float(detuning[-1] - detuning[0])
            fitter = FitLorentzian(
                da_win,
                inverted=inverted,
                bounds={'x0': (x_lo_b, x_hi_b), 'gamma': (0.0, gamma_max)},
            )
            try:
                result = fitter.fit()
                p = result.params
                popt = np.array([p['x0'].value, p['amplitude'].value,
                                 p['gamma'].value, p['offset'].value])
                perr = np.array([
                    p['x0'].stderr if p['x0'].stderr is not None else np.nan,
                    p['amplitude'].stderr if p['amplitude'].stderr is not None else np.nan,
                    p['gamma'].stderr if p['gamma'].stderr is not None else np.nan,
                    p['offset'].stderr if p['offset'].stderr is not None else np.nan,
                ])
            except Exception:
                # Fall back to initial guess
                center_guess = detuning[idx]
                amp_guess = signal_corrected[idx] if not inverted else -signal_corrected[idx]
                gamma_guess = abs(detuning[min(idx + max(int(est_width_pts // 2), 1), len(detuning) - 1)]
                                  - detuning[idx])
                if gamma_guess == 0:
                    gamma_guess = abs(detuning[1] - detuning[0]) * 5
                popt = np.array([center_guess, amp_guess, gamma_guess, 0.0])
                perr = np.full(4, np.nan)

            det_fit = popt[0]
            fwhm = 2 * abs(popt[2])

            peak_entry: Dict[str, Any] = {
                "detuning": float(det_fit),
                "amplitude": float(popt[1]),
                "fwhm": float(fwhm),
                "offset": float(popt[3]),
                "detuning_err": float(perr[0]),
                "amplitude_err": float(perr[1]),
                "fwhm_err": float(2 * perr[2]),
                "fit_x": x_win,
                "fit_y": lorentzian(x_win, *popt),
            }

            # Report absolute frequency if available
            if "full_freq" in dataset.coords:
                freq_vals = dataset.coords["full_freq"].values.ravel().astype(float)
                # Interpolate to find absolute frequency at this detuning
                peak_entry["full_freq"] = float(np.interp(det_fit, detuning, freq_vals))

            peaks_info.append(peak_entry)

        # Sort peaks by detuning
        peaks_info.sort(key=lambda p: p["detuning"])

        return {
            "signal": signal,
            "baseline": baseline,
            "signal_corrected": signal_corrected,
            "ref_iq": ref_iq,
            "inverted": inverted,
            "peaks": peaks_info,
        }

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the reference, polarity, and per-peak fit parameters; drop the
        full spectrum arrays and per-peak fit curves."""
        drop_peak = {"fit_x", "fit_y"}
        return {
            "ref_iq": results.get("ref_iq"),
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
        return {"spectrum": plot_spectrum(plot_data)}
