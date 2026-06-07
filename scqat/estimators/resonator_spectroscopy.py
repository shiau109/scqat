"""
Resonator Spectroscopy Estimator
===============================
Single-dip Lorentzian fitting for resonator spectroscopy data.

The estimator fits the readout **power** ``|IQdata|^2`` (``I^2 + Q^2``): it
subtracts a polynomial baseline and fits **one inverted Lorentzian** over the
swept detuning — the resonator shows up as a dip.  Power is the quantity that is
truly Lorentzian, so the fitted FWHM equals the cavity linewidth ``kappa`` and
the dip centre is the resonator frequency (the centre is the same whether you fit
power or amplitude; only the width is unbiased in power).  Figures display the
**amplitude** ``|IQdata|`` (square root of the fitted power) for readability.

This mirrors :class:`~scqat.estimators.qubit_spectroscopy.QubitSpectroscopyEstimator`
in structure (it reuses the same ``FitLorentzian`` model and baseline helper) but
fits the power instead of the distance from an IQ reference, and fits a single
dip rather than searching for an arbitrary number of peaks.  It is the simplest
of several possible resonator analyses — a starting point; a full complex
circle-fit of ``I + jQ`` is the rigorous upgrade.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - detuning : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq: (detuning,) absolute readout frequency (Hz). Optional;
                 if present, the resonator frequency is also reported in
                 absolute frequency.
Data variables:
    - IQdata   : (detuning,) – complex demodulated signal (I + iQ), **or**
    - I, Q     : (detuning,) – the two quadratures, combined into IQdata.

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_lorentzian import FitLorentzian, lorentzian


# ------------------------------------------------------------------
# Helper: baseline estimation (same approach as qubit_spectroscopy)
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

class ResonatorSpectroscopyEstimator(BaseEstimator):
    """
    Fit the resonator dip in 1-D resonator spectroscopy data.

    Results are returned per-qubit (single dip).  The result dict reports the
    dip ``detuning`` (and absolute ``full_freq`` when available), ``fwhm``,
    ``amplitude``, ``offset``, their fit errors, and a ``success`` flag.
    """

    estimator_name = "resonator_spectroscopy"

    # ------------------------------------------------------------------
    # IQ assembly + validation
    # ------------------------------------------------------------------
    @staticmethod
    def _with_iqdata(dataset: xr.Dataset) -> xr.Dataset:
        """Return a dataset that has an ``IQdata`` variable, building it from
        ``I``/``Q`` when only the quadratures are present."""
        if "IQdata" in dataset:
            return dataset
        if "I" in dataset and "Q" in dataset:
            return dataset.assign(IQdata=dataset["I"] + 1j * dataset["Q"])
        raise ValueError(
            "ResonatorSpectroscopyEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
        )

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "detuning" not in dataset.coords:
            raise ValueError(
                "ResonatorSpectroscopyEstimator requires a 'detuning' coordinate."
            )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "ResonatorSpectroscopyEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Subtract a baseline and fit a single inverted Lorentzian to the readout
        power ``|IQdata|^2`` in a single-qubit dataset.  The fitted FWHM is then
        the cavity linewidth ``kappa``.

        The dataset is expected to have the ``qubit`` dimension already removed
        (e.g. via ``repetition_data``).  The 1-D ``detuning`` coordinate is the
        only required dimension.

        Keyword arguments
        -----------------
        baseline_order : int, optional
            Polynomial order of the baseline fit (default 1, i.e. linear).
        baseline_quantile : float, optional
            Fraction of the quietest points used to estimate the baseline
            (default 0.25).

        Returns
        -------
        dict
            ``{signal, baseline, signal_corrected, detuning, full_freq,
            fwhm, amplitude, offset, *_err, fit_x, fit_y, success}``
        """
        baseline_order = kwargs.get("baseline_order", 1)
        baseline_quantile = kwargs.get("baseline_quantile", 0.25)

        ds = self._with_iqdata(dataset)
        detuning = ds.coords["detuning"].values.astype(float)

        # --- Power signal |IQ|^2 = I^2 + Q^2 (the truly-Lorentzian quantity) ---
        iq_complex = ds["IQdata"].values.ravel()
        signal = np.abs(iq_complex) ** 2

        # --- Baseline subtraction ---
        baseline = _estimate_baseline(
            detuning, signal, order=baseline_order, quantile=baseline_quantile
        )
        signal_corrected = signal - baseline

        # --- Fit a single inverted Lorentzian (the resonator dip) ---
        da = xr.DataArray(signal_corrected, coords={"x": detuning}, dims="x")
        gamma_max = float(detuning[-1] - detuning[0]) or 1.0
        fitter = FitLorentzian(
            da,
            inverted=True,
            bounds={"x0": (float(detuning.min()), float(detuning.max())),
                    "gamma": (0.0, abs(gamma_max))},
        )
        try:
            result = fitter.fit()
            p = result.params
            popt = np.array([p["x0"].value, p["amplitude"].value,
                             p["gamma"].value, p["offset"].value])
            perr = np.array([
                p["x0"].stderr if p["x0"].stderr is not None else np.nan,
                p["amplitude"].stderr if p["amplitude"].stderr is not None else np.nan,
                p["gamma"].stderr if p["gamma"].stderr is not None else np.nan,
                p["offset"].stderr if p["offset"].stderr is not None else np.nan,
            ])
            converged = bool(getattr(result, "success", False))
        except Exception:
            # Fall back to the minimum of the corrected signal
            idx = int(np.argmin(signal_corrected))
            center_guess = float(detuning[idx])
            amp_guess = float(signal_corrected[idx])
            gamma_guess = abs(detuning[1] - detuning[0]) * 5 if len(detuning) > 1 else 1.0
            popt = np.array([center_guess, amp_guess, gamma_guess, 0.0])
            perr = np.full(4, np.nan)
            converged = False

        det_fit = float(popt[0])
        fwhm = 2.0 * abs(float(popt[2]))

        # Fit curve on the corrected signal over the full grid
        fit_y = lorentzian(detuning, *popt)

        # Success: fit converged, dip centre within the swept span, sane width
        in_span = float(detuning.min()) <= det_fit <= float(detuning.max())
        success = bool(converged and in_span and np.isfinite(fwhm) and fwhm > 0)

        results: Dict[str, Any] = {
            "signal": signal,
            "baseline": baseline,
            "signal_corrected": signal_corrected,
            "detuning": det_fit,
            "amplitude": float(popt[1]),
            "fwhm": float(fwhm),
            "offset": float(popt[3]),
            "detuning_err": float(perr[0]),
            "amplitude_err": float(perr[1]),
            "fwhm_err": float(2 * perr[2]),
            "fit_x": detuning,
            "fit_y": fit_y,
            "success": success,
        }

        # Report absolute frequency if available
        if "full_freq" in ds.coords:
            freq_vals = ds.coords["full_freq"].values.ravel().astype(float)
            results["full_freq"] = float(np.interp(det_fit, detuning, freq_vals))

        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the scalar fit parameters; drop the full spectrum arrays and
        the fit curve (those belong in the plot data)."""
        drop = {"signal", "baseline", "signal_corrected", "fit_x", "fit_y"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the **amplitude** spectrum ``|IQ|`` and the amplitude-domain
        Lorentzian fit into one self-sufficient Dataset so the figure redraws
        with no refitting.  The fit is performed on the power ``|IQ|^2``; here it
        is converted back to amplitude (square root of the full power model) for
        display only.  ``full_freq`` and the (power) dip centre / FWHM are stored
        as a coordinate / attrs.
        """
        ds = self._with_iqdata(dataset)
        detuning = ds.coords["detuning"].values.astype(float)

        power = np.asarray(results["signal"], dtype=float)               # |IQ|^2
        baseline = np.asarray(results["baseline"], dtype=float)          # power baseline
        fit_power = np.asarray(results["fit_y"], dtype=float) + baseline  # full power model

        data_vars: Dict[str, Any] = {
            "amplitude": ("detuning", np.sqrt(np.clip(power, 0.0, None))),
            "amplitude_baseline": ("detuning", np.sqrt(np.clip(baseline, 0.0, None))),
            "amplitude_fit": ("detuning", np.sqrt(np.clip(fit_power, 0.0, None))),
        }
        coords: Dict[str, Any] = {"detuning": detuning}
        attrs: Dict[str, Any] = {
            "resonator_detuning": float(results["detuning"]),
            "fwhm": float(results["fwhm"]),
            "success": int(bool(results["success"])),
        }

        if "full_freq" in ds.coords:
            data_vars["full_freq"] = (
                "detuning", ds.coords["full_freq"].values.ravel().astype(float)
            )
            attrs["has_full_freq"] = 1
            if "full_freq" in results:
                attrs["resonator_frequency"] = float(results["full_freq"])
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
        """
        Single figure: the ``|IQ|`` amplitude vs detuning with the (power-fitted)
        Lorentzian overlaid in the amplitude domain and a marker at the resonator
        dip, drawn entirely from plot_data.
        """
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        detuning = plot_data.coords["detuning"].values.astype(float)
        amplitude = plot_data["amplitude"].values
        amplitude_baseline = plot_data["amplitude_baseline"].values
        amplitude_fit = plot_data["amplitude_fit"].values
        det_res = float(plot_data.attrs.get("resonator_detuning", np.nan))
        fwhm = float(plot_data.attrs.get("fwhm", np.nan))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

        det_mhz = detuning / 1e6
        ax.plot(det_mhz, amplitude, "-", lw=0.8, color="C0", label="|IQ| amplitude")
        ax.plot(det_mhz, amplitude_baseline, "--", color="gray", lw=1.0, label="baseline")
        # Power-domain Lorentzian fit, shown as amplitude (sqrt of the power model)
        ax.plot(det_mhz, amplitude_fit, "-", lw=1.8, color="C1",
                label=f"Lorentzian fit (power; FWHM={fwhm / 1e6:.3f} MHz)")
        if np.isfinite(det_res):
            ax.axvline(det_res / 1e6, color="C3", ls=":", lw=1.0,
                       label=f"f_res @ {det_res / 1e6:.3f} MHz")

        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("Amplitude |IQ| (arb. u.)")
        ax.legend(fontsize=8)
        ax.set_title("Resonator spectroscopy")

        # Add absolute-frequency twin axis if available
        if plot_data.attrs.get("has_full_freq", 0):
            freq_vals = plot_data["full_freq"].values / 1e9
            ax_freq = ax.twiny()
            ax_freq.set_xlim(freq_vals[0], freq_vals[-1])
            ax_freq.set_xlabel("RF frequency (GHz)")

        fig.tight_layout()
        return {"resonator_spectroscopy": fig}
