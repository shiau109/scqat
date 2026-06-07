"""
Resonator Spectroscopy vs Flux Estimator
=======================================
Reduce a 2-D resonator-spectroscopy-vs-flux map to a 1-D
``center_frequency(flux)`` curve by fitting the resonator dip **flux-by-flux**.

For every flux-bias slice the estimator delegates to
:class:`~scqat.estimators.resonator_spectroscopy.ResonatorSpectroscopyEstimator`
— the same single inverted-Lorentzian fit of the readout power ``|IQ|^2`` used
for 1-D resonator spectroscopy — and records the dip centre and linewidth.
Stacking those per-slice centres yields the resonator centre frequency as a
function of flux (and its FWHM), i.e. the 2-D ``(flux_bias, detuning)`` map is
collapsed onto a 1-D trace.

What to *do* with the resulting ``center_frequency(flux)`` curve (sweet-spot /
idle-offset / phi0 extraction via a cosine/arccos fit) is intentionally **out of
scope** here — this estimator only produces the trace and the 2-D map for
plotting.

Expected xarray.Dataset contract
---------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute readout frequency (Hz). Optional; when
                  present the centre trace is also reported in absolute frequency.
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.resonator_spectroscopy import ResonatorSpectroscopyEstimator


def _mad_outliers(values: np.ndarray, valid: np.ndarray, n_sigma: float, rel_floor: float = 0.25):
    """Robust (median / MAD) outlier flagging among the ``valid`` points.

    A point is an outlier when it deviates from the median **both** by more than
    ``n_sigma * 1.4826 * MAD`` (1.4826 makes the scaled MAD a consistent estimator
    of the standard deviation for normal data) **and** by more than ``rel_floor``
    times the median. The relative floor keeps an almost-noise-free trace (MAD
    near zero) from flagging points that differ only by a negligible amount.
    With fewer than 4 valid points, or a zero MAD, nothing is flagged.

    Returns ``(outlier_mask, median, mad)`` where ``outlier_mask`` has the shape
    of ``values`` and is True only on flagged valid points.
    """
    values = np.asarray(values, dtype=float)
    finite_valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    outlier = np.zeros(values.shape, dtype=bool)
    v = values[finite_valid]
    if v.size < 4:
        return outlier, float("nan"), float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        return outlier, med, mad
    dev = np.abs(values - med)
    z = dev / robust_sigma
    rel = dev / max(abs(med), 1e-300)
    outlier = finite_valid & (z > n_sigma) & (rel > rel_floor)
    return outlier, med, mad


class ResonatorSpectroscopyVsFluxEstimator(BaseEstimator):
    """
    Fit the resonator dip at every flux bias and report the resonator centre
    frequency as a function of flux.

    The result dict reports, per flux point, the dip ``center_detuning`` (and
    absolute ``center_full_freq`` when available), the ``fwhm`` and a per-point
    ``success`` flag, alongside the 2-D ``amplitude`` map kept for plotting.
    """

    estimator_name = "resonator_spectroscopy_vs_flux"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("flux_bias", "detuning"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"ResonatorSpectroscopyVsFluxEstimator requires a '{coord}' coordinate."
                )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "ResonatorSpectroscopyVsFluxEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the resonator dip in every ``flux_bias`` slice and stack the centres.

        Each slice is handed to ``ResonatorSpectroscopyEstimator.extract_parameters``
        (single inverted Lorentzian on the readout power ``|IQ|^2``); ``kwargs``
        such as ``baseline_order`` / ``baseline_quantile`` are forwarded to it.

        Acceptance per flux point happens in two stages: (1) the fitted dip centre
        must lie strictly **inside** the swept detuning window (drops fits that the
        bounded optimiser pinned to an edge), and (2) the dip ``fwhm`` and
        ``|dip_amplitude|`` must not be robust (median/MAD) outliers across flux.
        The surviving ``good`` points are what downstream frequency-vs-flux fits
        should use.

        Keyword arguments
        -----------------
        n_sigma : float, optional
            Robust-sigma threshold for the width / amplitude outlier test
            (default 3.0). Not forwarded to the per-slice estimator.

        Returns
        -------
        dict
            ``{flux_bias, detuning, full_freq?, center_detuning, center_full_freq?,
            fwhm, dip_amplitude, success, in_window, outlier, good,
            fwhm_median, fwhm_mad, dip_amplitude_median, dip_amplitude_mad,
            amplitude_map, n_flux, n_success, n_good, n_outlier}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))

        ds = ResonatorSpectroscopyEstimator._with_iqdata(dataset)
        flux = ds.coords["flux_bias"].values.astype(float)
        detuning = ds.coords["detuning"].values.astype(float)
        has_full_freq = "full_freq" in ds.coords

        single = ResonatorSpectroscopyEstimator()
        n_flux = len(flux)
        center_detuning = np.full(n_flux, np.nan)
        center_full_freq = np.full(n_flux, np.nan)
        fwhm = np.full(n_flux, np.nan)
        dip_amplitude = np.full(n_flux, np.nan)
        success = np.zeros(n_flux, dtype=bool)

        for k in range(n_flux):
            sl = ds.isel(flux_bias=k)
            try:
                r = single.extract_parameters(sl, **kwargs)
            except Exception:
                # Leave NaN / False for this flux point and carry on.
                continue
            center_detuning[k] = r["detuning"]
            fwhm[k] = r["fwhm"]
            dip_amplitude[k] = r["amplitude"]
            success[k] = bool(r["success"])
            if has_full_freq and "full_freq" in r:
                center_full_freq[k] = r["full_freq"]

        # (1) Strict window enforcement: the fitted dip centre must lie inside the
        # swept detuning window (a centre at an edge means the fit was pinned).
        det_lo, det_hi = float(detuning.min()), float(detuning.max())
        in_window = np.isfinite(center_detuning) & (center_detuning > det_lo) & (center_detuning < det_hi)
        valid = success & in_window

        # (2) Robust outlier rejection on the dip width and amplitude.
        outlier_fwhm, fwhm_med, fwhm_mad = _mad_outliers(fwhm, valid, n_sigma)
        outlier_amp, amp_med, amp_mad = _mad_outliers(np.abs(dip_amplitude), valid, n_sigma)
        outlier = valid & (outlier_fwhm | outlier_amp)
        good = valid & ~outlier

        # 2-D |IQ| amplitude map, oriented (flux_bias, detuning) for plotting.
        amplitude_map = np.abs(
            ds["IQdata"].transpose("flux_bias", "detuning").values
        )

        results: Dict[str, Any] = {
            "flux_bias": flux,
            "detuning": detuning,
            "center_detuning": center_detuning,
            "fwhm": fwhm,
            "dip_amplitude": dip_amplitude,
            "success": success,
            "in_window": in_window,
            "outlier": outlier,
            "good": good,
            "fwhm_median": fwhm_med,
            "fwhm_mad": fwhm_mad,
            "dip_amplitude_median": amp_med,
            "dip_amplitude_mad": amp_mad,
            "amplitude_map": amplitude_map,
            "n_flux": int(n_flux),
            "n_success": int(success.sum()),
            "n_good": int(good.sum()),
            "n_outlier": int(outlier.sum()),
        }
        if has_full_freq:
            results["full_freq"] = ds.coords["full_freq"].values.ravel().astype(float)
            results["center_full_freq"] = center_full_freq

        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the 1-D flux-indexed traces; drop the bulky 2-D map and the
        spectrum axes (those belong in the plot data)."""
        drop = {"amplitude_map", "detuning", "full_freq"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 2-D amplitude map and the extracted centre trace into one
        self-sufficient Dataset so the figure redraws with no refitting."""
        flux = np.asarray(results["flux_bias"], dtype=float)
        detuning = np.asarray(results["detuning"], dtype=float)
        amplitude = np.asarray(results["amplitude_map"], dtype=float)

        data_vars: Dict[str, Any] = {
            "amplitude": (("flux_bias", "detuning"), amplitude),
            "center_detuning": ("flux_bias", np.asarray(results["center_detuning"], float)),
            "fwhm": ("flux_bias", np.asarray(results["fwhm"], float)),
            "dip_amplitude": ("flux_bias", np.asarray(results["dip_amplitude"], float)),
            "success": ("flux_bias", np.asarray(results["success"], bool)),
            "good": ("flux_bias", np.asarray(results["good"], bool)),
            "outlier": ("flux_bias", np.asarray(results["outlier"], bool)),
        }
        coords: Dict[str, Any] = {"flux_bias": flux, "detuning": detuning}
        attrs: Dict[str, Any] = {
            "n_flux": int(results["n_flux"]),
            "n_success": int(results["n_success"]),
            "n_good": int(results["n_good"]),
            "n_outlier": int(results["n_outlier"]),
        }

        if "full_freq" in results:
            coords["full_freq"] = ("detuning", np.asarray(results["full_freq"], float))
            data_vars["center_full_freq"] = (
                "flux_bias", np.asarray(results["center_full_freq"], float)
            )
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
        """Single figure: the 2-D ``|IQ|`` amplitude map over (flux, frequency)
        with the fitted resonator-centre trace overlaid, drawn from plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        flux = plot_data.coords["flux_bias"].values.astype(float)
        amplitude = plot_data["amplitude"].values  # (flux, detuning)

        use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
        if use_full:
            yvals = plot_data["full_freq"].values.astype(float) / 1e9
            center = plot_data["center_full_freq"].values / 1e9
            ylabel = "RF frequency (GHz)"
        else:
            yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
            center = plot_data["center_detuning"].values / 1e6
            ylabel = "Detuning (MHz)"

        good = plot_data["good"].values.astype(bool)
        outlier = plot_data["outlier"].values.astype(bool)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        pcm = ax.pcolormesh(flux, yvals, amplitude.T, shading="auto", cmap="viridis")
        fig.colorbar(pcm, ax=ax, label="Amplitude |IQ| (arb. u.)")
        # Kept centres form the trace; rejected (out-of-window / outlier) shown as red x.
        ax.plot(flux[good], center[good], ".-", color="C3", ms=5, lw=1.0, label="centre (kept)")
        if outlier.any():
            ax.plot(flux[outlier], center[outlier], "x", color="red", ms=7, mew=1.5,
                    label="rejected (outlier)")
        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel(ylabel)
        n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
        n_flux = int(plot_data.attrs.get("n_flux", len(flux)))
        ax.set_title(f"Resonator spectroscopy vs flux (kept {n_good}/{n_flux})")
        ax.legend(fontsize=8)

        fig.tight_layout()
        return {"resonator_spectroscopy_vs_flux": fig}
