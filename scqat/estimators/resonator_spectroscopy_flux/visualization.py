"""Combined figure for the composite resonator-spectroscopy-vs-flux analysis.

Draws, on one absolute-frequency axis (or detuning when ``full_freq`` is absent):
  * the 2-D ``|IQ|`` amplitude map (raw data) over (flux, frequency),
  * the per-flux fitted resonator centres (kept points; rejected ones as red x),
  * the fitted ``center_frequency(flux)`` curve (dispersive or sine) and the
    sweet-spot marker.

Everything is read from the merged ``plot_data`` Dataset assembled by the composite
estimator — no recomputation — so any consumer redraws an identical figure.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_combined(plot_data: xr.Dataset, ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Build the combined resonator-vs-flux figure from merged ``plot_data``.

    Parameters
    ----------
    plot_data : xr.Dataset
        The merged Dataset from
        :meth:`ResonatorSpectroscopyFluxEstimator.build_plot_data` — carries the
        2-D ``amplitude`` map, the per-flux centres + ``good``/``outlier`` masks,
        the ``fit_freq`` curve, and the sweet-spot point in ``attrs``.
    ax : matplotlib.axes.Axes, optional
        Draw onto an existing axis; a new figure/axis is created when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    flux = plot_data.coords["flux_bias"].values.astype(float)
    amplitude = np.asarray(plot_data["amplitude"].values, dtype=float)  # (flux, detuning)
    good = plot_data["good"].values.astype(bool)
    outlier = plot_data["outlier"].values.astype(bool)

    # Absolute RF frequency axis when available, else detuning.
    use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
    if use_full:
        yvals = plot_data["full_freq"].values.astype(float) / 1e9
        centers = plot_data["center_full_freq"].values.astype(float) / 1e9
        scale, ylabel = 1e9, "RF frequency (GHz)"
    else:
        yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
        centers = plot_data["center_detuning"].values.astype(float) / 1e6
        scale, ylabel = 1e6, "Detuning (MHz)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    else:
        fig = ax.figure

    # (1) Raw 2-D |IQ| amplitude map.
    pcm = ax.pcolormesh(flux, yvals, amplitude.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Amplitude |IQ| (arb. u.)")

    # (2) Per-flux fitted resonator centres (kept) + rejected outliers.
    if good.any():
        ax.plot(flux[good], centers[good], "o", color="white", ms=4, mec="black", mew=0.4,
                label="centre (kept)")
    if outlier.any():
        ax.plot(flux[outlier], centers[outlier], "x", color="red", ms=7, mew=1.5,
                label="rejected")

    # (3) Flux-model fit curve + sweet-spot point.
    if "fit_flux" in plot_data.coords and "fit_freq" in plot_data:
        fit_flux = plot_data.coords["fit_flux"].values.astype(float)
        fit_freq = plot_data["fit_freq"].values.astype(float)
        if fit_freq.size and np.isfinite(fit_freq).any():
            method = str(plot_data.attrs.get("method", "dispersive"))
            # White halo under an orange line so it reads over the colormap.
            ax.plot(fit_flux, fit_freq / scale, "-", color="white", lw=3.0)
            ax.plot(fit_flux, fit_freq / scale, "-", color="C1", lw=1.5, label=f"{method} fit")
    mf_flux = float(plot_data.attrs.get("sweet_spot_flux", np.nan))
    mf_freq = float(plot_data.attrs.get("sweet_spot_res", np.nan))
    if np.isfinite(mf_flux) and np.isfinite(mf_freq):
        ax.plot([mf_flux], [mf_freq / scale], "*", color="yellow", ms=15, mec="black", mew=0.6,
                label="sweet spot")
    ml_flux = float(plot_data.attrs.get("sweet_spot_low_flux", np.nan))
    ml_freq = float(plot_data.attrs.get("sweet_spot_low_res", np.nan))
    if np.isfinite(ml_flux) and np.isfinite(ml_freq):
        ax.plot([ml_flux], [ml_freq / scale], "*", color="cyan", ms=13, mec="black", mew=0.6,
                label="sweet spot (low)")

    ax.set_xlim(float(flux.min()), float(flux.max()))
    ax.set_ylim(float(yvals.min()), float(yvals.max()))
    ax.set_xlabel("Flux bias (V)")
    ax.set_ylabel(ylabel)
    n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
    n_flux = int(plot_data.attrs.get("n_flux", flux.size))
    ax.set_title(f"Resonator spectroscopy vs flux (kept {n_good}/{n_flux})")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_flux_model(plot_data: xr.Dataset) -> plt.Figure:
    """Per-method diagnostic figure for the stage-2 trace fit alone: the fitted
    centre trace, the flux-model curve, and the sweet-spot marker — drawn
    entirely from a METHOD's ``plot_data`` (see ``methods/``), dispatching the
    annotations on ``attrs["method"]``.

    plot_data layout: coords ``flux_bias``/``fit_flux``; vars ``center_freq``
    (flux_bias), ``fit_freq`` (fit_flux); attrs ``method``, ``sweet_spot_flux``,
    ``sweet_spot_res``, ``dv_phi0``, ``success`` (+ ``f_r0``/``g``/``f_q_max``/
    ``ec``/``f_q_max_fixed`` for dispersive; ``amp``/``offset`` for sine).
    """
    method = str(plot_data.attrs.get("method", "dispersive"))
    flux = plot_data.coords["flux_bias"].values.astype(float)
    center = plot_data["center_freq"].values.astype(float)
    fit_flux = plot_data.coords["fit_flux"].values.astype(float)
    fit_freq = plot_data["fit_freq"].values.astype(float)
    mf_flux = float(plot_data.attrs.get("sweet_spot_flux", np.nan))
    mf_res = float(plot_data.attrs.get("sweet_spot_res", np.nan))
    dv_phi0 = float(plot_data.attrs.get("dv_phi0", np.nan))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(flux, center / 1e9, "o", ms=4, color="C0", label="fitted centre (data)")
    if np.isfinite(fit_freq).any():
        ax.plot(fit_flux, fit_freq / 1e9, "-", lw=1.8, color="C1",
                label=f"{method} fit (dv_phi0={dv_phi0:.4f} V)")
    if np.isfinite(mf_flux):
        ax.axvline(mf_flux, color="C3", ls=":", lw=1.0,
                   label=f"sweet spot @ {mf_flux:.4f} V, {mf_res / 1e9:.4f} GHz")
    ml_flux = float(plot_data.attrs.get("sweet_spot_low_flux", np.nan))
    ml_res = float(plot_data.attrs.get("sweet_spot_low_res", np.nan))
    if np.isfinite(ml_flux):
        ax.axvline(ml_flux, color="C9", ls="--", lw=1.0,
                   label=f"sweet spot (low) @ {ml_flux:.4f} V, {ml_res / 1e9:.4f} GHz")

    ax.set_xlabel("Flux bias (V)")
    ax.set_ylabel("Resonator centre frequency (GHz)")
    # dispersive-only caveat: g is conditional on the assumed f_q_max.
    cond = ""
    if method == "dispersive" and plot_data.attrs.get("f_q_max_fixed", 1) != 0:
        cond = " (g conditional on assumed f_q_max)"
    ax.set_title(f"Resonator flux model — {method}{cond}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
