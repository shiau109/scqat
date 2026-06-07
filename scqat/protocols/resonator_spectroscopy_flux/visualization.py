"""Combined figure for the composite resonator-spectroscopy-vs-flux analysis.

Draws, on one absolute-frequency axis (or detuning when ``full_freq`` is absent):
  * the 2-D ``|IQ|`` amplitude map (raw data) over (flux, frequency),
  * the per-flux fitted resonator centres (kept points; rejected ones as red x),
  * the dispersive ``center_frequency(flux)`` fit curve and the sweet-spot marker.

Everything is read from the merged ``plot_data`` Dataset assembled by the composite
analyzer — no recomputation — so any consumer redraws an identical figure.
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
        :meth:`ResonatorSpectroscopyFluxAnalyzer.build_plot_data` — carries the
        2-D ``amplitude`` map, the per-flux centres + ``good``/``outlier`` masks,
        the dispersive ``fit_freq`` curve, and the sweet spot in ``attrs``.
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

    # (3) Dispersive fit curve + sweet spot.
    if "fit_flux" in plot_data.coords and "fit_freq" in plot_data:
        fit_flux = plot_data.coords["fit_flux"].values.astype(float)
        fit_freq = plot_data["fit_freq"].values.astype(float)
        if fit_freq.size and np.isfinite(fit_freq).any():
            # White halo under an orange line so it reads over the colormap.
            ax.plot(fit_flux, fit_freq / scale, "-", color="white", lw=3.0)
            ax.plot(fit_flux, fit_freq / scale, "-", color="C1", lw=1.5, label="dispersive fit")
    ss_flux = float(plot_data.attrs.get("sweet_spot_flux", np.nan))
    ss_freq = float(plot_data.attrs.get("sweet_spot_freq", np.nan))
    if np.isfinite(ss_flux) and np.isfinite(ss_freq):
        ax.plot([ss_flux], [ss_freq / scale], "*", color="yellow", ms=15, mec="black", mew=0.6,
                label="sweet spot")

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
