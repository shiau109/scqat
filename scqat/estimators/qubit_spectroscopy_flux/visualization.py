"""
Qubit-spectroscopy-vs-flux plotting helper.

``plot_flux_map`` consumes the **plot_data** Dataset built by
``QubitSpectroscopyFluxEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``flux_bias``, ``detuning``, ``peak`` (+ optional ``full_freq``)
vars   : ``reduced`` + ``amplitude`` (flux_bias, detuning) — the background is
         ``reduced`` (the per-slice fitted signal; ``amplitude`` = raw |IQ| kept
         for reference and as the fallback for pre-``reduced`` plotdata);
         per-peak ``peak_flux`` / ``peak_detuning`` / ``peak_fwhm`` /
         ``peak_amplitude`` / ``good`` / ``outlier`` (+ optional ``peak_full_freq``)
attrs  : ``n_flux``, ``n_peaks``, ``n_good``, ``n_outlier``, ``has_full_freq``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_flux_map(plot_data: xr.Dataset) -> plt.Figure:
    """The 2-D signal map over (flux, frequency) with every kept qubit peak
    overlaid and outliers marked, drawn entirely from ``plot_data``."""
    flux = plot_data.coords["flux_bias"].values.astype(float)
    # Background = the per-slice REDUCED signal the peaks were fitted on; raw
    # "amplitude" only as fallback for plotdata saved before the reduced map
    # existed (immutable run data).
    background = plot_data["reduced"] if "reduced" in plot_data else plot_data["amplitude"]
    amplitude = background.values  # (flux, detuning)
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
    return fig
