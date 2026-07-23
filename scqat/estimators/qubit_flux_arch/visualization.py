"""Plotting for the qubit flux-arch composite — draws ONLY from plot_data."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_arch(plot_data: xr.Dataset) -> plt.Figure:
    """2-D signal map over (flux, absolute frequency) with the selected 0-1
    branch points and the fitted transmon arch (sweet spot marked)."""
    flux = plot_data["flux_bias"].values
    fig, ax = plt.subplots(figsize=(8, 5))

    # Clamp the y-axis to the MEASURED frequency window (the heatmap extent). The
    # arch fit extrapolates far past it — up toward the sweet spot and down to ~0
    # at the flux edges — so letting it drive autoscale squeezes the raw-data
    # heatmap into a thin strip.
    # Background = the per-slice REDUCED signal the peaks were fitted on (a weak
    # line is visible here even when it vanishes under the raw |IQ| ground offset
    # + chain ripple). Raw "amplitude" only as fallback for plotdata saved before
    # the reduced map existed (immutable run data).
    background = plot_data["reduced"] if "reduced" in plot_data else plot_data["amplitude"]
    ylim = None
    if int(plot_data.attrs.get("has_full_freq", 0)):
        freq_ghz = plot_data["full_freq"].values * 1e-9
        ax.pcolormesh(flux, freq_ghz, background.values.T, shading="auto")
        ax.set_ylabel("drive frequency (GHz)")
        ylim = (float(np.min(freq_ghz)), float(np.max(freq_ghz)))
    else:  # pragma: no cover - composite requires full_freq upstream
        ax.set_ylabel("detuning (Hz)")

    if "sel_flux" in plot_data:
        used = plot_data["sel_used"].values.astype(bool)
        sf = plot_data["sel_flux"].values
        sq = plot_data["sel_freq_hz"].values * 1e-9
        ax.plot(sf[used], sq[used], "o", mfc="none", color="w", label="0-1 branch")
        if (~used).any():
            ax.plot(sf[~used], sq[~used], "x", color="r", label="rejected")
        ax.plot(
            plot_data["fit_flux"].values,
            plot_data["fit_freq_hz"].values * 1e-9,
            "-", color="orange", label="arch fit",
        )
        ss = float(plot_data.attrs["sweet_spot_flux"])
        f01 = float(plot_data.attrs["f01_max_hz"]) * 1e-9
        # only mark the sweet spot when it falls inside the visible window, so the
        # legend never carries a star that has been clipped off-screen
        in_window = ylim is None or ylim[0] <= f01 <= ylim[1]
        if float(np.min(flux)) <= ss <= float(np.max(flux)) and in_window:
            ax.plot([ss], [f01], "*", color="yellow", markersize=14, label="sweet spot")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(
            f"f01(flux): sweet spot {ss:.4g} V, f01_max {f01:.4f} GHz, "
            f"Ej_sum {plot_data.attrs['ej_sum_ghz']:.1f} GHz"
        )
    else:
        ax.set_title("f01(flux): arch fit failed (point cloud only)")

    if ylim is not None:
        ax.set_ylim(*ylim)  # applied last so the arch-fit line cannot re-expand it
    ax.set_xlabel("flux bias (V)")
    fig.tight_layout()
    return fig
