"""Plotting for the qubit flux-arch composite — draws ONLY from plot_data."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_arch(plot_data: xr.Dataset) -> plt.Figure:
    """2-D signal map over (flux, absolute frequency) with the selected 0-1
    branch points and the fitted transmon arch (sweet spot marked)."""
    flux = plot_data["flux_bias"].values
    fig, ax = plt.subplots(figsize=(8, 5))

    if int(plot_data.attrs.get("has_full_freq", 0)):
        freq_ghz = plot_data["full_freq"].values * 1e-9
        ax.pcolormesh(flux, freq_ghz, plot_data["amplitude"].values.T, shading="auto")
        ax.set_ylabel("drive frequency (GHz)")
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
        if float(np.min(flux)) <= ss <= float(np.max(flux)):
            ax.plot([ss], [f01], "*", color="yellow", markersize=14, label="sweet spot")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(
            f"f01(flux): sweet spot {ss:.4g} V, f01_max {f01:.4f} GHz, "
            f"Ej_sum {plot_data.attrs['ej_sum_ghz']:.1f} GHz"
        )
    else:
        ax.set_title("f01(flux): arch fit failed (point cloud only)")

    ax.set_xlabel("flux bias (V)")
    fig.tight_layout()
    return fig
