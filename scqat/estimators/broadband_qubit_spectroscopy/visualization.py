"""Visualization for broadband qubit spectroscopy.

Renders standard two-tone qubit spectroscopy plots:
- Top panel: Raw radial signal |IQ - ref|, baseline, and overlaid Lorentzian fits at detected peaks.
- Bottom panel: Baseline-subtracted corrected signal with Lorentzian fit curves and zero reference line.
"""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_broadband_qubit_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Plot standard qubit spectroscopy figure drawn from plot_data."""
    freq = plot_data.coords["frequency"].values.astype(float)
    freq_ghz = freq / 1e9
    signal = plot_data["signal"].values.astype(float)
    baseline = plot_data["baseline"].values.astype(float)
    corrected = plot_data["signal_corrected"].values.astype(float)
    n_peaks = int(plot_data.attrs.get("n_peaks", 0))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 6), dpi=120,
    )

    # -- Top Panel: Raw signal + Baseline + Lorentzian peak fits --
    ax_top.plot(freq_ghz, signal, "-", lw=0.8, color="#1f77b4", label=r"Signal $|IQ - \mathrm{ref}|$")
    ax_top.plot(freq_ghz, baseline, "--", color="gray", lw=1.0, label="Baseline")

    for i in range(n_peaks):
        if "peak_fit" in plot_data:
            fit_y = plot_data["peak_fit"].isel(peak=i).values.astype(float)
            pk_f = float(plot_data["peak_freq"].values[i])
            pk_fwhm = float(plot_data["peak_fwhm"].values[i])
            color = f"C{i + 1}"

            # Only plot non-nan fit regions
            valid_mask = np.isfinite(fit_y)
            if np.any(valid_mask):
                ax_top.plot(
                    freq_ghz[valid_mask],
                    fit_y[valid_mask] + baseline[valid_mask],
                    "-",
                    lw=1.5,
                    color=color,
                    label=f"peak {i}: {pk_f / 1e9:.4f} GHz, FWHM={pk_fwhm / 1e6:.2f} MHz",
                )
            ax_top.axvline(pk_f / 1e9, color=color, ls=":", lw=1.0, alpha=0.7)

    ax_top.set_ylabel("Signal (arb. u.)", fontsize=11)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.set_title("Broadband Qubit Spectroscopy", fontsize=12, fontweight="bold")
    ax_top.grid(True, linestyle="--", alpha=0.5)

    # -- Bottom Panel: Baseline-subtracted corrected signal + Lorentzian fits --
    ax_bot.plot(freq_ghz, corrected, "-", lw=0.8, color="#1f77b4")
    for i in range(n_peaks):
        if "peak_fit" in plot_data:
            fit_y = plot_data["peak_fit"].isel(peak=i).values.astype(float)
            pk_f = float(plot_data["peak_freq"].values[i])
            color = f"C{i + 1}"
            valid_mask = np.isfinite(fit_y)
            if np.any(valid_mask):
                ax_bot.plot(freq_ghz[valid_mask], fit_y[valid_mask], "-", lw=1.5, color=color)
            ax_bot.axvline(pk_f / 1e9, color=color, ls=":", lw=1.0, alpha=0.7)

    ax_bot.axhline(0, color="k", lw=0.5)
    ax_bot.set_xlabel("Drive Frequency (GHz)", fontsize=11)
    ax_bot.set_ylabel("Corrected", fontsize=11)
    ax_bot.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
