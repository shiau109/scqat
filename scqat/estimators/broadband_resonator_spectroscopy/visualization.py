"""Visualization for broadband resonator spectroscopy.

Renders the concatenated wideband transmission spectrum (magnitude in dB and
unwrapped phase) with candidate resonator dips marked and annotated.
"""

from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_broadband_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the wideband transmission spectrum and mark candidate dips."""
    freq = plot_data.coords["frequency"].values.astype(float)
    freq_ghz = freq / 1e9
    mag_db = plot_data["mag_db"].values.astype(float)
    baseline_db = plot_data["baseline_db"].values.astype(float)
    phase_rad = plot_data["phase_rad"].values.astype(float)

    dips_json = str(plot_data.attrs.get("dips_json", "[]"))
    try:
        dips = json.loads(dips_json)
    except Exception:
        dips = []

    fig, (ax_mag, ax_phase) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    # Top Panel: Magnitude in dB
    ax_mag.plot(freq_ghz, mag_db, label=r"$|S_{21}|$ (dB)", color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax_mag.plot(freq_ghz, baseline_db, label="Baseline", color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8)

    y_min, y_max = np.nanmin(mag_db), np.nanmax(mag_db)
    y_range = max(y_max - y_min, 1.0)
    ax_mag.set_ylim(y_min - 0.08 * y_range, y_max + 0.15 * y_range)

    # Annotate dips
    colors = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i, d in enumerate(dips):
        c = colors[i % len(colors)]
        f_val = d["frequency_hz"]
        f_ghz = f_val / 1e9
        rank = d.get("rank", i + 1)
        depth = d.get("depth_db", 0.0)
        ql = d.get("ql", 0.0)
        prom = d.get("prominence_db", 0.0)

        # Mark with vertical line
        ax_mag.axvline(f_ghz, color=c, linestyle=":", alpha=0.85, linewidth=1.5)
        ax_phase.axvline(f_ghz, color=c, linestyle=":", alpha=0.85, linewidth=1.5)

        # Find closest point in mag_db for marker
        idx = int(np.argmin(np.abs(freq - f_val)))
        val_at_dip = mag_db[idx]
        ax_mag.plot(f_ghz, val_at_dip, "v", color=c, markersize=8, zorder=5)

        # Annotation text
        label_text = f"#{rank}: {f_ghz:.4f} GHz\n" f"Drop: {depth:.1f} dB (Q~{ql:.1e})" if ql > 0 else f"#{rank}: {f_ghz:.4f} GHz\nDrop: {depth:.1f} dB"
        ax_mag.annotate(
            label_text,
            xy=(f_ghz, val_at_dip),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color=c,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.85, lw=1),
            arrowprops=dict(arrowstyle="->", color=c, lw=1),
        )

    ax_mag.set_ylabel(r"$|S_{21}|$ (dB)", fontsize=11)
    ax_mag.set_title(
        f"Broadband Resonator Spectrum — {len(dips)} Candidate Dip(s) Marked",
        fontsize=12,
        fontweight="bold",
    )
    ax_mag.grid(True, linestyle="--", alpha=0.6)
    ax_mag.legend(loc="upper right", fontsize=9)

    # Bottom Panel: Detrended Phase (Electrical Delay Compensated)
    delay_s = float(plot_data.attrs.get("cable_delay_s", 0.0))
    delay_str = f" [$\\tau \\approx {delay_s * 1e9:.1f}$ ns]" if abs(delay_s) > 1e-12 else ""
    ax_phase.plot(freq_ghz, phase_rad, label=f"Detrended Phase{delay_str}", color="#2ca02c", linewidth=1.2)
    ax_phase.axhline(0, color="gray", linestyle=":", alpha=0.5)

    p_min, p_max = np.nanmin(phase_rad), np.nanmax(phase_rad)
    p_range = max(p_max - p_min, 0.5)
    ax_phase.set_ylim(p_min - 0.1 * p_range, p_max + 0.1 * p_range)

    for i, d in enumerate(dips):
        c = colors[i % len(colors)]
        f_val = d["frequency_hz"]
        f_ghz = f_val / 1e9
        idx = int(np.argmin(np.abs(freq - f_val)))
        p_at_dip = phase_rad[idx]
        ax_phase.plot(f_ghz, p_at_dip, "o", color=c, markersize=5, zorder=5)

    ax_phase.set_xlabel("Frequency (GHz)", fontsize=11)
    ax_phase.set_ylabel("Detrended Phase (rad)", fontsize=11)
    ax_phase.grid(True, linestyle="--", alpha=0.6)
    ax_phase.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig
