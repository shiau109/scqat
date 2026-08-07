"""ADE T1-tracking plotting helpers.

Every function consumes the **plot_data** Dataset built by
``QubitT1AdeEstimator.build_plot_data`` and draws without any recalculation.

plot_data layout
----------------
coords : ``block_idx``, ``lab_time_s`` (block_idx; NaN when no timestamps)
vars   : ``t1_s``, ``t1_sigma_s``, ``t1_boot_sigma_s`` (NaN when no bootstrap),
         ``dt_s``, ``clipped`` (0/1), ``gamma_fpga``, ``gamma_offline``
attrs  : ``t1_median_s``, ``t1_sigma_median_s``, ``t1_boot_sigma_median_s``,
         ``n_blocks``, ``n_valid``, ``n_clipped``, ``n_fpga_mismatch``,
         ``n_bootstrap``, ``success``, ``has_shots``, ``has_lab_time``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _time_axis(plot_data: xr.Dataset):
    """The x axis: hardware lab time when present, else the block index."""
    if int(plot_data.attrs.get("has_lab_time", 0)):
        return plot_data.coords["lab_time_s"].values, "Lab time (s)"
    return plot_data.coords["block_idx"].values.astype(float), "Block index"


def _annotate(ax, lines, loc="upper right") -> None:
    x, ha = (0.98, "right") if loc.endswith("right") else (0.02, "left")
    ax.text(
        x, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def plot_t1_trace(plot_data: xr.Dataset) -> plt.Figure:
    """T1 vs lab time with the analytic sigma band (and the bootstrap band
    when it was computed). Clipped blocks — outside the ADE validity domain,
    where the FPGA floors silently produce plausible numbers — are marked x
    and excluded from every drawn band."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    attrs = plot_data.attrs

    x, xlabel = _time_axis(plot_data)
    t1_us = plot_data["t1_s"].values * 1e6
    sigma_us = plot_data["t1_sigma_s"].values * 1e6
    boot_us = plot_data["t1_boot_sigma_s"].values * 1e6
    clipped = plot_data["clipped"].values.astype(bool)

    ok = np.isfinite(t1_us) & ~clipped
    # raw trace drawn unconditionally (finite subset); bands guarded below
    ax.plot(x[ok], t1_us[ok], "o-", markersize=3, alpha=0.7, label="T1 (ADE)")

    band = ok & np.isfinite(sigma_us)
    if np.any(band):
        ax.fill_between(x[band], (t1_us - sigma_us)[band], (t1_us + sigma_us)[band],
                        alpha=0.25, label="analytic $\\pm\\sigma$ (FPGA)")
    boot_band = ok & np.isfinite(boot_us)
    if np.any(boot_band):
        ax.plot(x[boot_band], (t1_us - boot_us)[boot_band], "--",
                color="tab:green", linewidth=1, label="bootstrap $\\pm\\sigma$")
        ax.plot(x[boot_band], (t1_us + boot_us)[boot_band], "--",
                color="tab:green", linewidth=1)
    if np.any(clipped):
        ax.plot(x[clipped], np.nan_to_num(t1_us[clipped]), "x", color="red",
                markersize=6, label=f"clipped ({int(np.sum(clipped))})")

    median_us = float(attrs.get("t1_median_s", float("nan"))) * 1e6
    if np.isfinite(median_us):
        ax.axhline(median_us, color="red", linestyle="--", linewidth=1)

    lines = [
        f"median T1 = {median_us:.2f} us" if np.isfinite(median_us) else "no valid estimate",
        f"median analytic sigma = "
        f"{float(attrs.get('t1_sigma_median_s', float('nan'))) * 1e6:.3g} us",
        f"{int(attrs.get('n_valid', 0))}/{int(attrs.get('n_blocks', 0))} blocks valid",
    ]
    boot_median = float(attrs.get("t1_boot_sigma_median_s", float("nan")))
    if np.isfinite(boot_median):
        lines.append(f"median bootstrap sigma = {boot_median * 1e6:.3g} us")
    mismatch = int(attrs.get("n_fpga_mismatch", 0))
    if mismatch:
        lines.append(f"FPGA vs offline mismatch on {mismatch} blocks")
    _annotate(ax, lines)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("T1 (us)", fontsize=12)
    ax.set_title("T1 tracking — Analytical Decay Estimation", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_dt_trace(plot_data: xr.Dataset) -> plt.Figure:
    """The delay spacing dt per block — flat on a fixed-dt run, a staircase
    when the probe adapted dt from the running estimate."""
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=100)
    x, xlabel = _time_axis(plot_data)
    dt_us = plot_data["dt_s"].values * 1e6
    finite = np.isfinite(dt_us)
    ax.step(x[finite], dt_us[finite], where="mid")
    adaptive = finite.sum() > 1 and np.nanstd(dt_us) > 0
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("dt (us)", fontsize=12)
    ax.set_title(
        f"ADE delay spacing per block ({'adaptive' if adaptive else 'fixed'})",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
