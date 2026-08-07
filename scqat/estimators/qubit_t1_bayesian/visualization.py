"""Adaptive Bayesian T1 plotting helpers.

Every function consumes the **plot_data** Dataset built by
``QubitT1BayesianEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``block_idx``, ``lab_time_s`` (block_idx; NaN when no timestamps);
         optional ``psd_freq_hz``, ``allan_tau_s``, ``evol_probe_idx``,
         ``t1_grid_s``, ``lin_wait_s``
vars   : ``t1_s``, ``k``, ``t1_ci_low_s``, ``t1_ci_high_s``; optional ``psd``,
         ``allan_dev``, ``t1_evol_s`` / ``k_evol`` / ``posterior_pdf``,
         ``p_lin`` / ``lin_best_fit``
attrs  : ``t1_median_s``, ``k_final_median``, ``ci``, ``n_blocks``,
         ``psd_dt_s``, ``t1_lin_s``, ``t1_lin_ratio``, ``success``,
         ``has_lab_time``, ``has_evolution``, ``has_validation``,
         ``validation_disagrees``
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
    """T1 vs lab time with the posterior credible band."""
    fig, ax = plt.subplots(figsize=(9, 5), dpi=100)
    attrs = plot_data.attrs

    x, xlabel = _time_axis(plot_data)
    t1_us = plot_data["t1_s"].values * 1e6
    lo_us = plot_data["t1_ci_low_s"].values * 1e6
    hi_us = plot_data["t1_ci_high_s"].values * 1e6

    finite = np.isfinite(t1_us)
    # raw trace drawn unconditionally (finite subset); band guarded below
    ax.plot(x[finite], t1_us[finite], "o-", markersize=3, alpha=0.7,
            label="T1 (adaptive Bayes)")
    band = finite & np.isfinite(lo_us) & np.isfinite(hi_us)
    if np.any(band):
        ci_pct = int(round(float(attrs.get("ci", 0.9)) * 100))
        ax.fill_between(x[band], lo_us[band], hi_us[band], alpha=0.25,
                        label=f"{ci_pct}% credible interval")

    t1_lin_us = float(attrs.get("t1_lin_s", float("nan"))) * 1e6
    if np.isfinite(t1_lin_us):
        ax.axhline(t1_lin_us, color="gray", linestyle=":", linewidth=1.2,
                   label=f"validation fit = {t1_lin_us:.1f} us")

    lines = [
        f"median T1 = {float(attrs.get('t1_median_s', float('nan'))) * 1e6:.2f} us",
        f"median final k = {float(attrs.get('k_final_median', float('nan'))):.1f}",
    ]
    if int(attrs.get("validation_disagrees", 0)):
        lines.append("!! adaptive vs validation disagree —")
        lines.append("   check t1_prior_s")
    _annotate(ax, lines)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("T1 (us)", fontsize=12)
    ax.set_title("T1 tracking — adaptive Bayesian estimation (u = 1/k)",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_posterior_evolution(plot_data: xr.Dataset) -> plt.Figure:
    """P(T1) sharpening probe-by-probe over the LAST block: the posterior
    over T1 = 1/Gamma1 is inverse-gamma(shape=k, scale=k*T1); watching it
    narrow is the convergence check."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    grid_us = plot_data.coords["t1_grid_s"].values * 1e6
    steps = plot_data.coords["evol_probe_idx"].values
    pdf = plot_data["posterior_pdf"].values
    pcm = ax.pcolormesh(grid_us, steps, np.nan_to_num(pdf), cmap="viridis",
                        shading="auto")
    fig.colorbar(pcm, ax=ax, label="P(T1)")

    t1_ev_us = plot_data["t1_evol_s"].values * 1e6
    finite = np.isfinite(t1_ev_us)
    if np.any(finite):
        ax.plot(t1_ev_us[finite], steps[finite], "w.-", lw=1, ms=3,
                label="T1 estimate")
        ax.legend(fontsize=8, loc="upper right")

    k_ev = plot_data["k_evol"].values
    if np.any(np.isfinite(k_ev)) and np.any(finite):
        ax.set_title(
            f"Posterior evolution (last block) — final T1 = "
            f"{t1_ev_us[finite][-1]:.1f} us, k = {k_ev[np.isfinite(k_ev)][-1]:.1f}",
            fontsize=10,
        )
    else:
        ax.set_title("Posterior evolution (last block)", fontsize=10)
    ax.set_xlabel("T1 (us)", fontsize=12)
    ax.set_ylabel("Bayesian update step", fontsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD of the T1 fluctuation trace — unfitted diagnostic."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    freq = plot_data.coords["psd_freq_hz"].values
    psd = plot_data["psd"].values
    ax.loglog(freq, psd, ".-", markersize=3)
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD of T1 (s$^2$/Hz)", fontsize=12)
    ax.set_title("T1 fluctuation spectrum (Welch, unfitted)", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_allan(plot_data: xr.Dataset) -> plt.Figure:
    """Allan deviation of the T1 trace: where averaging stops helping."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    tau = plot_data.coords["allan_tau_s"].values
    adev_us = plot_data["allan_dev"].values * 1e6
    mask = np.isfinite(tau) & np.isfinite(adev_us) & (tau > 0) & (adev_us > 0)
    if np.any(mask):
        ax.loglog(tau[mask], adev_us[mask], "o-")
    else:
        ax.set_title("Allan deviation — no finite points", fontsize=10)
    ax.set_xlabel("Averaging time (s)", fontsize=12)
    ax.set_ylabel("Allan deviation of T1 (us)", fontsize=12)
    ax.set_title("T1 Allan deviation", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_validation(plot_data: xr.Dataset) -> plt.Figure:
    """The interleaved NON-adaptive decay with its exponential fit — the
    classical cross-check on the adaptive estimate. A disagreement means the
    PRIOR was wrong, not the chip."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    attrs = plot_data.attrs

    wait_us = plot_data.coords["lin_wait_s"].values * 1e6
    p = plot_data["p_lin"].values
    # raw curve drawn unconditionally; fit overlay guarded
    ax.plot(wait_us, p, "o", markersize=4, label="interleaved non-adaptive")
    fit = plot_data["lin_best_fit"].values
    if np.any(np.isfinite(fit)):
        t1_lin_us = float(attrs.get("t1_lin_s", float("nan"))) * 1e6
        ax.plot(wait_us, fit, "--", color="gray",
                label=f"exp fit: T1 = {t1_lin_us:.1f} us")
    ax.axhline(0.5, color="k", linestyle=":", alpha=0.3)

    lines = [
        f"adaptive median T1 = "
        f"{float(attrs.get('t1_median_s', float('nan'))) * 1e6:.1f} us",
    ]
    ratio = float(attrs.get("t1_lin_ratio", float("nan")))
    if np.isfinite(ratio):
        lines.append(f"adaptive / validation = {ratio:.2f}"
                     + ("  <- disagreement" if int(attrs.get("validation_disagrees", 0))
                        else ""))
    _annotate(ax, lines)

    ax.set_xlabel("Wait time (us)", fontsize=12)
    ax.set_ylabel("P(|1>)", fontsize=12)
    ax.set_title("Interleaved validation: non-adaptive decay vs adaptive Bayes",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
