"""XY-Z delay plotting helper.

Consumes the **plot_data** Dataset built by ``XyzDelayEstimator.build_plot_data``
and draws without any recalculation.

plot_data layout
----------------
coords : ``relative_time``
vars   : ``difference`` (relative_time), ``best_fit`` (relative_time)
attrs  : ``t0_s``, ``t0_std_s``, ``amp``, ``offset``, ``half_width_s``,
         ``snr``, ``success``, ``reduction_method``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_delay_scan(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the |e> - |g> difference trace, the triangle fit, and the fitted
    delay, drawing strictly from ``plot_data`` so the figure reconstructs
    downstream without rerunning the analysis."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    relative_time = plot_data.coords["relative_time"].values
    ax.plot(relative_time, plot_data["difference"].values, ".", label="|e> - |g>", markersize=4)

    attrs = plot_data.attrs
    if "best_fit" in plot_data and bool(attrs.get("success", 0)):
        ax.plot(relative_time, plot_data["best_fit"].values, "-", label="triangle fit", linewidth=2)

    t0 = attrs.get("t0_s", float("nan"))
    ax.axvline(t0, color="k", linestyle="--", alpha=0.7, label=f"delay = {t0:.4g}")

    textstr = "\n".join([
        f"t0 = {t0:.4g}",
        f"t0 std = {attrs.get('t0_std_s', float('nan')):.4g}",
        f"SNR = {attrs.get('snr', float('nan')):.3g}",
        f"success = {bool(attrs.get('success', 0))}",
        f"reduction = {attrs.get('reduction_method', 'state')}",
    ])
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax.set_xlabel("Relative XY-Z time", fontsize=16)
    ax.set_ylabel("Readout contrast (|e> - |g>)", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig
