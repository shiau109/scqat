"""
Swap-oscillation (N-swap) plotting helper.

Consumes the **plot_data** Dataset built by ``SwapOscillationEstimator.build_plot_data``
and draws without any recalculation.

plot_data layout
----------------
coords : ``round``, ``round_dense``
vars   : ``signal`` (round), ``best_fit`` (round), ``best_fit_dense`` (round_dense)
attrs  : ``a``, ``f``, ``phi``, ``c``, ``swap_period``, ``success``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_rounds_fit(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the raw population-vs-N signal and cosine fit, annotating the extracted
    swap-oscillation frequency and period.

    Draws strictly from the ``plot_data`` Dataset produced by
    ``SwapOscillationEstimator.build_plot_data`` — variables ``signal`` and
    ``best_fit`` over the ``round`` coordinate, with the fit parameters in
    ``.attrs`` — so the figure can be reconstructed downstream without rerunning
    the analysis.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    rounds = plot_data.coords["round"].values
    ax.plot(rounds, plot_data["signal"].values, "o", label="Raw Data", markersize=4)

    # Prefer the dense curve: the sweep has few integer-N points, so the best-fit
    # sampled there draws as a jagged polyline.
    if "best_fit_dense" in plot_data:
        ax.plot(
            plot_data.coords["round_dense"].values,
            plot_data["best_fit_dense"].values,
            "-", label="Fit", linewidth=2,
        )
    elif "best_fit" in plot_data:
        ax.plot(rounds, plot_data["best_fit"].values, "-", label="Fit", linewidth=2)

    ax.legend()

    textstr = _build_param_text(plot_data.attrs)
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    ax.set_xlabel("Number of swaps N", fontsize=16)
    ax.set_ylabel("Signal", fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def _build_param_text(attrs: dict) -> str:
    """Build a formatted string of cosine-fit parameters for annotation, reading the
    parameters stored in ``plot_data.attrs``."""
    lines = [
        f"f = {attrs.get('f', float('nan')):.4g} /swap",
        f"period = {attrs.get('swap_period', float('nan')):.4g} swaps",
        f"a = {attrs.get('a', float('nan')):.4g}",
        f"ϕ = {attrs.get('phi', float('nan')):.4g}",
        f"c = {attrs.get('c', float('nan')):.4g}",
    ]
    return "\n".join(lines)
