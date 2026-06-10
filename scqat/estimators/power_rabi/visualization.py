"""
Power-Rabi plotting helper.

Consumes the **plot_data** Dataset built by ``PowerRabiEstimator.build_plot_data``
and draws without any recalculation.

plot_data layout
----------------
coords : ``amp_prefactor``
vars   : ``signal`` (amp_prefactor), ``best_fit`` (amp_prefactor)
attrs  : ``a``, ``f``, ``phi``, ``c``, ``opt_amp_prefactor``, ``success``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_amplitude_fit(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the raw power-Rabi signal and cosine fit, marking the optimal pi-pulse
    amplitude prefactor.

    Draws strictly from the ``plot_data`` Dataset produced by
    ``PowerRabiEstimator.build_plot_data`` — variables ``signal`` and ``best_fit``
    over the ``amp_prefactor`` coordinate, with the fit parameters in ``.attrs`` — so
    the figure can be reconstructed downstream without rerunning the analysis.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    amp_prefactor = plot_data.coords["amp_prefactor"].values
    ax.plot(amp_prefactor, plot_data["signal"].values, ".", label="Raw Data", markersize=3)

    if "best_fit" in plot_data:
        ax.plot(amp_prefactor, plot_data["best_fit"].values, "-", label="Fit", linewidth=2)

    opt = plot_data.attrs.get("opt_amp_prefactor", float("nan"))
    if opt == opt:  # not NaN
        ax.axvline(opt, color="red", linestyle="--", label=f"opt prefactor = {opt:.4g}")

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

    ax.set_xlabel("Amplitude prefactor", fontsize=16)
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
        f"opt prefactor = {attrs.get('opt_amp_prefactor', float('nan')):.4g}",
        f"a = {attrs.get('a', float('nan')):.4g}",
        f"f = {attrs.get('f', float('nan')):.4g}",
        f"ϕ = {attrs.get('phi', float('nan')):.4g}",
        f"c = {attrs.get('c', float('nan')):.4g}",
    ]
    return "\n".join(lines)
