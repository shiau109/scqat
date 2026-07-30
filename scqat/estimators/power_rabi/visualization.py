"""
Power-Rabi plotting helper.

Consumes the **plot_data** Dataset built by ``PowerRabiEstimator.build_plot_data``
and draws without any recalculation.

plot_data layout
----------------
coords : ``amp_prefactor``
vars   : ``signal`` (amp_prefactor), ``best_fit`` (amp_prefactor),
         optional ``twin`` (amp_prefactor) — a companion scale for the same points
attrs  : ``a``, ``f``, ``phi``, ``c``, ``opt_amp_prefactor``, ``success``;
         with a twin, also ``twin_label`` and ``opt_twin_value``
"""

import matplotlib.pyplot as plt
import xarray as xr

from scqat.estimators._twin_axis import add_twin_axis


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

    # the same sweep in the caller's companion scale (e.g. absolute amplitude),
    # drawn on top so the primary reading and the marker keep their meaning
    if "twin" in plot_data:
        add_twin_axis(
            ax, amp_prefactor, plot_data["twin"].values,
            str(plot_data.attrs.get("twin_label", "")),
        )

    fig.tight_layout()
    plt.close(fig)
    return fig


def _build_param_text(attrs: dict) -> str:
    """Build a formatted string of cosine-fit parameters for annotation, reading the
    parameters stored in ``plot_data.attrs``."""
    lines = [
        f"opt prefactor = {attrs.get('opt_amp_prefactor', float('nan')):.4g}",
    ]
    if "opt_twin_value" in attrs:  # the same answer in the companion scale,
        # which the secondary axis already names — keep the box narrow
        lines.append(f"opt abs = {attrs['opt_twin_value']:.4g}")
    lines += [
        f"a = {attrs.get('a', float('nan')):.4g}",
        f"f = {attrs.get('f', float('nan')):.4g}",
        f"ϕ = {attrs.get('phi', float('nan')):.4g}",
        f"c = {attrs.get('c', float('nan')):.4g}",
    ]
    return "\n".join(lines)
