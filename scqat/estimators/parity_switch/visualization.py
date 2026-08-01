"""Parity-switch plotting helpers.

Both functions consume the **plot_data** Dataset built by
``ParitySwitchEstimator.build_plot_data`` and draw without any recalculation.

plot_data layout
----------------
coords : ``shot_idx``, ``time_s`` (shot_idx), ``psd_freq_hz``
vars   : ``state`` (shot_idx, 0/1), ``psd`` / ``psd_fit`` (psd_freq_hz),
         optional ``iq_i`` / ``iq_q`` (iq_idx — the shared IQ-plane panel)
attrs  : ``parity_rate_hz``, ``psd_corner_hz``, ``psd_amplitude``,
         ``psd_white_floor``, ``n_transitions``, ``p_excited``, ``success``,
         ``dt_s``, ``state_source``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: shots drawn in the telegraph snippet — a 1e5-point step plot is unreadable
#: and slow; the first few thousand shots show the switching character.
_TRACE_SNIPPET = 2000


def plot_trace(plot_data: xr.Dataset) -> plt.Figure:
    """Step-plot the first shots of the 0/1 telegraph with the rate annotation."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    state = plot_data["state"].values
    t = plot_data.coords["time_s"].values
    n = int(min(state.size, _TRACE_SNIPPET))
    ax.step(t[:n] * 1e3, state[:n], where="post", lw=0.8)

    attrs = plot_data.attrs
    textstr = "\n".join([
        f"rate = {float(attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
        f"p_excited = {float(attrs.get('p_excited', float('nan'))):.3g}",
        f"transitions = {attrs.get('n_transitions', 'n/a')} / {state.size} shots",
    ])
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    ax.set_yticks([0, 1])
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("State", fontsize=14)
    ax.set_title(f"Parity telegraph — first {n} of {state.size} shots",
                 fontsize=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD with the Lorentzian-knee fit and the corner marker."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    freq = plot_data.coords["psd_freq_hz"].values
    ax.loglog(freq, plot_data["psd"].values, ".", markersize=3,
              label="Welch PSD")
    fit = plot_data["psd_fit"].values
    if np.any(np.isfinite(fit)):
        ax.loglog(freq, fit, "-", linewidth=2, label="Lorentzian knee fit")

    attrs = plot_data.attrs
    corner = float(attrs.get("psd_corner_hz", float("nan")))
    if np.isfinite(corner):
        rate = float(attrs.get("parity_rate_hz", float("nan")))
        ax.axvline(corner, color="red", linestyle="--", linewidth=1,
                   label=f"corner {corner:.4g} Hz -> rate {rate:.4g} Hz")

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (1/Hz)", fontsize=14)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig
