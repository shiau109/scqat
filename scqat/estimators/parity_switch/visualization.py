"""Parity-switch plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ParitySwitchEstimator.build_plot_data`` and draws without any recalculation.

Two time-domain figures, because they answer different questions:

* ``plot_trace`` — the MEASURED state per shot, i.e. which readout pole the
  qubit landed on.
* ``plot_parity`` — the DERIVED parity of each consecutive pair (even = the two
  shots agree, odd = they differ), which is what a parity switch actually looks
  like in the data. One shorter than the shot trace, and each point is timed
  between its two shots.

Both are SCATTER, never a step or line plot. At these shot counts a connected
trace fills solid black and shows nothing — a real run is ~1e6 shots, and even
a 2000-shot window renders as one filled block. Scattered markers keep the
occupancy of each level readable.

plot_data layout
----------------
coords : ``shot_idx``, ``time_s`` (shot_idx), ``pair_idx``,
         ``pair_time_s`` (pair_idx), ``psd_freq_hz``
vars   : ``state`` (shot_idx, 0/1), ``parity`` (pair_idx, 0/1),
         ``psd`` / ``psd_fit`` (psd_freq_hz),
         optional ``iq_i`` / ``iq_q`` (iq_idx — the shared IQ-plane panel)
attrs  : ``parity_rate_hz``, ``psd_corner_hz``, ``psd_amplitude``,
         ``psd_white_floor``, ``n_transitions``, ``p_excited``, ``success``,
         ``dt_s``, ``state_source``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: points drawn in the time-domain snippets. A full run is ~1e6 shots; past a
#: few thousand markers the panel saturates regardless of marker size, and the
#: PSD figure is the quantitative view anyway.
_TRACE_SNIPPET = 2000

_SCATTER = dict(s=4, alpha=0.5, edgecolors="none")


def _annotate(ax, lines) -> None:
    ax.text(
        0.98, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def plot_trace(plot_data: xr.Dataset) -> plt.Figure:
    """Scatter the first shots of the MEASURED 0/1 readout trace."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    state = plot_data["state"].values
    t = plot_data.coords["time_s"].values
    n = int(min(state.size, _TRACE_SNIPPET))
    ax.scatter(t[:n] * 1e3, state[:n], **_SCATTER)

    attrs = plot_data.attrs
    _annotate(ax, [
        f"rate = {float(attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
        f"p_excited = {float(attrs.get('p_excited', float('nan'))):.3g}",
        f"shot period = {float(attrs.get('dt_s', float('nan'))) * 1e6:.4g} us",
    ])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["|0>", "|1>"])
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Measured state", fontsize=14)
    ax.set_title(f"Measured state — first {n} of {state.size} shots", fontsize=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_parity(plot_data: xr.Dataset) -> plt.Figure:
    """Scatter the DERIVED parity of each consecutive shot pair.

    even (0) = the two shots agree, odd (1) = they differ. The odd fraction is
    annotated because it is the sanity check the PSD fit cannot do on its own:
    for a resolved telegraph it is the per-shot switching probability, well
    under 0.5. At ~0.5 consecutive shots are uncorrelated — the switching is
    faster than the shot cadence (or the discrimination is noise), the spectrum
    is white, and any fitted knee is meaningless.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    parity = plot_data["parity"].values
    if parity.size == 0:
        ax.set_title("Parity — too few shots (needs at least 2)", fontsize=10)
        fig.tight_layout()
        plt.close(fig)
        return fig

    t = plot_data.coords["pair_time_s"].values
    n = int(min(parity.size, _TRACE_SNIPPET))
    ax.scatter(t[:n] * 1e3, parity[:n], **_SCATTER)

    p_odd = float(np.mean(parity))
    lines = [
        f"rate = {float(plot_data.attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
        f"odd fraction = {p_odd:.3g}",
        f"odd = {int(np.count_nonzero(parity))} / {parity.size} pairs",
    ]
    if p_odd > 0.4:
        # not decoration: see the docstring — this is the regime where the
        # fitted rate stops meaning anything.
        lines.append("WARNING: ~uncorrelated shots,")
        lines.append("switching faster than the cadence?")
    _annotate(ax, lines)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["even", "odd"])
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Parity of pair (i, i+1)", fontsize=14)
    ax.set_title(f"Parity — first {n} of {parity.size} pairs", fontsize=10)
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
