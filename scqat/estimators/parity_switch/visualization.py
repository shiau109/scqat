"""Parity-switch plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ParitySwitchEstimator.build_plot_data`` and draws without any recalculation.

Four panels, and which series each shows matters:

* ``plot_trace`` — the raw READOUT per shot. Under this no-reset sequence that
  is the running XOR of the parity, not the parity itself.
* ``plot_parity`` — the PARITY, derived as the consecutive-pair difference.
  This is the measured telegraph and the series the rate comes from.
* ``plot_psd`` — the PARITY's spectrum with the Lorentzian-knee fit.
* ``plot_state_psd`` — the READOUT's spectrum, raw and deliberately UNFITTED
  (it is an integrated telegraph, so a knee on it would mean nothing).

The two time-domain panels are SCATTER, never a step or line plot. At these
shot counts a connected trace fills solid and shows nothing — a real run is
~1e6 shots, and even a 2000-shot window renders as one filled block.

plot_data layout
----------------
coords : ``shot_idx``, ``time_s`` (shot_idx), ``pair_idx``,
         ``pair_time_s`` (pair_idx), ``psd_freq_hz``, ``state_psd_freq_hz``
vars   : ``state`` (shot_idx, 0/1), ``parity`` (pair_idx, 0/1),
         ``psd`` / ``psd_fit`` (psd_freq_hz), ``state_psd`` (state_psd_freq_hz),
         optional ``iq_i`` / ``iq_q`` (iq_idx — the shared IQ-plane panel)
attrs  : ``parity_rate_hz``, ``psd_corner_hz``, ``psd_amplitude``,
         ``psd_white_floor``, ``n_transitions``, ``p_switch``, ``p_high``,
         ``p_parity_odd``, ``p_state_high``, ``success``, ``dt_s``,
         ``state_source``

Three level-fractions appear here and they are NOT interchangeable:
``p_parity_odd`` (== ``p_high``) is the PARITY's level, ~0.5 on a healthy run;
``p_state_high`` is the raw READOUT's mean; ``p_switch`` is how often the parity
CHANGES. Only the last says anything about the rate.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: below this much low-frequency headroom the plateau is thinly sampled and a
#: longer record would tighten the fit. Advisory only — the tool does not gate
#: on it, because the remedy is the caller's decision.
_LOW_MARGIN_ADVISORY = 5.0

#: points drawn in the time-domain snippets. A full run is ~1e6 shots; past a
#: thousand markers the panel saturates regardless of marker size, and the PSD
#: figure is the quantitative view anyway.
_TRACE_SNIPPET = 1000

_SCATTER = dict(s=4, alpha=0.5, edgecolors="none")


def _annotate(ax, lines) -> None:
    ax.text(
        0.98, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def plot_trace(plot_data: xr.Dataset) -> plt.Figure:
    """Scatter the first shots of the raw READOUT.

    Not the parity: with no qubit reset each outcome inverts with the pole the
    previous shot left behind, so this trace is the running XOR of the parity.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)

    state = plot_data["state"].values
    t = plot_data.coords["time_s"].values
    n = int(min(state.size, _TRACE_SNIPPET))
    ax.scatter(t[:n] * 1e3, state[:n], **_SCATTER)

    attrs = plot_data.attrs
    lines = [
        f"shot period = {float(attrs.get('dt_s', float('nan'))) * 1e6:.4g} us",
        f"shots = {state.size}",
    ]
    # the mean over the WHOLE trace, not just the window drawn — it is the
    # statistic, and a window-local mean would move as the snippet changed
    mean = float(attrs.get("p_state_high", float("nan")))
    if np.isfinite(mean):
        ax.axhline(mean, color="tab:orange", linestyle="--", linewidth=1.2,
                   label=f"mean = {mean:.4g} (all {state.size} shots)")
        ax.legend(fontsize=9, loc="center right")
        lines.append(f"mean readout = {mean:.4g}")
    _annotate(ax, lines)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["|0>", "|1>"])
    # headroom for the annotation box: at (-0.3, 1.3) it sat ON the upper row
    # and hid the last third of the trace
    ax.set_ylim(-0.35, 2.0)
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Readout", fontsize=14)
    ax.set_title(f"Raw readout (running XOR of the parity) — first {n} "
                 f"of {state.size} shots", fontsize=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_parity(plot_data: xr.Dataset) -> plt.Figure:
    """Scatter the PARITY — the consecutive-pair difference, which under this
    no-reset sequence is the measured telegraph.

    Two different fractions are annotated and they must not be confused:
    ``p_parity_odd`` is how often the chip sits in the odd parity (~0.5 is
    HEALTHY, it carries no rate information), while ``p_switch`` is how often
    the parity CHANGES between neighbouring samples — that one saturates at 0.5
    and is what decides whether a rate is recoverable at all.
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

    attrs = plot_data.attrs
    _annotate(ax, [
        f"rate = {float(attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
        f"odd parity = {float(attrs.get('p_parity_odd', float('nan'))):.3g}"
        f"  (~0.5 expected)",
        f"switch fraction = {float(attrs.get('p_switch', float('nan'))):.4g}",
    ])

    ax.set_yticks([0, 1])
    # NOT "even"/"odd": this measurement fixes the parity only up to a global
    # label, so which value is the even charge parity is unknowable from it.
    # The plotted quantity is literally s[i] XOR s[i+1].
    ax.set_yticklabels(["XOR=0", "XOR=1"])
    # headroom for the annotation box: at (-0.3, 1.3) it sat ON the upper row
    # and hid the last third of the trace
    ax.set_ylim(-0.35, 2.0)
    ax.set_xlabel("Time (ms)", fontsize=14)
    ax.set_ylabel("Charge parity", fontsize=14)
    ax.set_title(f"Parity (measured telegraph) — first {n} of "
                 f"{parity.size} samples", fontsize=10)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD **of the parity** with the Lorentzian-knee fit."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    freq = plot_data.coords["psd_freq_hz"].values
    ax.loglog(freq, plot_data["psd"].values, ".", markersize=3,
              label="parity PSD (Welch)")
    fit = plot_data["psd_fit"].values
    if np.any(np.isfinite(fit)):
        ax.loglog(freq, fit, "-", linewidth=2, label="Lorentzian knee fit")

    attrs = plot_data.attrs
    corner = float(attrs.get("psd_corner_hz", float("nan")))
    if np.isfinite(corner):
        rate = float(attrs.get("parity_rate_hz", float("nan")))
        ax.axvline(corner, color="red", linestyle="--", linewidth=1,
                   label=f"corner {corner:.4g} Hz -> rate {rate:.4g} Hz")

    # the low-frequency edge: how slow a rate this record length could see at
    # all. Drawn because a corner sitting near it is the readable symptom of
    # "record for longer".
    f_min = float(attrs.get("psd_freq_min_hz", float("nan")))
    margin = float(attrs.get("corner_margin_low", float("nan")))
    if np.isfinite(f_min):
        ax.axvline(f_min, color="tab:green", linestyle=":", linewidth=1.2,
                   label=f"lowest bin {f_min:.3g} Hz (= 8 / record time)")

    lines = [f"contrast A/B = {float(attrs.get('psd_contrast', float('nan'))):.4g}"]
    if np.isfinite(margin):
        lines.append(f"low-freq headroom = {margin:.2f}x")
        if margin < _LOW_MARGIN_ADVISORY:
            lines.append("thin plateau — a LONGER RECORD")
            lines.append("would tighten this fit")
    _annotate(ax, lines)

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (1/Hz)", fontsize=14)
    ax.set_title("Parity spectrum — this is the measurement", fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_state_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD of the raw READOUT — diagnostic, NEVER fitted.

    The readout is the running XOR of the parity, i.e. an integrated telegraph,
    so its spectrum rises toward low frequency and has no Lorentzian corner to
    read. It is shown because the raw spectrum is still worth eyeballing for
    pickup, drift and readout artefacts; deliberately no fit line is drawn, so
    it cannot be mistaken for the measurement.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if "state_psd" not in plot_data.data_vars \
            or plot_data["state_psd"].size == 0:
        ax.set_title("Raw readout spectrum — unavailable", fontsize=10)
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = plot_data.coords["state_psd_freq_hz"].values
    ax.loglog(freq, plot_data["state_psd"].values, ".", markersize=3,
              color="tab:gray", label="readout PSD (Welch, unfitted)")

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (1/Hz)", fontsize=14)
    ax.set_title("Raw readout spectrum — DIAGNOSTIC ONLY (integrated "
                 "telegraph; the rate comes from the parity)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig
