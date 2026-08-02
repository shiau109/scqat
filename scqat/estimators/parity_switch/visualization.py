"""Parity-switch plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ParitySwitchEstimator.build_plot_data`` and draws without any recalculation.

Three panels, and which series each shows matters:

* ``plot_timetrace`` — a DEBUG panel: the raw READOUT and the derived PARITY
  stacked on a shared time axis. Under this no-reset sequence the readout is the
  running XOR of the parity, not the parity itself, so seeing the two together
  is the point. Eyeball only — nothing quantitative is read off it.
* ``plot_psd`` — the PARITY's spectrum with the Lorentzian-knee fit. **This is
  the measurement.**
* ``plot_state_psd`` — the READOUT's spectrum, raw and deliberately UNFITTED
  (it is an integrated telegraph, so a knee on it would mean nothing).

The time-domain panels are SCATTER, never a step or line plot. A real run is
~1e6 shots; a connected trace fills solid and shows nothing.

plot_data layout
----------------
coords : ``shot_idx``, ``time_s`` (shot_idx), ``pair_idx``,
         ``pair_time_s`` (pair_idx), ``psd_freq_hz``, ``state_psd_freq_hz``
vars   : ``state`` (shot_idx, 0/1), ``parity`` (pair_idx, 0/1),
         ``psd`` / ``psd_fit`` (psd_freq_hz), ``state_psd`` (state_psd_freq_hz),
         optional ``iq_i`` / ``iq_q`` (iq_idx — the shared IQ-plane panel)
attrs  : ``parity_rate_hz``, ``psd_corner_hz``, ``psd_amplitude``,
         ``psd_white_floor``, ``n_transitions``, ``p_switch``, ``p_high``,
         ``p_parity_odd``, ``p_state_high``, ``mapping_fidelity``,
         ``mapping_fidelity_floor``, ``mapping_fidelity_ratio``, ``psd_model``,
         ``psd_fit_residual``, ``success``, ``dt_s``, ``state_source``

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

#: points drawn in the time-domain debug panel. Deliberately small: the panel
#: is for confirming the two series look like a telegraph and its running XOR,
#: and the PSD figure is the quantitative view. At a ~30 us shot this window is
#: ~3 ms, far shorter than one switching period — do NOT try to count
#: transitions here.
_TRACE_SNIPPET = 100

#: markers are large because only ~100 points are drawn; at the old 1000-point
#: window they had to be tiny to stay legible.
_SCATTER = dict(s=16, alpha=0.75, edgecolors="none")


def _annotate(ax, lines, loc="upper right") -> None:
    x, ha = (0.98, "right") if loc.endswith("right") else (0.02, "left")
    ax.text(
        x, 0.98, "\n".join(lines),
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )


def plot_timetrace(plot_data: xr.Dataset) -> plt.Figure:
    """Debug panel: raw READOUT over the derived PARITY, shared time axis.

    The two series are one differencing step apart, which is the whole reason
    they are drawn together — the sequence carries no qubit reset and is a
    unitary, so each outcome inverts with the pole the previous shot left
    behind (``s[i] = s[i-1] XOR parity[i]``). The top panel is therefore the
    running XOR of the bottom one, and only the bottom one is fitted.

    Diagnostic only. The window is ~100 shots, orders of magnitude shorter than
    a switching period on a real chip, so the parity row will usually look
    constant — that is expected, not a fault.
    """
    # no gridspec hspace here — it makes the axes incompatible with the
    # tight_layout call below and matplotlib warns
    fig, (ax_s, ax_p) = plt.subplots(2, 1, figsize=(9, 5), dpi=100, sharex=True)
    attrs = plot_data.attrs

    state = plot_data["state"].values
    t = plot_data.coords["time_s"].values
    n = int(min(state.size, _TRACE_SNIPPET))
    ax_s.scatter(t[:n] * 1e3, state[:n], **_SCATTER)

    # the mean over the WHOLE trace, not just the window drawn — it is the
    # statistic, and a window-local mean would move as the snippet changed
    mean = float(attrs.get("p_state_high", float("nan")))
    if np.isfinite(mean):
        ax_s.axhline(mean, color="tab:orange", linestyle="--", linewidth=1.2,
                     label=f"mean = {mean:.4g} (all {state.size} shots)")
        ax_s.legend(fontsize=8, loc="lower right")
    _annotate(ax_s, [
        f"shot period = {float(attrs.get('dt_s', float('nan'))) * 1e6:.4g} us",
        f"{state.size} shots total",
    ])
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["|0>", "|1>"])
    # headroom for the annotation box: flush against the rows it hides them
    ax_s.set_ylim(-0.4, 2.1)
    ax_s.set_ylabel("Readout", fontsize=12)
    ax_s.set_title(f"Debug: raw readout (running XOR) and the derived parity "
                   f"— first {n} points", fontsize=10)

    parity = plot_data["parity"].values
    if parity.size:
        tp = plot_data.coords["pair_time_s"].values
        m = int(min(parity.size, _TRACE_SNIPPET))
        ax_p.scatter(tp[:m] * 1e3, parity[:m], color="tab:green", **_SCATTER)
        _annotate(ax_p, [
            f"rate = {float(attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
            f"XOR=1 fraction = "
            f"{float(attrs.get('p_parity_odd', float('nan'))):.3g}"
            f"  (~0.5 expected)",
            f"switch fraction = "
            f"{float(attrs.get('p_switch', float('nan'))):.4g}",
        ])
    else:
        ax_p.set_title("Parity — too few shots (needs at least 2)", fontsize=9)

    ax_p.set_yticks([0, 1])
    # NOT "even"/"odd": this measurement fixes the parity only up to a global
    # label, so which value is the even charge parity is unknowable from it.
    # The plotted quantity is literally s[i] XOR s[i+1].
    ax_p.set_yticklabels(["XOR=0", "XOR=1"])
    ax_p.set_ylim(-0.4, 2.1)
    ax_p.set_xlabel("Time (ms)", fontsize=12)
    ax_p.set_ylabel("Charge parity", fontsize=12)

    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD **of the parity** with the Lorentzian-knee fit.

    The fitted ``A / (1 + (f/f_c)^2) + B`` is the reference RTS spectrum
    ``4F^2*G / ((2G)^2 + (2*pi*f)^2) + (1-F^2)*dt`` in different variables. Under
    the ``constrained`` model F is a single shared parameter (dt fixed); under
    ``independent`` the plateau and floor each give F on their own and their
    ratio is the model-consistency check — see the ``telegraph_psd`` docstring.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    attrs = plot_data.attrs
    model = str(attrs.get("psd_model", ""))

    freq = plot_data.coords["psd_freq_hz"].values
    ax.loglog(freq, plot_data["psd"].values, ".", markersize=3,
              label="parity PSD (Welch)")
    fit = plot_data["psd_fit"].values
    if np.any(np.isfinite(fit)):
        fit_label = {"constrained": "shared-F fit (F, $\\Gamma$; dt fixed)",
                     "independent": "free-floor fit (A, $f_c$, B)"}.get(
                         model, "Lorentzian knee fit")
        ax.loglog(freq, fit, "-", linewidth=2, label=fit_label)

    corner = float(attrs.get("psd_corner_hz", float("nan")))
    if np.isfinite(corner):
        rate = float(attrs.get("parity_rate_hz", float("nan")))
        ax.axvline(corner, color="red", linestyle="--", linewidth=1,
                   label=f"corner $f_c$ = {corner:.4g} Hz "
                         f"-> $\\Gamma = \\pi f_c$ = {rate:.4g} Hz")

    # the low-frequency edge: how slow a rate this record length could see at
    # all. Drawn because a corner sitting near it is the readable symptom of
    # "record for longer".
    f_min = float(attrs.get("psd_freq_min_hz", float("nan")))
    margin = float(attrs.get("corner_margin_low", float("nan")))
    if np.isfinite(f_min):
        ax.axvline(f_min, color="tab:green", linestyle=":", linewidth=1.2,
                   label=f"lowest bin {f_min:.3g} Hz (= 8 / record time)")

    model_label = {"constrained": "model: constrained (shared F)",
                   "independent": "model: independent (free floor)"}.get(model)
    lines = [model_label] if model_label else []
    lines.append(f"contrast A/B = {float(attrs.get('psd_contrast', float('nan'))):.4g}")
    # constrained: ONE fitted F (floor/ratio are NaN -> the two lines below drop
    # out). independent: F read TWICE from the SAME fit and printed together,
    # because their gap IS the diagnostic — a Lorentzian-plus-WHITE-floor model
    # that does not fit (correlated readout error does exactly that) shows a big
    # ratio the contrast number cannot see.
    f_amp = float(attrs.get("mapping_fidelity", float("nan")))
    f_floor = float(attrs.get("mapping_fidelity_floor", float("nan")))
    if np.isfinite(f_amp):
        lines.append(f"F {'(plateau)' if np.isfinite(f_floor) else ''}= {f_amp:.3g}")
    if np.isfinite(f_floor):
        lines.append(f"F (floor)   = {f_floor:.3g}")
    ratio = float(attrs.get("mapping_fidelity_ratio", float("nan")))
    if np.isfinite(ratio):
        lines.append(f"F ratio = {ratio:.3g}" +
                     ("" if 0.5 <= ratio <= 2.0 else "  <- model mismatch"))
    resid = float(attrs.get("psd_fit_residual", float("nan")))
    if np.isfinite(resid):
        lines.append(f"fit residual = {resid:.3g}")
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
