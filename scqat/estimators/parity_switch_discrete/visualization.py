"""Discrete parity-switch plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ParitySwitchDiscreteEstimator.build_plot_data`` and draws without any
recalculation.

Three panels, and which series each shows matters:

* ``plot_timetrace`` — a DEBUG panel: the two per-cycle measurements M1 and M2
  and the derived PARITY (``m1 XOR m2``) stacked on a shared time axis. The
  parity is per-cycle here — each sample is computed within its own cycle, no
  chain across cycles. Eyeball only.
* ``plot_psd`` — the PARITY's spectrum with the Lorentzian-knee fit. **This is
  the measurement.**
* ``plot_state_psd`` — the raw M1 trace's spectrum, deliberately UNFITTED
  (absent errors, M1 is the running XOR of the parity — the QND chain
  integrates it — so a knee on it would mean nothing).

The time-domain panels are SCATTER, never a step or line plot. A real run is
~1e6 cycles; a connected trace fills solid and shows nothing.

plot_data layout
----------------
coords : ``shot_idx`` (cycle index), ``time_s`` (shot_idx), ``psd_freq_hz``,
         ``state_psd_freq_hz``
vars   : ``m1`` / ``m2`` / ``parity`` (shot_idx, 0/1),
         ``psd`` / ``psd_fit`` (psd_freq_hz), ``state_psd`` (state_psd_freq_hz),
         optional ``iq_i`` / ``iq_q`` (iq_idx — the shared IQ-plane panel)
attrs  : ``parity_rate_hz``, ``psd_corner_hz``, ``psd_amplitude``,
         ``psd_white_floor``, ``n_transitions``, ``p_switch``, ``p_high``,
         ``p_parity_odd``, ``p_intercycle_flip``, ``p_m1_high``, ``p_m2_high``,
         ``mapping_fidelity``, ``mapping_fidelity_floor``,
         ``mapping_fidelity_ratio``, ``psd_model``, ``psd_fit_residual``,
         ``success``, ``dt_s``, ``state_source``

The level-fractions are NOT interchangeable: ``p_parity_odd`` (== ``p_high``)
is the PARITY's level, ~0.5 on a healthy run; ``p_m1_high``/``p_m2_high`` are
the two measurement slots' means; ``p_switch`` is how often the parity CHANGES
(the only one about the rate); ``p_intercycle_flip`` is how often the QND chain
breaks between cycles (health check, not a parity quantity).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: below this much low-frequency headroom the plateau is thinly sampled and a
#: longer record would tighten the fit. Advisory only — the tool does not gate
#: on it, because the remedy is the caller's decision.
_LOW_MARGIN_ADVISORY = 5.0

#: points drawn in the time-domain debug panel. Deliberately small: the panel
#: is for confirming the three series hang together (m1 XOR m2 = parity), and
#: the PSD figure is the quantitative view — do NOT try to count transitions
#: here.
_TRACE_SNIPPET = 100

#: markers are large because only ~100 points are drawn.
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
    """Debug panel: M1 and M2 over the derived per-cycle PARITY, shared axis.

    ``parity[i] = m1[i] XOR m2[i]`` within each cycle — the bottom row is the
    XOR of the two above it, and only the bottom row is fitted. Diagnostic
    only: the window is ~100 cycles, usually far shorter than a switching
    period, so the parity row will often look constant — expected, not a
    fault.
    """
    fig, (ax_1, ax_2, ax_p) = plt.subplots(
        3, 1, figsize=(9, 6.5), dpi=100, sharex=True)
    attrs = plot_data.attrs

    t = plot_data.coords["time_s"].values
    m1 = plot_data["m1"].values
    m2 = plot_data["m2"].values
    n = int(min(m1.size, _TRACE_SNIPPET))

    ax_1.scatter(t[:n] * 1e3, m1[:n], **_SCATTER)
    _annotate(ax_1, [
        f"cycle period = {float(attrs.get('dt_s', float('nan'))) * 1e6:.4g} us",
        f"{m1.size} cycles total",
        f"inter-cycle flip = "
        f"{float(attrs.get('p_intercycle_flip', float('nan'))):.3g}"
        f"  (0 = clean QND chain)",
    ])
    ax_1.set_yticks([0, 1])
    ax_1.set_yticklabels(["|0>", "|1>"])
    ax_1.set_ylim(-0.4, 2.1)
    ax_1.set_ylabel("M1", fontsize=12)
    ax_1.set_title(f"Debug: the two per-cycle measurements and their XOR "
                   f"(= the parity) — first {n} cycles", fontsize=10)

    ax_2.scatter(t[:n] * 1e3, m2[:n], color="tab:orange", **_SCATTER)
    _annotate(ax_2, [
        f"p_m1_high = {float(attrs.get('p_m1_high', float('nan'))):.3g}",
        f"p_m2_high = {float(attrs.get('p_m2_high', float('nan'))):.3g}",
    ])
    ax_2.set_yticks([0, 1])
    ax_2.set_yticklabels(["|0>", "|1>"])
    ax_2.set_ylim(-0.4, 2.1)
    ax_2.set_ylabel("M2", fontsize=12)

    parity = plot_data["parity"].values
    ax_p.scatter(t[:n] * 1e3, parity[:n], color="tab:green", **_SCATTER)
    _annotate(ax_p, [
        f"rate = {float(attrs.get('parity_rate_hz', float('nan'))):.4g} Hz",
        f"XOR=1 fraction = "
        f"{float(attrs.get('p_parity_odd', float('nan'))):.3g}"
        f"  (~0.5 expected)",
        f"switch fraction = "
        f"{float(attrs.get('p_switch', float('nan'))):.4g}",
    ])
    ax_p.set_yticks([0, 1])
    # NOT "even"/"odd": the measurement fixes the parity only up to a global
    # label. The plotted quantity is literally m1[i] XOR m2[i].
    ax_p.set_yticklabels(["XOR=0", "XOR=1"])
    ax_p.set_ylim(-0.4, 2.1)
    ax_p.set_xlabel("Time (ms)", fontsize=12)
    ax_p.set_ylabel("Charge parity", fontsize=12)

    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_psd(plot_data: xr.Dataset) -> plt.Figure:
    """Log-log Welch PSD **of the parity** with the Lorentzian-knee fit.

    The fitted ``A / (1 + (f/f_c)^2) + B`` is the reference RTS spectrum in
    different variables — see the ``telegraph_psd`` docstring. The timebase is
    the CYCLE period, so a longer cycle (bigger ``cycle_period_ns``) lowers the
    spectral window without more acquisition bins.
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
        # name the FITTED parameter first: 'constrained' fits Gamma directly,
        # 'independent' fits the corner (see the continuous sibling).
        if model == "constrained":
            corner_label = (f"$\\Gamma$ = {rate:.4g} Hz (fitted) "
                            f"-> $f_c = \\Gamma/\\pi$ = {corner:.4g} Hz")
        else:
            corner_label = (f"corner $f_c$ = {corner:.4g} Hz "
                            f"-> $\\Gamma = \\pi f_c$ = {rate:.4g} Hz")
        ax.axvline(corner, color="red", linestyle="--", linewidth=1,
                   label=corner_label)

    # the low-frequency edge: how slow a rate this record length could see at
    # all. A corner near it is the readable symptom of "record for longer".
    f_min = float(attrs.get("psd_freq_min_hz", float("nan")))
    margin = float(attrs.get("corner_margin_low", float("nan")))
    if np.isfinite(f_min):
        ax.axvline(f_min, color="tab:green", linestyle=":", linewidth=1.2,
                   label=f"lowest bin {f_min:.3g} Hz (= 8 / record time)")

    model_label = {"constrained": "model: constrained (shared F)",
                   "independent": "model: independent (free floor)"}.get(model)
    lines = [model_label] if model_label else []
    lines.append(f"contrast A/B = {float(attrs.get('psd_contrast', float('nan'))):.4g}")
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
    flip = float(attrs.get("p_intercycle_flip", float("nan")))
    if np.isfinite(flip):
        lines.append(f"inter-cycle flip = {flip:.3g}")
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
    """Log-log Welch PSD of the raw M1 trace — diagnostic, NEVER fitted.

    Absent errors the QND chain makes M1 the running XOR of the parity, i.e.
    an integrated telegraph, so its spectrum rises toward low frequency and
    has no Lorentzian corner to read. Shown for pickup/drift/readout-artefact
    eyeballing; deliberately no fit line, so it cannot be mistaken for the
    measurement.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if "state_psd" not in plot_data.data_vars \
            or plot_data["state_psd"].size == 0:
        ax.set_title("Raw M1 spectrum — unavailable", fontsize=10)
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = plot_data.coords["state_psd_freq_hz"].values
    ax.loglog(freq, plot_data["state_psd"].values, ".", markersize=3,
              color="tab:gray", label="M1 PSD (Welch, unfitted)")

    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (1/Hz)", fontsize=14)
    ax.set_title("Raw M1 spectrum — DIAGNOSTIC ONLY (integrated telegraph; "
                 "the rate comes from the parity)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    plt.close(fig)
    return fig
