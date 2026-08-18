"""
Resonator-spectroscopy-vs-power plotting helper.

``plot_power_map`` consumes the **plot_data** Dataset built by
``ResonatorSpectroscopyPowerEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``power``, ``detuning`` (+ optional ``full_freq``)
vars   : ``amplitude`` + power-normalized ``amplitude_db`` (power, detuning);
         per-power ``center_detuning`` / ``fwhm`` / ``dip_amplitude`` /
         ``success`` / ``good`` / ``outlier`` / ``branch_class``
         (+ optional ``center_full_freq``;
         + optional chain provenance ``digital_amp`` / ``chain_setting`` over
         ``power`` — drawn as an amp/chain subplot under the map, the ONE shared
         figure form both scqo punchouts emit)
attrs  : ``n_power``, ``n_success``, ``n_good``, ``n_outlier``, ``has_full_freq``,
         ``optimal_power``, ``crossing_power``, ``frequency_shift``, the two
         punchout branches ``f_dress0`` / ``f_bare`` with ``lamb_shift`` and
         ``branch_success`` (drawn only on the absolute axis) and their plateau
         boundary powers ``dress_max_power`` / ``bare_min_power`` (vertical
         dashed lines bracketing the transition)
         (+ optional ``resonator_frequency``;
         + optional ``chain_name`` labeling ``chain_setting``, ``power_axis_kind``
         for the x-label, and ``mode_label`` tagging the mechanism in the title)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_power_map(plot_data: xr.Dataset) -> plt.Figure:
    """The 2-D power-normalized ``20*log10|IQ| - power`` map over (power,
    frequency) with the fitted resonator-centre trace and optimal-power marker
    overlaid, drawn entirely from ``plot_data``. With chain provenance
    (``digital_amp``/``chain_setting`` per power) a second row shares the power
    axis and shows what the instrument was actually doing at every point."""
    power = plot_data.coords["power"].values.astype(float)
    if "amplitude_db" in plot_data:
        amplitude_db = plot_data["amplitude_db"].values  # (power, detuning)
    else:
        # Older plotdata files carry only the linear map; normalize here.
        amplitude = plot_data["amplitude"].values
        amplitude_db = (
            20.0 * np.log10(np.maximum(amplitude, np.finfo(float).tiny)) - power[:, None]
        )

    use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
    if use_full:
        yvals = plot_data["full_freq"].values.astype(float) / 1e9
        center = plot_data["center_full_freq"].values / 1e9
        ylabel = "RF frequency (GHz)"
    else:
        yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
        center = plot_data["center_detuning"].values / 1e6
        ylabel = "Detuning (MHz)"

    good = plot_data["good"].values.astype(bool)
    outlier = plot_data["outlier"].values.astype(bool)

    x_kind = str(plot_data.attrs.get("power_axis_kind", "dB") or "dB")
    x_label = f"Readout power ({x_kind})"
    opt_unit = "dBm" if "absolute" in x_kind.lower() else "dB"

    # Chain provenance -> a second row sharing the power axis.
    has_steps = "digital_amp" in plot_data.data_vars
    if has_steps:
        fig, (ax, ax_chain) = plt.subplots(
            2, 1, figsize=(10, 7.5), dpi=120, sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        ax_chain = None

    pcm = ax.pcolormesh(power, yvals, amplitude_db.T, shading="auto", cmap="viridis")
    cbar_axes = [ax, ax_chain] if has_steps else ax  # span both rows so widths match
    fig.colorbar(pcm, ax=cbar_axes, label="20 log10 |IQ| - power (dB)")
    # Kept centres form the trace; rejected (out-of-window / outlier) shown as red x.
    ax.plot(power[good], center[good], ".-", color="C3", ms=5, lw=1.0, label="centre (kept)")
    if outlier.any():
        ax.plot(power[outlier], center[outlier], "x", color="red", ms=7, mew=1.5,
                label="rejected (outlier)")

    # Optimal-power marker (vertical line) when the pick succeeded.
    optimal_power = float(plot_data.attrs.get("optimal_power", np.nan))
    if bool(plot_data.attrs.get("optimal_success", 0)) and np.isfinite(optimal_power):
        ax.axvline(optimal_power, color="magenta", ls="--", lw=1.5,
                   label=f"optimal power = {optimal_power:.1f} {opt_unit}")

    # The two punchout branches. Absolute Hz, so they can only be drawn on the
    # absolute axis — which is exactly the axis `full_freq` provides, the same
    # gate the estimator uses to report them at all. Each is guarded on its own:
    # a punchout that only reached the dispersive regime still shows its dressed
    # line rather than losing both.
    for attr, color, name in (("f_dress0", "cyan", "f_dress0 (qubit |0>)"),
                              ("f_bare", "white", "f_bare (saturated)")):
        value = float(plot_data.attrs.get(attr, np.nan))
        if use_full and np.isfinite(value):
            ax.axhline(value / 1e9, color=color, ls=":", lw=1.6,
                       label=f"{name} = {value / 1e9:.5f} GHz")
    lamb = float(plot_data.attrs.get("lamb_shift", np.nan))
    if use_full and np.isfinite(lamb):
        ax.plot([], [], " ", label=f"Lamb shift = {lamb / 1e6:.2f} MHz")

    # The plateau boundary powers bracketing the transition: everything at or
    # below dress_max is the dressed run, at or above bare_min the bare run —
    # the points between fed NEITHER branch. Each guarded on its own (a window
    # that never saturated has no bare boundary).
    for attr, color, name in (("dress_max_power", "cyan", "dressed ≤"),
                              ("bare_min_power", "white", "bare ≥")):
        value = float(plot_data.attrs.get(attr, np.nan))
        if np.isfinite(value):
            ax.axvline(value, color=color, ls="--", lw=1.2,
                       label=f"{name} {value:.1f} {opt_unit}")

    ax.set_ylabel(ylabel)
    n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
    n_power = int(plot_data.attrs.get("n_power", len(power)))
    title = f"Resonator spectroscopy vs power (kept {n_good}/{n_power})"

    if has_steps:
        # Per-point digital amplitude (left) + the used chain setting (right),
        # shared power x-axis with the map above.
        digital_amp = plot_data["digital_amp"].values.astype(float)
        ax_chain.plot(power, digital_amp, ".-", color="C0", ms=4, lw=1.0)
        ax_chain.axhline(0.5, color="gray", ls=":", lw=1.0)  # the canonical point
        ax_chain.set_ylabel("digital amp", color="C0", fontsize=9)
        ax_chain.tick_params(axis="y", labelcolor="C0")
        ax_chain.set_ylim(0.0, 1.05)
        if "chain_setting" in plot_data.data_vars:
            ax_setting = ax_chain.twinx()
            ax_setting.step(power, plot_data["chain_setting"].values.astype(float),
                            where="mid", color="C1", lw=1.2)
            ax_setting.set_ylabel(str(plot_data.attrs.get("chain_name", "chain setting")),
                                  color="C1", fontsize=9)
            ax_setting.tick_params(axis="y", labelcolor="C1")
        ax_chain.set_xlabel(x_label)
    else:
        ax.set_xlabel(x_label)

    # Title second line: the mechanism (mode_label).
    mode_label = str(plot_data.attrs.get("mode_label", "") or "")
    if mode_label:
        title += f"\n{mode_label}"

    ax.set_title(title)
    ax.legend(fontsize=8)

    if not has_steps:  # constrained layout already handles the two-row figure
        fig.tight_layout()
    return fig
