"""Spectroscopy cryoscope plotting helpers.

Both functions consume the **plot_data** Dataset built by
``SpectroscopyCryoscopeEstimator.build_plot_data`` and draw without any
recalculation.

plot_data layout
----------------
coords : ``wait_time`` (s), ``detuning`` (Hz)
vars   : ``signal`` (wait_time, detuning), ``center_detuning`` /
         ``step_response`` / ``best_fit`` (wait_time)
attrs  : ``success``, ``a_dc``, ``rms_residual``, ``n_components``,
         ``fit_component_amps``, ``fit_component_taus_s``, ``stable_tail_frac``,
         ``center_offset_hz``, ``n_peaks_found``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _component_text(attrs: dict) -> str:
    amps = np.atleast_1d(attrs.get("fit_component_amps", []))
    taus = np.atleast_1d(attrs.get("fit_component_taus_s", []))
    a_dc = float(attrs.get("a_dc", float("nan")))
    lines = [
        f"success: {bool(attrs.get('success', 0))}",
        f"a_dc = {a_dc:.4g}",
    ]
    for i, (amp, tau) in enumerate(zip(amps, taus), start=1):
        rel = amp / a_dc if a_dc else float("nan")
        lines.append(f"A{i} = {rel:.4g}   τ{i} = {tau * 1e9:.4g} ns")
    lines.append(f"rms = {float(attrs.get('rms_residual', float('nan'))):.2e}")
    return "\n".join(lines)


def plot_step_response(plot_data: xr.Dataset) -> plt.Figure:
    """The normalized long-time step response and its multi-exponential fit, on
    linear and log time axes, with the tap components annotated."""
    t_ns = plot_data.coords["wait_time"].values * 1e9
    step = plot_data["step_response"].values
    best_fit = plot_data["best_fit"].values

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 5), dpi=100)

    for ax in (ax_lin, ax_log):
        ax.plot(t_ns, step, ".", markersize=4, label="step response")
        ax.plot(t_ns, best_fit, "-", linewidth=2, label="fit")
        ax.axhline(1.0, color="0.6", linestyle="--", linewidth=1)
        ax.set_xlabel("Wait time into flux pulse (ns)", fontsize=14)
    ax_lin.set_ylabel("Normalized flux (step response)", fontsize=14)
    ax_log.set_xscale("log")
    ax_lin.set_title("linear time")
    ax_log.set_title("log time")
    ax_lin.legend()

    ax_lin.text(
        0.98, 0.98, _component_text(plot_data.attrs),
        transform=ax_lin.transAxes, fontsize=11,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_spectrogram(plot_data: xr.Dataset) -> plt.Figure:
    """Diagnostic: the raw spectroscopy map (drive detuning vs wait time) with the
    fitted peak-center track overlaid (drawn from ``plot_data`` only)."""
    t_ns = plot_data.coords["wait_time"].values * 1e9
    det_mhz = plot_data.coords["detuning"].values * 1e-6
    signal = plot_data["signal"].values  # (wait_time, detuning)
    center_mhz = plot_data["center_detuning"].values * 1e-6

    fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
    mesh = ax.pcolormesh(t_ns, det_mhz, signal.T, shading="auto", cmap="viridis")
    ax.plot(t_ns, center_mhz, ".-", color="red", markersize=4, linewidth=1,
            label="fitted center")
    ax.set_xlabel("Wait time into flux pulse (ns)", fontsize=14)
    ax.set_ylabel("Drive detuning (MHz)", fontsize=14)
    ax.set_title("spectroscopy vs time after flux pulse")
    ax.legend(loc="upper right")
    fig.colorbar(mesh, ax=ax, label="signal")
    fig.tight_layout()
    plt.close(fig)
    return fig
