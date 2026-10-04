"""Figures for qubit_ramsey_flux_crosstalk - drawn from plot_data only.

Three views of one run: the raw fringe maps (one panel per source amplitude), the
``delta_f`` curves they reduce to (one parabola per source amplitude, sliding
sideways when there is crosstalk), and the apex position against the source
amplitude with its line. The raw maps are drawn unconditionally; every fit overlay
only when finite.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

#: the most source amplitudes drawn as separate fringe panels
MAX_PANELS = 9


def _source_colors(n: int):
    return plt.get_cmap("viridis")(np.linspace(0.05, 0.9, max(n, 1)))


def plot_fringe_maps(plot_data: xr.Dataset) -> plt.Figure:
    """The raw signal over (own flux, idle time), one panel per source amplitude."""
    b = plot_data["source_flux"].values
    x = plot_data["flux_bias"].values
    t = plot_data["idle_time"].values
    sig = plot_data["signal"].transpose("source_flux", "flux_bias", "idle_time").values
    picks = (np.arange(b.size) if b.size <= MAX_PANELS
             else np.unique(np.round(np.linspace(0, b.size - 1, MAX_PANELS)).astype(int)))
    n = max(len(picks), 1)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False,
                             sharex=True, sharey=True)
    finite = sig[np.isfinite(sig)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    mesh = None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(picks):
            ax.axis("off")
            continue
        j = int(picks[k])
        mesh = ax.pcolormesh(t * 1e6, x * 1e3, sig[j], shading="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"source {b[j] * 1e3:+.1f} mV", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("idle time (us)")
    for ax in axes[:, 0]:
        ax.set_ylabel("own flux (mV)")
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="signal", shrink=0.9)
    fig.suptitle("Ramsey fringes vs own flux, per source amplitude "
                 f"(detuning {plot_data.attrs.get('ramp_detuning_hz', np.nan) / 1e6:+.3f} MHz)",
                 fontsize=10)
    return fig


def plot_flux_curves(plot_data: xr.Dataset) -> plt.Figure:
    """delta_f = f_q - f_drive vs own flux, one series per source amplitude."""
    b = plot_data["source_flux"].values
    x = plot_data["flux_bias"].values * 1e3
    d = plot_data["delta_f_hz"].values / 1e6
    valid = plot_data["valid"].values.astype(bool)
    fine = plot_data["flux_fine"].values * 1e3
    curves = plot_data["fit_delta_f_hz"].values / 1e6
    apex = plot_data["apex_flux"].values * 1e3
    height = plot_data["apex_delta_f_hz"].values / 1e6
    used = plot_data["source_valid"].values.astype(bool)
    colors = _source_colors(b.size)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for j in range(b.size):
        ax.plot(x[valid[j]], d[j][valid[j]], "o", color=colors[j], ms=4,
                label=f"{b[j] * 1e3:+.1f} mV")
        if (~valid[j]).any():
            ax.plot(x[~valid[j]], d[j][~valid[j]], "x", color="0.6", ms=4)
        if np.isfinite(curves[j]).any():
            ax.plot(fine, curves[j], "-", color=colors[j], lw=1)
        if used[j] and np.isfinite(apex[j]) and np.isfinite(height[j]):
            ax.plot(apex[j], height[j], "v", color=colors[j], mec="k", ms=7)
    ax.set_xlabel("own flux (mV)")
    ax.set_ylabel("f_q - f_drive (MHz)")
    ax.set_title("local arch per source amplitude (triangles: apex)")
    if b.size:
        ax.legend(title="source", fontsize=7, title_fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_apex_vs_source(plot_data: xr.Dataset) -> plt.Figure:
    """Apex position (the crosstalk line) and apex height vs source amplitude."""
    attrs = plot_data.attrs
    b = plot_data["source_flux"].values * 1e3
    apex = plot_data["apex_flux"].values * 1e3
    apex_err = plot_data["apex_flux_stderr"].values * 1e3
    height = plot_data["apex_delta_f_hz"].values / 1e3
    height_err = plot_data["apex_delta_f_stderr_hz"].values / 1e3
    used = plot_data["source_valid"].values.astype(bool)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.2), sharex=True)

    if used.any():
        top.errorbar(b[used], apex[used], yerr=apex_err[used], fmt="o", label="apex")
    rejected = ~used & np.isfinite(apex)
    if rejected.any():
        top.plot(b[rejected], apex[rejected], "x", color="0.5", label="rejected")
    line = plot_data["fit_apex_flux"].values * 1e3
    if np.isfinite(line).all():
        top.plot(plot_data["source_fine"].values * 1e3, line, "-", color="C1", label="line")
    top.set_ylabel("apex position, own flux (mV)")
    m, m_err = attrs.get("flux_crosstalk", np.nan), attrs.get("flux_crosstalk_stderr", np.nan)
    title = "flux crosstalk: "
    title += (f"m = {m * 100:+.3f} % +- {m_err * 100:.3f} %" if np.isfinite(m) else "no fit")
    if not attrs.get("success"):
        title += " (FAILED)"
    if attrs.get("nonlinear_suspected"):
        title += " (nonlinear)"
    if attrs.get("fold_suspected"):
        title += " (fold suspected)"
    top.set_title(title)
    if top.get_legend_handles_labels()[0]:
        top.legend(fontsize=8)

    if used.any():
        bottom.errorbar(b[used], height[used], yerr=height_err[used], fmt="s", color="C2")
    if rejected.any():
        bottom.plot(b[rejected], height[rejected], "x", color="0.5")
    bottom.set_xlabel("source flux (mV)")
    bottom.set_ylabel("apex height, f_q - f_drive (kHz)")
    bottom.set_title("apex height (moves without crosstalk, e.g. a coupler's shift)",
                     fontsize=9)
    fig.tight_layout()
    return fig
