"""Plotting for the AC-Stark phase echo estimator — draws ONLY from plot_data.

Each plotter draws its raw measured arrays UNCONDITIONALLY and guards the
fit overlay/annotations behind a finiteness check, so a failed fit still
produces its figure (scqat's "raw data must always be plottable" rule).
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_phase_vs_amp(plot_data: xr.Dataset) -> plt.Figure:
    """Recovered AC-Stark phase vs stark amplitude, with the phi ~ k*amp^2 fit."""
    amp = plot_data["stark_amp"].values
    phase_deg = np.degrees(plot_data["phase"].values)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(amp, phase_deg, ".", label="phi (data)", alpha=0.8)

    best = plot_data["best_fit"].values
    coeff = plot_data.attrs.get("stark_coeff", np.nan)
    if np.any(np.isfinite(best)):
        ax.plot(amp, np.degrees(best), "-", label="k*amp^2 fit")

    ax.set_xlabel("stark amplitude (factor of baked stark amp)")
    ax.set_ylabel("AC-Stark phase (deg)")
    title = "AC-Stark phase echo"
    if np.isfinite(coeff):
        title += f": k = {coeff:.3g} rad / amp^2"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_phasor(plot_data: xr.Dataset) -> plt.Figure:
    """Expectation-value phasor: <X> (~ cos phi) on x vs <Y> (~ sin phi) on y.

    For a discriminated readout the reduced signal is a population P, so the
    expectation value is <Z> = 1 - 2*P -- the -y90 basis gives <X>, the x90 basis
    gives <Y>, and the point (<X>, <Y>) traces the UNIT circle as the AC-Stark
    phase winds. For raw I/Q the axial projection is mapped to the same unit circle
    via the fitted circle (which reduces to 1 - 2*P for a population). Coloured by
    stark amplitude, anchor (smallest |amp|, phi ~ 0) marked. Pure raw-data view
    (no phi fit), so it always renders."""
    amp = plot_data["stark_amp"].values
    s_cos = plot_data["s_cos"].values   # -y90 close -> ~ cos phi
    s_sin = plot_data["s_sin"].values   # x90  close -> ~ sin phi
    cx0 = plot_data.attrs.get("circle_cx", np.nan)
    cy0 = plot_data.attrs.get("circle_cy", np.nan)
    r0 = plot_data.attrs.get("circle_r", np.nan)
    is_pop = plot_data.attrs.get("reduction_method", "signal") == "signal"

    if is_pop or not (np.isfinite(r0) and r0 > 0):
        # <Z> = 1 - 2*P (exact for a discriminated population).
        x = 1.0 - 2.0 * s_cos
        y = 1.0 - 2.0 * s_sin
        cx, cy, r = 1.0 - 2.0 * cx0, 1.0 - 2.0 * cy0, 2.0 * r0
    else:
        # raw I/Q: map the fitted circle onto the unit circle -> proper <X>, <Y>.
        x = (cx0 - s_cos) / r0
        y = (cy0 - s_sin) / r0
        cx, cy, r = 0.0, 0.0, 1.0

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.axhline(0.0, color="0.9", lw=0.6, zorder=0)
    ax.axvline(0.0, color="0.9", lw=0.6, zorder=0)
    ax.plot(x, y, "-", color="0.8", lw=0.8, zorder=1)          # trajectory in sweep order

    if np.isfinite(cx) and np.isfinite(cy) and np.isfinite(r) and r > 0:
        t = np.linspace(0.0, 2 * np.pi, 200)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), "-", color="tab:orange",
                lw=1.0, alpha=0.7, zorder=2, label="fitted circle")
        ax.plot(cx, cy, "+", color="tab:orange", ms=11, mew=1.6, zorder=4)

    sc = ax.scatter(x, y, c=amp, cmap="viridis", s=28, zorder=3)
    i0 = int(np.argmin(np.abs(amp)))                           # anchor: phi ~ 0
    ax.plot(x[i0], y[i0], "o", mfc="none", mec="red", ms=13, mew=1.6, zorder=5,
            label=f"anchor (amp={amp[i0]:.2g}, phi=0)")
    fig.colorbar(sc, ax=ax, label="stark amplitude (factor)")
    ax.set_xlabel(r"$\langle X\rangle$  (~ cos phi)")
    ax.set_ylabel(r"$\langle Y\rangle$  (~ sin phi)")
    ax.set_title("AC-Stark phasor (phase tomography)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_quadratures(plot_data: xr.Dataset) -> plt.Figure:
    """The two measured quadratures (x90 -> sin, -y90 -> cos) vs stark amplitude."""
    amp = plot_data["stark_amp"].values
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(amp, plot_data["s_sin"].values, ".-", label="x90  (~ sin phi)", alpha=0.8)
    ax.plot(amp, plot_data["s_cos"].values, ".-", label="-y90 (~ cos phi)", alpha=0.8)
    ax.set_xlabel("stark amplitude (factor of baked stark amp)")
    ax.set_ylabel("reduced signal")
    ax.set_title("Measurement-basis quadratures")
    ax.legend()
    fig.tight_layout()
    return fig
