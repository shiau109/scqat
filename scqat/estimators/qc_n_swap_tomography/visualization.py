"""Figures for ``qc_n_swap_tomography``, drawn from ``plot_data`` only.

``plot_bloch_swap_frame``
    per stark amplitude, the single-excitation Bloch vector (conditioned on the
    subspace) in the SWAP frame - each amplitude's fitted frame turn
    ``a_off + N * frame_step`` undone, so every round is the same exchange about
    x. Face on (y-z) a compensated swap is the great circle through the poles;
    edge on (x-z) it is the vertical line x = 0, and the per-step phase tilts it.
    Measured points coloured by swap count, the fitted channel as a line.
``plot_bloch_drive_frame``
    the same vectors as recorded, in the drive frames (x-y and x-z): every point
    also carries the frames' turn, so the trajectory zig-zags; this is what the
    tomography saw before any fit.
``plot_components``
    per stark amplitude, the conditioned x, y, z against N, and the populations
    in / out of the subspace (p_sub, p00, p11) against N, with the fit.
``plot_compensation``
    the fitted per-step phase and angle against the stark amplitude, with the
    phase's zero crossing (the compensating amplitude) and the angle reported.

plot_data layout
----------------
coords : ``stark_amp``, ``swap_count``
vars   : ``x y z p_sub p00 p11 purity``, ``fit_x fit_y fit_z fit_p_sub
         fit_p00`` and the swap-frame ``x_swap_frame y_swap_frame
         fit_x_swap_frame fit_y_swap_frame`` (unnormalized, like ``x``; NaN where
         the fit failed) over (stark_amp, swap_count); per-amplitude columns
         ``theta_per_amp theta_err_per_amp phase_per_step_rad phase_err_rad
         frame_offset_rad t1_loss_high_per_amp t1_loss_low_per_amp
         dephasing_per_amp prep_error_per_amp fit_rms_per_amp
         leak_to_11_per_amp fit_success_per_amp``
attrs  : the scalar metadata (``theta_rad``, ``compensating_stark_amp``, ...),
         ``readout_correction``, ``high_name``, ``low_name``
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

FIG_BLOCH_SWAP = "bloch_swap_frame"
FIG_BLOCH_DRIVE = "bloch_drive_frame"
FIG_COMPONENTS = "components"
FIG_COMPENSATION = "compensation"

#: at most this many stark amplitudes get a row in the per-amplitude figures
MAX_ROWS = 6


def _rows(pd: xr.Dataset) -> list[int]:
    n = int(pd.sizes["stark_amp"])
    if n <= MAX_ROWS:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, MAX_ROWS)})


def _conditioned(pd: xr.Dataset, i: int, prefix: str = "") -> tuple[np.ndarray, ...]:
    return tuple(_component(pd, i, c, prefix) for c in ("x", "y", "z"))


def _component(pd: xr.Dataset, i: int, name: str, prefix: str = "") -> np.ndarray:
    """One Bloch component of row ``i``, conditioned on staying in the subspace."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd[f"{prefix}{name}"].values[i] / pd[f"{prefix}p_sub"].values[i]


def _title(pd: xr.Dataset) -> str:
    return (f"{pd.attrs.get('high_name', 'high')}-{pd.attrs.get('low_name', 'low')}  "
            f"(readout correction: {pd.attrs.get('readout_correction', 'none')})")


def _row_title(pd: xr.Dataset, i: int) -> str:
    title = f"stark amp {float(pd['stark_amp'].values[i]):.3f}"
    phi = float(pd["phase_per_step_rad"].values[i])
    return title + (f", phase/step {phi:+.3f} rad" if np.isfinite(phi) else "")


#: a view = (horizontal variable, its label, vertical variable, its label, panel name)
_DRIVE_VIEWS = (("x", "x", "y", "y", "top"), ("x", "x", "z", "z", "side"))
_SWAP_VIEWS = (("y_swap_frame", "y", "z", "z", "face on"),
               ("x_swap_frame", "x", "z", "z", "edge on"))


def _plot_bloch(pd: xr.Dataset, views, suptitle: str, edge_guide: bool) -> plt.Figure:
    rows = _rows(pd)
    counts = pd["swap_count"].values
    fig, axes = plt.subplots(len(rows), 2, figsize=(8.5, 3.9 * len(rows)), squeeze=False)
    circle = np.linspace(0, 2 * np.pi, 200)
    for r, i in enumerate(rows):
        for c, (h, hl, v, vl, panel) in enumerate(views):
            ax = axes[r, c]
            ax.plot(np.cos(circle), np.sin(circle), color="0.8", lw=0.8)
            if edge_guide and c == 1:
                ax.plot([0.0, 0.0], [-1.0, 1.0], color="0.55", ls=":", lw=1.0)
            dh, dv = _component(pd, i, h), _component(pd, i, v)
            sc = ax.scatter(dh, dv, c=counts, cmap="viridis", s=22, zorder=3)
            if not np.isfinite(dh).any():
                ax.text(0.0, 0.0, "no fitted frame:\nsee the drive-frame figure",
                        ha="center", va="center", fontsize=9, color="0.4")
            fh, fv = _component(pd, i, h, "fit_"), _component(pd, i, v, "fit_")
            if np.isfinite(fh).any() and np.isfinite(fv).any():
                ax.plot(fh, fv, color="C3", lw=1.2, zorder=2)
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_aspect("equal")
            ax.set_xlabel(f"{hl} (in subspace)")
            ax.set_ylabel(f"{vl} (in subspace)")
            ax.set_title(f"{_row_title(pd, i)}\n{panel} ({hl}-{vl})", fontsize=10)
        fig.colorbar(sc, ax=axes[r, 1], label="swap count N")
    fig.suptitle(suptitle + "\n" + _title(pd), fontsize=11)
    fig.tight_layout()
    return fig


def plot_bloch_swap_frame(pd: xr.Dataset) -> plt.Figure:
    return _plot_bloch(
        pd, _SWAP_VIEWS,
        "Bloch vector after N swaps, swap frame (fitted frame turn undone)\n"
        "dots: data (colour = N), red: channel fit, dotted: compensated (edge on)",
        edge_guide=True)


def plot_bloch_drive_frame(pd: xr.Dataset) -> plt.Figure:
    return _plot_bloch(
        pd, _DRIVE_VIEWS,
        "Bloch vector after N swaps, drive frames (as recorded)\n"
        "dots: data (colour = N), red: channel fit", edge_guide=False)


def plot_components(pd: xr.Dataset) -> plt.Figure:
    rows = _rows(pd)
    counts = pd["swap_count"].values
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.4 * len(rows)), squeeze=False)
    for r, i in enumerate(rows):
        amp = float(pd["stark_amp"].values[i])
        x, y, z = _conditioned(pd, i)
        fx, fy, fz = _conditioned(pd, i, "fit_")
        ax = axes[r, 0]
        for k, (data, fit, name) in enumerate(((x, fx, "x"), (y, fy, "y"), (z, fz, "z"))):
            ax.plot(counts, data, "o", ms=4, color=f"C{k}", label=name)
            if np.isfinite(fit).any():
                ax.plot(counts, fit, "-", lw=1.1, color=f"C{k}")
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("Bloch component (in subspace)")
        ax.set_title(f"stark amp {amp:.3f}")
        ax.legend(fontsize=8, ncol=3)
        ax = axes[r, 1]
        for k, name in enumerate(("p_sub", "p00", "p11")):
            ax.plot(counts, pd[name].values[i], "o", ms=4, color=f"C{k + 3}", label=name)
        for k, name in enumerate(("fit_p_sub", "fit_p00")):
            fit = pd[name].values[i]
            if np.isfinite(fit).any():
                ax.plot(counts, fit, "-", lw=1.1, color=f"C{k + 3}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("population")
        ax.legend(fontsize=8, ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel("swap count N")
    fig.suptitle("Tomography components after N swaps\n" + _title(pd), fontsize=11)
    fig.tight_layout()
    return fig


def plot_compensation(pd: xr.Dataset) -> plt.Figure:
    amps = pd["stark_amp"].values
    fig, (ax_phi, ax_theta) = plt.subplots(1, 2, figsize=(11, 4))
    phi = pd["phase_per_step_rad"].values
    phi_err = pd["phase_err_rad"].values
    theta = pd["theta_per_amp"].values
    theta_err = pd["theta_err_per_amp"].values
    if np.isfinite(phi).any():
        ax_phi.errorbar(amps, phi, yerr=np.nan_to_num(phi_err), fmt="o", color="C0")
    ax_phi.axhline(0.0, color="0.6", lw=0.8)
    root = float(pd.attrs.get("compensating_stark_amp", np.nan))
    if np.isfinite(root):
        style = "--" if int(pd.attrs.get("compensation_extrapolated", 0)) else "-"
        ax_phi.axvline(root, color="C3", ls=style, lw=1.2,
                       label=f"compensation {root:.4f}")
        ax_phi.legend(fontsize=8)
    ax_phi.set_xlabel("stark amplitude")
    ax_phi.set_ylabel("phase per step (rad, wrapped)")
    ax_phi.set_title("relative Z phase per step")
    if np.isfinite(theta).any():
        ax_theta.errorbar(amps, theta, yerr=np.nan_to_num(theta_err), fmt="o", color="C1")
    reported = float(pd.attrs.get("theta_rad", np.nan))
    if np.isfinite(reported):
        ax_theta.axhline(reported, color="C3", lw=1.0,
                         label=f"theta {reported:.4f} +- "
                               f"{float(pd.attrs.get('theta_rad_err', np.nan)):.4f}")
        ax_theta.legend(fontsize=8)
    ax_theta.set_xlabel("stark amplitude")
    ax_theta.set_ylabel("exchange angle per step (rad)")
    ax_theta.set_title("exchange angle (must not depend on the stark amplitude)")
    beta = float(pd.attrs.get("frame_step_rad", np.nan))
    beta_pred = float(pd.attrs.get("predicted_frame_step_rad", np.nan))
    frame = ""
    if np.isfinite(beta):
        frame = f"\nframe step {np.degrees(beta):+.1f} deg per round"
        if np.isfinite(beta_pred):
            frame += f" (predicted {np.degrees(beta_pred):+.1f})"
    fig.suptitle(_title(pd) + frame, fontsize=11)
    fig.tight_layout()
    return fig
