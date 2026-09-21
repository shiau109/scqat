"""Figures for the ``qc_swap_flux_stark`` map and its compensation ridge.

Two figures. The map is the shared 2x2 joint-population drawing; the alias keeps
the per-estimator import path stable while the drawing lives once in
``_pair_swap_maps``. The ridge figure is this estimator's own: WHERE along the
stark axis each flux row peaks (that locus is ``phi = 0``), the same optima with
their ``2*pi`` wraps undone, and the transfer read along them.

There is no global fitted curve to draw, on purpose -- the ridge is straight in
PHASE and curved in amplitude, so the compensation is interpolated from the rows
around the resonance and only that local segment is drawn.

All three panels put the FLUX on x, and the two that show a stark amplitude put
it on y, so a feature reads straight across the figure. That is also the
orientation of the sibling 2x2 population map (``plot_pair_swap_map`` draws
``axis0`` on x), so the two PNGs of one run overlay without a mental transpose.

Both draw from ``plot_data`` only, and both draw their raw data unconditionally
-- the overlays are guarded, so a run whose read failed still renders.

plot_data contract (produced by the estimator's ``build_plot_data``):
  vars   : ``p00``/``p01``/``p10``/``p11`` and ``transfer`` over
           ``(flux_amp_v, stark_amp)``; ``row_stark_amp`` (as measured) /
           ``row_contrast`` / ``ridge_stark_amp`` (unwrapped) /
           ``ridge_transfer`` / ``ridge_theta_rad`` / ``row_ok`` over
           ``(flux_amp_v,)``
  attrs  : the estimator's scalar read (``compensating_stark_amp``,
           ``resonance_flux_amp_v``, ``swap_angle_rad_refined``, the gates,
           ``ridge_slope_per_v`` / ``ridge_wrap_amp``) plus ``swap_count`` /
           ``min_row_contrast``
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_qc_swap_flux_stark", "plot_swap_flux_stark_ridge"]

plot_qc_swap_flux_stark = plot_pair_swap_map

AXIS0 = "flux_amp_v"
AXIS1 = "stark_amp"

_RIDGE_COLOR = "tab:red"
_ROW_COLOR = "tab:blue"
_DEAD_COLOR = "0.65"


def _column(plot_data: xr.Dataset, name: str, size: int) -> np.ndarray:
    """A per-row column, NaN-filled when the read never wrote it."""
    if name not in plot_data:
        return np.full(size, np.nan)
    return np.asarray(plot_data[name].values, dtype=float)


def _attr(plot_data: xr.Dataset, name: str, default=float("nan")):
    value = plot_data.attrs.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _suptitle(plot_data: xr.Dataset) -> str:
    bits = []
    swaps = _attr(plot_data, "swap_count")
    if np.isfinite(swaps):
        bits.append(f"N = {int(swaps)}")
    prior = _attr(plot_data, "swap_angle_rad_prior")
    if np.isfinite(prior):
        bits.append(f"prior theta = {prior:.3f} rad")
    comp = _attr(plot_data, "compensating_stark_amp")
    if np.isfinite(comp):
        bits.append(f"compensating stark = {comp:.4g}")
    refined = _attr(plot_data, "swap_angle_rad_refined")
    if np.isfinite(refined):
        bits.append(f"refined theta = {refined:.3f} rad")
    wrap = _attr(plot_data, "ridge_wrap_amp")
    if np.isfinite(wrap):
        bits.append(f"2pi wrap = {wrap:.3g}")
    bits.append(f"ridge_ok={int(_attr(plot_data, 'ridge_ok', 0))} "
                f"branch_ok={int(_attr(plot_data, 'branch_ok', 0))}")
    if int(_attr(plot_data, "resonance_in_gap", 0)):
        bits.append("RESONANCE IN A DEAD BAND — refused")
    return "AC-Stark compensation  —  " + "   ".join(bits)


def _draw_map(ax, flux, stark, transfer, row_star, row_ok, comp, resonance) -> None:
    """The normalized transfer, with the measured per-row optima over it.

    Flux on x and stark on y, matching BOTH the sibling 2x2 population map
    (``plot_pair_swap_map`` draws ``axis0`` on x) and the two panels beside this
    one. The three panels of this figure then share an x axis, and a feature
    reads straight across them.
    """
    mesh = ax.pcolormesh(flux, stark, transfer.T, cmap="viridis",
                         vmin=0.0, vmax=1.0, shading="nearest")
    ax.figure.colorbar(mesh, ax=ax, label="normalized transfer")
    used = np.asarray(row_ok, dtype=bool)
    if np.isfinite(row_star).any():
        ax.plot(flux[used], row_star[used], "o", ms=4, mfc="none",
                mec="w", mew=1.2, label="row optimum")
    if np.isfinite(resonance):
        ax.axvline(resonance, color="w", ls="--", lw=1.2, label="resonance")
    if np.isfinite(comp) and np.isfinite(resonance):
        ax.plot(resonance, comp, "*", ms=18, color=_RIDGE_COLOR, mec="k", mew=0.6,
                label=f"comp = {comp:.3g}")
    ax.set_xlim(float(np.min(flux)), float(np.max(flux)))
    ax.set_ylim(float(np.min(stark)), float(np.max(stark)))
    ax.set_xlabel("flux amplitude (V)")
    ax.set_ylabel("AC-Stark amplitude")
    ax.set_title("normalized transfer + the per-row optimum")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)


def _draw_optima(ax, flux, row_star, ridge_star, row_ok, resonance,
                 min_contrast) -> None:
    """The optima as measured and as unwrapped — the wrap made visible."""
    used = np.asarray(row_ok, dtype=bool)
    drawn = False
    if np.isfinite(row_star).any():
        ax.plot(flux[~used], row_star[~used], "o", ms=4, color=_DEAD_COLOR,
                label="row rejected")
        ax.plot(flux[used], row_star[used], "o", ms=5, color=_ROW_COLOR,
                label="row optimum (as measured)")
        drawn = True
    if np.isfinite(ridge_star).any() and not np.allclose(
            np.nan_to_num(ridge_star), np.nan_to_num(row_star), equal_nan=True):
        ax.plot(flux[used], ridge_star[used], "s", ms=4, mfc="none",
                color=_RIDGE_COLOR, label="unwrapped")
        drawn = True
    if np.isfinite(resonance):
        ax.axvline(resonance, color="k", ls="--", lw=1)
    ax.set_xlabel("flux amplitude (V)")
    ax.set_ylabel("stark amplitude at the row peak")
    title = "per-row optimum  (= phi = 0, mod 2pi)"
    if np.isfinite(min_contrast):
        title += f"\nrows kept above contrast {min_contrast:.2g}"
    ax.set_title(title)
    if drawn:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "no row optimum found", ha="center", va="center",
                transform=ax.transAxes, color=_DEAD_COLOR)
    ax.grid(alpha=0.3)


def _draw_along_ridge(ax, flux, ridge_transfer, ridge_theta, resonance) -> None:
    """Transfer and angle read on the ridge — a pure-fit panel."""
    drawn = False
    if np.isfinite(ridge_transfer).any():
        ax.plot(flux, ridge_transfer, "o-", ms=4, color=_ROW_COLOR,
                label="transfer at the row optimum")
        ax.set_ylim(0.0, 1.05)
        drawn = True
    ax.set_xlabel("flux amplitude (V)")
    ax.set_ylabel("transfer on the ridge")
    if np.isfinite(ridge_theta).any():
        twin = ax.twinx()
        twin.plot(flux, ridge_theta, "-", color=_RIDGE_COLOR, lw=1.5)
        twin.set_ylabel("theta per swap (rad)", color=_RIDGE_COLOR)
        twin.tick_params(axis="y", labelcolor=_RIDGE_COLOR)
        drawn = True
    if np.isfinite(resonance):
        ax.axvline(resonance, color="k", ls="--", lw=1)
    ax.set_title("along the ridge: phase-compensated transfer,\n"
                 "and the angle it implies (red)")
    if drawn:
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no ridge to sample", ha="center", va="center",
                transform=ax.transAxes, color=_DEAD_COLOR)


def plot_swap_flux_stark_ridge(plot_data: Optional[xr.Dataset]) -> plt.Figure:
    """1x3: the map with the optima, the optima themselves, and the ridge cut."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    if plot_data is None:
        for ax in axes:
            ax.text(0.5, 0.5, "no plot data", ha="center", va="center",
                    transform=ax.transAxes, color=_DEAD_COLOR)
        return fig

    flux = np.asarray(plot_data[AXIS0].values, dtype=float)
    stark = np.asarray(plot_data[AXIS1].values, dtype=float)
    transfer = (np.asarray(plot_data["transfer"].transpose(AXIS0, AXIS1).values,
                           dtype=float)
                if "transfer" in plot_data
                else np.full((flux.size, stark.size), np.nan))
    row_star = _column(plot_data, "row_stark_amp", flux.size)
    ridge_star = _column(plot_data, "ridge_stark_amp", flux.size)
    ridge_transfer = _column(plot_data, "ridge_transfer", flux.size)
    ridge_theta = _column(plot_data, "ridge_theta_rad", flux.size)
    row_ok = _column(plot_data, "row_ok", flux.size) > 0
    resonance = _attr(plot_data, "resonance_flux_amp_v")
    comp = _attr(plot_data, "compensating_stark_amp")
    min_contrast = _attr(plot_data, "min_row_contrast")

    _draw_map(axes[0], flux, stark, transfer, row_star, row_ok, comp, resonance)
    _draw_optima(axes[1], flux, row_star, ridge_star, row_ok, resonance,
                 min_contrast)
    _draw_along_ridge(axes[2], flux, ridge_transfer, ridge_theta, resonance)
    fig.suptitle(_suptitle(plot_data))
    return fig
