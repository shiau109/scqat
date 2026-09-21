"""Figures for the ``qc_swap_flux_stark`` map and its compensation ridge.

Two figures. The map is the shared 2x2 joint-population drawing; the alias keeps
the per-estimator import path stable while the drawing lives once in
``_pair_swap_maps``. The ridge figure is this estimator's own: WHERE along the
stark axis each flux row peaks (that locus is ``phi = 0``), the same optima with
their ``2*pi`` wraps undone, and the transfer read along them with the arch
fitted through it.

The only fitted curve is that arch, in the FLUX direction. There is none along
the stark optima, on purpose -- the ridge is straight in PHASE and curved in
amplitude, so the compensation is interpolated from the rows around the
resonance and nothing global is drawn there.

Three flux markers, one per meaning: the fitted resonance (dashed) with its
one-sigma band, and the largest row (dotted). On a flat-topped arch they differ,
and that difference is the reason the fit exists. A REFUSED resonance draws no
dashed line; the arch is still drawn, and the title says why.

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
           ``(flux_amp_v,)``; ``arch_fit`` over ``(flux_amp_v_dense,)``
  attrs  : the estimator's scalar read (``compensating_stark_amp`` /
           ``compensating_stark_err``, ``resonance_flux_amp_v`` /
           ``resonance_flux_err_v``, ``swap_angle_rad_refined`` /
           ``swap_angle_err_rad``, the gates and the three refusal flags,
           ``ridge_slope_per_v`` / ``ridge_wrap_amp``) plus ``swap_count`` /
           ``min_row_contrast``
"""

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_qc_swap_flux_stark", "plot_swap_flux_stark_ridge"]

plot_qc_swap_flux_stark = plot_pair_swap_map

AXIS0 = "flux_amp_v"
AXIS1 = "stark_amp"
DENSE_AXIS = "flux_amp_v_dense"

_RIDGE_COLOR = "tab:red"
_ROW_COLOR = "tab:blue"
_ARCH_COLOR = "k"
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


def _pm(value: float, err: float) -> str:
    """``value ± err``, the value rounded to the error's second digit."""
    if not np.isfinite(err) or err <= 0:
        return f"{value:.5g}"
    digits = max(0, 1 - int(math.floor(math.log10(err))))
    return f"{value:.{digits}f} ± {err:.{digits}f}"


def _flag(plot_data: xr.Dataset, name: str) -> bool:
    return bool(int(_attr(plot_data, name, 0)))


def _suptitle(plot_data: xr.Dataset) -> str:
    """Two lines: what the run was, then what it read -- or why it read nothing."""
    bits = []
    swaps = _attr(plot_data, "swap_count")
    if np.isfinite(swaps):
        bits.append(f"N = {int(swaps)}")
    prior = _attr(plot_data, "swap_angle_rad_prior")
    if np.isfinite(prior):
        bits.append(f"prior theta = {prior:.3f} rad")
    wrap = _attr(plot_data, "ridge_wrap_amp")
    if np.isfinite(wrap):
        bits.append(f"2pi wrap = {wrap:.3g}")
    bits.append(f"ridge_ok={int(_attr(plot_data, 'ridge_ok', 0))} "
                f"branch_ok={int(_attr(plot_data, 'branch_ok', 0))}")
    head = "AC-Stark compensation  —  " + "   ".join(bits)

    err = _attr(plot_data, "resonance_flux_err_v")
    detail = f" (± {err:.2g} V)" if np.isfinite(err) else ""
    if not _flag(plot_data, "ridge_ok"):
        return head + "\ngate shut: no resonance read"
    if _flag(plot_data, "resonance_at_edge"):
        return (head + "\nREFUSED: the fitted resonance is outside the live "
                f"rows{detail} — move the flux window onto it")
    if _flag(plot_data, "resonance_unresolved"):
        return (head + f"\nREFUSED: the resonance is unresolved{detail} — "
                "widen the flux window or average more")
    read = []
    resonance = _attr(plot_data, "resonance_flux_amp_v")
    if np.isfinite(resonance):
        read.append(f"resonance {_pm(resonance, err)} V")
    comp = _attr(plot_data, "compensating_stark_amp")
    if np.isfinite(comp):
        read.append("compensating stark "
                    f"{_pm(comp, _attr(plot_data, 'compensating_stark_err'))}")
    elif _flag(plot_data, "compensation_in_gap"):
        read.append("compensation WITHHELD: the resonance sits in a dead band")
    refined = _attr(plot_data, "swap_angle_rad_refined")
    if np.isfinite(refined):
        read.append("refined theta "
                    f"{_pm(refined, _attr(plot_data, 'swap_angle_err_rad'))} rad")
    return head + ("\n" + "   ".join(read) if read else "")


def _mark_flux(ax, resonance, resonance_err, peak_flux, color="k") -> None:
    """One vertical marker per meaning: the fitted resonance, and the raw peak.

    The dashed line is the arch's centre with its one-sigma band; the dotted
    one is the largest row. Where they disagree the top of the arch is flat
    and the dotted line is the one to distrust.
    """
    if np.isfinite(peak_flux):
        ax.axvline(peak_flux, color=color, ls=":", lw=1.2,
                   label=f"max transfer: {peak_flux:.5g} V")
    if np.isfinite(resonance):
        ax.axvline(resonance, color=color, ls="--", lw=1.2,
                   label=f"resonance: {_pm(resonance, resonance_err)} V")
        if np.isfinite(resonance_err) and resonance_err > 0:
            ax.axvspan(resonance - resonance_err, resonance + resonance_err,
                       color=color, alpha=0.18, lw=0)


def _draw_map(ax, flux, stark, transfer, row_star, row_ok, comp, resonance,
              resonance_err, peak_flux) -> None:
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
    _mark_flux(ax, resonance, resonance_err, peak_flux, color="w")
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
                 resonance_err, peak_flux, min_contrast) -> None:
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
    _mark_flux(ax, resonance, resonance_err, peak_flux)
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


def _draw_along_ridge(ax, flux, ridge_transfer, ridge_theta, dense, arch,
                      resonance, resonance_err, peak_flux) -> None:
    """Transfer and angle read on the ridge, with the arch fitted through it.

    The peak is marked whether or not a prior opened the gates, because WHERE
    the compensated transfer is largest is a measurement. It is only the claim
    that the peak IS the resonance that needs the prior — past ``N*theta =
    pi/2`` the same peak is one of two flanks, and the arch then dips between
    them.
    """
    drawn = False
    if np.isfinite(ridge_transfer).any():
        ax.plot(flux, ridge_transfer, "o", ms=4, color=_ROW_COLOR,
                label="transfer at the row optimum")
        ax.set_ylim(0.0, 1.05)
        drawn = True
    if np.isfinite(arch).any():
        ax.plot(dense, arch, "-", lw=1.5, color=_ARCH_COLOR, label="arch fit")
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
    _mark_flux(ax, resonance, resonance_err, peak_flux)
    ax.set_title("along the ridge: phase-compensated transfer + arch fit,\n"
                 "and the angle each row implies (red)")
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
    if DENSE_AXIS in plot_data.coords and "arch_fit" in plot_data:
        dense = np.asarray(plot_data[DENSE_AXIS].values, dtype=float)
        arch = np.asarray(plot_data["arch_fit"].values, dtype=float)
    else:  # a plotdata.nc written before the arch fit existed
        dense = arch = np.full(1, np.nan)
    resonance = _attr(plot_data, "resonance_flux_amp_v")
    resonance_err = _attr(plot_data, "resonance_flux_err_v")
    peak_flux = _attr(plot_data, "ridge_peak_flux_amp_v")
    comp = _attr(plot_data, "compensating_stark_amp")
    min_contrast = _attr(plot_data, "min_row_contrast")

    _draw_map(axes[0], flux, stark, transfer, row_star, row_ok, comp, resonance,
              resonance_err, peak_flux)
    _draw_optima(axes[1], flux, row_star, ridge_star, row_ok, resonance,
                 resonance_err, peak_flux, min_contrast)
    _draw_along_ridge(axes[2], flux, ridge_transfer, ridge_theta, dense, arch,
                      resonance, resonance_err, peak_flux)
    fig.suptitle(_suptitle(plot_data))
    return fig
