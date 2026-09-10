"""Figures for ``qc_n_stark_amp``: the raw maps, and the compensation read-off.

Two figures, both drawn from ``plot_data`` only:

``plot_qc_n_stark_amp``
    the shared 2x2 joint-population map -- an alias, so the drawing lives once in
    ``_pair_swap_maps`` while the per-estimator import path stays stable.
``plot_stark_compensation``
    the transfer map with the picked amplitude marked, the two criteria as curves
    over the stark amplitude, and the picked amplitude's own oscillation in N
    with its fitted cosine on a dense axis.

plot_data layout
----------------
coords : ``stark_amp``, ``swap_count``, ``swap_count_dense``
vars   : ``p00`` / ``p01`` / ``p10`` / ``p11`` (the joint basis maps, drawn by
         the shared ``plot_pair_swap_map``), ``transfer`` and ``transfer_fit``
         (``transfer_fit`` over ``(stark_amp, swap_count_dense)``), and
         ``osc_contrast`` / ``osc_amplitude`` / ``osc_period`` /
         ``osc_r_squared`` / ``osc_success`` over ``stark_amp``
attrs  : ``compensating_stark_amp`` (+ ``_refined`` /
         ``compensating_is_refined``), ``compensation_score``,
         ``compensating_osc_contrast``, ``compensating_osc_period``,
         ``compensating_theta_rad``, ``max_osc_contrast_stark_amp`` /
         ``max_osc_contrast``, ``max_osc_period_stark_amp`` /
         ``max_osc_period``, ``min_osc_period``, ``osc_criteria_agree``,
         ``n_osc_ok`` (plus the shared map attrs)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_qc_n_stark_amp", "plot_stark_compensation"]

plot_qc_n_stark_amp = plot_pair_swap_map

AXIS0 = "stark_amp"
AXIS1 = "swap_count"
DENSE_AXIS = "swap_count_dense"

#: two counts per cycle: the integer-N Nyquist period, and the pi/2-per-swap
#: limit the whole reading assumes no row crosses.
_NYQUIST_PERIOD = 2.0

_AMP_COLOR = "tab:blue"
_PERIOD_COLOR = "tab:green"
_PICK_COLOR = "tab:red"


def _column(plot_data: xr.Dataset, name: str, size: int) -> np.ndarray:
    """A plot_data column, NaN-filled when the variable is absent.

    Keeps this plotter drawable against a plot_data written by an older run that
    predates a variable, rather than raising and taking the raw map down with it.
    """
    if name in plot_data:
        return np.asarray(plot_data[name].values, dtype=float)
    return np.full(size, np.nan)


def _nearest(amps: np.ndarray, value: float) -> int:
    """Index of the swept amplitude closest to ``value`` (already on the grid)."""
    return int(np.argmin(np.abs(amps - value)))


def plot_stark_compensation(plot_data: xr.Dataset) -> plt.Figure:
    """Where the swap oscillation is strongest and slowest -- the compensation point.

    Left: the raw transfer marginal over (stark amplitude, N), the picture the
    fits are made from. Middle: the two criteria -- the measured contrast and the
    fitted period -- against the stark amplitude, with the pi/2 limit marked on
    the period axis. Right: the picked amplitude's transfer against N with its
    fitted cosine, so the oscillation the two criteria were read from is visible
    rather than asserted.

    The raw map is drawn UNCONDITIONALLY; every fit-derived overlay is guarded,
    so a run in which every row failed still produces this figure.
    """
    fig, (ax_map, ax_curve, ax_trace) = plt.subplots(
        1, 3, figsize=(17, 5.2), constrained_layout=True
    )

    amps = np.asarray(plot_data[AXIS0].values, dtype=float)
    counts = np.asarray(plot_data[AXIS1].values, dtype=float)
    transfer = np.asarray(plot_data["transfer"].values, dtype=float)  # (amp, N)
    dense = (np.asarray(plot_data[DENSE_AXIS].values, dtype=float)
             if DENSE_AXIS in plot_data.coords else counts)
    fit_map = _column2d(plot_data, "transfer_fit", (amps.size, dense.size))

    contrast = _column(plot_data, "osc_contrast", amps.size)
    period = _column(plot_data, "osc_period", amps.size)
    ok = _column(plot_data, "osc_success", amps.size).astype(bool)

    pick = float(plot_data.attrs.get("compensating_stark_amp", float("nan")))
    refined = float(plot_data.attrs.get("compensating_stark_amp_refined", float("nan")))
    pick_amp = float(plot_data.attrs.get("compensating_osc_contrast", float("nan")))
    pick_period = float(plot_data.attrs.get("compensating_osc_period", float("nan")))
    theta = float(plot_data.attrs.get("compensating_theta_rad", float("nan")))
    agree = int(plot_data.attrs.get("osc_criteria_agree", 0))
    fastest = float(plot_data.attrs.get("min_osc_period", float("nan")))
    n_ok = int(plot_data.attrs.get("n_osc_ok", int(ok.sum())))

    # --- left: the raw transfer map (always drawn) -----------------------
    mesh = ax_map.pcolormesh(
        *np.meshgrid(amps, counts), transfer.T,
        shading="auto", cmap="viridis", vmin=0.0, vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax_map, label="transfer population")
    ax_map.set_xlabel("AC-Stark amplitude")
    ax_map.set_ylabel("swap count N")
    ax_map.set_title("transfer vs (stark amplitude, N)")
    if np.isfinite(pick):
        ax_map.axvline(pick, color=_PICK_COLOR, lw=1.4, ls="--")

    # --- middle: the two criteria (guarded) ------------------------------
    _criteria_panel(ax_curve, amps, contrast, period, ok, pick, refined)

    # --- right: the picked amplitude trace + its fitted cosine -----------
    _plot_traces(ax_trace, amps, counts, dense, transfer, fit_map, pick)

    bits = [f"{n_ok} of {amps.size} amplitudes fitted"]
    if np.isfinite(pick):
        pick_text = f"pick {pick:.4g}"
        if np.isfinite(refined) and refined != pick:
            pick_text += f" (refined {refined:.4g})"
        bits.append(pick_text)
        bits.append(f"contrast {pick_amp:.3f}")
        bits.append(f"period {pick_period:.2f} swaps")
        if np.isfinite(theta):
            bits.append(rf"$\theta_{{\mathrm{{eff}}}}$ {theta:.3f} rad")
        bits.append("both criteria agree" if agree
                    else "criteria DISAGREE — see the curves")
    if np.isfinite(fastest):
        # the standing assumption's dashboard: 2 counts per cycle is pi/2 a swap
        note = f"fastest row {fastest:.2f} swaps/cycle"
        if fastest <= _NYQUIST_PERIOD * 1.1:
            note += " — AT the limit, anything past it aliases"
        bits.append(note)
    fig.suptitle(
        "AC-Stark compensation — strongest and slowest swap oscillation\n"
        + "     ·     ".join(bits),
        fontsize=12,
    )
    return fig


def _criteria_panel(ax, amps, contrast, period, ok, pick, refined) -> None:
    """The two criteria against the stark amplitude, on twinned y axes.

    Rows whose cosine fit was rejected are marked apart, since they win nothing;
    everything else competes, the map being read under the standing assumption
    that no row swaps by more than pi/2 (estimator module docstring).
    """
    ax_period = ax.twinx()
    if np.isfinite(contrast).any() or np.isfinite(period).any():
        ax.plot(amps[ok], contrast[ok], "o-", color=_AMP_COLOR,
                markersize=5, label="contrast")
        ax_period.plot(amps[ok], period[ok], "s-", color=_PERIOD_COLOR,
                       markersize=5, label="period")
        if (~ok).any():
            ax.plot(amps[~ok], contrast[~ok], "x", color="0.6", markersize=5,
                    label="fit rejected")
    else:
        ax.text(0.5, 0.5, "no oscillation could be fitted\n(raw map at left)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color=_PICK_COLOR)

    if np.isfinite(pick):
        ax.axvline(pick, color=_PICK_COLOR, lw=1.4, ls="--")
        ax.annotate(f"{pick:.4g}", xy=(pick, ax.get_ylim()[0]), xytext=(4, 6),
                    textcoords="offset points", fontsize=9, color=_PICK_COLOR)
    if np.isfinite(refined) and refined != pick:
        ax.axvline(refined, color=_PICK_COLOR, lw=1.0, ls=":", alpha=0.8)

    # The pi/2 limit, in the period axis's own units: a row reaching TWO counts
    # per cycle is at the standing assumption's edge, and anything past it aliases
    # instead of slowing down. Drawn after the data with the data's own limits
    # restored, so a far-off bound cannot squash the curve it annotates.
    ylim = ax_period.get_ylim()
    ax_period.axhline(_NYQUIST_PERIOD, color=_PERIOD_COLOR, lw=1.0, ls=":",
                      alpha=0.6)
    ax_period.set_ylim(ylim)

    ax.set_xlabel("AC-Stark amplitude")
    ax.set_ylabel("oscillation contrast", color=_AMP_COLOR)
    ax.tick_params(axis="y", labelcolor=_AMP_COLOR)
    ax_period.set_ylabel("fitted period (swaps per cycle)", color=_PERIOD_COLOR)
    ax_period.tick_params(axis="y", labelcolor=_PERIOD_COLOR)
    ax.set_title("the two criteria vs stark amplitude")
    handles, labels = ax.get_legend_handles_labels()
    extra = ax_period.get_legend_handles_labels()
    if handles or extra[0]:
        ax.legend(handles + extra[0], labels + extra[1], loc="best", fontsize=8)


def _plot_traces(ax, amps, counts, dense, transfer, fit_map, pick) -> None:
    """The picked amplitude's own oscillation in N, with its fitted cosine.

    The raw points are drawn unconditionally and the fit is overlaid only where
    the stored curve is finite, so a failed run still shows its trace. The fit
    rides the dense axis it was stored on -- a swap oscillation runs several
    cycles across a handful of integer counts, and joining the fit at those counts
    alone would draw a zigzag that looks like the data rather than through it.
    """
    ax.set_xlabel("swap count N")
    ax.set_ylabel("transfer population")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("oscillation at the picked amplitude")
    if not np.isfinite(pick):
        # Nothing was picked: show the raw traces anyway, faintly, so the panel
        # is never an empty box.
        for row in transfer:
            ax.plot(counts, row, "-", color="0.85", lw=0.8)
        ax.text(0.5, 0.5, "no amplitude picked\n(every row's fit was rejected)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color=_PICK_COLOR)
        return

    i_pick = _nearest(amps, pick)
    ax.plot(counts, transfer[i_pick], "o", color=_PICK_COLOR, markersize=6,
            label=f"picked amplitude ({amps[i_pick]:.4g})")
    if np.isfinite(fit_map[i_pick]).any():
        ax.plot(dense, fit_map[i_pick], "-", color=_PICK_COLOR, lw=1.6,
                label="fitted cosine")
    ax.legend(loc="best", fontsize=9)


def _column2d(plot_data: xr.Dataset, name: str, shape) -> np.ndarray:
    """A 2-D plot_data map, NaN-filled when the variable is absent."""
    if name in plot_data:
        return np.asarray(plot_data[name].values, dtype=float)
    return np.full(shape, np.nan)
