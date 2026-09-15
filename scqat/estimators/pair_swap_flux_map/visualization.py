"""Figures for ``pair_swap_flux_map``.

Consumes the **plot_data** Dataset built by
``PairSwapFluxMapEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``qubit_flux_v``, ``coupler_flux_v``
vars   : ``p00`` / ``p01`` / ``p10`` / ``p11`` (the joint basis maps, drawn by
         the shared ``plot_pair_swap_map``), ``transfer_norm`` and
         ``transfer_fit`` over ``(qubit_flux_v, coupler_flux_v)``, and
         ``theta_rad`` / ``j_hz`` / ``peak_transfer`` /
         ``resonance_qubit_flux_v`` / ``hwhm_qubit_flux_v`` / ``fit_r_squared``
         / ``fit_success`` / ``branch_warn`` / ``j_poly_curve`` /
         ``resonance_poly_curve`` over ``coupler_flux_v``
attrs  : ``swap_time_ns``, ``coupler_off_v``, ``j_at_off_hz``, ``j_max_hz``,
         ``j_max_coupler_flux_v``, ``theta_max_rad``, ``poly_degree``,
         ``off_is_interpolated``, ``n_j_ok``, ``n_branch_warn``,
         ``n_poly_rows``, plus the conversion formulas
         ``j2_poly_coeffs`` / ``resonance_poly_coeffs`` / ``poly_window_v`` /
         ``poly_prefer_v`` / ``resonance_poly_degree`` /
         ``resonance_poly_r_squared`` (plus the shared map attrs)

Every raw array is drawn UNCONDITIONALLY and every fit-derived overlay is
guarded, so a run in which every column failed still produces all three figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_pair_swap_flux_map", "plot_coupling_curve", "plot_transfer_fit"]

AXIS0 = "qubit_flux_v"
AXIS1 = "coupler_flux_v"

#: reference angles worth seeing on the theta axis, as (radians, label).
_LANDMARKS = (
    (np.pi / 2, r"$\pi/2$  (full swap)"),
    (np.pi / 4, r"$\pi/4$  ($\sqrt{\mathrm{iSWAP}}$)"),
)

#: how many representative columns the diagnostic figure cuts through.
_N_CUTS = 3

#: The fixed-time map's own figure is the shared 2x2 joint-population map; the
#: alias keeps the per-estimator import path stable while the drawing lives once
#: in ``_pair_swap_maps``.
plot_pair_swap_flux_map = plot_pair_swap_map


def _column(plot_data: xr.Dataset, name: str) -> np.ndarray:
    """One per-coupler column as float, all-NaN when the variable is absent."""
    size = int(plot_data.sizes[AXIS1])
    if name not in plot_data:
        return np.full(size, np.nan)
    return np.asarray(plot_data[name].values, dtype=float)


def _fit_window(plot_data: xr.Dataset, knob: np.ndarray) -> np.ndarray:
    """Mask of the coupler range the conversion polynomials were FITTED over.

    Both curves are interpolating fits, and the sweep axis is normally wider than
    the columns that actually fitted — so drawing a polynomial across the whole
    axis shows an extrapolation with no data under it, which on a two-branch
    ``J^2`` parabola climbs to a coupling nothing measured. Draw only where it
    is supported.
    """
    window = plot_data.attrs.get("poly_window_v", None)
    values = np.asarray(window, dtype=float).ravel() if window is not None else None
    if values is None or values.size != 2 or not np.isfinite(values).all():
        return np.ones(knob.shape, dtype=bool)
    return (knob >= values.min()) & (knob <= values.max())


def _annotate_empty(ax, message: str) -> None:
    """Say why a fit panel is blank instead of drawing an empty box."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.4")


def plot_coupling_curve(plot_data: xr.Dataset) -> plt.Figure:
    """The J(coupler flux) calibration curve, its resonance line and diagnostics.

    Three panels over one shared coupler-flux axis:

    * **top** — the fitted coupling. ``j_hz`` when the played duration was known,
      otherwise the dimensionless angle (the ``theta`` axis is always drawn on
      the right). The ``J^2`` polynomial and the decouple point it locates are
      overlaid, and columns flagged ``branch_warn`` are ringed: past a full swap
      ``sin^2`` folds and J is UNDER-reported there (see the estimator's module
      docstring).
    * **middle** — ``resonance_qubit_flux_v``, i.e. where the pair is resonant at
      each coupler bias. This is the resonance LINE, which bends with the
      coupler because the coupler pulse pulls both members.
    * **bottom** — the fitted peak height and half-width. The width is a
      DIAGNOSTIC: at fixed duration it is set by the pulse's Fourier limit and
      narrows as the coupling grows, so it is never converted into a coupling.
    """
    knob = np.asarray(plot_data[AXIS1].values, dtype=float)
    theta = _column(plot_data, "theta_rad")
    j_hz = _column(plot_data, "j_hz")
    peak = _column(plot_data, "peak_transfer")
    hwhm = _column(plot_data, "hwhm_qubit_flux_v")
    resonance = _column(plot_data, "resonance_qubit_flux_v")
    poly = _column(plot_data, "j_poly_curve")
    ok = _column(plot_data, "fit_success") > 0
    warn = _column(plot_data, "branch_warn") > 0

    has_hz = bool(np.isfinite(j_hz).any())
    t_ns = float(plot_data.attrs.get("swap_time_ns", float("nan")))

    fig, (ax_j, ax_res, ax_diag) = plt.subplots(
        3, 1, figsize=(9, 10), sharex=True, constrained_layout=True
    )

    # --- top: the coupling curve ---------------------------------------
    # Plotted in MHz when the duration is known and in radians otherwise. The two
    # are exactly proportional (theta = 2*pi*J*t), so the OTHER one rides along as
    # a linear secondary axis and the landmark angles are placed in whichever
    # unit the primary axis carries.
    scale = (2.0 * np.pi * t_ns * 1e-9 * 1e6) if (has_hz and np.isfinite(t_ns)
                                                  and t_ns > 0) else None
    primary = (j_hz / 1e6) if has_hz else theta
    if ok.any():
        ax_j.plot(knob[ok], primary[ok], "o-", color="C0",
                  label="fitted |J|" if has_hz else r"fitted $\theta$")
        ax_j.set_ylabel("|J| (MHz)" if has_hz else r"$\theta$ (rad)")
        if has_hz and np.isfinite(poly).any():
            inside = _fit_window(plot_data, knob)
            ax_j.plot(knob[inside], poly[inside] / 1e6, "-", color="C3", lw=1.2,
                      alpha=0.8,
                      label=f"$J^2$ poly (deg "
                            f"{int(plot_data.attrs.get('poly_degree', 0))})")
        if scale is not None:
            secondary = ax_j.secondary_yaxis(
                "right", functions=(lambda v: v * scale, lambda v: v / scale))
            secondary.set_ylabel(r"$\theta$ (rad)")
        for level, label in _LANDMARKS:
            ax_j.axhline(level / scale if scale is not None else level,
                         color="0.7", ls=":", lw=1)
            ax_j.annotate(label,
                          xy=(knob[0], level / scale if scale is not None else level),
                          fontsize=8, color="0.45", va="bottom")
        if warn.any():
            ax_j.plot(knob[warn], primary[warn], "o", mfc="none", mec="C3",
                      ms=11, mew=1.6, label=r"suspect fold ($\theta>\pi/2$)")
        off_v = float(plot_data.attrs.get("coupler_off_v", float("nan")))
        if np.isfinite(off_v):
            interpolated = int(plot_data.attrs.get("off_is_interpolated", 0))
            ax_j.axvline(off_v, color="C2", ls="--", lw=1.4,
                         label=f"decouple point {off_v:.4g} V"
                               f"{'' if interpolated else '  (grid)'}")
        ax_j.legend(loc="best", fontsize=8)
    else:
        _annotate_empty(ax_j, "no coupler column passed the peak fit")
        ax_j.set_ylabel("|J| (MHz)" if has_hz else r"$\theta$ (rad)")
    title = "exchange coupling vs coupler flux"
    if np.isfinite(t_ns):
        title += f"   (pulse {t_ns:.0f} ns)"
    else:
        title += "   (duration unknown — angle only)"
    ax_j.set_title(title)

    # --- middle: the resonance line + its own conversion polynomial -----
    resonance_poly = _column(plot_data, "resonance_poly_curve")
    if np.isfinite(resonance).any():
        ax_res.plot(knob[ok], resonance[ok], "o", color="C4", ms=5,
                    label="fitted per column")
        if np.isfinite(resonance_poly).any():
            degree = int(plot_data.attrs.get("resonance_poly_degree", 0))
            r2 = float(plot_data.attrs.get("resonance_poly_r_squared", float("nan")))
            inside = _fit_window(plot_data, knob)
            ax_res.plot(knob[inside], resonance_poly[inside], "-", color="C3",
                        lw=1.2, alpha=0.8,
                        label=f"poly (deg {degree}, $R^2$={r2:.3f})")
        ax_res.legend(loc="best", fontsize=8)
    else:
        _annotate_empty(ax_res, "no fitted resonance point")
    ax_res.set_ylabel("resonance qubit flux (V)")

    # --- bottom: the diagnostics ---------------------------------------
    if np.isfinite(peak).any():
        ax_diag.plot(knob[ok], peak[ok], "o-", color="C1", label="peak transfer")
    ax_diag.set_ylabel("peak transfer (0-1)")
    ax_diag.set_ylim(0.0, 1.05)
    ax_w = ax_diag.twinx()
    if np.isfinite(hwhm).any():
        ax_w.plot(knob[ok], hwhm[ok], "s--", color="0.45", ms=4,
                  label="HWHM (diagnostic)")
    ax_w.set_ylabel("HWHM qubit flux (V)")
    if not np.isfinite(peak).any() and not np.isfinite(hwhm).any():
        _annotate_empty(ax_diag, "no fitted peaks")
    handles = ax_diag.get_legend_handles_labels()[0] + ax_w.get_legend_handles_labels()[0]
    if handles:
        ax_diag.legend(handles=handles, loc="best", fontsize=8)
    ax_diag.set_xlabel("coupler flux (V)")

    fig.suptitle(
        "pair swap flux map — coupling calibration\n"
        "peak height carries J; the HWHM is Fourier-limited and is NOT a coupling",
        fontsize=11,
    )
    return fig


def plot_transfer_fit(plot_data: xr.Dataset) -> plt.Figure:
    """The normalized transfer map with the fitted resonance line, plus cuts.

    Left: ``transfer_norm`` — the single-excitation-normalized transfer the fit
    actually consumes — with the fitted resonance line drawn over it. Right: a
    few representative coupler columns with their Lorentzian fits, which is the
    "did the per-column fit work?" view. The map is drawn unconditionally.
    """
    qubit_flux = np.asarray(plot_data[AXIS0].values, dtype=float)
    knob = np.asarray(plot_data[AXIS1].values, dtype=float)
    transfer = np.asarray(plot_data["transfer_norm"].values, dtype=float)
    best_fit = (np.asarray(plot_data["transfer_fit"].values, dtype=float)
                if "transfer_fit" in plot_data else np.full(transfer.shape, np.nan))
    resonance = _column(plot_data, "resonance_qubit_flux_v")
    theta = _column(plot_data, "theta_rad")
    ok = _column(plot_data, "fit_success") > 0

    fig, (ax_map, ax_cuts) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )

    mesh = ax_map.pcolormesh(
        *np.meshgrid(knob, qubit_flux), transfer,
        shading="auto", cmap="viridis", vmin=0.0, vmax=1.0,
    )
    fig.colorbar(mesh, ax=ax_map, label="normalized transfer")
    if np.isfinite(resonance).any():
        ax_map.plot(knob[ok], resonance[ok], "-", color="w", lw=1.6, alpha=0.85)
        ax_map.plot(knob[ok], resonance[ok], ".", color="C3", ms=4,
                    label="fitted resonance")
        ax_map.legend(loc="best", fontsize=8)
    ax_map.set_xlabel("coupler flux (V)")
    ax_map.set_ylabel("qubit flux (V)")
    ax_map.set_title(r"$p_{\rm transfer}/(p_{01}+p_{10})$")

    # --- right: representative columns and their fits --------------------
    good = np.flatnonzero(ok)
    picks = (good[np.linspace(0, good.size - 1, min(_N_CUTS, good.size)).astype(int)]
             if good.size else
             np.linspace(0, knob.size - 1, min(_N_CUTS, knob.size)).astype(int))
    for n, i in enumerate(picks):
        colour = f"C{n}"
        ax_cuts.plot(qubit_flux, transfer[:, int(i)], "o", ms=4, color=colour,
                     label=f"{knob[int(i)]:.4g} V"
                           + (f"  ($\\theta$={theta[int(i)]:.2f})"
                              if np.isfinite(theta[int(i)]) else "  (no fit)"))
        if np.isfinite(best_fit[:, int(i)]).any():
            ax_cuts.plot(qubit_flux, best_fit[:, int(i)], "-", lw=1.4, color=colour)
    if not good.size:
        _annotate_empty(ax_cuts, "no column passed the peak fit\n(raw cuts shown)")
    ax_cuts.set_xlabel("qubit flux (V)")
    ax_cuts.set_ylabel("normalized transfer")
    ax_cuts.set_ylim(-0.05, 1.05)
    ax_cuts.legend(loc="best", fontsize=8, title="coupler flux")
    ax_cuts.set_title("representative columns + central-peak fit")
    return fig
