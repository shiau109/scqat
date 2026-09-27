"""Figures for ``pair_coupler_crossing``.

Consumes the **plot_data** Dataset built by
``PairCouplerCrossingEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``coupler_flux_v``, ``joint_state``, ``member`` (``high``/``low``),
         ``flux_fine``
vars   : ``joint_population`` (joint_state, coupler_flux_v); ``marginal`` /
         ``normalized`` / ``in_dip`` (member, coupler_flux_v);
         ``arch_frequency_hz`` (flux_fine, all-NaN when the arch was not solved)
attrs  : every scalar of the results (crossings, symmetry points, the center and
         its kind, the arch, the flags, ``high_name``/``low_name``) plus
         ``dip_centers_<role>_v``

The raw populations are drawn UNCONDITIONALLY; every fit-derived overlay is
guarded, so a run with no dips, or an unsolved arch, still renders both figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_crossings", "plot_arch"]

AXIS = "coupler_flux_v"
_COLORS = {"high": "tab:red", "low": "tab:blue"}


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _label(plot_data: xr.Dataset, role: str) -> str:
    return f"{plot_data.attrs.get(f'{role}_name', role)} ({role})"


def _mark_center(ax, plot_data: xr.Dataset) -> None:
    at = plot_data.attrs.get("bracket_at", 0.0)
    if _finite(at):
        ax.axvline(float(at), color="0.5", lw=0.8, ls=":", label="idle")
    center = plot_data.attrs.get("center_from_idle_v")
    if _finite(center):
        kind = plot_data.attrs.get("center_kind", "unknown")
        ax.axvline(float(center), color="k", lw=1.0, ls="--", label=f"center ({kind})")


def plot_crossings(plot_data: xr.Dataset) -> plt.Figure:
    """Top: the four joint populations. Bottom: each measured member's normalized
    marginal with its dips shaded, the threshold, the paired crossings and the
    symmetry point."""
    x = np.asarray(plot_data[AXIS].values, dtype=float)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for label in plot_data["joint_state"].values:
        top.plot(x, plot_data["joint_population"].sel(joint_state=label).values,
                 lw=1.0, label=f"P{label}")
    top.set_ylabel("joint population")
    top.set_title(f"coupler crossings (digits: {plot_data.attrs.get('high_name', 'high')}, "
                  f"{plot_data.attrs.get('low_name', 'low')})")
    top.legend(loc="best", fontsize=8, ncol=4)

    for role in plot_data["member"].values:
        role = str(role)
        if not int(plot_data.attrs.get(f"measured_{role}", 0)):
            continue
        s = np.asarray(plot_data["normalized"].sel(member=role).values, dtype=float)
        color = _COLORS.get(role, None)
        bottom.plot(x, s, lw=1.0, color=color, label=_label(plot_data, role))
        in_dip = np.asarray(plot_data["in_dip"].sel(member=role).values, dtype=bool)
        if in_dip.any():
            bottom.fill_between(x, 0, 1, where=in_dip, color=color, alpha=0.12,
                                transform=bottom.get_xaxis_transform(), step="mid")
        for key in ("lower", "upper"):
            xc = plot_data.attrs.get(f"crossing_{role}_{key}_v")
            if _finite(xc):
                bottom.axvline(float(xc), color=color, lw=1.2)
    threshold = plot_data.attrs.get("dip_threshold")
    if _finite(threshold):
        bottom.axhline(float(threshold), color="0.4", lw=0.8, ls="-.", label="dip threshold")
    _mark_center(bottom, plot_data)
    bottom.set_xlabel("coupler flux (V)")
    bottom.set_ylabel("normalized P(excited)")
    bottom.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_arch(plot_data: xr.Dataset) -> plt.Figure:
    """The solved coupler arch against the two member frequencies; an annotated
    empty panel when the arch was not solved."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fine = np.asarray(plot_data["flux_fine"].values, dtype=float)
    arch = np.asarray(plot_data["arch_frequency_hz"].values, dtype=float)
    ax.set_xlabel("coupler flux (V)")
    ax.set_ylabel("frequency (GHz)")
    if not np.isfinite(arch).any():
        ax.text(0.5, 0.5, "arch not solved\n(needs both members' crossings, a known "
                "center kind\nand both member frequencies)",
                ha="center", va="center", transform=ax.transAxes, color="0.4")
        ax.set_title("coupler arch")
        return fig
    ax.plot(fine, arch / 1e9, color="k", lw=1.2, label="coupler arch (model)")
    for role in ("high", "low"):
        f_hz = plot_data.attrs.get(f"f_{role}_hz")
        if not _finite(f_hz):
            continue
        color = _COLORS[role]
        ax.axhline(float(f_hz) / 1e9, color=color, lw=0.8, ls="--", label=_label(plot_data, role))
        for key in ("lower", "upper"):
            xc = plot_data.attrs.get(f"crossing_{role}_{key}_v")
            if _finite(xc):
                ax.plot([float(xc)], [float(f_hz) / 1e9], "o", color=color)
    _mark_center(ax, plot_data)
    top = plot_data.attrs.get("f_c_max_hz")
    period = plot_data.attrs.get("period_v")
    if _finite(top) and _finite(period):
        ax.set_title(f"coupler arch: f_c_max = {float(top) / 1e9:.2f} GHz, "
                     f"period = {float(period):.3f} V")
    finite = arch[np.isfinite(arch)]
    ax.set_ylim(max(0.0, float(np.min(finite)) / 1e9 - 0.2), float(np.max(finite)) / 1e9 + 0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
