"""Shared IQ-plane diagnostic panel — one uniform figure for every IQ-reducing
experiment.

Shows the RAW averaged IQ cloud together with the reference the reduction
actually used, so the pre-processing is visible per run (the companion of the
``reduction_method`` / ``ref_iq`` provenance stamps):

* **radial** experiments (spectroscopy family): the reference *point*
  (``|IQ - ref|`` is the fitted signal) — a star, labeled with its source
  (auto median vs supplied);
* **axial** experiments (power_rabi / ramsey / T1 / echo): the projection *axis*
  through the cloud with its positive direction, labeled with how the axis was
  resolved (pca / angle / positions);
* **2-D maps** (vs flux): all slices colored by the slow axis, plus the
  per-slice reference trajectory (the ground point moves with flux — the reason
  the reference is per-slice).

This is a plain shared FUNCTION consumed by several estimators' figure steps
(function-level sharing is what the estimator-layering rule permits; it lives
outside ``tools/`` because it is presentation, not math). It draws ONLY from
``plot_data``:

vars   : ``iq_i`` / ``iq_q`` — float pairs (netCDF-safe), 1-D over the sweep dim
         or 2-D over (slow, fast); optional ``ref_i`` / ``ref_q`` over the slow
         dim (per-slice reference trajectory).
attrs  : radial — ``ref_iq_real`` / ``ref_iq_imag`` (+ ``ref_source``);
         axial — ``reduction_method`` + ``reduction_angle``.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_iq_plane"]


def has_iq_plane(plot_data: xr.Dataset) -> bool:
    """Whether ``plot_data`` carries the IQ-plane vars (absent when the probe
    returned a pre-reduced real ``signal`` — no cloud exists to draw)."""
    return "iq_i" in plot_data.data_vars and "iq_q" in plot_data.data_vars


def _reduction_kind(plot_data: xr.Dataset) -> str:
    """Which reference geometry the reduction used: ``"radial"`` (a point),
    ``"radial-per-slice"`` (a point per slow-axis slice), ``"axial"`` (an axis),
    or ``""`` (bare cloud, nothing resolvable)."""
    if plot_data["iq_i"].ndim == 2:
        return "radial-per-slice"
    attrs = plot_data.attrs
    if "ref_iq_real" in attrs and "ref_iq_imag" in attrs:
        return "radial"
    angle = float(attrs.get("reduction_angle", np.nan))
    method = str(attrs.get("reduction_method", ""))
    if np.isfinite(angle) and method and method != "signal":
        return "axial"
    return ""


_KIND_TITLES = {
    "radial": "IQ plane — RADIAL reduction: signal = |IQ - ref|",
    "radial-per-slice": "IQ plane — RADIAL reduction (per slice): signal = |IQ - ref(slice)|",
    "axial": "IQ plane — AXIAL reduction: signal = projection onto the |0>-|1> axis",
}


def plot_iq_plane(plot_data: xr.Dataset) -> plt.Figure:
    """Draw the raw IQ cloud + the reference used by the reduction (see module
    docstring for the plot_data contract). The title states whether the
    reference is RADIAL (a point) or AXIAL (an axis) and what the fitted
    signal therefore is."""
    if not has_iq_plane(plot_data):
        raise ValueError("plot_data has no 'iq_i'/'iq_q' variables — no IQ cloud to draw.")
    iq_i = plot_data["iq_i"]
    iq_q = plot_data["iq_q"]

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    if iq_i.ndim == 2:
        _draw_2d(ax, fig, plot_data, iq_i, iq_q)
    else:
        _draw_1d(ax, fig, plot_data, iq_i, iq_q)

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(_KIND_TITLES.get(_reduction_kind(plot_data), "IQ plane: raw cloud"),
                 fontsize=10)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
def _scatter(ax, fig, i, q, c, label: str, cbar_label: str):
    sc = ax.scatter(i, q, c=c, s=8, cmap="viridis", label=label)
    fig.colorbar(sc, ax=ax, label=cbar_label)


def _draw_1d(ax, fig, plot_data: xr.Dataset, iq_i: xr.DataArray, iq_q: xr.DataArray) -> None:
    dim = iq_i.dims[0]
    sweep = (
        plot_data.coords[dim].values.astype(float)
        if dim in plot_data.coords else np.arange(iq_i.sizes[dim], dtype=float)
    )
    i = iq_i.values.astype(float)
    q = iq_q.values.astype(float)
    _scatter(ax, fig, i, q, sweep, "raw IQ", str(dim))

    attrs = plot_data.attrs
    if "ref_iq_real" in attrs and "ref_iq_imag" in attrs:
        # radial: the reference POINT the distance signal is measured from
        source = str(attrs.get("ref_source", "median"))
        ax.plot(
            [float(attrs["ref_iq_real"])], [float(attrs["ref_iq_imag"])],
            "*", color="red", markersize=15, mec="black",
            label=f"radial ref ({source})",
        )
        return

    angle = float(attrs.get("reduction_angle", np.nan))
    method = str(attrs.get("reduction_method", ""))
    if np.isfinite(angle) and method and method != "signal":
        # axial: the projection AXIS through the cloud, positive direction marked.
        # axial() returns Re(z * e^{i*a}) so the +projection direction in the IQ
        # plane is e^{-i*a} = (cos a, -sin a).
        c0 = complex(float(np.mean(i)), float(np.mean(q)))
        dvec = np.array([np.cos(angle), -np.sin(angle)])
        half = 0.55 * max(float(np.ptp(i)), float(np.ptp(q))) or 1.0
        p0 = np.array([c0.real, c0.imag]) - half * dvec
        p1 = np.array([c0.real, c0.imag]) + half * dvec
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "--", color="red", lw=1.4,
                label=f"axial axis ({method})")
        ax.annotate(
            "", xy=tuple(np.array([c0.real, c0.imag]) + 0.5 * half * dvec),
            xytext=(c0.real, c0.imag),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.4),
        )


def _draw_2d(ax, fig, plot_data: xr.Dataset, iq_i: xr.DataArray, iq_q: xr.DataArray) -> None:
    slow_dim, fast_dim = iq_i.dims
    slow = (
        plot_data.coords[slow_dim].values.astype(float)
        if slow_dim in plot_data.coords else np.arange(iq_i.sizes[slow_dim], dtype=float)
    )
    i = iq_i.values.astype(float)
    q = iq_q.values.astype(float)
    c = np.repeat(slow, i.shape[1])
    _scatter(ax, fig, i.ravel(), q.ravel(), c, "raw IQ (per slice)", str(slow_dim))

    if "ref_i" in plot_data.data_vars and "ref_q" in plot_data.data_vars:
        ri = plot_data["ref_i"].values.astype(float)
        rq = plot_data["ref_q"].values.astype(float)
        ok = np.isfinite(ri) & np.isfinite(rq)
        if ok.any():
            ax.plot(ri[ok], rq[ok], "-", color="red", lw=1.0, alpha=0.7)
            ax.plot(ri[ok], rq[ok], "*", color="red", markersize=10, mec="black",
                    label="per-slice radial ref (median)")
