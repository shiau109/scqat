"""Shared secondary x-axis — one sweep drawn in two scales at once.

A sweep is often *driven* in one scale and *understood* in another. The
amplitude family is the reference case: the probe plays a dimensionless
prefactor of the target's stored pulse amplitude (so one array serves every
qubit in a multiplexed run), but what a reader wants off the figure months
later is the ABSOLUTE amplitude that prefactor stood for. The caller supplies
the second scale as an extra coordinate over the same points; this module
draws it as a secondary top axis.

SECONDARY, never a relabel. The fit answer, the vertical marker and the
annotation box all live in the PRIMARY frame; relabelling the one axis would
silently change what those numbers mean. A top axis shows both frames at once
and leaves the primary reading intact.

The estimator stays frame-AGNOSTIC: it knows only "there is a swept
coordinate, and there may be a second coordinate over the same points". It
never learns which one is the ratio, so nothing here assumes a direction, a
unit, or an affine relation — :func:`add_twin_axis` interpolates, so any
strictly monotone companion works.

A twin that cannot be drawn is simply absent: :func:`twin_values` returns
``None`` for a missing, non-finite or non-monotone companion, and every caller
treats that as "draw the primary axis only". A figure must never fail over a
decoration.

This is a plain shared FUNCTION module consumed by several estimators' figure
steps — the same arrangement as :mod:`scqat.estimators._iq_plane`, and for the
same reason: it is presentation, not math, so it lives outside ``tools/``.

plot_data contract (what a carrying estimator writes)
-----------------------------------------------------
vars  : ``twin`` — the companion scale, 1-D over the sweep dim.
attrs : ``twin_label`` — the axis label; optionally ``opt_twin_value`` /
        ``best_twin_value`` — the answer expressed in the companion scale.
"""

from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["TWIN_KNOBS", "twin_values", "add_twin_axis", "twin_at"]

#: the kwarg names an estimator accepts for this feature. Estimators validate
#: their own flat kwarg surface, so each one unions this into its whitelist
#: rather than loosening the check.
TWIN_KNOBS = frozenset({"twin_coord", "twin_label"})


def twin_values(
    dataset: xr.Dataset, sweep_coord: str, twin_coord: Optional[str]
) -> Optional[np.ndarray]:
    """The companion scale over the swept points, or ``None`` when undrawable.

    ``None`` (never an exception) for every degenerate case, because the twin
    is a decoration and must not be able to fail an analysis:

    * ``twin_coord`` is ``None`` or absent from ``dataset``;
    * it is not 1-D, or its length disagrees with the sweep;
    * any value is non-finite — a partially-known scale would draw a plausible
      axis with wrong ticks, which is worse than no axis;
    * it is not strictly monotone, so it has no invertible axis transform.
    """
    if not twin_coord or twin_coord not in dataset.coords:
        return None
    values = np.asarray(dataset.coords[twin_coord].values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        return None
    if sweep_coord in dataset.coords:
        if values.size != np.asarray(dataset.coords[sweep_coord].values).size:
            return None
    if values.size < 2:
        return None
    step = np.diff(values)
    if not ((step > 0).all() or (step < 0).all()):
        return None
    return values


def twin_at(sweep: np.ndarray, twin: np.ndarray, value: float) -> float:
    """The companion-scale value at one primary-scale point (e.g. the fitted
    optimum), so the annotation can state the answer in both frames."""
    if not np.isfinite(value):
        return float("nan")
    return float(_interp(sweep, twin)(np.asarray([value], dtype=float))[0])


def add_twin_axis(
    ax: plt.Axes, sweep: np.ndarray, twin: np.ndarray, label: str
) -> None:
    """Draw ``twin`` as a secondary TOP x-axis over the same points.

    Interpolating both ways rather than assuming a scale factor, so a
    non-linear companion is drawn correctly; the ends extrapolate linearly, so
    matplotlib can place ticks slightly outside the data range (it asks for
    transforms over the full axis limits, which are padded).
    """
    sweep = np.asarray(sweep, dtype=float)
    twin = np.asarray(twin, dtype=float)
    forward: Callable = _interp(sweep, twin)
    inverse: Callable = _interp(twin, sweep)
    secondary = ax.secondary_xaxis("top", functions=(forward, inverse))
    secondary.set_xlabel(label, fontsize=12)

    # The title sits where the secondary label now is, and matplotlib does not
    # reflow it — so lift it out of the way. Font properties are re-applied
    # explicitly because set_title() resets them to the rcParams defaults.
    title = ax.get_title()
    if title:
        ax.set_title(title, pad=30, fontsize=ax.title.get_fontsize(),
                     fontweight=ax.title.get_fontweight())


# ----------------------------------------------------------------------
def _interp(x: np.ndarray, y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Monotone piecewise-linear map with LINEAR EXTRAPOLATION past both ends.

    ``np.interp`` clamps outside its range, which would flatten the transform
    at the edges and pile ticks up there; matplotlib evaluates these functions
    over the padded axis limits, so the ends must keep their slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x[0] > x[-1]:  # np.interp needs an increasing table
        x, y = x[::-1], y[::-1]

    def _edge_slope(i: int, j: int) -> float:
        return (y[j] - y[i]) / (x[j] - x[i]) if x[j] != x[i] else 0.0

    lo_slope, hi_slope = _edge_slope(0, 1), _edge_slope(-2, -1)

    def mapper(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        out = np.interp(values, x, y)
        out = np.where(values < x[0], y[0] + (values - x[0]) * lo_slope, out)
        out = np.where(values > x[-1], y[-1] + (values - x[-1]) * hi_slope, out)
        return out

    return mapper
