"""A swept axis's traversal ORDER is acquisition provenance, never analysis input.

An acquisition layer may walk a swept window in either direction - SCQO's
``start_*``/``end_*`` fields mean exactly that order, because consecutive points
are not always independent (heating, long flux tails, hysteresis) and the
direction has to stay visible in the stored data. The dataset therefore arrives
in its REALIZED order, possibly descending, and is never re-sorted upstream.

The contract on this side: **the same (x, y) pairs in either order give the same
result.** Positional code breaks it silently, and none of these raise:

* a span or step taken by position - ``x[-1] - x[0]``, ``x[1] - x[0]`` - is
  negative on a descending axis, so a bound built from it inverts (``peak_fit``
  fitted a 4 MHz line as a ~180 MHz one this way);
* ``np.interp`` with a decreasing ``xp`` returns garbage, and ``searchsorted``
  assumes ascending input;
* "the first point" standing for "the lowest value" (``best_fit[0]`` as the
  zero-amplitude reference), or a neighbour walk seeded from index 0.

A TOOL fixes those in place, because it returns per-point arrays its caller
pairs with its own axis. An ESTIMATOR takes the simpler, total route:
:func:`ascending` canonicalizes the swept dims on entry to both
``extract_parameters`` and ``build_plot_data``, so everything downstream sees one
order and the direction cannot leak into any number. Only the analysis VIEW is
sorted; the caller's dataset - and the stored ``dataset.nc`` - keep the order the
instrument walked.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["ascending"]


def ascending(dataset: xr.Dataset, *dims: str) -> xr.Dataset:
    """``dataset`` with each named swept dim in ascending coordinate order.

    Every variable and coordinate along the dim moves with its label, so each
    (x, y) pair is preserved and only the traversal order is discarded. A dim
    that is absent, has no 1-D coordinate, or is already non-decreasing is
    left alone (no copy). Sorting is stable, so a non-monotone axis (an explicit
    point list) is canonicalized too - repeated values keep their relative order.
    """
    for dim in dims:
        if dim not in dataset.dims or dim not in dataset.coords:
            continue
        values = np.asarray(dataset.coords[dim].values)
        if values.ndim != 1 or values.size < 2:
            continue
        if bool(np.all(np.diff(values) >= 0)):
            continue
        dataset = dataset.sortby(dim)
    return dataset
