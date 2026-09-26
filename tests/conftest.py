"""Shared pytest configuration.

Force the non-interactive ``Agg`` matplotlib backend for the whole test
session so figure-generating code never tries to open a GUI window.  This keeps
the suite reproducible on headless machines / CI and avoids Tcl/Tk errors when
the active environment lacks a working interactive backend.

Also the ``order_free`` fixture - the one check every estimator that reads a
swept flux, detuning or amplitude axis carries (see
``scqat.tools.sweep_order``).
"""

import math

import matplotlib
import pytest
import xarray as xr

matplotlib.use("Agg")

from scqat.core.base_estimator import _json_safe  # noqa: E402 - after the backend pin


def _same(a, b, rtol, atol, path="metadata"):
    """Recursive equality over ``_json_safe`` output; floats within
    ``rtol``/``atol`` (exact when ``rtol`` is None)."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for key in a:
            _same(a[key], b[key], rtol, atol, f"{path}.{key}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, rtol, atol, f"{path}[{i}]")
    elif rtol is not None and isinstance(a, float) and isinstance(b, float):
        assert math.isclose(a, b, rel_tol=rtol, abs_tol=atol), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _assert_order_free(estimator, dataset, dims, *, rtol=None, atol=0.0, **kwargs):
    """The SAME data walked the other way must give the SAME answer.

    Reverses ``dims`` - every (x, y) pair is kept, only the traversal order
    flips - and requires identical metadata and identical plot data. Exact by
    default, not approximate: an estimator canonicalizes its swept dims on
    entry, so a tolerance would only hide a place that forgot to. ``rtol`` /
    ``atol`` are for an estimator whose answer already wobbles between two calls on the SAME
    data (say so where it is passed). Returns the ascending run's results so the
    caller can also assert the answer is RIGHT (two equal wrong answers would
    pass the symmetry check alone).
    """
    dims = (dims,) if isinstance(dims, str) else tuple(dims)
    flipped = dataset.isel({dim: slice(None, None, -1) for dim in dims})
    for dim in dims:  # the fixture must really present a descending axis
        values = flipped.coords[dim].values
        assert values[0] > values[-1], f"{dim} did not flip - check the fixture"

    up = estimator.extract_parameters(dataset, **dict(kwargs))
    down = estimator.extract_parameters(flipped, **dict(kwargs))
    _same(_json_safe(estimator.extract_metadata(up)),
          _json_safe(estimator.extract_metadata(down)), rtol, atol)

    plot_up = estimator.build_plot_data(dataset, up, **dict(kwargs))
    plot_down = estimator.build_plot_data(flipped, down, **dict(kwargs))
    if plot_up is None:
        assert plot_down is None
    elif rtol is None:
        xr.testing.assert_identical(plot_up, plot_down)
    else:
        xr.testing.assert_allclose(plot_up, plot_down, rtol=rtol, atol=atol)
        assert plot_up.attrs.keys() == plot_down.attrs.keys()
    return up


@pytest.fixture
def order_free():
    """``order_free(estimator, dataset, dims, **kwargs)`` -> ascending results."""
    return _assert_order_free
