"""``tools.sweep_order.ascending`` - the one canonicalization every estimator
that reads a swept flux, detuning or amplitude axis applies on entry."""

import numpy as np
import xarray as xr

from scqat.tools.sweep_order import ascending


def _map(x, y):
    """(x, y) map whose VALUE at each point encodes both labels, so any
    mis-pairing after a reorder shows up as a wrong number."""
    data = x[:, None] * 1000.0 + y[None, :]
    return xr.Dataset(
        {"signal": (("x", "y"), data), "twin": ("x", 2.0 * x)},
        coords={"x": x, "y": y, "full": ("y", y + 5.0)},
    )


def test_descending_dims_come_back_ascending_with_every_pair_intact():
    ds = _map(np.array([3.0, 2.0, 1.0]), np.array([30.0, 20.0, 10.0, 0.0]))
    out = ascending(ds, "x", "y")
    np.testing.assert_array_equal(out.x.values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out.y.values, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_array_equal(out.twin.values, 2.0 * out.x.values)
    np.testing.assert_array_equal(out.full.values, out.y.values + 5.0)
    np.testing.assert_array_equal(
        out.signal.values, out.x.values[:, None] * 1000.0 + out.y.values[None, :])
    # the input is untouched: the caller's dataset keeps the realized order
    np.testing.assert_array_equal(ds.x.values, [3.0, 2.0, 1.0])


def test_ascending_input_is_returned_as_is():
    ds = _map(np.array([1.0, 2.0]), np.array([0.0, 5.0]))
    assert ascending(ds, "x", "y") is ds


def test_only_the_named_dims_are_touched_and_absent_ones_are_ignored():
    ds = _map(np.array([2.0, 1.0]), np.array([9.0, 3.0]))
    out = ascending(ds, "x", "not_a_dim")
    np.testing.assert_array_equal(out.x.values, [1.0, 2.0])
    np.testing.assert_array_equal(out.y.values, [9.0, 3.0])


def test_a_non_monotone_point_list_is_sorted_stably():
    ds = _map(np.array([1.0, 0.5, 2.0, 0.5]), np.array([0.0]))
    out = ascending(ds, "x")
    np.testing.assert_array_equal(out.x.values, [0.5, 0.5, 1.0, 2.0])
    np.testing.assert_array_equal(out.signal.values[:, 0], out.x.values * 1000.0)
