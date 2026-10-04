"""Tests for tools.flux_crosstalk - the apex-position line whose slope is -m."""

import numpy as np
import pytest

from scqat.tools.flux_crosstalk import fit_crosstalk_line


def test_crosstalk_line_is_exact_on_a_line_and_order_free():
    b = np.array([-0.05, -0.025, 0.0, 0.025, 0.05])
    apex = 0.4e-3 - 0.031 * b
    err = np.full(b.size, 1e-5)
    r = fit_crosstalk_line(b, apex, err)
    assert r["success"] and r["n_used"] == 5
    assert r["crosstalk"] == pytest.approx(0.031, abs=1e-9)
    assert r["apex_at_zero"] == pytest.approx(0.4e-3, abs=1e-9)
    assert r["max_residual"] < 1e-12
    perm = np.array([3, 0, 4, 1, 2])
    p = fit_crosstalk_line(b[perm], apex[perm], err[perm])
    assert p["crosstalk"] == pytest.approx(r["crosstalk"], rel=1e-12)
    # residuals come back per INPUT point, in the caller's order
    assert np.allclose(p["residuals"], r["residuals"][perm], atol=1e-12)


def test_crosstalk_line_honours_the_mask_and_refuses_two_points():
    b = np.array([-0.05, 0.0, 0.05, 0.1])
    apex = np.array([1e-3, 0.0, -1e-3, 5e-3])  # the last one is an outlier
    err = np.full(4, 1e-5)
    masked = fit_crosstalk_line(b, apex, err, np.array([1, 1, 1, 0], dtype=bool))
    assert masked["crosstalk"] == pytest.approx(0.02, abs=1e-9)
    assert masked["used"].tolist() == [1, 1, 1, 0] and np.isnan(masked["residuals"][3])
    two = fit_crosstalk_line(b, apex, err, np.array([1, 0, 1, 0], dtype=bool))
    assert two["success"] is False and np.isnan(two["crosstalk"])
