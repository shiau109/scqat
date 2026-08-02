"""Tests for the triangle-peak fit (XY-Z delay reduction).

Synthetic triangle-overlap trace ``offset + amp*max(0, 1-|t-t0|/hw)`` — the
shape two equal-length pulses' overlap produces as one slides past the other.
The tool must recover the peak position ``t0`` (in the unit of the passed time
axis — ns and seconds both exercised, the tool is unit-agnostic), flag a clean
peak ``success=True``, reject a noise-only trace, and degrade to an honest
``success=False`` argmax fallback on a bad fit rather than raising.
"""

import numpy as np
import pytest

from scqat.tools.fit_triangle import MIN_SAMPLES, fit_triangle, triangle_peak


def _trace(t, amp, t0, hw, offset=0.0, noise=0.0, seed=0):
    y = triangle_peak(t, amp, t0, hw, offset)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, t.size)
    return y


class TestFitTriangle:

    def test_clean_peak_recovered(self):
        t = np.arange(-60.0, 60.0)  # ns, like the QM default scan
        y = _trace(t, amp=0.9, t0=12.0, hw=40.0, offset=0.05, noise=0.01)
        res = fit_triangle(t, y, xy_duration=40.0)
        assert res["success"] is True
        assert res["t0"] == pytest.approx(12.0, abs=1.0)
        assert res["amp"] == pytest.approx(0.9, rel=0.1)
        assert res["snr"] > 3.0
        assert res["best_fit"].shape == t.shape

    def test_unit_agnostic_seconds(self):
        """The same samples on a seconds axis return t0 scaled by 1e-9."""
        t_ns = np.arange(-60.0, 60.0)
        y = _trace(t_ns, amp=0.9, t0=12.0, hw=40.0, offset=0.05, noise=0.01)
        res_s = fit_triangle(t_ns * 1e-9, y, xy_duration=40e-9)
        assert res_s["success"] is True
        assert res_s["t0"] == pytest.approx(12e-9, abs=1e-9)

    def test_noise_only_is_not_significant(self):
        t = np.arange(-60.0, 60.0)
        y = np.random.default_rng(1).normal(0.0, 0.05, t.size)
        res = fit_triangle(t, y, xy_duration=40.0)
        assert res["success"] is False
        assert np.isnan(res["t0_std"])
        assert np.all(np.isnan(res["best_fit"]))

    def test_peak_near_edge_rejected(self):
        """A peak within xy_duration of a scan edge fails the edge gate and
        falls back to argmax (success=False)."""
        t = np.arange(-60.0, 60.0)
        y = _trace(t, amp=0.9, t0=55.0, hw=40.0, offset=0.05, noise=0.01)
        res = fit_triangle(t, y, xy_duration=40.0)
        assert res["success"] is False

    def test_fallback_t0_is_argmax(self):
        """On a rejected fit, t0 is the argmax of the baseline-subtracted trace."""
        t = np.arange(-60.0, 60.0)
        y = _trace(t, amp=0.9, t0=50.0, hw=40.0, offset=0.05)  # near-edge -> rejected
        res = fit_triangle(t, y, xy_duration=40.0)
        assert res["success"] is False
        baseline = np.percentile(y, 10)
        assert res["t0"] == pytest.approx(float(t[np.argmax(y - baseline)]))

    def test_max_t0_std_gate(self):
        """A tiny max_t0_std rejects an otherwise-clean fit."""
        t = np.arange(-60.0, 60.0)
        y = _trace(t, amp=0.9, t0=12.0, hw=40.0, offset=0.05, noise=0.01)
        res = fit_triangle(t, y, xy_duration=40.0, max_t0_std=1e-6)
        assert res["success"] is False

    def test_degenerate_input_fails_without_raising(self):
        t = np.arange(0.0, float(MIN_SAMPLES) - 1)  # one sample short
        res = fit_triangle(t, np.ones_like(t), xy_duration=40.0)
        assert res["success"] is False
        assert np.isnan(res["t0"])

    def test_all_nan_input_fails_without_raising(self):
        t = np.arange(-60.0, 60.0)
        res = fit_triangle(t, np.full_like(t, np.nan), xy_duration=40.0)
        assert res["success"] is False
