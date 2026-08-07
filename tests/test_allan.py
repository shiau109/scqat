"""overlapping_allan_deviation: white-noise scaling, degraded behavior."""

import numpy as np
import pytest

from scqat.tools.allan import overlapping_allan_deviation


class TestAllanDeviation:
    def test_white_noise_averages_down(self):
        """White estimation noise: adev(tau) falls like 1/sqrt(tau)."""
        rng = np.random.default_rng(11)
        series = rng.normal(0.0, 1.0, size=4096)
        tau, adev = overlapping_allan_deviation(series, 1.0)
        assert tau.size >= 10
        # first point ~ sigma of a single sample; a decade later it must have
        # dropped by roughly sqrt(10) (loose factor-2 window)
        i10 = np.argmin(np.abs(tau - 10.0))
        expected = adev[0] / np.sqrt(10.0)
        assert adev[i10] == pytest.approx(expected, rel=0.6)
        # and it decreases overall
        assert adev[i10] < adev[0]

    def test_linear_drift_grows(self):
        """A pure drift: adev grows with tau instead of averaging down."""
        series = np.linspace(0.0, 1.0, 2048)
        tau, adev = overlapping_allan_deviation(series, 1.0)
        assert adev[-1] > adev[0]

    def test_units_scale_with_dt(self):
        rng = np.random.default_rng(3)
        series = rng.standard_normal(512)
        tau_a, _ = overlapping_allan_deviation(series, 1.0)
        tau_b, _ = overlapping_allan_deviation(series, 2.5)
        np.testing.assert_allclose(tau_b, tau_a * 2.5)

    def test_short_series_returns_empty(self):
        tau, adev = overlapping_allan_deviation(np.array([1.0, 2.0, 3.0]), 1.0)
        assert tau.size == 0 and adev.size == 0

    def test_bad_dt_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            overlapping_allan_deviation(np.ones(16), -1.0)
