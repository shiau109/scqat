"""timeseries_psd: spectrum sanity, knob validation, degraded behavior."""

import numpy as np
import pytest

from scqat.tools.timeseries_psd import (
    TIMESERIES_PSD_KNOBS,
    timeseries_psd,
    validate_timeseries_psd_kwargs,
)


class TestTimeseriesPsd:
    def test_sinusoid_peak_at_the_right_frequency(self):
        dt = 1e-3
        f0 = 40.0
        t = np.arange(8192) * dt
        rng = np.random.default_rng(5)
        series = np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(t.size)
        freq, psd = timeseries_psd(series, dt)
        assert freq.size > 0
        assert freq[np.argmax(psd)] == pytest.approx(f0, rel=0.05)

    def test_dc_and_nonpositive_bins_dropped(self):
        freq, psd = timeseries_psd(np.random.default_rng(0).standard_normal(1024), 1.0)
        assert np.all(freq > 0)
        assert np.all(psd > 0)
        assert np.all(np.isfinite(psd))

    def test_mean_is_subtracted(self):
        """A large DC offset must not leak into the spectrum floor."""
        rng = np.random.default_rng(1)
        base = rng.standard_normal(2048)
        f1, p1 = timeseries_psd(base, 1.0)
        f2, p2 = timeseries_psd(base + 1e6, 1.0)
        np.testing.assert_allclose(p1, p2, rtol=1e-9)

    def test_short_series_still_returns_arrays(self):
        freq, psd = timeseries_psd(np.array([1.0, 2.0, 1.5, 2.5]), 0.1)
        assert freq.ndim == 1 and psd.ndim == 1

    def test_bad_dt_raises(self):
        with pytest.raises(ValueError, match="dt_s"):
            timeseries_psd(np.ones(16), 0.0)
        with pytest.raises(ValueError, match="dt_s"):
            timeseries_psd(np.ones(16), float("nan"))

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty"):
            timeseries_psd(np.array([]), 1.0)

    def test_validate_rejects_unknown_knob(self):
        with pytest.raises(ValueError, match="Unknown timeseries-PSD knob"):
            validate_timeseries_psd_kwargs({"npersegg": 128})
        validate_timeseries_psd_kwargs({k: None for k in TIMESERIES_PSD_KNOBS})
