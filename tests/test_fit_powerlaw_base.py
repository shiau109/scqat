import numpy as np
import pytest
import xarray as xr

from scqat.math_tools import get_fitter
from scqat.math_tools.fit_powerlaw_base import FitBasePowerLaw


def _make_powerlaw(a=0.5, base=0.9, c=0.5, n_points=60, noise_std=0.0):
    """Generate a synthetic base power-law (e.g. RB decay) as a DataArray."""
    x = np.arange(n_points, dtype=float)
    y = a * (base ** x) + c
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitBasePowerLaw:
    """Tests for the FitBasePowerLaw fitter."""

    def test_noiseless_recovery(self):
        a, base, c = 0.5, 0.9, 0.5
        da = _make_powerlaw(a=a, base=base, c=c)
        result = FitBasePowerLaw(da).fit()

        assert result.success
        assert result.params['base'].value == pytest.approx(base, rel=0.01)
        assert result.params['a'].value == pytest.approx(a, rel=0.05)
        assert result.params['c'].value == pytest.approx(c, abs=0.02)

    def test_noisy_recovery(self):
        a, base, c = 0.45, 0.92, 0.5
        da = _make_powerlaw(a=a, base=base, c=c, n_points=80, noise_std=0.005)
        result = FitBasePowerLaw(da).fit()

        assert result.success
        assert result.params['base'].value == pytest.approx(base, rel=0.03)

    def test_registered_in_factory(self):
        fitter = get_fitter('powerlaw_base', _make_powerlaw())
        assert isinstance(fitter, FitBasePowerLaw)

    def test_accepts_raw_array(self):
        fitter = FitBasePowerLaw(np.array([1.0, 0.9, 0.81]))
        assert np.allclose(fitter.x, [0, 1, 2])
        assert np.allclose(fitter.y, [1.0, 0.9, 0.81])
