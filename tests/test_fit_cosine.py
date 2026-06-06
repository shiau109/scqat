import numpy as np
import pytest
import xarray as xr

from scqat.math_tools import get_fitter
from scqat.math_tools.fit_cosine import FitCosine


def _make_cosine(a=1.0, f=2.0, phi=0.3, c=0.5, n_points=200, noise_std=0.0, x_max=2.0):
    """Generate a synthetic cosine as an xr.DataArray with coord 'x'."""
    x = np.linspace(0.0, x_max, n_points)
    y = a * np.cos(2 * np.pi * f * x + phi) + c
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitCosine:
    """Tests for the FitCosine fitter."""

    def test_noiseless_recovery(self):
        a, f, c = 1.0, 2.0, 0.5
        da = _make_cosine(a=a, f=f, phi=0.3, c=c)
        result = FitCosine(da).fit()

        assert result.success
        assert result.params['a'].value == pytest.approx(a, rel=0.02)
        assert result.params['f'].value == pytest.approx(f, rel=0.02)
        assert result.params['c'].value == pytest.approx(c, abs=0.02)

    def test_noisy_recovery(self):
        a, f, c = 0.8, 1.5, 0.2
        da = _make_cosine(a=a, f=f, phi=-0.4, c=c, n_points=400, noise_std=0.02)
        result = FitCosine(da).fit()

        assert result.success
        assert result.params['f'].value == pytest.approx(f, rel=0.05)
        assert result.params['a'].value == pytest.approx(a, rel=0.15)

    def test_registered_in_factory(self):
        fitter = get_fitter('cosine', _make_cosine())
        assert isinstance(fitter, FitCosine)

    def test_accepts_raw_array(self):
        fitter = FitCosine(np.array([1.0, 0.0, -1.0]))
        assert np.allclose(fitter.x, [0, 1, 2])
        assert np.allclose(fitter.y, [1.0, 0.0, -1.0])

    def test_accepts_raw_xy(self):
        x = np.linspace(0.0, 1.0, 3)
        fitter = FitCosine(np.array([1.0, 0.0, -1.0]), x=x)
        assert np.allclose(fitter.x, x)
