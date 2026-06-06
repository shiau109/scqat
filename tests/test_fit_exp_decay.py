import numpy as np
import pytest
import xarray as xr

from scqat.math_tools import get_fitter
from scqat.math_tools.fit_exp_decay import FitExponentialDecay


def _make_decay(a=1.0, tau=0.5, c=0.1, n_points=200, noise_std=0.0, x_max=2.0):
    """Generate a synthetic exponential decay as an xr.DataArray with coord 'x'."""
    x = np.linspace(0.0, x_max, n_points)
    y = a * np.exp(-x / tau) + c
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitExponentialDecay:
    """Tests for the FitExponentialDecay fitter."""

    def test_noiseless_recovery(self):
        """Fit noiseless data and check that parameters are recovered."""
        a, tau, c = 1.0, 0.5, 0.1
        da = _make_decay(a=a, tau=tau, c=c)
        result = FitExponentialDecay(da).fit()

        assert result.success
        assert result.params['a'].value == pytest.approx(a, rel=0.02)
        assert result.params['tau'].value == pytest.approx(tau, rel=0.02)
        assert result.params['c'].value == pytest.approx(c, abs=0.02)

    def test_noisy_recovery(self):
        """Fit noisy data and check the decay constant is close."""
        a, tau, c = 0.8, 0.7, 0.2
        da = _make_decay(a=a, tau=tau, c=c, n_points=400, noise_std=0.01)
        result = FitExponentialDecay(da).fit()

        assert result.success
        assert result.params['tau'].value == pytest.approx(tau, rel=0.15)
        assert result.params['c'].value == pytest.approx(c, abs=0.05)

    def test_registered_in_factory(self):
        """The fitter is discoverable via the get_fitter factory."""
        da = _make_decay()
        fitter = get_fitter('exp_decay', da)
        assert isinstance(fitter, FitExponentialDecay)

    def test_accepts_raw_array(self):
        """A bare y array is accepted; x defaults to the sample index."""
        fitter = FitExponentialDecay(np.array([3.0, 2.0, 1.0]))
        assert np.allclose(fitter.x, [0, 1, 2])
        assert np.allclose(fitter.y, [3.0, 2.0, 1.0])

    def test_accepts_raw_xy(self):
        """Raw (x, y) arrays are accepted without wrapping in a DataArray."""
        x = np.linspace(0.0, 1.0, 3)
        fitter = FitExponentialDecay(np.array([3.0, 2.0, 1.0]), x=x)
        assert np.allclose(fitter.x, x)
