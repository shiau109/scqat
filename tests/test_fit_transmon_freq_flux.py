import numpy as np
import pytest
import xarray as xr

from scqat.math_tools import get_fitter
from scqat.math_tools.fit_transmon_freq_flux import FitTransmonFrequencyFlux


def _make_transmon(Ej_sum=20.0, period=0.8, offset=0.0, Ec=0.2, n_points=41,
                   x_range=0.35, noise_std=0.0):
    """Generate symmetric-transmon frequency vs flux (d=0)."""
    x = np.linspace(-x_range, x_range, n_points)
    q = (x - offset) / period
    ej_eff = Ej_sum * np.abs(np.cos(np.pi * q))
    y = np.sqrt(8 * Ec * ej_eff) - Ec
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitTransmonFrequencyFlux:
    """Tests for the FitTransmonFrequencyFlux fitter."""

    def test_noiseless_recovery(self):
        Ej_sum, period = 20.0, 0.8
        da = _make_transmon(Ej_sum=Ej_sum, period=period, offset=0.0)
        result = FitTransmonFrequencyFlux(da).fit()

        assert result.success
        assert result.params['Ej_sum'].value == pytest.approx(Ej_sum, rel=0.05)
        assert result.params['period'].value == pytest.approx(period, rel=0.1)
        # Ec is fixed by default
        assert result.params['Ec'].vary is False
        assert result.params['Ec'].value == pytest.approx(0.2)

    def test_fit_reproduces_data(self):
        da = _make_transmon(Ej_sum=18.0, period=0.7, offset=0.03, noise_std=0.005)
        result = FitTransmonFrequencyFlux(da).fit()
        assert result.success
        # the fitted curve should track the data closely
        assert np.sqrt(np.mean((result.best_fit - da.values) ** 2)) < 0.05

    def test_registered_in_factory(self):
        fitter = get_fitter('transmon_freq_flux', _make_transmon())
        assert isinstance(fitter, FitTransmonFrequencyFlux)
