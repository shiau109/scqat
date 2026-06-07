import numpy as np
import pytest
import xarray as xr

from scqat.tools.fit_qubit_decoherence import (
    FitQubitDecoherence,
    decoherence_G,
    rho11_model,
    rho10_model,
)
from scqat.tools import get_fitter


# Reparametrisation:
#     gamma   = 2 * Lambda
#     lambda_ = sqrt(Gamma * Lambda / 2)
# Critical damping: gamma = 4 * lambda_ (i.e. Lambda = 2*Gamma).


def _time_axis(n=120, t_max=8.0):
    return np.linspace(0.0, t_max, n)


class TestFitQubitDecoherence:

    def test_rho11_overdamped_recovery(self):
        # Overdamped: gamma > 4*lambda_  (was Lambda > 2*Gamma).
        # Old (Gamma=0.3, Lambda=1.0) -> (gamma=2.0, lambda_=sqrt(0.15)).
        gamma, lambda_, rho0 = 2.0, np.sqrt(0.15), 0.95
        t = _time_axis()
        y = rho11_model(t, gamma, lambda_, 0.0, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.1)
        assert result.params['lambda_'].value == pytest.approx(lambda_, rel=0.1)
        assert result.params['rho_0'].value == pytest.approx(rho0, rel=0.05)

    def test_rho10_recovery(self):
        # Old (Gamma=0.4, Lambda=1.5) -> (gamma=3.0, lambda_=sqrt(0.3)).
        gamma, lambda_, rho0 = 3.0, np.sqrt(0.3), 0.5
        t = _time_axis()
        y = rho10_model(t, gamma, lambda_, 0.0, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_10').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.1)
        assert result.params['lambda_'].value == pytest.approx(lambda_, rel=0.1)

    def test_noisy_rho11(self):
        # Old (Gamma=0.5, Lambda=2.0) -> (gamma=4.0, lambda_=sqrt(0.5)).
        gamma, lambda_, rho0 = 4.0, np.sqrt(0.5), 1.0
        t = _time_axis(n=200, t_max=6.0)
        rng = np.random.default_rng(0)
        y = rho11_model(t, gamma, lambda_, 0.0, rho0) + rng.normal(0, 0.01, size=t.shape)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.25)

    def test_factory_registration(self):
        t = _time_axis()
        y = rho11_model(t, 2.0, np.sqrt(0.15), 0.0, 1.0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        fitter = get_fitter('qubit_decoherence', data=da, component='rho_11')
        assert isinstance(fitter, FitQubitDecoherence)

    def test_invalid_component(self):
        t = _time_axis()
        da = xr.DataArray(np.zeros_like(t), coords={'x': t}, dims='x')
        with pytest.raises(ValueError, match="component"):
            FitQubitDecoherence(da, component='bad')

    def test_decoherence_G_zero_time(self):
        # G(0) = 1 regardless of (gamma, lambda_, Delta).
        assert decoherence_G(0.0, 2.0, np.sqrt(0.25), 0.0) == pytest.approx(1.0)

    def test_rho11_underdamped_recovery(self):
        # Underdamped: gamma < 4*lambda_.
        # Old (Gamma=2.0, Lambda=0.10) -> (gamma=0.20, lambda_=sqrt(0.1)).
        gamma, lambda_, rho0 = 0.20, np.sqrt(0.1), 1.0
        t = np.linspace(0.0, 10.0, 200)
        y = rho11_model(t, gamma, lambda_, 0.0, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.2)
        assert result.params['lambda_'].value == pytest.approx(lambda_, rel=0.2)
        assert result.params['rho_0'].value == pytest.approx(rho0, rel=0.05)

    def test_rho11_underdamped_noisy(self):
        gamma, lambda_, rho0 = 0.20, np.sqrt(0.1), 1.0
        t = np.linspace(0.0, 10.0, 200)
        rng = np.random.default_rng(42)
        y = rho11_model(t, gamma, lambda_, 0.0, rho0) + rng.normal(0, 0.02, size=t.shape)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.3)
        assert result.params['lambda_'].value == pytest.approx(lambda_, abs=0.1)

    def test_rho11_critical_recovery(self):
        # Critical damping: gamma = 4*lambda_ (was Lambda = 2*Gamma).
        # Old (Gamma=0.5, Lambda=1.0) -> (gamma=2.0, lambda_=0.5).
        gamma, lambda_, rho0 = 2.0, 0.5, 1.0
        t = np.linspace(0.0, 10.0, 200)
        y = rho11_model(t, gamma, lambda_, 0.0, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.15)
        assert result.params['lambda_'].value == pytest.approx(lambda_, rel=0.15)
