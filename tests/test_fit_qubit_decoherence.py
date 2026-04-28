import numpy as np
import pytest
import xarray as xr

from scqat.math_tools.fit_qubit_decoherence import (
    FitQubitDecoherence,
    decoherence_G,
    rho11_model,
    rho10_model,
)
from scqat.math_tools import get_fitter


def _time_axis(n=120, t_max=8.0):
    return np.linspace(0.0, t_max, n)


class TestFitQubitDecoherence:

    def test_rho11_overdamped_recovery(self):
        # Overdamped: Lambda > 2*Gamma
        Gamma, Lambda, rho0 = 0.3, 1.0, 0.95
        t = _time_axis()
        y = rho11_model(t, Gamma, Lambda, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.1)
        assert result.params['Lambda'].value == pytest.approx(Lambda, rel=0.1)
        assert result.params['rho_0'].value == pytest.approx(rho0, rel=0.05)

    def test_rho10_recovery(self):
        Gamma, Lambda, rho0 = 0.4, 1.5, 0.5
        t = _time_axis()
        y = rho10_model(t, Gamma, Lambda, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_10').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.1)
        assert result.params['Lambda'].value == pytest.approx(Lambda, rel=0.1)

    def test_noisy_rho11(self):
        Gamma, Lambda, rho0 = 0.5, 2.0, 1.0
        t = _time_axis(n=200, t_max=6.0)
        rng = np.random.default_rng(0)
        y = rho11_model(t, Gamma, Lambda, rho0) + rng.normal(0, 0.01, size=t.shape)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.25)

    def test_factory_registration(self):
        t = _time_axis()
        y = rho11_model(t, 0.3, 1.0, 1.0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        fitter = get_fitter('qubit_decoherence', data=da, component='rho_11')
        assert isinstance(fitter, FitQubitDecoherence)

    def test_invalid_component(self):
        t = _time_axis()
        da = xr.DataArray(np.zeros_like(t), coords={'x': t}, dims='x')
        with pytest.raises(ValueError, match="component"):
            FitQubitDecoherence(da, component='bad')

    def test_decoherence_G_zero_time(self):
        assert decoherence_G(0.0, 0.5, 1.0) == pytest.approx(1.0)

    def test_rho11_underdamped_recovery(self):
        # Deep underdamped (Gamma >> Lambda/2): Lambda(Lambda - 2 Gamma) < 0.
        # Previously the guess was overdamped-only and the fit failed.
        Gamma, Lambda, rho0 = 2.0, 0.10, 1.0
        t = np.linspace(0.0, 10.0, 200)
        y = rho11_model(t, Gamma, Lambda, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.1)
        assert result.params['Lambda'].value == pytest.approx(Lambda, rel=0.2)
        assert result.params['rho_0'].value == pytest.approx(rho0, rel=0.05)

    def test_rho11_underdamped_noisy(self):
        Gamma, Lambda, rho0 = 2.0, 0.10, 1.0
        t = np.linspace(0.0, 10.0, 200)
        rng = np.random.default_rng(42)
        y = rho11_model(t, Gamma, Lambda, rho0) + rng.normal(0, 0.02, size=t.shape)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.25)
        assert result.params['Lambda'].value == pytest.approx(Lambda, abs=0.1)

    def test_rho11_critical_recovery(self):
        # Critical damping: Lambda = 2*Gamma, d^2 = 0.
        Gamma, Lambda, rho0 = 0.5, 1.0, 1.0
        t = np.linspace(0.0, 10.0, 200)
        y = rho11_model(t, Gamma, Lambda, rho0)
        da = xr.DataArray(y, coords={'x': t}, dims='x')
        result = FitQubitDecoherence(da, component='rho_11').fit()
        assert result.success
        assert result.params['Gamma'].value == pytest.approx(Gamma, rel=0.15)
        assert result.params['Lambda'].value == pytest.approx(Lambda, rel=0.15)
