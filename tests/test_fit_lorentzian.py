import numpy as np
import pytest
import xarray as xr

from scqat.math_tools.fit_lorentzian import FitLorentzian, lorentzian
from scqat.math_tools import get_fitter


def _make_lorentzian_data(x0=0.0, amplitude=1.0, gamma=0.1, offset=0.05,
                          n_points=201, x_lo=-1.0, x_hi=1.0, noise_std=0.0):
    x = np.linspace(x_lo, x_hi, n_points)
    y = lorentzian(x, x0, amplitude, gamma, offset)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitLorentzian:

    def test_noiseless_peak(self):
        x0, amp, gamma, off = 0.2, 1.5, 0.08, 0.1
        da = _make_lorentzian_data(x0=x0, amplitude=amp, gamma=gamma, offset=off)
        result = FitLorentzian(da).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=0.01)
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.05)
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.1)
        assert result.params['offset'].value == pytest.approx(off, abs=0.05)

    def test_noiseless_dip(self):
        x0, amp, gamma, off = -0.3, -0.8, 0.05, 0.5
        da = _make_lorentzian_data(x0=x0, amplitude=amp, gamma=gamma, offset=off)
        result = FitLorentzian(da, inverted=True).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=0.01)
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.1)

    def test_noisy_recovery(self):
        x0, amp, gamma, off = 0.0, 1.0, 0.1, 0.0
        da = _make_lorentzian_data(x0=x0, amplitude=amp, gamma=gamma, offset=off,
                                   n_points=400, noise_std=0.02)
        result = FitLorentzian(da).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=0.02)
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.2)

    def test_factory_registration(self):
        da = _make_lorentzian_data()
        fitter = get_fitter('lorentzian', data=da)
        assert isinstance(fitter, FitLorentzian)

    def test_rejects_non_dataarray(self):
        with pytest.raises(ValueError, match="xarray.DataArray"):
            FitLorentzian(np.array([1, 2, 3]))
