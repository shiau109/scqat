import numpy as np
import pytest
import xarray as xr

from scqat.math_tools.fit_abscos import FitAbsCos


def _make_abscos_data(amplitude=0.5, frequency=2.0, phase=0.1, n_points=100, noise_std=0.0):
    """Generate a synthetic |cos| signal as an xr.DataArray with coord 'x'."""
    x = np.linspace(-1, 1, n_points)
    y = amplitude * np.abs(np.cos(2 * np.pi * frequency * (x - phase)))
    if noise_std > 0:
        rng = np.random.default_rng(42)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitAbsCos:
    """Tests for the FitAbsCos fitter."""

    def test_noiseless_recovery(self):
        """Fit noiseless data and check that parameters are recovered."""
        amp, freq, phase = 0.5, 2.0, 0.1
        da = _make_abscos_data(amplitude=amp, frequency=freq, phase=phase)
        fitter = FitAbsCos(da)
        result = fitter.fit()

        assert result.success
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.05)
        assert result.params['frequency'].value == pytest.approx(freq, rel=0.05)
        # |cos| is periodic with half-period 1/(2*freq); phase is only defined modulo that
        half_period = 1.0 / (2.0 * freq)
        phase_diff = (result.params['phase'].value - phase) % half_period
        assert min(phase_diff, half_period - phase_diff) == pytest.approx(0.0, abs=0.05)

    def test_noisy_recovery(self):
        """Fit noisy data and check parameters are close."""
        amp, freq, phase = 0.8, 1.5, -0.2
        da = _make_abscos_data(amplitude=amp, frequency=freq, phase=phase,
                               n_points=200, noise_std=0.02)
        fitter = FitAbsCos(da)
        result = fitter.fit()

        assert result.success
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.15)
        assert result.params['frequency'].value == pytest.approx(freq, rel=0.15)

    def test_rejects_non_dataarray(self):
        """Should raise ValueError if input is not a DataArray."""
        with pytest.raises(ValueError, match="xarray.DataArray"):
            FitAbsCos(np.array([1, 2, 3]))
