import numpy as np
import pytest
import xarray as xr

from scqat.tools.fit_notch_circle import FitNotchCircle, notch_s21
from scqat.tools import get_fitter


def _make_notch_data(fr=7.2e9, Ql=8000.0, absQc=10000.0, phi0=0.25,
                     a=0.9, alpha=0.6, delay=42e-9,
                     n_points=401, span_fwhm=10.0, noise_std=0.0, seed=42):
    """Synthetic notch S21 over ``span_fwhm`` linewidths around fr."""
    span = span_fwhm * fr / Ql
    f = np.linspace(fr - span / 2, fr + span / 2, n_points)
    z = notch_s21(f, fr, Ql, absQc, phi0, a=a, alpha=alpha, delay=delay)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        z = z + rng.normal(0, noise_std, n_points) + 1j * rng.normal(0, noise_std, n_points)
    return xr.DataArray(z, coords={'x': f}, dims='x')


class TestFitNotchCircle:

    def test_ideal_no_environment(self):
        """a=1, alpha=0, delay=0: pure notch model must be recovered tightly."""
        fr, Ql, absQc, phi0 = 7.2e9, 8000.0, 10000.0, 0.25
        da = _make_notch_data(fr=fr, Ql=Ql, absQc=absQc, phi0=phi0,
                              a=1.0, alpha=0.0, delay=0.0)
        res = FitNotchCircle(da).fit()
        assert res["success"]
        assert res["fr"] == pytest.approx(fr, abs=1e3)
        assert res["Ql"] == pytest.approx(Ql, rel=0.02)
        assert res["absQc"] == pytest.approx(absQc, rel=0.02)
        assert res["phi0"] == pytest.approx(phi0, abs=0.02)

    def test_full_environment_with_noise(self):
        """Amplitude, phase offset, cable delay and noise together."""
        fr, Ql, absQc, phi0 = 7.2e9, 8000.0, 10000.0, 0.25
        a, delay = 0.9, 42e-9
        da = _make_notch_data(fr=fr, Ql=Ql, absQc=absQc, phi0=phi0,
                              a=a, alpha=0.6, delay=delay, noise_std=0.009)
        res = FitNotchCircle(da).fit()
        assert res["success"]
        assert res["fr"] == pytest.approx(fr, abs=5e4)      # ~5% of FWHM
        assert res["Ql"] == pytest.approx(Ql, rel=0.1)
        assert res["absQc"] == pytest.approx(absQc, rel=0.15)
        assert res["phi0"] == pytest.approx(phi0, abs=0.15)
        assert res["a"] == pytest.approx(a, rel=0.05)
        assert res["delay"] == pytest.approx(delay, rel=0.1)
        # derived physics
        qi_true = 1.0 / (1.0 / Ql - np.cos(phi0) / absQc)
        assert res["Qi_dia_corr"] == pytest.approx(qi_true, rel=0.2)
        # errors are finite and sane
        assert np.isfinite(res["fr_err"]) and res["fr_err"] > 0
        assert np.isfinite(res["Ql_err"]) and res["Ql_err"] > 0

    def test_large_delay_low_q_noisy(self):
        """Regression for real hardware (5Q4C): a large cable delay (many phase
        wraps) over a narrow span with a low-Q resonance in noise. Here the
        circle-tightness delay criterion finds the WRONG global minimum; the
        full-model residual must select the phase-slope delay instead."""
        fr, Ql, absQc, phi0 = 5.94e9, 2500.0, 3000.0, 0.15
        delay = 390e-9
        f = np.linspace(fr - 10e6, fr + 10e6, 501)  # 20 MHz span, ~8 wraps
        z = notch_s21(f, fr, Ql, absQc, phi0, a=0.9, alpha=0.5, delay=delay)
        rng = np.random.default_rng(4)
        scale = np.median(np.abs(z))
        z = z + (rng.normal(0, 0.03 * scale, f.size)
                 + 1j * rng.normal(0, 0.03 * scale, f.size))
        res = FitNotchCircle(z, x=f).fit()
        assert res["success"]
        assert res["fr"] == pytest.approx(fr, abs=2e5)     # within ~1 linewidth
        assert res["Ql"] == pytest.approx(Ql, rel=0.2)
        assert res["delay"] == pytest.approx(delay, rel=0.05)

    def test_phaseless_line_is_not_success(self):
        """A magnitude-only sweep (Q = noise) collapses onto a line in the
        complex plane: no resonance circle, so no Qi/Qc — must not be success."""
        n = 301
        det = np.linspace(-5e6, 5e6, n)
        mag = 1.0 - 0.8 / (1.0 + (det / 0.7e6) ** 2)
        rng = np.random.default_rng(2)
        z = (mag + rng.normal(0, 0.01, n)) + 1j * rng.normal(0, 0.01, n)
        res = FitNotchCircle(z, x=det + 6e9).fit()
        assert res["success"] is False
        assert res["circularity"] < 0.3

    def test_negative_mismatch_angle(self):
        da = _make_notch_data(phi0=-0.35, noise_std=0.005)
        res = FitNotchCircle(da).fit()
        assert res["success"]
        assert res["phi0"] == pytest.approx(-0.35, abs=0.1)

    def test_fixed_delay(self):
        """A user-supplied delay is honoured, not refitted."""
        delay = 42e-9
        da = _make_notch_data(delay=delay)
        res = FitNotchCircle(da, delay=delay).fit()
        assert res["success"]
        assert res["delay"] == pytest.approx(delay)
        assert res["fr"] == pytest.approx(7.2e9, abs=1e4)

    def test_kappa_from_ql(self):
        """fwhm = fr/Ql is the total linewidth the estimator will report."""
        fr, Ql = 7.2e9, 8000.0
        da = _make_notch_data(fr=fr, Ql=Ql, noise_std=0.005)
        res = FitNotchCircle(da).fit()
        assert res["fr"] / res["Ql"] == pytest.approx(fr / Ql, rel=0.1)

    def test_fit_curve_matches_data(self):
        """z_fit reproduces the raw data (environment included)."""
        da = _make_notch_data()
        res = FitNotchCircle(da).fit()
        rms = np.sqrt(np.mean(np.abs(res["z_fit"] - da.values) ** 2))
        assert rms < 0.02  # noiseless data: near-exact reproduction

    def test_factory_registration(self):
        da = _make_notch_data()
        fitter = get_fitter('notch_circle', data=da)
        assert isinstance(fitter, FitNotchCircle)

    def test_accepts_raw_arrays(self):
        f = np.linspace(7.19e9, 7.21e9, 11)
        z = notch_s21(f, 7.2e9, 8000.0, 10000.0, 0.1)
        fitter = FitNotchCircle(z, x=f)
        assert np.allclose(fitter.x, f)
        assert np.allclose(fitter.z, z)
        assert fitter.x.dtype == np.float64


def test_descending_frequency_axis_fits_the_same_resonator():
    """Regression: the delay scan's bracket and the Lorentzian seed's width bound
    were built from ``f[-1] - f[0]``; an axis swept high -> low raised
    ``The lower bound exceeds the upper bound``. The same points in the other
    order must give the same resonator."""
    da = _make_notch_data(noise_std=0.003)
    up = FitNotchCircle(da).fit()
    down = FitNotchCircle(da.isel(x=slice(None, None, -1))).fit()
    assert up["success"] and down["success"]
    for key in ("fr", "Ql", "absQc", "phi0", "delay"):
        assert down[key] == pytest.approx(up[key], rel=1e-6)
