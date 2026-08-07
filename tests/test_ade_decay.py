"""ade_decay tools: closed-form recovery, validity domain, sigma, bootstrap."""

import numpy as np
import pytest

from scqat.tools.ade_decay import ade_bootstrap_sigma_t1, ade_gamma, ade_sigma_gamma


def _pops(gamma, t0, dt, a=0.93, b=0.05):
    """Exact three-delay populations of ``a*exp(-gamma*t)+b``."""
    t = np.array([t0, t0 + dt, t0 + 3 * dt])
    return a * np.exp(-gamma * t) + b


class TestAdeGamma:
    def test_exact_recovery_offset_and_amplitude_cancel(self):
        gamma = 1.0 / 40e-6
        dt = 30e-6
        for a, b in [(0.93, 0.05), (0.5, 0.3), (1.0, 0.0)]:
            p0, p1, p3 = _pops(gamma, t0=1e-6, dt=dt, a=a, b=b)
            got, valid = ade_gamma(p0, p1, p3, dt)
            assert bool(valid)
            assert float(got) == pytest.approx(gamma, rel=1e-9)

    def test_vectorized_over_blocks(self):
        gammas = 1.0 / np.array([20e-6, 40e-6, 80e-6])
        dt = np.full(3, 30e-6)
        p = np.array([_pops(g, 1e-6, 30e-6) for g in gammas])
        got, valid = ade_gamma(p[:, 0], p[:, 1], p[:, 2], dt)
        assert valid.all()
        np.testing.assert_allclose(got, gammas, rtol=1e-9)

    def test_invalid_domains_report_nan_not_a_number(self):
        # c <= 1: no decay resolved (P1 == P3)
        g, valid = ade_gamma(0.9, 0.5, 0.5, 30e-6)
        assert not bool(valid) and np.isnan(float(g))
        # zero denominator (P0 == P1)
        g, valid = ade_gamma(0.5, 0.5, 0.3, 30e-6)
        assert not bool(valid) and np.isnan(float(g))
        # c >= 3: gamma <= 0 (growing "decay")
        g, valid = ade_gamma(0.5, 0.6, 0.9, 30e-6)
        assert not bool(valid) and np.isnan(float(g))
        # non-positive dt
        g, valid = ade_gamma(*_pops(1.0 / 40e-6, 1e-6, 30e-6), 0.0)
        assert not bool(valid) and np.isnan(float(g))


class TestAdeSigmaGamma:
    def test_matches_numeric_propagation(self):
        """The chain-rule sigma equals brute-force finite-difference propagation."""
        gamma = 1.0 / 40e-6
        dt = 30e-6
        n_avg = 200
        p0, p1, p3 = _pops(gamma, t0=1e-6, dt=dt)
        sigma = float(ade_sigma_gamma(p0, p1, p3, dt, n_avg))
        assert np.isfinite(sigma) and sigma > 0

        eps = 1e-7
        var = 0.0
        for i, p in enumerate((p0, p1, p3)):
            args_hi = [p0, p1, p3]
            args_lo = [p0, p1, p3]
            args_hi[i] += eps
            args_lo[i] -= eps
            g_hi, _ = ade_gamma(*args_hi, dt)
            g_lo, _ = ade_gamma(*args_lo, dt)
            dg_dp = (float(g_hi) - float(g_lo)) / (2 * eps)
            var += dg_dp**2 * p * (1 - p) / n_avg
        assert sigma == pytest.approx(np.sqrt(var), rel=1e-4)

    def test_nan_on_invalid_block(self):
        sigma = ade_sigma_gamma(0.5, 0.5, 0.3, 30e-6, 100)
        assert np.isnan(float(sigma))

    def test_shrinks_with_shots(self):
        p0, p1, p3 = _pops(1.0 / 40e-6, 1e-6, 30e-6)
        s100 = float(ade_sigma_gamma(p0, p1, p3, 30e-6, 100))
        s400 = float(ade_sigma_gamma(p0, p1, p3, 30e-6, 400))
        assert s400 == pytest.approx(s100 / 2.0, rel=1e-9)


class TestAdeBootstrap:
    def _shots(self, gamma=1.0 / 40e-6, dt=30e-6, n_blocks=4, n_avg=400, seed=7):
        rng = np.random.default_rng(seed)
        p = _pops(gamma, t0=1e-6, dt=dt)
        shots = [rng.binomial(1, p[i], size=(n_blocks, n_avg)) for i in range(3)]
        return shots, np.full(n_blocks, dt)

    def test_bootstrap_sigma_agrees_with_analytic(self):
        gamma, dt, n_avg = 1.0 / 40e-6, 30e-6, 400
        (s0, s1, s3), dts = self._shots(gamma, dt, n_blocks=4, n_avg=n_avg)
        boot = ade_bootstrap_sigma_t1(s0, s1, s3, dts, n_bootstrap=300, seed=1)
        assert np.all(np.isfinite(boot))
        p0, p1, p3 = _pops(gamma, 1e-6, dt)
        sigma_t1 = float(ade_sigma_gamma(p0, p1, p3, dt, n_avg)) / gamma**2
        # same statistic measured two ways — loose factor-2 agreement
        assert np.median(boot) == pytest.approx(sigma_t1, rel=1.0)

    def test_deterministic_for_a_seed(self):
        (s0, s1, s3), dts = self._shots()
        a = ade_bootstrap_sigma_t1(s0, s1, s3, dts, 50, seed=3)
        b = ade_bootstrap_sigma_t1(s0, s1, s3, dts, 50, seed=3)
        np.testing.assert_array_equal(a, b)

    def test_no_bootstrap_returns_nan(self):
        (s0, s1, s3), dts = self._shots(n_blocks=2)
        out = ade_bootstrap_sigma_t1(s0, s1, s3, dts, 0)
        assert np.all(np.isnan(out))

    def test_shape_mismatch_raises(self):
        (s0, s1, s3), dts = self._shots(n_blocks=2)
        with pytest.raises(ValueError, match="shape"):
            ade_bootstrap_sigma_t1(s0, s1[:1], s3, dts, 10)
