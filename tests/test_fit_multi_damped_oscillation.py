import numpy as np
import xarray as xr
import pytest

from scqat.tools.fit_multi_damped_oscillation import (
    FitMultiDampedOscillation,
    multi_damped_osc_eval,
)


def _make_signal(t, modes, c=0.0, noise=0.0, seed=0):
    y = multi_damped_osc_eval(
        t,
        [{"a": a, "k": k, "f": f, "phi": phi} for (a, k, f, phi) in modes],
        c=c,
    )
    if noise > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise, size=y.shape)
    return y


def _hankel_seed(modes_truth, jitter=0.05, rng_seed=1):
    """Return a Hankel-like seed list close to the truth."""
    rng = np.random.default_rng(rng_seed)
    seeds = []
    for (a, k, f, phi) in modes_truth:
        seeds.append({
            "amplitude": a * (1 + jitter * rng.standard_normal()),
            "decay_rate": k * (1 + jitter * rng.standard_normal()),
            "freq_hz": f * (1 + jitter * rng.standard_normal()),
            "phase": phi + jitter * rng.standard_normal(),
        })
    return seeds


class TestFitMultiDampedOscillation:
    def test_two_mode_recovery(self):
        t = np.linspace(0.0, 4.0, 400)
        truth = [
            (0.6, -0.4, 1.0, 0.2),
            (0.3, -0.8, 2.5, -0.5),
        ]
        c_true = 0.05
        y = _make_signal(t, truth, c=c_true, noise=1e-3, seed=0)
        da = xr.DataArray(y, coords={"x": t}, dims="x")

        seeds = _hankel_seed(truth, jitter=0.05)
        fitter = FitMultiDampedOscillation(da, modes=seeds)
        fitter.guess()
        result = fitter.fit()

        assert result.success
        modes_fit = fitter.unpack_modes(result)

        # Sort by frequency for stable comparison
        modes_fit.sort(key=lambda m: m["f"])
        truth_sorted = sorted(truth, key=lambda m: m[2])

        for fit_m, (a, k, f, phi) in zip(modes_fit, truth_sorted):
            assert fit_m["a"] == pytest.approx(a, rel=0.05, abs=0.02)
            assert fit_m["k"] == pytest.approx(k, rel=0.1, abs=0.05)
            assert fit_m["f"] == pytest.approx(f, rel=0.02, abs=0.02)
        assert float(result.params["c"].value) == pytest.approx(c_true, abs=0.02)

    def test_requires_modes(self):
        t = np.linspace(0.0, 1.0, 50)
        da = xr.DataArray(np.zeros_like(t), coords={"x": t}, dims="x")
        with pytest.raises(ValueError):
            FitMultiDampedOscillation(da, modes=[])
