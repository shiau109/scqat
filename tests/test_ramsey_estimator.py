"""Tests for the RamseyEstimator three-way model selection.

The lab sequence is x90 -> idle -> y90, so the fringe is a damped **sine** seeded at
phase 0. The estimator picks the model with a frequency gate then a BIC comparison:

  * a single damped sine (the common case),
  * a two-frequency beat (charge dispersion) — mean of the two frequencies calibrates
    the qubit,
  * a pure exponential decay (relaxation) when the fringe frequency is ~0, reported with
    ``f_1 == 0``.

These tests synthesise each case over an ``idle_time`` coordinate and check the selected
``model_type``, the recovered frequency / T2*, the ``force_model`` override, and that
``analyze`` round-trips its metadata + plot-data artifacts.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import RamseyEstimator
from scqat.estimators.ramsey import RamseyEstimator as SubpkgEstimator


def _idle(n=201, t_max=2000.0):
    return np.linspace(0.0, t_max, n)


def _make_single(f=0.0025, tau=1000.0, a=0.4, c=0.5, phi=0.0, noise=2e-3, seed=0,
                 n=201, t_max=2000.0):
    """Single damped sine: ~5 oscillations across the idle window."""
    t = _idle(n, t_max)
    rng = np.random.default_rng(seed)
    signal = c + a * np.exp(-t / tau) * np.sin(2 * np.pi * f * t + phi)
    signal = signal + noise * rng.standard_normal(n)
    return xr.Dataset({"signal": ("idle_time", signal)}, coords={"idle_time": t})


def _make_beat(f1=0.002, f2=0.004, tau=1200.0, a1=0.25, a2=0.25, c=0.5, noise=2e-3,
               seed=1, n=201, t_max=2000.0):
    """Two well-separated damped sines (charge dispersion)."""
    t = _idle(n, t_max)
    rng = np.random.default_rng(seed)
    signal = (
        c
        + a1 * np.exp(-t / tau) * np.sin(2 * np.pi * f1 * t)
        + a2 * np.exp(-t / tau) * np.sin(2 * np.pi * f2 * t)
    )
    signal = signal + noise * rng.standard_normal(n)
    return xr.Dataset({"signal": ("idle_time", signal)}, coords={"idle_time": t})


def _make_relaxation(tau=600.0, a=0.4, c=0.5, noise=2e-3, seed=2, n=201, t_max=2000.0):
    """Pure exponential decay — no resolvable fringe frequency."""
    t = _idle(n, t_max)
    rng = np.random.default_rng(seed)
    signal = c + a * np.exp(-t / tau) + noise * rng.standard_normal(n)
    return xr.Dataset({"signal": ("idle_time", signal)}, coords={"idle_time": t})


class TestRamseyEstimator:

    def test_imports_match(self):
        assert RamseyEstimator is SubpkgEstimator
        assert RamseyEstimator.estimator_name == "ramsey"

    def test_single_model(self):
        ds = _make_single(f=0.0025, tau=1000.0)
        res = RamseyEstimator().extract_parameters(ds)
        assert res["model_type"] == "single"
        assert res["success"] is True
        assert res["f_1"] == pytest.approx(0.0025, rel=0.1)
        assert res["tau_1"] == pytest.approx(1000.0, rel=0.25)
        assert "a_2" not in res  # single model carries no second component

    def test_beat_model(self):
        ds = _make_beat(f1=0.002, f2=0.004)
        res = RamseyEstimator().extract_parameters(ds)
        assert res["model_type"] == "beat"
        freqs = sorted([res["f_1"], res["f_2"]])
        assert freqs[0] == pytest.approx(0.002, rel=0.15)
        assert freqs[1] == pytest.approx(0.004, rel=0.15)
        # The mean of the two frequencies is what calibrates the qubit.
        assert 0.5 * (res["f_1"] + res["f_2"]) == pytest.approx(0.003, rel=0.1)

    def test_relaxation_model(self):
        ds = _make_relaxation(tau=600.0)
        res = RamseyEstimator().extract_parameters(ds)
        assert res["model_type"] == "relaxation"
        assert res["f_1"] == 0.0
        assert res["tau_1"] == pytest.approx(600.0, rel=0.25)

    def test_force_model_overrides_selection(self):
        single_ds = _make_single()
        relax_ds = _make_relaxation()

        # Force relaxation on oscillating data -> frequency treated as 0.
        forced_relax = RamseyEstimator().extract_parameters(single_ds, force_model="relaxation")
        assert forced_relax["model_type"] == "relaxation"
        assert forced_relax["f_1"] == 0.0

        # Force single on decay-only data -> a single damped-sine fit is attempted.
        forced_single = RamseyEstimator().extract_parameters(relax_ds, force_model="single")
        assert forced_single["model_type"] == "single"

        # Force beat -> a two-frequency result with both components present.
        forced_beat = RamseyEstimator().extract_parameters(single_ds, force_model="beat")
        assert forced_beat["model_type"] == "beat"
        assert "f_2" in forced_beat

    def test_force_model_rejects_unknown(self):
        with pytest.raises(ValueError):
            RamseyEstimator().extract_parameters(_make_single(), force_model="bogus")

    def test_check_data_requires_signal_and_coord(self):
        est = RamseyEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"I": ("idle_time", [0, 1])},
                                       coords={"idle_time": [0, 1]}))
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"signal": ("x", [0, 1])}, coords={"x": [0, 1]}))

    def test_metadata_drops_arrays(self):
        ds = _make_single()
        est = RamseyEstimator()
        res = est.extract_parameters(ds)
        meta = est.extract_metadata(res)
        for k in ("best_fit", "fft_freq", "fft_amp", "fit_report"):
            assert k not in meta
        assert {"model_type", "f_1", "tau_1", "success"} <= set(meta)

    def test_plot_data_layout(self):
        ds = _make_single()
        est = RamseyEstimator()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        assert "signal" in pd and "best_fit" in pd and "fft_amp" in pd
        assert pd["signal"].dims == ("idle_time",)
        assert pd["fft_amp"].dims == ("fft_freq",)
        assert pd.attrs["model_type"] == "single"

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_single()
        est = RamseyEstimator()
        res, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "ramsey_metadata.json").exists()
        assert (tmp_path / "ramsey_plotdata.nc").exists()
        assert set(figs) == {"time_domain", "fft_spectrum"}
        assert isinstance(figs["time_domain"], plt.Figure)
        assert isinstance(figs["fft_spectrum"], plt.Figure)
        plt.close("all")
