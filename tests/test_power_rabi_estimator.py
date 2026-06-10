"""Tests for the PowerRabiEstimator cosine fit / pi-amplitude extraction.

Synthesises a single-pulse power-Rabi amplitude sweep and checks the estimator
recovers the pi-pulse amplitude prefactor, exposes the ``opt_amp_prefactor`` /
``success`` fields the LCHQMDriver node relies on, and that ``analyze`` round-trips
its metadata + plot-data artifacts.

Convention: ``signal`` is high at zero amplitude (ground-state readout) and dips at
the pi pulse — i.e. ``0.5 + 0.5*cos(pi * x / factor_pi)`` — matching the reused
qualibration_libs derivation ``opt = (pi - phase) / (2*pi*f)`` (first minimum of the
fitted cosine). The pi pulse then sits at ``x = factor_pi``.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import PowerRabiEstimator
from scqat.estimators.power_rabi import PowerRabiEstimator as SubpkgEstimator


def _make_ds(factor_pi=0.8, n=200, amp_min=0.0, amp_max=1.99, noise=2e-3, seed=0, rising=False):
    """Power-Rabi sweep with a pi pulse at ``x = factor_pi``.

    ``rising=False`` is the ground-readout convention (signal high at zero amplitude,
    dipping at the pi pulse); ``rising=True`` is the opposite sign (signal low at zero,
    peaking at the pi pulse) — the case that traps a fixed phi=0 / a>=0 fit at a~0.
    """
    amp_prefactor = np.linspace(amp_min, amp_max, n)
    sign = -1.0 if rising else 1.0
    signal = 0.5 + sign * 0.5 * np.cos(np.pi * amp_prefactor / factor_pi)
    rng = np.random.default_rng(seed)
    signal = signal + noise * rng.standard_normal(n)
    return xr.Dataset(
        {"signal": ("amp_prefactor", signal)},
        coords={"amp_prefactor": amp_prefactor},
    )


class TestPowerRabiEstimator:

    def test_imports_match(self):
        assert PowerRabiEstimator is SubpkgEstimator
        assert PowerRabiEstimator.estimator_name == "power_rabi"

    @pytest.mark.parametrize("rising", [False, True])
    def test_recovers_pi_prefactor(self, rising):
        # Both readout signs must recover the pi prefactor: 'rising' is the case that
        # traps a fixed phi=0 / a>=0 cosine fit at a flat (a~0) line.
        factor_pi = 0.8
        ds = _make_ds(factor_pi=factor_pi, rising=rising)
        results = PowerRabiEstimator().extract_parameters(ds)
        assert results["success"] is True
        assert results["opt_amp_prefactor"] == pytest.approx(factor_pi, rel=0.05)
        # f corresponds to a half-period at the pi pulse: f = 1 / (2 * factor_pi).
        assert results["f"] == pytest.approx(1.0 / (2 * factor_pi), rel=0.05)
        # The fit must actually track the oscillation (no degenerate a~0 line).
        assert abs(results["a"]) > 0.2

    def test_check_data_requires_signal_and_coord(self):
        est = PowerRabiEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"I": ("amp_prefactor", [0, 1])},
                                       coords={"amp_prefactor": [0, 1]}))
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"signal": ("x", [0, 1])}, coords={"x": [0, 1]}))

    def test_metadata_drops_arrays(self):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results = est.extract_parameters(ds)
        meta = est.extract_metadata(results)
        assert "best_fit" not in meta and "fit_report" not in meta
        assert {"a", "f", "phi", "c", "opt_amp_prefactor", "success"} <= set(meta)

    def test_plot_data_layout(self):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, results)
        assert "signal" in pd and "best_fit" in pd
        assert pd["signal"].dims == ("amp_prefactor",)
        assert pd.attrs["opt_amp_prefactor"] == pytest.approx(results["opt_amp_prefactor"])

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "power_rabi_metadata.json").exists()
        assert (tmp_path / "power_rabi_plotdata.nc").exists()
        assert set(figs) == {"amplitude"}
        assert isinstance(figs["amplitude"], plt.Figure)
        plt.close("all")
