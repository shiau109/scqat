"""Tests for the SwapOscillationEstimator cosine fit / swap-period extraction.

Synthesises a swap-chain (N-swap) sweep — population vs the integer number of swaps —
and checks the estimator recovers the swap-oscillation frequency ``f`` (cycles per
swap) and ``swap_period = 1/f``, exposes the ``success`` field the LCHQMDriver node
relies on, and that ``analyze`` round-trips its metadata + plot-data artifacts.

Convention: the swap pair's control qubit starts in |1> (signal high at N=0, dipping
as population swaps away) — ``0.5 + 0.5*cos(2*pi*f*N)``; the swap target starts in
|0> (the opposite sign). Both must fit.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import SwapOscillationEstimator
from scqat.estimators.swap_oscillation import SwapOscillationEstimator as SubpkgEstimator


def _make_ds(f=0.3, n_max=20, noise=2e-3, seed=0, rising=False):
    """Swap-chain sweep with oscillation frequency ``f`` (cycles per swap) over N=0..n_max.

    ``rising=False`` is the control-qubit convention (signal high at N=0, dipping as the
    excitation swaps away); ``rising=True`` is the swap target (signal low at N=0) — the
    case that traps a fixed phi=0 / a>=0 fit at a~0.
    """
    rounds = np.arange(0, n_max + 1)
    sign = -1.0 if rising else 1.0
    signal = 0.5 + sign * 0.5 * np.cos(2 * np.pi * f * rounds)
    rng = np.random.default_rng(seed)
    signal = signal + noise * rng.standard_normal(rounds.size)
    return xr.Dataset(
        {"signal": ("round", signal)},
        coords={"round": rounds},
    )


class TestSwapOscillationEstimator:

    def test_imports_match(self):
        assert SwapOscillationEstimator is SubpkgEstimator
        assert SwapOscillationEstimator.estimator_name == "swap_oscillation"

    @pytest.mark.parametrize("rising", [False, True])
    def test_recovers_swap_frequency(self, rising):
        # Both measured qubits must fit: 'rising' (the swap target, low at N=0) is the
        # case that traps a fixed phi=0 / a>=0 cosine fit at a flat (a~0) line.
        f = 0.3
        ds = _make_ds(f=f, rising=rising)
        results = SwapOscillationEstimator().extract_parameters(ds)
        assert results["success"] is True
        assert results["f"] == pytest.approx(f, rel=0.05)
        assert results["swap_period"] == pytest.approx(1.0 / f, rel=0.05)
        # The fit must actually track the oscillation (no degenerate a~0 line).
        assert abs(results["a"]) > 0.2
        assert results["r_squared"] > 0.9

    def test_flat_signal_fails(self):
        # A no-op swap macro gives a flat curve; the contrast guard must report failure.
        rounds = np.arange(0, 21)
        rng = np.random.default_rng(1)
        signal = 0.5 + 1e-4 * rng.standard_normal(rounds.size)
        ds = xr.Dataset({"signal": ("round", signal)}, coords={"round": rounds})
        results = SwapOscillationEstimator().extract_parameters(ds)
        assert results["success"] is False

    def test_check_data_requires_signal_and_coord(self):
        est = SwapOscillationEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"state": ("round", [0, 1])},
                                       coords={"round": [0, 1]}))
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"signal": ("x", [0, 1])}, coords={"x": [0, 1]}))

    def test_metadata_drops_arrays(self):
        ds = _make_ds()
        est = SwapOscillationEstimator()
        results = est.extract_parameters(ds)
        meta = est.extract_metadata(results)
        assert "best_fit" not in meta and "fit_report" not in meta
        assert "round_dense" not in meta and "best_fit_dense" not in meta
        assert {"a", "f", "phi", "c", "swap_period", "r_squared", "success"} <= set(meta)

    def test_plot_data_layout(self):
        ds = _make_ds()
        est = SwapOscillationEstimator()
        results = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, results)
        assert "signal" in pd and "best_fit" in pd and "best_fit_dense" in pd
        assert pd["signal"].dims == ("round",)
        # The dense fit curve (for a smooth plotted line) lives on its own fine axis.
        assert pd["best_fit_dense"].dims == ("round_dense",)
        assert pd.sizes["round_dense"] > 10 * pd.sizes["round"]
        assert pd.attrs["swap_period"] == pytest.approx(results["swap_period"])

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_ds()
        est = SwapOscillationEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "swap_oscillation_metadata.json").exists()
        assert (tmp_path / "swap_oscillation_plotdata.nc").exists()
        assert set(figs) == {"rounds"}
        assert isinstance(figs["rounds"], plt.Figure)
        plt.close("all")
