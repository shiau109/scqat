"""T1RelaxationEstimator: recovery, metadata projection, artifact naming."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.t1_relaxation import T1RelaxationEstimator


def _make_decay(t1=40e-6, a=0.9, c=0.05, n=101, t_max=200e-6, noise_std=0.0):
    t = np.linspace(0.0, t_max, n)
    y = a * np.exp(-t / t1) + c
    if noise_std > 0:
        rng = np.random.default_rng(7)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.Dataset({"signal": ("wait_time", y)}, coords={"wait_time": t})


class TestT1RelaxationEstimator:
    def test_noiseless_recovery(self):
        t1 = 40e-6
        results, _ = T1RelaxationEstimator().analyze(_make_decay(t1=t1), skip_figures=True)
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.05)
        assert results["amplitude"] == pytest.approx(0.9, rel=0.05)
        assert results["offset"] == pytest.approx(0.05, abs=0.02)

    def test_noisy_recovery(self):
        t1 = 60e-6
        results, _ = T1RelaxationEstimator().analyze(
            _make_decay(t1=t1, n=201, noise_std=0.02), skip_figures=True
        )
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.15)

    def test_metadata_drops_arrays(self):
        est = T1RelaxationEstimator()
        results = est.extract_parameters(_make_decay())
        metadata = est.extract_metadata(results)
        assert "best_fit" not in metadata
        assert {"t1", "amplitude", "offset", "success"} <= set(metadata)

    def test_artifacts_and_figure_name(self, tmp_path):
        """Single-figure idiom: the file is t1_relaxation.png, not doubled."""
        T1RelaxationEstimator().analyze(_make_decay(), output_dir=str(tmp_path))
        names = {p.name for p in tmp_path.iterdir()}
        assert "t1_relaxation_metadata.json" in names
        assert "t1_relaxation_plotdata.nc" in names
        assert "t1_relaxation.png" in names

    def test_check_data_rejects_missing(self):
        with pytest.raises(ValueError):
            T1RelaxationEstimator().analyze(xr.Dataset({"other": ("x", [1.0])}, coords={"x": [0.0]}))
