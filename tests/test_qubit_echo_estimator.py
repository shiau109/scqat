"""QubitEchoEstimator: recovery, metadata projection, artifact naming."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_echo import QubitEchoEstimator


def _make_decay(t2e=50e-6, a=0.8, c=0.1, n=101, t_max=250e-6, noise_std=0.0):
    t = np.linspace(0.0, t_max, n)
    y = a * np.exp(-t / t2e) + c
    if noise_std > 0:
        rng = np.random.default_rng(11)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.Dataset({"signal": ("idle_time", y)}, coords={"idle_time": t})


class TestQubitEchoEstimator:
    def test_noiseless_recovery(self):
        t2e = 50e-6
        results, _ = QubitEchoEstimator().analyze(_make_decay(t2e=t2e), skip_figures=True)
        assert results["success"]
        assert results["t2_echo"] == pytest.approx(t2e, rel=0.05)
        assert results["amplitude"] == pytest.approx(0.8, rel=0.05)
        assert results["offset"] == pytest.approx(0.1, abs=0.02)

    def test_noisy_recovery(self):
        t2e = 70e-6
        results, _ = QubitEchoEstimator().analyze(
            _make_decay(t2e=t2e, n=201, noise_std=0.02), skip_figures=True
        )
        assert results["success"]
        assert results["t2_echo"] == pytest.approx(t2e, rel=0.15)

    def test_metadata_drops_arrays(self):
        est = QubitEchoEstimator()
        results = est.extract_parameters(_make_decay())
        metadata = est.extract_metadata(results)
        assert "best_fit" not in metadata
        assert {"t2_echo", "amplitude", "offset", "success"} <= set(metadata)

    def test_artifacts_and_figure_name(self, tmp_path):
        """Single-figure idiom: the file is qubit_echo.png, not doubled."""
        QubitEchoEstimator().analyze(_make_decay(), output_dir=str(tmp_path))
        names = {p.name for p in tmp_path.iterdir()}
        assert "qubit_echo_metadata.json" in names
        assert "qubit_echo_plotdata.nc" in names
        assert "qubit_echo.png" in names

    def test_check_data_rejects_missing(self):
        with pytest.raises(ValueError):
            QubitEchoEstimator().analyze(xr.Dataset({"other": ("x", [1.0])}, coords={"x": [0.0]}))
