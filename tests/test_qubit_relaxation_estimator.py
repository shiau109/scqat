"""QubitRelaxationEstimator: recovery, metadata projection, artifact naming."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_relaxation import QubitRelaxationEstimator


def _make_decay(t1=40e-6, a=0.9, c=0.05, n=101, t_max=200e-6, noise_std=0.0):
    t = np.linspace(0.0, t_max, n)
    y = a * np.exp(-t / t1) + c
    if noise_std > 0:
        rng = np.random.default_rng(7)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.Dataset({"signal": ("wait_time", y)}, coords={"wait_time": t})


def _make_decay_iq(t1=40e-6, theta=1.2, n=101, t_max=200e-6, sep=3.0, c_frac=0.05,
                   noise_std=0.0, seed=3):
    """T1 decay placed in the IQ plane at readout rotation ``theta``: excited fraction
    ``P = exp(-t/t1)`` decays the cloud from |1> back to |0> along the g->e axis."""
    t = np.linspace(0.0, t_max, n)
    P = c_frac + (1.0 - c_frac) * np.exp(-t / t1)
    d = sep * np.exp(1j * theta)
    pos0 = 1.0 - 0.5j
    z = pos0 + P * d
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        z = z + noise_std * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return xr.Dataset(
        {"I": ("wait_time", np.real(z)), "Q": ("wait_time", np.imag(z))},
        coords={"wait_time": t},
    )


class TestQubitRelaxationEstimator:
    def test_noiseless_recovery(self):
        t1 = 40e-6
        results, _ = QubitRelaxationEstimator().analyze(_make_decay(t1=t1), skip_figures=True)
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.05)
        assert results["amplitude"] == pytest.approx(0.9, rel=0.05)
        assert results["offset"] == pytest.approx(0.05, abs=0.02)

    def test_noisy_recovery(self):
        t1 = 60e-6
        results, _ = QubitRelaxationEstimator().analyze(
            _make_decay(t1=t1, n=201, noise_std=0.02), skip_figures=True
        )
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.15)

    @pytest.mark.parametrize("theta", [0.4, 1.6, -2.2])
    def test_recovery_from_iq(self, theta):
        t1 = 40e-6
        results, _ = QubitRelaxationEstimator().analyze(
            _make_decay_iq(t1=t1, theta=theta), skip_figures=True
        )
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.05)
        assert results["reduction_method"] == "pca"

    def test_metadata_drops_arrays(self):
        est = QubitRelaxationEstimator()
        results = est.extract_parameters(_make_decay())
        metadata = est.extract_metadata(results)
        assert "best_fit" not in metadata
        assert {"t1", "amplitude", "offset", "success"} <= set(metadata)

    def test_artifacts_and_figure_name(self, tmp_path):
        """Single-figure idiom: the file is qubit_relaxation.png, not doubled."""
        QubitRelaxationEstimator().analyze(_make_decay(), output_dir=str(tmp_path))
        names = {p.name for p in tmp_path.iterdir()}
        assert "qubit_relaxation_metadata.json" in names
        assert "qubit_relaxation_plotdata.nc" in names
        assert "qubit_relaxation.png" in names

    def test_check_data_rejects_missing(self):
        with pytest.raises(ValueError):
            QubitRelaxationEstimator().analyze(xr.Dataset({"other": ("x", [1.0])}, coords={"x": [0.0]}))

    def test_stored_positions_resolve_axis(self):
        """ref_pos_* variables resolve the axis deterministically (method
        'positions') and the blob centers reach the plotdata attrs."""
        t1, theta = 40e-6, 1.2
        ds = _make_decay_iq(t1=t1, theta=theta)
        pos0 = 1.0 - 0.5j                      # the fixture's ground center
        pos1 = pos0 + 3.0 * np.exp(1j * theta)  # + sep * e^{i theta}
        ds["ref_pos_g_i"], ds["ref_pos_g_q"] = float(pos0.real), float(pos0.imag)
        ds["ref_pos_e_i"], ds["ref_pos_e_q"] = float(pos1.real), float(pos1.imag)
        est = QubitRelaxationEstimator()
        results = est.extract_parameters(ds)
        assert results["success"]
        assert results["t1"] == pytest.approx(t1, rel=0.05)
        assert results["reduction_method"] == "positions"
        pd = est.build_plot_data(ds, results)
        assert pd.attrs["reduction_method"] == "positions"
        assert pd.attrs["pos_e_i"] == pytest.approx(pos1.real)
