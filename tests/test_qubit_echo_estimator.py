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


def _make_decay_iq(t2e=50e-6, theta=0.9, n=101, t_max=250e-6, sep=3.0, c_frac=0.1,
                   noise_std=0.0, seed=13):
    """Echo decay placed in the IQ plane at readout rotation ``theta``."""
    t = np.linspace(0.0, t_max, n)
    P = c_frac + (1.0 - c_frac) * np.exp(-t / t2e)
    d = sep * np.exp(1j * theta)
    pos0 = -0.4 + 0.9j
    z = pos0 + P * d
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        z = z + noise_std * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return xr.Dataset(
        {"I": ("idle_time", np.real(z)), "Q": ("idle_time", np.imag(z))},
        coords={"idle_time": t},
    )


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

    @pytest.mark.parametrize("theta", [0.3, 1.4, -2.0])
    def test_recovery_from_iq(self, theta):
        t2e = 50e-6
        results, _ = QubitEchoEstimator().analyze(
            _make_decay_iq(t2e=t2e, theta=theta), skip_figures=True
        )
        assert results["success"]
        assert results["t2_echo"] == pytest.approx(t2e, rel=0.05)
        assert results["reduction_method"] == "pca"

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

    def test_stored_positions_resolve_axis(self):
        """ref_pos_* variables resolve the axis deterministically (method
        'positions') and the blob centers reach the plotdata attrs."""
        t2e, theta = 50e-6, 0.9
        ds = _make_decay_iq(t2e=t2e, theta=theta)
        pos0 = -0.4 + 0.9j                     # the fixture's ground center
        pos1 = pos0 + 3.0 * np.exp(1j * theta)  # + sep * e^{i theta}
        ds["ref_pos_g_i"], ds["ref_pos_g_q"] = float(pos0.real), float(pos0.imag)
        ds["ref_pos_e_i"], ds["ref_pos_e_q"] = float(pos1.real), float(pos1.imag)
        est = QubitEchoEstimator()
        results = est.extract_parameters(ds)
        assert results["success"]
        assert results["t2_echo"] == pytest.approx(t2e, rel=0.05)
        assert results["reduction_method"] == "positions"
        pd = est.build_plot_data(ds, results)
        assert pd.attrs["reduction_method"] == "positions"
        assert pd.attrs["pos_g_i"] == pytest.approx(pos0.real)
