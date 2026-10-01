"""qc_n_swap_tomography: the channel read off synthetic tomography data.

The synthetic data come from the channel model itself, projected onto the nine
bases by an independent implementation of the pre-rotations, folded through a
two-member readout confusion and sampled shot by shot. The planted phase per
step depends on the stark amplitude as ``2 pi (a^2 - A_COMP^2)``, so the
compensation sits at ``A_COMP``; the planted angle does not depend on it.
"""

import json

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_n_swap_tomography import QcNSwapTomographyEstimator
from scqat.tools import swap_channel as sc
from scqat.tools import two_qubit_tomography as tq

THETA = 0.2
A_COMP = 0.45
AMPS = np.array([0.40, 0.45, 0.50])
COUNTS = np.arange(0, 13)
DECAY = dict(p_high=0.0166, p_low=0.0182, lam=0.049, eps=0.03)
FID_HIGH, FID_LOW = (0.952, 0.9235), (0.954, 0.929)
LABELS = ["00", "01", "10", "11"]

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PRE = {"z": I2,
       "x": np.cos(-np.pi / 4) * I2 - 1j * np.sin(-np.pi / 4) * Y,
       "y": np.cos(np.pi / 4) * I2 - 1j * np.sin(np.pi / 4) * X}


def _dataset(shots=2000, calibrate=True, theta=THETA, seed=11, drive_high=True):
    rng = np.random.default_rng(seed)
    m = tq.confusion_from_fidelities(FID_HIGH, FID_LOW)
    jp = np.zeros((4, AMPS.size, COUNTS.size, len(tq.BASIS_LABELS)))
    for i, a in enumerate(AMPS):
        phi = 2 * np.pi * (a ** 2 - A_COMP ** 2)
        params = [theta, phi, 0.3, DECAY["p_high"], DECAY["p_low"], DECAY["lam"], DECAY["eps"]]
        states = sc.channel_states(params, COUNTS.max(), excite_high=drive_high)
        for k, label in enumerate(tq.BASIS_LABELS):
            u = np.kron(PRE[label[0]], PRE[label[1]])
            for j, n in enumerate(COUNTS):
                p = np.clip(np.real(np.diag(u @ states[n] @ u.conj().T)), 0, None)
                p = m @ (p / p.sum())
                jp[:, i, j, k] = rng.multinomial(shots, p / p.sum()) / shots
    data = {"joint_population": (("joint_state", "stark_amp", "swap_count", "basis"), jp)}
    coords = {"joint_state": LABELS, "stark_amp": AMPS, "swap_count": COUNTS,
              "basis": list(tq.BASIS_LABELS)}
    if calibrate:
        cal = np.stack([rng.multinomial(4 * shots, m[:, k]) / (4 * shots) for k in range(4)])
        data["calibration_population"] = (("prepared_state", "joint_state"), cal)
        coords["prepared_state"] = LABELS
    return xr.Dataset(data, coords=coords)


@pytest.fixture(scope="module")
def results():
    return QcNSwapTomographyEstimator().extract_parameters(_dataset(), drive_side="high")


def test_angle_and_compensation_are_recovered(results):
    assert results["success"] == 1 and results["n_fit_ok"] == AMPS.size
    assert results["readout_correction"] == "calibration"
    assert abs(results["theta_rad"] - THETA) < max(4 * results["theta_rad_err"], 0.005)
    assert results["theta_stark_amp"] == pytest.approx(A_COMP)
    assert results["compensating_stark_amp"] == pytest.approx(A_COMP, abs=0.005)
    assert results["compensation_extrapolated"] == 0
    assert results["theta_spread_rad"] < 0.02
    assert results["theta_consistency_sigma"] < 3.0
    planted = [2 * np.pi * (a ** 2 - A_COMP ** 2) for a in AMPS]
    np.testing.assert_allclose(results["phase_per_step_rad"], planted, atol=0.03)


def test_decay_is_recovered(results):
    assert results["t1_loss_per_step_high"] == pytest.approx(DECAY["p_high"], abs=0.004)
    assert results["dephasing_per_step"] == pytest.approx(DECAY["lam"], abs=0.015)
    assert results["prep_error"] == pytest.approx(DECAY["eps"], abs=0.015)
    assert abs(results["leak_to_11_per_step"]) < 0.005


def test_readout_correction_fallbacks():
    est = QcNSwapTomographyEstimator()
    ds = _dataset(shots=400, calibrate=False)
    stored = est.extract_parameters(ds, drive_side="high", fid_high=FID_HIGH, fid_low=FID_LOW)
    assert stored["readout_correction"] == "stored_product"
    bare = est.extract_parameters(ds, drive_side="high")
    assert bare["readout_correction"] == "none"
    # uncorrected readout loses contrast, which the channel can only read as prep error
    assert bare["prep_error"] > stored["prep_error"]


def test_low_member_drive():
    res = QcNSwapTomographyEstimator().extract_parameters(
        _dataset(shots=800, drive_high=False), drive_side="low")
    assert abs(res["theta_rad"] - THETA) < 0.01


def test_predictions_from_t1_t2(results):
    res = QcNSwapTomographyEstimator().extract_parameters(
        _dataset(shots=400), drive_side="high", round_duration_ns=368.0,
        t1_high_s=22e-6, t1_low_s=20e-6, t2_star_high_s=12.4e-6, t2_star_low_s=10e-6)
    assert res["predicted_t1_loss_high"] == pytest.approx(1 - np.exp(-0.368 / 22), rel=1e-9)
    assert res["predicted_dephasing"] == pytest.approx(0.049, abs=0.002)
    assert np.isfinite(res["excess_dephasing_per_step"])
    assert np.isnan(results["predicted_dephasing"])


def test_order_free(order_free):
    est = QcNSwapTomographyEstimator()
    out = order_free(est, _dataset(shots=400), ("stark_amp", "swap_count"), drive_side="high")
    assert out["success"] == 1


def test_metadata_is_json_and_artifacts_are_written(tmp_path):
    est = QcNSwapTomographyEstimator()
    res, figures = est.analyze(_dataset(shots=400), output_dir=str(tmp_path), drive_side="high")
    meta = json.loads((tmp_path / "qc_n_swap_tomography_metadata.json").read_text())
    assert meta["estimator_name"] == "qc_n_swap_tomography"
    assert (tmp_path / "qc_n_swap_tomography_plotdata.nc").exists()
    assert set(figures) == {"bloch", "components", "compensation"}


def test_figures_render_on_a_failed_fit():
    ds = _dataset(shots=50)
    ds["joint_population"][:] = 0.25            # no information: every fit fails or is junk
    est = QcNSwapTomographyEstimator()
    res = est.extract_parameters(ds, drive_side="high")
    plot_data = est.build_plot_data(ds, res)
    figures = est.generate_figures(ds, res, plot_data=plot_data)
    assert len(figures) == 3


def test_check_data_refuses_missing_bases():
    ds = _dataset(shots=50).isel(basis=slice(0, 8))
    with pytest.raises(ValueError, match="basis"):
        QcNSwapTomographyEstimator().analyze(ds)
