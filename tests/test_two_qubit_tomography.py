"""Two-qubit tomography tool: noiseless round trips and conventions."""

import numpy as np
import pytest

from scqat.tools import two_qubit_tomography as tq

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

#: the pre-rotation that brings each measured axis onto Z (the probes' convention)
PRE = {
    "z": I2,
    "x": np.cos(-np.pi / 4) * I2 - 1j * np.sin(-np.pi / 4) * Y,   # -Y90
    "y": np.cos(np.pi / 4) * I2 - 1j * np.sin(np.pi / 4) * X,     # +X90
}


def measured(rho, confusion=None):
    """The joint distribution of every basis setting, computed independently."""
    out = {}
    for label in tq.BASIS_LABELS:
        u = np.kron(PRE[label[0]], PRE[label[1]])
        p = np.real(np.diag(u @ rho @ u.conj().T))
        out[label] = p if confusion is None else confusion @ p
    return out


def random_rho(rng):
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_noiseless_states_reconstruct_exactly():
    rng = np.random.default_rng(0)
    for _ in range(5):
        rho = random_rho(rng)
        got = tq.rho_from_paulis(tq.pauli_expectations(measured(rho)))
        np.testing.assert_allclose(got, rho, atol=1e-12)


def test_readout_correction_inverts_a_known_confusion():
    rng = np.random.default_rng(1)
    rho = random_rho(rng)
    m = tq.confusion_from_fidelities((0.95, 0.92), (0.96, 0.90))
    corrected = {b: tq.correct_readout(p, m) for b, p in measured(rho, m).items()}
    np.testing.assert_allclose(tq.rho_from_paulis(tq.pauli_expectations(corrected)), rho,
                               atol=1e-12)


def test_confusion_from_calibration_is_the_transposed_normalized_block():
    m = tq.confusion_from_fidelities((0.95, 0.92), (0.96, 0.90))
    calibration = 2.0 * m.T                      # rows = prepared, unnormalized
    np.testing.assert_allclose(tq.confusion_from_calibration(calibration), m)
    with pytest.raises(ValueError):
        tq.confusion_from_calibration(np.ones((3, 4)))
    with pytest.raises(ValueError):
        tq.confusion_from_calibration(np.zeros((4, 4)))


def test_confusion_from_fidelities_orders_member_zero_first():
    m = tq.confusion_from_fidelities((1.0, 0.8), (1.0, 1.0))
    # |10> prepared (index 2): member 0 decays to 0 in 20 % of the shots -> |00>
    np.testing.assert_allclose(m[:, 2], [0.2, 0.0, 0.8, 0.0])


def test_dual_rail_conventions():
    alpha = 0.7
    psi = np.zeros(4, dtype=complex)
    psi[2], psi[1] = 1 / np.sqrt(2), np.exp(1j * alpha) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    d = tq.dual_rail(rho)
    # (|up> + e^{i alpha}|down>)/sqrt2 sits on the equator at azimuth alpha
    np.testing.assert_allclose([d["x"], d["y"], d["z"]], [np.cos(alpha), np.sin(alpha), 0.0],
                               atol=1e-12)
    assert d["p_sub"] == pytest.approx(1.0)
    up = np.zeros((4, 4))
    up[2, 2] = 1.0
    d = tq.dual_rail(up)
    assert (d["x"], d["y"], d["z"], d["p00"], d["p11"]) == (0.0, 0.0, 1.0, 0.0, 0.0)
    assert tq.purity(up) == pytest.approx(1.0)


def test_vectorized_over_leading_axes():
    rng = np.random.default_rng(2)
    rhos = np.stack([random_rho(rng) for _ in range(6)]).reshape(2, 3, 4, 4)
    probs = {b: np.zeros((2, 3, 4)) for b in tq.BASIS_LABELS}
    for i in range(2):
        for j in range(3):
            for b, p in measured(rhos[i, j]).items():
                probs[b][i, j] = p
    np.testing.assert_allclose(tq.rho_from_paulis(tq.pauli_expectations(probs)), rhos,
                               atol=1e-12)
