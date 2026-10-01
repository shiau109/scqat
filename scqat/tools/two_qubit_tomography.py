"""Two-qubit state tomography from discriminated joint readout - pure math.

A pair is read out jointly: every shot gives one of the four outcomes
``00, 01, 10, 11`` (digit 0 = the FIRST member, the pair's ``high`` member in
SCQO's order). State tomography measures the pair in nine bases: before the
readout each member is rotated so that the axis to be measured lands on Z
(``z``: nothing; ``x``: -Y90; ``y``: +X90). A basis label is two characters, one
per member in the same order, e.g. ``"xz"`` = member 0 measured along X,
member 1 along Z.

Conventions (shared with every caller):

* basis states ``|00>, |01>, |10>, |11>`` at indices 0..3, index = 2*m0 + m1;
* readout level 0 is Z = +1, level 1 is Z = -1;
* ``rho = 1/4 sum_{P,Q in I,X,Y,Z} <P (x) Q> P (x) Q``.

The functions here are vectorized over any leading axes: a probability array of
shape ``(..., 4)`` maps to expectations of shape ``(...,)`` and a density
matrix of shape ``(..., 4, 4)``.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

#: the nine measurement bases, first character = member 0.
BASIS_LABELS: tuple[str, ...] = tuple(a + b for a in "zxy" for b in "zxy")

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

# Signs of the three observables a basis setting yields, over the outcomes
# (00, 01, 10, 11): <A (x) I>, <I (x) B>, <A (x) B>.
_SIGN_FIRST = np.array([1.0, 1.0, -1.0, -1.0])
_SIGN_SECOND = np.array([1.0, -1.0, 1.0, -1.0])
_SIGN_BOTH = np.array([1.0, -1.0, -1.0, 1.0])


def confusion_from_calibration(calibration: np.ndarray) -> np.ndarray:
    """The 4x4 assignment matrix ``M[measured, prepared]`` from a calibration
    block ``calibration[prepared, measured]`` (each row a measured distribution
    of one prepared basis state, rows in the order 00, 01, 10, 11).

    Rows are renormalized; a row that sums to zero raises.
    """
    cal = np.asarray(calibration, dtype=float)
    if cal.shape != (4, 4):
        raise ValueError(f"calibration must be 4x4 (prepared, measured), got {cal.shape}")
    sums = cal.sum(axis=1, keepdims=True)
    if np.any(sums <= 0) or not np.all(np.isfinite(cal)):
        raise ValueError("calibration has an empty or non-finite row")
    return (cal / sums).T


def confusion_from_fidelities(fid_first: tuple[float, float],
                              fid_second: tuple[float, float]) -> np.ndarray:
    """The 4x4 assignment matrix of two INDEPENDENT readouts, from each member's
    ``(P(0|0), P(1|1))``. Ignores readout crosstalk between the members."""
    def single(fg: float, fe: float) -> np.ndarray:
        return np.array([[fg, 1.0 - fe], [1.0 - fg, fe]])   # [measured, prepared]
    return np.kron(single(*fid_first), single(*fid_second))


def correct_readout(probabilities: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    """Invert the readout: solve ``M p_true = p_measured`` along the last axis.

    Plain linear inversion - the result can leave the simplex by the noise, which
    the linear tomography below tolerates by construction.
    """
    p = np.asarray(probabilities, dtype=float)
    inv = np.linalg.inv(np.asarray(confusion, dtype=float))
    return p @ inv.T


def pauli_expectations(probabilities: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The 15 non-trivial two-qubit Pauli expectations from the nine bases.

    ``probabilities[label]`` is the (readout-corrected) joint distribution of one
    basis setting, shape ``(..., 4)``. A single-member observable (``"XI"``,
    ``"IZ"``, ...) is measured in three settings and averaged. Keys are two
    upper-case characters, member 0 first (``"XY"`` = X on member 0, Y on 1).
    """
    missing = [b for b in BASIS_LABELS if b not in probabilities]
    if missing:
        raise ValueError(f"missing tomography bases {missing}")
    sums: dict[str, list[np.ndarray]] = {}
    for label in BASIS_LABELS:
        p = np.asarray(probabilities[label], dtype=float)
        a, b = label[0].upper(), label[1].upper()
        sums.setdefault(a + "I", []).append(p @ _SIGN_FIRST)
        sums.setdefault("I" + b, []).append(p @ _SIGN_SECOND)
        sums.setdefault(a + b, []).append(p @ _SIGN_BOTH)
    return {key: np.mean(values, axis=0) for key, values in sums.items()}


def rho_from_paulis(expectations: Mapping[str, np.ndarray]) -> np.ndarray:
    """Linear-inversion density matrix ``(..., 4, 4)`` from the 15 expectations."""
    first = next(iter(expectations.values()))
    shape = np.shape(first)
    rho = np.zeros((*shape, 4, 4), dtype=complex)
    rho += np.eye(4) / 4.0
    for key, value in expectations.items():
        op = np.kron(_PAULI[key[0]], _PAULI[key[1]])
        rho += np.asarray(value, dtype=float)[..., None, None] * op / 4.0
    return rho


def dual_rail(rho: np.ndarray) -> dict[str, np.ndarray]:
    """The single-excitation ("dual-rail") view of a two-member state.

    Up = ``|10>`` (member 0 excited), down = ``|01>``. Returns the UNNORMALIZED
    Bloch components ``x, y, z`` of that subspace (``x - i y = 2 rho_{10,01}``,
    ``z = rho_{10,10} - rho_{01,01}``), the subspace population ``p_sub`` and the
    two populations outside it, ``p00`` and ``p11``. Divide x, y, z by ``p_sub``
    for the Bloch vector conditioned on staying in the subspace.
    """
    r = np.asarray(rho)
    c = r[..., 2, 1]
    return {
        "x": 2.0 * np.real(c),
        "y": -2.0 * np.imag(c),
        "z": np.real(r[..., 2, 2] - r[..., 1, 1]),
        "p_sub": np.real(r[..., 2, 2] + r[..., 1, 1]),
        "p00": np.real(r[..., 0, 0]),
        "p11": np.real(r[..., 3, 3]),
    }


def purity(rho: np.ndarray) -> np.ndarray:
    """``tr(rho^2)`` along the leading axes."""
    r = np.asarray(rho)
    return np.real(np.einsum("...ij,...ji->...", r, r))
