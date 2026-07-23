"""IQ -> 1-D signal reductions — the two ways a 2-D readout cloud becomes a fit signal.

Every averaged experiment must collapse the complex demodulated readout ``I + iQ``
into the 1-D real signal its fitter consumes. There are exactly two geometries, and
they are NOT interchangeable:

* :func:`radial` — ``|IQ - ref|``, the distance from a reference point. Correct when
  the sweep sits mostly at ONE cluster (off-resonance ~ ground) with a narrow
  excursion: the spectroscopy family, where ``ref`` is the off-resonance/ground
  cluster (auto-estimated as the complex median). Unsigned, peak-shaped.
* :func:`axial` — the signed projection onto the ``|0>-|1>`` line, i.e. rotate the
  cloud so that axis lands on the real axis and take the real part. Correct when the
  population is swept along the FULL ground->excited segment: the coherent-drive
  family (power_rabi, ramsey, echo, T1). Robust to the readout rotation (uses both
  quadratures), so it is a clean cosine/decay for ANY readout phase — unlike taking a
  single raw quadrature, whose contrast dies as ``cos(readout_rotation)``.

The axis of :func:`axial` is resolved by the best available reference (priority
order): a supplied ``angle`` (a calibrated readout rotation), else two per-state
``positions`` (the ``|0>``/``|1>`` IQ centroids), else PCA of the sweep itself
(rotate-to-max-variance). Nothing but math lives here — no dataset/estimator imports
(the module must stay importable by an external simulation repo and pass the
no-estimator-layering rule).

Convention
----------
``axial`` returns ``I*cos(a) - Q*sin(a) = Re((I + iQ) * exp(i*a))`` for the resolved
angle ``a``. The angle is chosen so the ``|0>->|1>`` (g->e) vector lands on the
positive real axis: ``a = -angle(pos1 - pos0)``. Population then reads out monotonic
in the projection. The overall SIGN of a PCA-resolved projection is otherwise
arbitrary (PCA gives an axis, not a direction), so it is pinned by ``pca_sign`` — the
downstream fitters are sign-robust, so this is a cosmetic-but-deterministic choice.
"""

from typing import Any, Optional, Sequence

import numpy as np

#: knobs of :func:`axial` (the coherent-drive reduction)
AXIAL_KNOBS = frozenset({"angle", "positions", "pca_sign"})
#: knobs of :func:`radial` (the spectroscopy reduction)
RADIAL_KNOBS = frozenset({"ref"})
#: union — the full reduction knob surface, for dict-collecting callers to validate
IQ_REDUCE_KNOBS = AXIAL_KNOBS | RADIAL_KNOBS


def validate_iq_reduce_kwargs(knobs: dict, *, allowed: frozenset = IQ_REDUCE_KNOBS) -> None:
    """Raise ValueError for an unknown reduction knob — call BEFORE any slice loop.

    Pass ``allowed=AXIAL_KNOBS`` / ``RADIAL_KNOBS`` to restrict to one reduction.
    """
    unknown = set(knobs) - allowed
    if unknown:
        raise ValueError(
            f"Unknown IQ-reduction knob(s) {sorted(unknown)}; valid: {sorted(allowed)}"
        )


# ----------------------------------------------------------------------
# radial: distance from a reference point (spectroscopy family)
# ----------------------------------------------------------------------
def ground_ref(I: np.ndarray, Q: np.ndarray) -> complex:
    """Robust estimate of the dominant (off-resonance / ground) cluster: the complex
    median of the IQ cloud. Works when most sweep points sit off-resonance."""
    I = np.asarray(I, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    return complex(float(np.median(I)), float(np.median(Q)))


def radial(I: np.ndarray, Q: np.ndarray, *, ref: Optional[complex] = None) -> np.ndarray:
    """``|IQ - ref|`` — distance from a reference point in the IQ plane.

    Parameters
    ----------
    I, Q : array_like
        The two quadratures (same shape).
    ref : complex, optional
        Reference point. Defaults to :func:`ground_ref` (the complex median), the
        off-resonance/ground cluster of a spectroscopy sweep.
    """
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if ref is None:
        ref = ground_ref(I, Q)
    ref = complex(ref)
    return np.abs((I + 1j * Q) - ref)


# ----------------------------------------------------------------------
# axial: signed projection onto the |0>-|1> axis (coherent-drive family)
# ----------------------------------------------------------------------
def _as_complex_positions(positions: Sequence[Any]) -> np.ndarray:
    """Normalise ``positions`` to a 1-D complex array of per-state IQ centroids.

    Accepts a sequence of complex, a sequence of ``(I, Q)`` pairs, or an ``(N, 2)``
    array. Raises for >2 states — a 0/1/2 qutrit is not collinear, so a single axis
    cannot separate it (use 2-D state discrimination instead)."""
    arr = np.asarray(positions)
    if np.iscomplexobj(arr):
        pts = arr.ravel().astype(complex)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        pts = arr[:, 0].astype(float) + 1j * arr[:, 1].astype(float)
    else:
        pts = np.asarray(arr, dtype=complex).ravel()
    if pts.size < 2:
        raise ValueError("axial: `positions` needs at least the |0> and |1> centroids.")
    if pts.size > 2:
        raise ValueError(
            "axial: >2 `positions` given — a 0/1/2 qutrit is not collinear and cannot "
            "be reduced to one axis; use 2-D state discrimination "
            "(scqat.tools.discriminate.discriminate_states)."
        )
    return pts


def _pca_angle(I: np.ndarray, Q: np.ndarray) -> float:
    """Angle that rotates the cloud's principal (max-variance) axis onto the real
    axis: ``a = -atan2(v_Q, v_I)`` for the top eigenvector ``v`` of cov([I, Q])."""
    Ic = np.asarray(I, dtype=float).ravel() - float(np.mean(I))
    Qc = np.asarray(Q, dtype=float).ravel() - float(np.mean(Q))
    cov = np.cov(np.vstack([Ic, Qc]))
    # Degeneracy check must be SCALE-FREE (trace == 0 iff the cloud is a single
    # point): an absolute allclose(cov, 0) silently zeroes the angle for
    # volt-scale readout data (cov ~ 1e-9), degrading axial back to raw I.
    if not np.all(np.isfinite(cov)) or float(np.trace(cov)) <= 0.0:
        return 0.0
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return float(-np.arctan2(v[1], v[0]))


def axis_angle(
    I: np.ndarray,
    Q: np.ndarray,
    *,
    angle: Optional[float] = None,
    positions: Optional[Sequence[Any]] = None,
) -> float:
    """Resolve the rotation angle that puts the state axis on the real axis.

    Priority: explicit ``angle`` -> ``positions`` (angle of the g->e vector) -> PCA of
    the ``(I, Q)`` cloud. The returned ``a`` is defined so that
    ``Re((I + iQ) * exp(i*a))`` runs along the ``pos0 -> pos1`` direction.
    """
    if angle is not None:
        return float(angle)
    if positions is not None:
        p0, p1 = _as_complex_positions(positions)[:2]
        return float(-np.angle(p1 - p0))
    return _pca_angle(I, Q)


def axial(
    I: np.ndarray,
    Q: np.ndarray,
    *,
    angle: Optional[float] = None,
    positions: Optional[Sequence[Any]] = None,
    pca_sign: str = "first_low",
) -> np.ndarray:
    """Signed projection of the IQ cloud onto the ``|0>-|1>`` axis.

    Returns ``I*cos(a) - Q*sin(a)`` for ``a = axis_angle(I, Q, angle=, positions=)``.

    When the axis comes from PCA the direction is arbitrary; ``pca_sign`` pins it:
    ``"first_low"`` (default) orients the projection so its first sample sits at the
    low end of the range (natural for sweeps that start in the ground state);
    ``"none"`` leaves the raw sign. Only applied when neither ``angle`` nor
    ``positions`` fixes the direction.
    """
    I = np.asarray(I, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float).ravel()
    a = axis_angle(I, Q, angle=angle, positions=positions)
    proj = I * np.cos(a) - Q * np.sin(a)

    if angle is None and positions is None and pca_sign == "first_low" and proj.size:
        lo, hi = float(np.min(proj)), float(np.max(proj))
        if abs(proj[0] - hi) < abs(proj[0] - lo):  # first sample nearer the top -> flip
            proj = -proj
    return proj
