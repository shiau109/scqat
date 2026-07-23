"""Tests for the shared IQ -> 1-D reductions (scqat.tools.iq_reduce).

`axial` (signed projection onto the |0>-|1> axis) must recover a clean cosine for
ANY readout rotation — this is the fix for taking a single raw quadrature, whose
contrast dies as cos(readout_rotation). `radial` (distance from the ground cluster)
must peak for a spectroscopy excursion but *fold* on a Rabi trajectory (why a
statistic reference fails for power_rabi).
"""

import numpy as np
import pytest

from scqat.tools.iq_reduce import (
    AXIAL_KNOBS,
    RADIAL_KNOBS,
    axial,
    axis_angle,
    ground_ref,
    radial,
    validate_iq_reduce_kwargs,
)


def _rabi_iq(theta, f=0.5, n=201, x_max=2.0, sep=3.0, ground=1.3 - 0.4j, noise=0.0, seed=0):
    """A Rabi amplitude sweep placed in the IQ plane at readout rotation ``theta``.

    Population ``P = (1 - cos(2*pi*f*x))/2`` runs 0 -> 1 -> 0 over ``x in [0, x_max]``;
    every IQ point is ``ground + P*(sep*e^{i*theta})``. Returns
    ``(x, P, I, Q, pos0, pos1)``.
    """
    x = np.linspace(0.0, x_max, n)
    P = 0.5 * (1.0 - np.cos(2.0 * np.pi * f * x))
    d = sep * np.exp(1j * theta)
    pos0 = complex(ground)
    pos1 = pos0 + d
    z = pos0 + P * d
    if noise:
        rng = np.random.default_rng(seed)
        z = z + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return x, P, np.real(z), np.imag(z), pos0, pos1


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, 9))
def test_axial_recovers_population_at_any_rotation(theta):
    """The whole point: a clean, full-contrast signal regardless of readout phase."""
    _, P, I, Q, *_ = _rabi_iq(theta=theta, noise=1e-3, seed=1)
    s = axial(I, Q)
    assert abs(_corr(s, P)) > 0.99


def test_radial_folds_where_axial_recovers():
    """On a Rabi trajectory `radial` folds (median lands mid-segment -> corr ~ 0),
    while `axial` tracks the population."""
    _, P, I, Q, *_ = _rabi_iq(theta=1.1)
    assert abs(_corr(axial(I, Q), P)) > 0.99
    assert abs(_corr(radial(I, Q), P)) < 0.3


def test_pca_sign_first_low_orients_first_sample_low():
    _, _P, I, Q, *_ = _rabi_iq(theta=0.9)
    s = axial(I, Q, pca_sign="first_low")
    # first sample is the ground state (P=0) -> at the low end of the range
    assert abs(s[0] - s.min()) < abs(s[0] - s.max())


def test_angle_positions_and_pca_agree_on_line_data():
    _, _P, I, Q, pos0, pos1 = _rabi_iq(theta=0.7)
    a_pos = axis_angle(I, Q, positions=[pos0, pos1])
    a_pca = axis_angle(I, Q)
    # same axis up to a pi ambiguity (PCA has no direction)
    delta = (a_pos - a_pca) % np.pi
    assert min(delta, np.pi - delta) < 1e-6
    # positions and an explicit angle give identical projections
    assert np.allclose(axial(I, Q, angle=a_pos), axial(I, Q, positions=[pos0, pos1]))
    # and they agree with PCA up to sign
    assert abs(_corr(axial(I, Q, positions=[pos0, pos1]), axial(I, Q))) > 0.999


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e5])
def test_axial_is_scale_invariant(scale):
    """Volt-scale readout clouds (ptp ~ 1e-4 -> cov ~ 1e-9) must resolve the same
    axis as O(1) data: the PCA degeneracy guard must be scale-free. (Regression:
    an absolute allclose(cov, 0) silently zeroed the angle for small clouds,
    degrading axial back to raw I.)"""
    theta = 1.9  # separation mostly in Q — raw I would lose most contrast
    _, P, I, Q, *_ = _rabi_iq(theta=theta, noise=1e-3, seed=4)
    s = axial(I * scale, Q * scale)
    assert abs(_corr(s, P)) > 0.99
    assert axis_angle(I * scale, Q * scale) != 0.0


def test_radial_matches_median_distance_and_ground_ref():
    rng = np.random.default_rng(0)
    I = rng.standard_normal(64)
    Q = rng.standard_normal(64)
    ref = complex(float(np.median(I)), float(np.median(Q)))
    assert ground_ref(I, Q) == ref
    assert np.allclose(radial(I, Q), np.abs((I + 1j * Q) - ref))


def test_radial_peaks_on_spectroscopy_excursion():
    n = 201
    idx = np.arange(n)
    cluster = 2.0 - 1.0j
    bump = 1.0 / (1.0 + ((idx - 100) / 3.0) ** 2)  # narrow Lorentzian, peak at 100
    z = cluster + (0.6 + 0.8j) * bump
    I, Q = np.real(z), np.imag(z)
    assert abs(ground_ref(I, Q) - cluster) < 0.05
    assert abs(int(np.argmax(radial(I, Q))) - 100) <= 2


def test_axial_rejects_qutrit_positions():
    I = np.array([0.0, 1.0, 2.0])
    Q = np.array([0.0, 0.5, 0.0])
    with pytest.raises(ValueError):
        axial(I, Q, positions=[0 + 0j, 1 + 0j, 0.5 + 1j])


def test_validate_knobs():
    validate_iq_reduce_kwargs({"angle": 1.0, "pca_sign": "none"}, allowed=AXIAL_KNOBS)
    validate_iq_reduce_kwargs({"ref": 0}, allowed=RADIAL_KNOBS)
    with pytest.raises(ValueError):
        validate_iq_reduce_kwargs({"ref": 0}, allowed=AXIAL_KNOBS)  # ref is a radial knob
    with pytest.raises(ValueError):
        validate_iq_reduce_kwargs({"bogus": 1})
