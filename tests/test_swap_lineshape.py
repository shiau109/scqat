"""Tests for the fixed-time swap central-peak reduction.

The model is the two-level exchange kernel with the time frozen:
``(2J)^2/omega^2 * sin^2(pi*omega*t)``, whose on-resonance height is
``sin^2(theta)`` with ``theta = 2*pi*J*t``. These tests pin the angle recovery,
the branch that ``arcsin`` cannot see, and every degrade-don't-raise path.
"""

import numpy as np
import pytest

from scqat.tools.swap_lineshape import fit_swap_peak, j_hz_from_theta, theta_from_peak

T_NS = 100.0
#: a full swap at this duration; theta = pi/2 exactly here.
J_FULL_HZ = 1.0 / (4 * T_NS * 1e-9)


def _trace(j_hz, v0=0.10, n=31, noise=0.0, seed=0, j_ref_hz=2e6):
    """One column of a fixed-time swap map: the kernel sampled over member flux.

    The detuning slope is drawn RELATIVE to the grid, the way the SCQO simulator
    does it, so the peak is a few points wide — i.e. the sweep resolves it.
    """
    vq = np.linspace(0.04, 0.16, n)
    step = float(vq[1] - vq[0])
    kappa = 2 * j_ref_hz / (2 * step)  # Hz per V
    delta = kappa * (vq - v0)
    omega = np.sqrt(delta ** 2 + (2 * j_hz) ** 2)
    y = (2 * j_hz) ** 2 / omega ** 2 * np.sin(np.pi * omega * T_NS * 1e-9) ** 2
    if noise:
        y = y + np.random.default_rng(seed).normal(0, noise, y.shape)
    return vq, np.clip(y, 0.0, 1.0)


def test_theta_from_peak_is_the_arcsin_branch():
    assert theta_from_peak(1.0) == pytest.approx(np.pi / 2)
    assert theta_from_peak(0.5) == pytest.approx(np.pi / 4)
    assert theta_from_peak(0.0) == pytest.approx(0.0)
    # shot noise just past a full swap saturates rather than returning NaN
    assert theta_from_peak(1.02) == pytest.approx(np.pi / 2)
    assert np.isnan(theta_from_peak(float("nan")))


@pytest.mark.parametrize("fraction", [0.2, 0.4, 0.7, 1.0])
def test_recovers_the_angle_below_a_full_swap(fraction):
    j_hz = fraction * J_FULL_HZ
    vq, y = _trace(j_hz, noise=0.005)
    fit = fit_swap_peak(vq, y)
    assert fit["success"] is True
    expected = 2 * np.pi * j_hz * T_NS * 1e-9
    # The central peak is not exactly Lorentzian, so the fit carries a few-%
    # systematic — documented in the module docstring, pinned here.
    assert fit["theta_rad"] == pytest.approx(expected, rel=0.10)
    assert float(j_hz_from_theta(fit["theta_rad"], T_NS)) == pytest.approx(j_hz, rel=0.10)
    assert fit["x0"] == pytest.approx(0.10, abs=0.005)


def test_reports_the_principal_branch_past_a_full_swap():
    """theta > pi/2 folds back and is UNDER-reported — by design, not by bug."""
    j_hz = 1.4 * J_FULL_HZ
    true_theta = 2 * np.pi * j_hz * T_NS * 1e-9
    assert true_theta > np.pi / 2
    vq, y = _trace(j_hz, noise=0.005)
    fit = fit_swap_peak(vq, y)
    assert fit["success"] is True
    assert fit["theta_rad"] < np.pi / 2
    # it is the reflection pi - theta that comes back, which is what makes the
    # fold invisible without the caller's structural check
    assert fit["theta_rad"] == pytest.approx(np.pi - true_theta, rel=0.10)


def test_j_hz_needs_the_played_duration():
    assert np.isnan(j_hz_from_theta(np.pi / 2, None)).all()
    assert np.isnan(j_hz_from_theta(np.pi / 2, 0.0)).all()
    assert float(j_hz_from_theta(np.pi / 2, T_NS)) == pytest.approx(J_FULL_HZ)


def test_rejects_an_unresolved_peak():
    """A peak narrower than the grid is one sample, not a measurement."""
    # a 20x steeper detuning slope packs the whole peak inside one grid step
    vq, y = _trace(0.5 * J_FULL_HZ, j_ref_hz=2e6 * 20, noise=0.005)
    assert fit_swap_peak(vq, y)["success"] is False


def test_rejects_noise_and_flat_traces():
    vq = np.linspace(0.04, 0.16, 31)
    noise = np.random.default_rng(1).normal(0.02, 0.01, vq.shape)
    assert fit_swap_peak(vq, noise)["success"] is False
    assert fit_swap_peak(vq, np.full(vq.size, 0.3))["success"] is False


def test_degrades_without_raising():
    vq = np.linspace(0.04, 0.16, 31)
    for x, y in (
        (vq, np.full(vq.size, np.nan)),          # failed acquisition
        (np.arange(3.0), np.arange(3.0)),        # too few points
        (np.zeros(8), np.ones(8)),               # degenerate axis
    ):
        fit = fit_swap_peak(x, y)
        assert fit["success"] is False
        assert np.isnan(fit["theta_rad"])
        assert np.isnan(fit["best_fit"]).all()
        assert np.shape(fit["best_fit"]) == np.shape(x)


def test_best_fit_covers_the_full_axis_not_just_the_window():
    """The caller stores best_fit as one column of the map, so it must span x."""
    vq, y = _trace(0.5 * J_FULL_HZ, noise=0.005)
    fit = fit_swap_peak(vq, y, window_factor=1.0)
    assert fit["success"] is True
    assert fit["best_fit"].shape == vq.shape
    assert np.isfinite(fit["best_fit"]).all()
