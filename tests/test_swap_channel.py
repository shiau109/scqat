"""The repeated-swap channel: closed forms, the fit, and the phase root."""

import numpy as np
import pytest

from scqat.tools import swap_channel as sc


def test_ideal_channel_rotates_about_x_at_zero_phase():
    theta = 0.23
    states = sc.channel_states([theta, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 12)
    f = sc.features_of(states)
    n = np.arange(13)
    np.testing.assert_allclose(f[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(f[:, 1], -np.sin(2 * n * theta), atol=1e-12)
    np.testing.assert_allclose(f[:, 2] - f[:, 3], np.cos(2 * n * theta), atol=1e-12)
    np.testing.assert_allclose(f[:, 2] + f[:, 3], 1.0, atol=1e-12)


def test_population_only_sees_the_composite_angle():
    """cos(omega/2) = cos(theta) cos(phi/2): the transfer oscillates at omega."""
    theta, phi = 0.2, 1.1
    f = sc.features_of(sc.channel_states([theta, phi, 0.0, 0, 0, 0, 0], 40))
    omega = 2 * np.arccos(np.cos(theta) * np.cos(phi / 2))
    n = np.arange(41)
    # the down population (transfer) of a rotation by omega about a tilted axis
    n_perp2 = (np.sin(theta) ** 2) / np.sin(omega / 2) ** 2
    np.testing.assert_allclose(f[:, 3], n_perp2 * np.sin(n * omega / 2) ** 2, atol=1e-12)


def test_low_member_excited_mirrors_the_high_one():
    p = [0.3, 0.4, 0.0, 0.01, 0.02, 0.03, 0.0]
    hi = sc.features_of(sc.channel_states(p, 8, excite_high=True))
    lo = sc.features_of(sc.channel_states(p, 8, excite_high=False))
    np.testing.assert_allclose(hi[0, 2], 1.0)
    np.testing.assert_allclose(lo[0, 3], 1.0)


@pytest.mark.parametrize("truth", [
    [0.126, 1.3, 0.4, 0.0166, 0.0182, 0.049, 0.03],
    [0.40, -0.6, -1.0, 0.02, 0.01, 0.06, 0.02],
    [0.20, 0.0, 0.0, 0.015, 0.015, 0.04, 0.0],
])
def test_fit_recovers_noiseless_parameters(truth):
    counts = np.arange(0, 16)
    f = sc.channel_features(truth, counts)
    fit = sc.fit_swap_channel(counts, f)
    assert fit["success"]
    for name, value in zip(sc.PARAM_NAMES, truth):
        assert fit[name] == pytest.approx(value, abs=2e-4), name
    assert fit["rms"] < 1e-6


def test_fit_recovers_noisy_parameters_within_errors():
    rng = np.random.default_rng(5)
    truth = [0.15, 0.9, 0.2, 0.017, 0.018, 0.05, 0.03]
    counts = np.arange(0, 21)
    f = sc.channel_features(truth, counts) + rng.normal(0, 0.015, (counts.size, 5))
    fit = sc.fit_swap_channel(counts, f)
    assert fit["success"]
    for name in ("theta", "phi", "lam"):
        value = truth[sc.PARAM_NAMES.index(name)]
        assert abs(fit[name] - value) < 4 * fit[f"{name}_err"] + 1e-3, name


def test_fit_fails_gracefully_on_too_few_counts():
    fit = sc.fit_swap_channel([0, 1, 2], np.zeros((3, 5)))
    assert fit["success"] is False
    assert np.isnan(fit["theta"]) and fit["model"].shape == (3, 5)


def test_phase_root_inside_and_extrapolated():
    root, extra = sc.phase_root([0.4, 0.45, 0.5], [0.3, 0.05, -0.25])
    assert not extra and root == pytest.approx(0.4583, abs=2e-3)
    root, extra = sc.phase_root([0.1, 0.2], [0.5, 0.3])
    assert extra and root == pytest.approx(0.35)
    # wrapped phases across +-pi still give one smooth crossing
    phases = np.angle(np.exp(1j * (2 * np.pi * (np.array([0.40, 0.45, 0.5]) ** 2 - 0.45 ** 2)
                                   + 2 * np.pi)))
    root, extra = sc.phase_root([0.40, 0.45, 0.5], phases)
    assert not extra and root == pytest.approx(0.45, abs=1e-6)
    assert np.isnan(sc.phase_root([0.4], [0.1])[0])
