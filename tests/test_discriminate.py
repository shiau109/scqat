"""Direct tests of the family-shared IQ discrimination ``tools.discriminate``.

The reduction behind state_discrimination / the readout-fidelity sweeps /
qubit-tomography training — tested straight on numpy arrays, no estimator
involved. Includes the degenerate-data case whose ragged ``direct_counts``
used to crash consumers indexing ``counts[1, 1]``.
"""

import numpy as np
import pytest

from scqat.tools.discriminate import (
    DISCRIMINATE_KNOBS,
    discriminate_states,
    validate_discriminate_kwargs,
)


def _two_blobs(n_shot=400, sep=6.0, sigma=1.0, mislabel=0.02, seed=0):
    """Two prepared states, two well-separated Gaussian blobs, a small
    mis-assigned fraction in each (realistic thermal/decay population)."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0]])
    I = np.empty((2, n_shot))
    Q = np.empty((2, n_shot))
    for s in range(2):
        other = rng.random(n_shot) < mislabel
        c = np.where(other[:, None], centers[1 - s], centers[s])
        I[s] = c[:, 0] + sigma * rng.standard_normal(n_shot)
        Q[s] = c[:, 1] + sigma * rng.standard_normal(n_shot)
    return I, Q, centers, sigma


def test_two_blob_training_recovers_truth():
    I, Q, centers, sigma = _two_blobs()
    r = discriminate_states(I, Q)

    mean = np.asarray(r["trained_paras"]["mean"])
    # Each true centre is identified by one trained centre (order-free match).
    # The coarse-histogram-argmax seed is refined onto each blob's density peak by
    # mean-shift, so the centre is now SUB-sigma accurate (was bin-limited ~1 sigma
    # when the seed was frozen). Symmetric Gaussian blobs => peak == truth.
    d = np.linalg.norm(mean[:, None, :] - centers[None, :, :], axis=-1)
    assert d.min(axis=0).max() < 0.4
    assert r["trained_paras"]["std"] == pytest.approx(sigma, rel=0.3)

    # Confusion diagonal ~ 1 - mislabel; labels/outliers per-shot shaped.
    counts = r["direct_counts"]
    assert counts.shape == (2, 2)
    order = np.argmin(d, axis=1)  # trained centre -> true state
    diag = [counts[s, int(np.flatnonzero(order == s)[0])] for s in range(2)]
    assert min(diag) > 0.9
    assert r["state_label"].shape == I.shape
    assert r["outlier_mask"].shape == I.shape
    # Few-percent outliers: the 2% cross-blob shots plus the >3-sigma tail
    # (inflated a little by the bin-limited pinned centre).
    assert np.all(r["outlier_probability"] < 0.10)
    assert np.all(np.isfinite(r["norm_res"]))


def _skewed_blobs(n_shot=2000, sep=6.0, sigma=1.0, tail_frac=0.25,
                  tail_shift=(0.0, 2.5), seed=0):
    """Two prepared states, each a dense Gaussian CORE at its centre plus a
    low-density TAIL offset perpendicular to the g->e axis (models T1 smear /
    |2> leakage). The core peak sits at the centre; the sample centroid is pulled
    toward the tail — so mode != centroid, the case that separates the two."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0]])
    I = np.empty((2, n_shot))
    Q = np.empty((2, n_shot))
    for s in range(2):
        is_tail = rng.random(n_shot) < tail_frac
        cx = centers[s, 0] + np.where(is_tail, tail_shift[0], 0.0)
        cy = centers[s, 1] + np.where(is_tail, tail_shift[1], 0.0)
        I[s] = cx + sigma * rng.standard_normal(n_shot)
        Q[s] = cy + sigma * rng.standard_normal(n_shot)
    return I, Q, centers, sigma


def test_mode_refine_finds_peak_not_tail():
    """A skewed blob (dense core + low-density tail): the trained centre must land
    on the core density PEAK (mode), not drift toward the tail like the centroid
    would. This is the fix for the coarse-argmax seed being frozen off the peak."""
    I, Q, centers, sigma = _skewed_blobs()
    r = discriminate_states(I, Q)
    mean = np.asarray(r["trained_paras"]["mean"])

    order = np.argmin(np.linalg.norm(mean[:, None, :] - centers[None, :, :], axis=-1), axis=0)
    for s in range(2):
        refined = mean[order[s]]
        centroid = np.array([I[s].mean(), Q[s].mean()])
        peak = centers[s]
        # mean-shift lands on the core peak...
        assert np.linalg.norm(refined - peak) < 0.4 * sigma
        # ...and is meaningfully closer to it than the tail-pulled centroid
        assert np.linalg.norm(refined - peak) < np.linalg.norm(centroid - peak)
        # sanity: the tail really does displace the centroid (mode != centroid here)
        assert abs(centroid[1] - peak[1]) > 0.4 * sigma


def test_direct_counts_fixed_shape_when_a_center_captures_nothing():
    """All shots sit in ONE blob while a pinned second centre is far away —
    the empty centre must appear as a ZERO COLUMN, not shrink the matrix
    (the ragged (2, 1) shape used to crash qubit_tomography's counts[1, 1])."""
    rng = np.random.default_rng(1)
    I = rng.standard_normal((2, 300)) * 0.5
    Q = rng.standard_normal((2, 300)) * 0.5
    r = discriminate_states(I, Q, user_mean=[[0.0, 0.0], [50.0, 50.0]], user_std=0.5)

    counts = r["direct_counts"]
    assert counts.shape == (2, 2)
    np.testing.assert_allclose(counts[:, 0], 1.0)   # everything on centre 0
    np.testing.assert_allclose(counts[:, 1], 0.0)   # empty centre = zero column
    assert counts[1, 1] == 0.0  # the exact read that used to IndexError


def test_user_pinned_model_skips_training():
    I, Q, centers, sigma = _two_blobs(seed=2)
    r = discriminate_states(I, Q, user_mean=centers.tolist(), user_std=sigma)
    np.testing.assert_allclose(np.asarray(r["trained_paras"]["mean"]), centers)
    assert r["trained_paras"]["std"] == sigma
    assert r["direct_counts"][0, 0] > 0.9
    assert r["direct_counts"][1, 1] > 0.9


def _ground_only(n_shot=4000, sep=6.0, sigma=1.0, population=0.03, seed=5):
    """ONE prepared state (the ground state) whose cloud carries a small
    residual population in the excited blob — the thermal-population case."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0]])
    excited = rng.random(n_shot) < population
    c = np.where(excited[:, None], centers[1], centers[0])
    I = (c[:, 0] + sigma * rng.standard_normal(n_shot))[None, :]
    Q = (c[:, 1] + sigma * rng.standard_normal(n_shot))[None, :]
    return I, Q, centers, sigma, float(np.mean(excited))


def test_pinned_centers_without_user_std_keep_all_centers():
    """Pinning MORE centres than there are prepared states: one ground-state
    cloud against a stored g/e reference. Every pinned centre must survive into
    ``trained_paras`` (unpacking n_state of them used to drop |e> silently), and
    both the counted and the fitted residual population must come back."""
    I, Q, centers, sigma, planted = _ground_only()
    r = discriminate_states(I, Q, user_mean=centers.tolist())   # NO user_std

    mean = np.asarray(r["trained_paras"]["mean"])
    assert mean.shape == (2, 2)
    np.testing.assert_allclose(mean, centers)
    # width comes from this run's own shots (MAD), not a pinned value
    assert r["trained_paras"]["std"] == pytest.approx(sigma, rel=0.3)

    assert r["direct_counts"].shape == (1, 2)
    assert r["gaussian_norms"].shape == (1, 2)
    assert r["direct_counts"][0, 1] == pytest.approx(planted, abs=0.015)
    assert r["gaussian_norms"][0, 1] == pytest.approx(planted, abs=0.015)


def test_validation_fails_loudly():
    with pytest.raises(ValueError, match="outlier_sgima"):
        validate_discriminate_kwargs({"outlier_sgima": 3})   # deliberate typo
    assert DISCRIMINATE_KNOBS == {"user_mean", "user_std", "outlier_sigma"}
    I, Q, _, _ = _two_blobs(n_shot=50, seed=3)
    with pytest.raises(TypeError):
        discriminate_states(I, Q, outlier_sgima=3)
