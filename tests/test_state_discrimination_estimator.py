"""The state-discrimination estimator's plot-data projection.

The fit itself is covered by ``test_discriminate.py`` (the shared reduction);
what this file pins is the estimator-level contract on top of it — chiefly the
shared axis window, which has to stay usable when one prepared state holds
nearly every shot.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.state_discrimination import StateDiscriminationEstimator

SEP, SIGMA = 5.0, 1.0
CENTERS = [[0.0, 0.0], [SEP, 0.0]]


def _cloud(populations, n_shot=4000, seed=0):
    """One row per prepared state; ``populations[k]`` = that row's |e> fraction."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(CENTERS)
    I = np.empty((len(populations), n_shot))
    Q = np.empty_like(I)
    for k, p_e in enumerate(populations):
        c = centers[(rng.random(n_shot) < p_e).astype(int)]
        I[k] = c[:, 0] + SIGMA * rng.standard_normal(n_shot)
        Q[k] = c[:, 1] + SIGMA * rng.standard_normal(n_shot)
    return xr.Dataset(
        {"I": (["prepared_state", "shot_idx"], I), "Q": (["prepared_state", "shot_idx"], Q)},
        coords={"prepared_state": np.arange(len(populations)), "shot_idx": np.arange(n_shot)},
    )


def _limits(plot_data):
    a = plot_data.attrs
    return (a["lim_I_low"], a["lim_I_high"]), (a["lim_Q_low"], a["lim_Q_high"])


def _window_holds_every_centre(plot_data, n_std=3):
    """Every trained centre AND its n-sigma circle inside the drawn frame."""
    (lo_I, hi_I), (lo_Q, hi_Q) = _limits(plot_data)
    mean = plot_data["trained_mean"].values
    r = n_std * plot_data.attrs["trained_std"]
    return bool(np.all((mean[:, 0] - r >= lo_I) & (mean[:, 0] + r <= hi_I)
                       & (mean[:, 1] - r >= lo_Q) & (mean[:, 1] + r <= hi_Q)))


def test_axis_window_holds_the_sparse_blob():
    """A ground-state-only cloud: 99% of the shots sit in ONE blob, so the pooled
    spread collapses to the blob WIDTH and a plain 5-sigma window crops the |e>
    blob — the very thing the run measures — off the edge. The window must be
    widened by the trained centres, or the figure hides the reported population."""
    est = StateDiscriminationEstimator()
    ds = _cloud([0.01])
    results = est.extract_parameters(ds, user_mean=CENTERS)
    plot_data = est.build_plot_data(ds, results)

    (lo_I, hi_I), _ = _limits(plot_data)
    pooled_high = float(ds["I"].values.mean() + 5 * ds["I"].values.std())
    assert pooled_high < SEP + 3 * SIGMA, "fixture no longer reproduces the crop"
    assert hi_I >= pooled_high, "limits may only grow"
    assert _window_holds_every_centre(plot_data)
    assert lo_I < 0.0  # the |g> blob keeps its own margin


def test_axis_window_unchanged_when_both_states_are_populated():
    """The widening is a floor, not a rewrite: a balanced two-state run is
    already comfortably framed by the pooled spread, and must keep that window."""
    est = StateDiscriminationEstimator()
    ds = _cloud([0.02, 0.97])
    results = est.extract_parameters(ds, user_mean=CENTERS)
    plot_data = est.build_plot_data(ds, results)

    (lo_I, hi_I), (lo_Q, hi_Q) = _limits(plot_data)
    I, Q = ds["I"].values, ds["Q"].values
    assert lo_I == pytest.approx(I.mean() - 5 * I.std())
    assert hi_I == pytest.approx(I.mean() + 5 * I.std())
    assert lo_Q == pytest.approx(Q.mean() - 5 * Q.std())
    assert hi_Q == pytest.approx(Q.mean() + 5 * Q.std())
    assert _window_holds_every_centre(plot_data)


def test_plot_data_shape_and_figures_survive_a_single_prepared_state():
    """The |g>-only case end to end: the projection keeps its (n_state, n_center)
    shapes with n_state=1, and every figure still draws."""
    est = StateDiscriminationEstimator()
    ds = _cloud([0.01])
    results = est.extract_parameters(ds, user_mean=CENTERS)
    plot_data = est.build_plot_data(ds, results)

    assert plot_data["direct_counts"].shape == (1, 2)
    assert plot_data["gaussian_norms"].shape == (1, 2)
    assert plot_data["trained_mean"].shape == (2, 2)
    np.testing.assert_allclose(plot_data["trained_mean"].values, CENTERS)

    figures = est.generate_figures(None, None, plot_data=plot_data)
    assert set(figures) == {"raw", "2DHist", "outliers", "fit_residue"}
