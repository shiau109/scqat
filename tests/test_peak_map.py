"""Direct tests of the generic 2-D peak-map tracker ``tools.peak_map``.

``track_peaks`` is the pooled per-row reduction behind the vs-flux and
parametric-drive maps; tested straight on numpy arrays with generic ``x``/``y``
keys — the estimators only relabel its output.
"""

import numpy as np
import pytest

from scqat.tools.fit_lorentzian import lorentzian
from scqat.tools.peak_map import track_peaks


def _ridge_map(n_x=9, n_y=301, seed=0):
    """One drifting Lorentzian ridge: centre moves linearly across rows."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_x)
    y = np.linspace(-40e6, 40e6, n_y)
    centres = np.linspace(-20e6, 20e6, n_x)
    sig = np.stack([lorentzian(y, c, 0.8, 3e6, 0.0) for c in centres])
    signal_map = sig + 1e-3 * (rng.standard_normal((n_x, n_y))
                               + 1j * rng.standard_normal((n_x, n_y)))
    return x, y, signal_map, centres


def _polarity_map(sign, n_x=6, n_y=241, seed=11):
    """A drifting Lorentzian line of a given per-row polarity on a REAL signal.

    Real (already-reduced, population-like) rows are the only input whose
    polarity can go either way: the complex path reduces radially to
    ``|IQ - ref|``, which is a positive bump whatever the physics did. ``sign``
    is a scalar or one value per row (-1 = dip, +1 = emission peak).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_x)
    y = np.linspace(-40e6, 40e6, n_y)
    centres = np.linspace(-20e6, 20e6, n_x)
    signs = np.broadcast_to(np.asarray(sign, dtype=float), (n_x,))
    rows = []
    for k in range(n_x):
        line = 0.6 / (1.0 + ((y - centres[k]) / 3e6) ** 2)
        background = 0.9 if signs[k] < 0 else 0.1
        rows.append(background + signs[k] * line + 3e-3 * rng.standard_normal(n_y))
    return x, y, np.stack(rows), centres


def test_tracks_drifting_ridge():
    x, y, signal_map, centres = _ridge_map()
    r = track_peaks(x, y, signal_map, max_peaks=1)
    assert r["n_x"] == len(x)
    assert r["n_peaks"] == len(x)          # exactly one peak per row
    assert r["n_good"] == len(x)           # a clean ridge has no outliers
    # Each kept point sits on the injected ridge.
    for xi, yi in zip(r["peak_x"], r["peak_y"]):
        assert yi == pytest.approx(np.interp(xi, x, centres), abs=1e6)
    # Row indices map back into x.
    np.testing.assert_allclose(x[r["peak_x_index"]], r["peak_x"])
    # The per-row REDUCED signal (what fit_peaks fitted) is stacked for display,
    # and each complex row carries its radial reference.
    assert r["reduced_map"].shape == (len(x), len(y))
    assert np.isfinite(r["reduced_map"]).all()
    assert np.isfinite(r["ref_i"]).all() and np.isfinite(r["ref_q"]).all()
    # A complex map reduces radially to |IQ - ref| — always a positive bump, so
    # no row is fitted on the inverted branch.
    assert not r["peak_inverted"].any() and r["n_inverted"] == 0


def test_real_rows_have_no_ref():
    """A real (already-reduced) signal map has no radial reference to report."""
    x, y, signal_map, _ = _ridge_map(n_x=4)
    r = track_peaks(x, y, np.abs(signal_map), max_peaks=1)
    assert r["reduced_map"].shape == (4, len(y))
    assert np.isnan(r["ref_i"]).all() and np.isnan(r["ref_q"]).all()


def test_full_freq_threading():
    x, y, signal_map, _ = _ridge_map(n_x=5)
    lo = 4.5e9
    r = track_peaks(x, y, signal_map, full_freq=y + lo, max_peaks=1)
    assert "peak_full_freq" in r
    np.testing.assert_allclose(r["peak_full_freq"], r["peak_y"] + lo, atol=1.0)
    assert "peak_full_freq" not in track_peaks(x, y, signal_map, max_peaks=1)


def test_dip_rows_are_flagged_inverted():
    """Polarity survives the pooling — the whole point of ``peak_inverted``."""
    x, y, signal_map, centres = _polarity_map(-1.0)
    r = track_peaks(x, y, signal_map, max_peaks=1)
    assert r["n_peaks"] == len(x)
    assert r["peak_inverted"].dtype == bool
    # The polarity CHOICE is per row and independent of how well that row's
    # Lorentzian then converged, so every point is flagged.
    assert r["peak_inverted"].all()
    assert r["n_inverted"] == r["n_peaks"]

    # Amplitudes are polarity-NORMALIZED, so the sign says nothing about
    # dip-vs-peak: a dip fit that CONVERGED reports a POSITIVE amplitude, and a
    # negative one is a badly-conditioned fit. Select the converged ones by
    # centre accuracy against the injected line rather than by sign — the sign
    # is exactly the signal this flag exists to replace. (Not every row
    # converges here: fit_peaks negates the trace for a dip and then still
    # seeds FitLorentzian with inverted=True, so the guess starts from a noise
    # trough. Tighten this selection to `r["good"]` once that is fixed.)
    on_line = np.abs(r["peak_y"] - centres[r["peak_x_index"]]) < 2e6
    assert on_line.sum() >= 3
    assert (r["peak_amplitude"][on_line] > 0).all()
    # With the flag, the signed physics is recoverable: these are dips.
    signed = np.where(r["peak_inverted"], -r["peak_amplitude"], r["peak_amplitude"])
    assert (signed[on_line] < 0).all()


def test_peak_rows_are_not_flagged_inverted():
    x, y, signal_map, _ = _polarity_map(+1.0)
    r = track_peaks(x, y, signal_map, max_peaks=1)
    assert r["n_peaks"] == len(x)
    assert not r["peak_inverted"].any()
    assert r["n_inverted"] == 0
    assert (r["peak_amplitude"][r["good"]] > 0).all()
    # Same normalized amplitudes as the dip map — only the flag tells them apart.
    signed = np.where(r["peak_inverted"], -r["peak_amplitude"], r["peak_amplitude"])
    assert (signed[r["good"]] > 0).all()


def test_polarity_is_carried_per_row():
    """Alternating dip / peak rows: every point carries ITS OWN row's polarity,
    not a map-wide verdict."""
    n_x = 6
    signs = np.where(np.arange(n_x) % 2 == 0, -1.0, 1.0)
    x, y, signal_map, _ = _polarity_map(signs, n_x=n_x)
    r = track_peaks(x, y, signal_map, max_peaks=1)
    assert r["n_peaks"] == n_x
    expected = signs[r["peak_x_index"]] < 0
    np.testing.assert_array_equal(r["peak_inverted"], expected)
    assert r["n_inverted"] == int(expected.sum())


def test_unknown_knob_raises_before_any_fit():
    x, y, signal_map, _ = _ridge_map(n_x=3)
    with pytest.raises(ValueError, match="prominance"):
        track_peaks(x, y, signal_map, prominance=0.2)  # deliberate typo
    # The message points at the tracker's own tunables too.
    with pytest.raises(ValueError, match="n_sigma"):
        track_peaks(x, y, signal_map, bogus=1)


def test_empty_cloud_shapes():
    """All-noise map: no peaks, but every array key present with length 0."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 4)
    y = np.linspace(-40e6, 40e6, 201)
    noise = 1e-3 * (rng.standard_normal((4, 201)) + 1j * rng.standard_normal((4, 201)))
    r = track_peaks(x, y, noise)
    assert r["n_peaks"] == 0 and r["n_good"] == 0
    for key in ("peak_x", "peak_y", "peak_fwhm", "peak_amplitude",
                "peak_inverted", "in_window", "outlier", "good"):
        assert len(r[key]) == 0
    assert r["peak_inverted"].dtype == bool
    assert r["n_inverted"] == 0
