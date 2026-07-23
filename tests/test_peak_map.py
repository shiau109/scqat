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
                "in_window", "outlier", "good"):
        assert len(r[key]) == 0
