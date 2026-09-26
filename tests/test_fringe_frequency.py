"""Tests for tools.fringe_frequency - the frequency of one Ramsey-type fringe.

The reduction must recover the fringe frequency at the kHz level at a LARGE
virtual detuning (where tools.ramsey_fit can lock onto the wrong line), for any
fringe phase (no phase seed), and independently of the order the trace arrives in.
"""

import numpy as np
import pytest

from scqat.tools import fringe_frequency


def _trace(f_hz=4.0e6, tau_s=10e-6, phi=0.0, a=0.45, c=0.5, noise=0.01, n=201,
           t_max=4000e-9, seed=0):
    t = np.linspace(16e-9, t_max, n)
    rng = np.random.default_rng(seed)
    y = c + a * np.exp(-t / tau_s) * np.cos(2 * np.pi * f_hz * t + phi)
    return t, y + noise * rng.standard_normal(n)


@pytest.mark.parametrize("f_hz", [1.2e6, 4.0e6, 7.3e6])
def test_recovers_frequency_within_its_stderr(f_hz):
    t, y = _trace(f_hz=f_hz)
    r = fringe_frequency(t, y)
    assert r["success"] and r["method"] == "fit"
    assert abs(r["frequency_hz"] - f_hz) < max(5 * r["frequency_stderr_hz"], 2e3)
    assert r["frequency_stderr_hz"] < 5e3


@pytest.mark.parametrize("phi", [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
def test_any_fringe_phase(phi):
    """No phase seed: an anti-phase trace must not collapse the amplitude."""
    t, y = _trace(phi=phi)
    r = fringe_frequency(t, y)
    assert r["method"] == "fit"
    assert abs(r["frequency_hz"] - 4.0e6) < 2e3
    assert r["amplitude"] == pytest.approx(0.45, rel=0.1)


def test_order_does_not_matter():
    t, y = _trace()
    ref = fringe_frequency(t, y)
    perm = np.random.default_rng(1).permutation(t.size)
    for tt, yy in ((t[::-1], y[::-1]), (t[perm], y[perm])):
        r = fringe_frequency(tt, yy)
        assert r["frequency_hz"] == ref["frequency_hz"]
        assert r["frequency_stderr_hz"] == ref["frequency_stderr_hz"]


def test_noise_only_has_low_snr():
    rng = np.random.default_rng(3)
    t = np.linspace(16e-9, 4000e-9, 201)
    r = fringe_frequency(t, 0.5 + 0.01 * rng.standard_normal(t.size))
    assert r["snr"] < 30


def test_band_limits_the_search():
    t, y = _trace(f_hz=4.0e6)
    r = fringe_frequency(t, y, f_min=5e6, f_max=20e6)
    assert r["f_min_hz"] == 5e6
    assert r["frequency_hz"] >= 5e6 - 1.0 / (t[-1] - t[0])


def test_too_few_points_fails_cleanly():
    r = fringe_frequency(np.arange(5) * 1e-8, np.ones(5))
    assert r["success"] is False
    assert np.isnan(r["frequency_hz"])
