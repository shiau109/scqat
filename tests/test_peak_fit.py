"""Direct tests of the family-shared per-trace peak fit ``tools.peak_fit``.

The reduction behind qubit_spectroscopy and every "qubit line vs <axis>" map —
tested straight on numpy arrays, no estimator involved (the estimators are
covered by their own suites and must agree with these numbers, since they are
thin wrappers).
"""

import inspect

import numpy as np
import pytest

from scqat.tools.fit_lorentzian import lorentzian
from scqat.tools.peak_fit import (
    PEAK_KNOBS,
    _window_height,
    fit_peaks,
    robust_noise,
    validate_peak_kwargs,
)


def _trace(peaks, n=801, span=100e6, noise=1e-3, seed=0, complex_signal=True):
    """One synthetic spectrum trace: ``peaks`` = list of (x0, amplitude, gamma)."""
    rng = np.random.default_rng(seed)
    detuning = np.linspace(-span / 2, span / 2, n)
    sig = np.zeros(n)
    for x0, amp, gamma in peaks:
        sig += lorentzian(detuning, x0, amp, gamma, 0.0)
    if complex_signal:
        signal = (sig + noise * rng.standard_normal(n)
                  + 1j * noise * rng.standard_normal(n))
    else:
        signal = sig + noise * rng.standard_normal(n)
    return detuning, signal


def test_knobs_frozenset_matches_signature():
    """PEAK_KNOBS is the single source of truth callers validate against — it
    must equal fit_peaks' keyword-only parameters exactly."""
    kw_only = {
        name for name, p in inspect.signature(fit_peaks).parameters.items()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    }
    assert PEAK_KNOBS == frozenset(kw_only)


def test_single_peak_complex_and_real_agree():
    detuning, iq = _trace([(10e6, 0.8, 3e6)])
    r_c = fit_peaks(detuning, iq)
    # Real input: the already-|IQ-ref|-like magnitude signal, used as-is.
    r_r = fit_peaks(detuning, np.abs(iq - r_c["ref_iq"]))

    for r in (r_c, r_r):
        assert len(r["peaks"]) == 1
        pk = r["peaks"][0]
        assert pk["detuning"] == pytest.approx(10e6, abs=0.5e6)
        assert pk["fwhm"] == pytest.approx(2 * 3e6, rel=0.15)
    # Provenance of the signal convention: complex input reports its IQ
    # reference, real input reports None.
    assert isinstance(r_c["ref_iq"], complex)
    assert r_r["ref_iq"] is None
    assert r_c["peaks"][0]["detuning"] == pytest.approx(
        r_r["peaks"][0]["detuning"], abs=0.2e6
    )


def test_noise_only_finds_no_peaks():
    rng = np.random.default_rng(3)
    detuning = np.linspace(-50e6, 50e6, 801)
    iq = 1e-3 * (rng.standard_normal(801) + 1j * rng.standard_normal(801))
    assert fit_peaks(detuning, iq)["peaks"] == []


def test_robust_noise_is_unbiased_on_pure_noise():
    """The min_snr gate exists to reject noise-only traces, so its sigma must be
    the real sigma — this is the property the estimator may not trade away for
    immunity to wide lines."""
    rng = np.random.default_rng(0)
    for sigma in (1e-3, 3e-2):
        est = np.mean([robust_noise(rng.normal(0, sigma, 51)) for _ in range(400)])
        assert est == pytest.approx(sigma, rel=0.06)


def test_robust_noise_ignores_the_line_however_wide():
    """A line is smooth, so a POINT-TO-POINT estimator cannot see it. The value
    spread it replaced could: a line filling much of the window made
    ``1.4826 * MAD(values)`` measure the LINE, and min_snr * sigma then exceeded
    the whole signal span (5Q4C q1, 2026-08-19 — three drive envelopes at one
    setting; the two whose line filled 41-45% of the window returned NO peak
    while carrying 0.89 contrast at 2.5x lower noise than the one that passed)."""
    detuning = np.linspace(-15e6, 15e6, 51)
    sigma = 0.02
    rng = np.random.default_rng(1)
    noise = rng.normal(0, sigma, 51)
    for gamma in (2e6, 8e6):  # narrow line, then one filling most of the window
        trace = lorentzian(detuning, 0.0, 0.9, gamma, 0.0) + noise
        assert robust_noise(trace) == pytest.approx(sigma, rel=0.6)


def test_wide_line_is_still_detected():
    """The regression: a clean, high-contrast line filling ~half the sweep must be
    found. Widening the line is a legitimate experimental choice (a shorter or
    smoother drive pulse), and it must not silently return zero peaks."""
    detuning = np.linspace(-15e6, 15e6, 51)
    rng = np.random.default_rng(2)
    trace = lorentzian(detuning, 1.5e6, 0.9, 7e6, 0.0) + rng.normal(0, 0.02, 51)
    peaks = fit_peaks(detuning, trace, max_peaks=1)["peaks"]
    assert len(peaks) == 1
    assert peaks[0]["detuning"] == pytest.approx(1.5e6, abs=0.6e6)


def test_two_peaks_and_max_peaks_cap():
    detuning, iq = _trace([(-30e6, 0.8, 3e6), (25e6, 0.5, 3e6)])
    assert len(fit_peaks(detuning, iq)["peaks"]) == 2
    capped = fit_peaks(detuning, iq, max_peaks=1)["peaks"]
    assert len(capped) == 1
    # The cap keeps the larger-area line.
    assert capped[0]["detuning"] == pytest.approx(-30e6, abs=0.5e6)


def test_full_freq_interp_post_step():
    lo = 4.5e9
    detuning, iq = _trace([(10e6, 0.8, 3e6)])
    r = fit_peaks(detuning, iq, full_freq=detuning + lo)
    pk = r["peaks"][0]
    assert pk["full_freq"] == pytest.approx(pk["detuning"] + lo, abs=1.0)
    # Without the axis the key is absent, not NaN.
    assert "full_freq" not in fit_peaks(detuning, iq)["peaks"][0]


def test_validation_fails_loudly():
    with pytest.raises(ValueError, match="prominance"):
        validate_peak_kwargs({"prominance": 0.2})   # deliberate typo
    # The pure function's own signature rejects unknown knobs natively.
    detuning, iq = _trace([(0.0, 0.8, 3e6)])
    with pytest.raises(TypeError):
        fit_peaks(detuning, iq, prominance=0.2)


def test_descending_axis_fits_the_same_line():
    """Regression: the width bound was ``detuning[-1] - detuning[0]``, negative on
    an axis swept high -> low, and a 4 MHz line came back ~180 MHz wide with no
    flag. The same (x, y) pairs in either order must give the same peak."""
    detuning, iq = _trace([(7e6, 0.8, 2e6)], n=201, noise=2e-2, seed=1,
                          complex_signal=False)
    up = fit_peaks(detuning, iq)["peaks"]
    down = fit_peaks(detuning[::-1], iq[::-1])["peaks"]
    assert len(up) == len(down) == 1
    assert up[0]["fwhm"] == pytest.approx(4e6, rel=0.1)
    for key in ("detuning", "fwhm", "amplitude"):
        assert down[0][key] == pytest.approx(up[0][key], rel=1e-6)
    # the absolute centre too - the interp post-step sorts its table
    lo = 4.5e9
    up_f = fit_peaks(detuning, iq, full_freq=detuning + lo)["peaks"][0]["full_freq"]
    down_f = fit_peaks(detuning[::-1], iq[::-1],
                       full_freq=(detuning + lo)[::-1])["peaks"][0]["full_freq"]
    assert down_f == pytest.approx(up_f, abs=1.0)


def test_auto_polarity_keeps_two_equal_lines_as_peaks():
    """Regression: two lines of equal height. In the inverted trace a noise point
    between them has both lines as its bases, so its PROMINENCE rivals theirs, and
    the old rule picked the dip side and returned no line (seed 8; 8 of 60 seeds).
    Compared by excursion from the baseline, that point is noise."""
    detuning, sig = _trace([(-20e6, 0.3, 2e6), (20e6, 0.3, 2e6)], n=401, noise=1e-2,
                           seed=8, complex_signal=False)
    for polarity in ("auto", "peak"):
        r = fit_peaks(detuning, sig, polarity=polarity)
        assert not r["inverted"]
        assert [p["detuning"] for p in r["peaks"]] == [
            pytest.approx(-20e6, abs=0.2e6), pytest.approx(20e6, abs=0.2e6)]
    for seed in range(60):
        detuning, sig = _trace([(-20e6, 0.3, 2e6), (20e6, 0.3, 2e6)], n=401,
                               noise=1e-2, seed=seed, complex_signal=False)
        assert not fit_peaks(detuning, sig)["inverted"], seed
    assert fit_peaks(-detuning, -sig, polarity="dip")["inverted"]
    with pytest.raises(ValueError, match="polarity"):
        fit_peaks(detuning, sig, polarity="up")


def _dip_trace(dips, seed, n=201, span=100e6, noise=0.01, level=0.85):
    """A real trace of ``dips`` = (x0, depth, gamma) below ``level``."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-span / 2, span / 2, n)
    y = level + noise * rng.standard_normal(n)
    for x0, depth, gamma in dips:
        y = y - lorentzian(x, x0, depth, gamma, 0.0)
    return x, y


@pytest.mark.parametrize("fit_window_factor", [5.0, 2.0])
def test_a_dip_is_fitted_on_its_own_centre(fit_window_factor):
    """Regression: the dip trace was negated and THEN fitted with inverted=True,
    whose guess seeds x0 at the window's minimum - on the negated trace a noise
    point on the shoulder. Seed 0 came back at +24.5 MHz with amplitude -0.049 and
    24 of 40 seeds were wrong, 3 empty. A dip is fitted as a peak of the negated
    trace and reports a POSITIVE amplitude."""
    for seed in range(40):
        x, y = _dip_trace([(5e6, 0.25, 2.5e6)], seed)
        r = fit_peaks(x, y, fit_window_factor=fit_window_factor)
        assert r["inverted"], seed
        assert len(r["peaks"]) == 1, seed
        pk = r["peaks"][0]
        assert pk["detuning"] == pytest.approx(5e6, abs=0.3e6), seed
        assert pk["amplitude"] == pytest.approx(0.25, abs=0.04), seed
        assert pk["fwhm"] == pytest.approx(5e6, rel=0.25), seed


def test_two_equal_dips_are_both_fitted():
    """The mirror of the two-equal-lines case, through the dip path (the old
    seeding got both dips right in 17 of 100 seeds)."""
    for seed in range(40):
        x, y = _dip_trace([(-20e6, 0.3, 2e6), (20e6, 0.3, 2e6)], seed, n=401,
                          level=0.9)
        r = fit_peaks(x, y)
        assert r["inverted"], seed
        assert [p["detuning"] for p in r["peaks"]] == [
            pytest.approx(-20e6, abs=0.3e6), pytest.approx(20e6, abs=0.3e6)], seed
        assert all(p["amplitude"] > 0 for p in r["peaks"]), seed


def test_every_reported_fit_clears_the_noise_gate():
    """A fit whose own amplitude is below min_snr sigma is not a line. A weak hump
    passes the prominence gate on the noise riding it and then fits broad and
    sub-noise: with the merge off, the old code reported such a fit in 6 of these
    30 seeds (seed 17: 0.016 x 24 MHz against a 0.061 gate)."""
    for seed in range(30):
        detuning, sig = _trace([(20e6, 0.26, 1.3e6), (-60e6, 0.02, 25e6)], n=501,
                               span=500e6, noise=0.01, seed=seed,
                               complex_signal=False)
        r = fit_peaks(detuning, sig, merge_factor=0)
        sigma = robust_noise(r["signal_corrected"])
        assert any(abs(p["detuning"] - 20e6) < 1e6 for p in r["peaks"]), seed
        assert all(p["amplitude"] >= 6.0 * sigma for p in r["peaks"]), seed


def test_a_broad_weak_fit_never_swallows_a_narrow_line():
    """Regression: the merge kept the larger-AREA of two overlapping fits, so a
    broad fit of a weak hump - or one wider than its own window, whose amplitude
    trades freely against its offset - swallowed a narrow line beside it (5Q4C
    coupler swap, 2026-09-27: 0.015 x 178 MHz beat 0.26 x 2.6 MHz; a 0.92 x 401 MHz
    fit took two real lines). Here the old merge lost the line in 2 of 40 seeds
    (seeds 25 and 30); the taller fit now survives."""
    for seed in range(40):
        detuning, sig = _trace([(20e6, 0.26, 1.3e6), (-40e6, 0.03, 75e6)], n=501,
                               span=500e6, noise=0.01, seed=seed,
                               complex_signal=False)
        peaks = fit_peaks(detuning, sig)["peaks"]
        assert any(abs(p["detuning"] - 20e6) < 1e6 and p["fwhm"] < 6e6
                   for p in peaks), seed


def test_window_height_is_the_rise_inside_the_window():
    """``height`` = the fit's rise inside its own window: the amplitude for a line
    the window resolves, a sliver of it for a fit far wider than the window."""
    assert _window_height(0.0, 0.3, 1e6, -50e6, 50e6) == pytest.approx(0.3, rel=1e-3)
    wide = _window_height(0.0, 16.0, 500e6, -15e6, 15e6)
    assert wide == pytest.approx(16.0 * 0.0009 / 1.0009)
    assert _window_height(0.0, -0.2, 1e6, -50e6, 50e6) < 0
    # a fit reports it
    detuning, iq = _trace([(10e6, 0.8, 3e6)])
    pk = fit_peaks(detuning, iq)["peaks"][0]
    assert pk["height"] == pytest.approx(pk["amplitude"], rel=0.02)
