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
from scqat.tools.peak_fit import PEAK_KNOBS, fit_peaks, validate_peak_kwargs


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
