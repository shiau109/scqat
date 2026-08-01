"""Tests for the telegraph-PSD reduction (parity switching rate).

Synthetic Markov telegraph: from either state the per-step flip probability is
``rate * dt`` (per-direction rate), so the discrete autocorrelation is
``(1 - 2*p)^k ~ exp(-2*Gamma*dt*k)`` and the PSD knee sits at
``f_c = Gamma / pi`` — the tool must report ``parity_rate_hz == Gamma``.

Readout errors XOR the trace shot-by-shot: they inflate the transition count
(each error fakes two flips) but only raise the white floor of the PSD, so the
fitted rate must survive them — that immunity is why the PSD method is THE
method.
"""

import numpy as np
import pytest

from scqat.tools.telegraph_psd import (
    MAX_ODD_FRACTION,
    TELEGRAPH_PSD_KNOBS,
    fit_telegraph_psd,
    lorentzian_knee,
    validate_telegraph_psd_kwargs,
)

GAMMA = 50.0     # per-direction switching rate, Hz
DT = 1e-4        # shot period, s (10 kHz cadence)
N = 100_000


def _telegraph(rate_up_hz=GAMMA, rate_down_hz=None, dt_s=DT, n=N, seed=0,
               p_err=0.0):
    """Markov 0/1 telegraph; ``rate_down_hz=None`` means symmetric."""
    rng = np.random.default_rng(seed)
    if rate_down_hz is None or rate_down_hz == rate_up_hz:
        states = np.cumsum(rng.random(n) < rate_up_hz * dt_s) % 2
    else:
        p = (rate_up_hz * dt_s, rate_down_hz * dt_s)  # p[s] = flip prob in state s
        flips = rng.random(n)
        states = np.empty(n, dtype=np.int64)
        s = 0
        for i in range(n):
            states[i] = s
            if flips[i] < p[s]:
                s = 1 - s
    if p_err:
        states = states ^ (rng.random(n) < p_err)
    return states.astype(np.int8)


class TestTelegraphPsd:

    def test_knobs_declared(self):
        assert TELEGRAPH_PSD_KNOBS == {"nperseg", "window", "detrend"}
        validate_telegraph_psd_kwargs({"nperseg": 1024})
        with pytest.raises(ValueError):
            validate_telegraph_psd_kwargs({"npersegg": 1024})

    def test_symmetric_rate_recovery(self):
        res = fit_telegraph_psd(_telegraph(), DT)
        assert res["success"] is True
        assert res["method"] == "welch_lorentzian"
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.15)
        # convention: rate = pi * corner
        assert res["parity_rate_hz"] == pytest.approx(
            np.pi * res["psd_corner_hz"])
        assert res["p_excited"] == pytest.approx(0.5, abs=0.05)

    def test_readout_errors_raise_the_floor_not_the_rate(self):
        clean = fit_telegraph_psd(_telegraph(seed=1), DT)
        noisy = fit_telegraph_psd(_telegraph(seed=1, p_err=0.05), DT)
        assert noisy["success"] is True
        assert noisy["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.25)
        # each readout error fakes two transitions -> the count inflates ...
        assert noisy["n_transitions"] > 1.5 * clean["n_transitions"]
        # ... while the white floor absorbs them and the knee stays put
        assert noisy["psd_white_floor"] > 3 * clean["psd_white_floor"]

    def test_asymmetric_reports_the_mean_per_direction_rate(self):
        res = fit_telegraph_psd(
            _telegraph(rate_up_hz=80.0, rate_down_hz=20.0, seed=2), DT)
        assert res["success"] is True
        # corner = Gamma_up + Gamma_down -> reported rate = the mean (50 Hz)
        assert res["parity_rate_hz"] == pytest.approx(50.0, rel=0.25)
        assert res["p_excited"] == pytest.approx(0.8, abs=0.07)

    def test_flat_trace_fails_softly(self):
        res = fit_telegraph_psd(np.zeros(5000), DT)
        assert res["success"] is False
        assert np.isnan(res["parity_rate_hz"])
        assert res["n_transitions"] == 0
        assert res["p_excited"] == 0.0
        for key in ("psd_freq_hz", "psd", "psd_fit"):
            assert key in res  # arrays present even on failure

    def test_diagnostics_on_a_tiny_trace(self):
        res = fit_telegraph_psd(np.array([0, 0, 1, 1, 0, 1]), DT)
        assert res["n_transitions"] == 3
        assert res["p_excited"] == pytest.approx(0.5)
        assert res["success"] is False  # far too short to resolve a knee

    def test_nperseg_knob(self):
        res = fit_telegraph_psd(_telegraph(seed=3), DT, nperseg=4096)
        assert res["success"] is True
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)

    def test_bad_dt_raises(self):
        trace = _telegraph(n=1000)
        with pytest.raises(ValueError):
            fit_telegraph_psd(trace, 0.0)
        with pytest.raises(ValueError):
            fit_telegraph_psd(trace, float("nan"))
        with pytest.raises(ValueError):
            fit_telegraph_psd(trace, None)

    def test_model_function(self):
        f = np.array([0.0, 10.0])
        assert lorentzian_knee(f, 2.0, 10.0, 0.5)[0] == pytest.approx(2.5)
        assert lorentzian_knee(f, 2.0, 10.0, 0.5)[1] == pytest.approx(1.5)


class TestUnresolvedTraceIsRefused:
    """A trace whose consecutive shots are UNCORRELATED carries no rate.

    p_odd = (1 - exp(-2*Gamma*dt))/2 saturates at 0.5, so p_odd near 0.5 means
    the shots are independent and the spectrum is white — but curve_fit still
    lands a finite corner on white noise, so the Lorentzian cannot tell on its
    own. Motivated by a real 1e6-shot run that reported 535053 transitions
    (p_odd = 0.535, past the ceiling) alongside a confident-looking 1708 Hz,
    and PASSED the outcome gate — proposing that rate as a chip fact.
    """

    def test_uncorrelated_trace_fails_with_a_nan_rate(self):
        rng = np.random.default_rng(11)
        res = fit_telegraph_psd(rng.integers(0, 2, 60_000).astype(np.int8), DT)
        assert res["success"] is False
        assert np.isnan(res["parity_rate_hz"])
        assert res["p_odd"] == pytest.approx(0.5, abs=0.02)
        # still diagnosable: the corner and the arrays survive the refusal
        assert np.isfinite(res["psd_corner_hz"])
        assert res["psd_freq_hz"].size > 0

    def test_the_real_run_signature_is_refused(self):
        """p_odd slightly ABOVE 0.5 — past the Markov ceiling entirely."""
        rng = np.random.default_rng(12)
        trace = rng.integers(0, 2, 40_000).astype(np.int8)
        trace[::2] = 1 - trace[1::2][:trace[::2].size]  # nudge anti-correlated
        res = fit_telegraph_psd(trace, DT)
        assert res["p_odd"] > 0.5
        assert res["success"] is False

    def test_p_odd_matches_the_transition_count(self):
        trace = _telegraph(seed=13)
        res = fit_telegraph_psd(trace, DT)
        assert res["p_odd"] == pytest.approx(
            res["n_transitions"] / (trace.size - 1))

    def test_a_resolved_trace_still_passes(self):
        """The guard must not reject merely FAST switching that is still
        sampled well enough to fit."""
        res = fit_telegraph_psd(_telegraph(seed=14), DT)
        assert res["success"] is True
        assert res["p_odd"] < MAX_ODD_FRACTION
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)

    def test_just_under_the_threshold_still_passes(self):
        """A boundary, not a blanket rejection of fast switching.

        ``_telegraph`` flips with per-step probability ``rate * dt``, and for a
        Bernoulli flip chain that probability IS p_odd — so a target odd
        fraction is ``rate = target / dt``. (The chain's true Gamma is the
        continuous-time inversion ``-ln(1 - 2*p)/(2*dt)``, which is ~2x larger
        here; the two only coincide for small p, which is why the other
        recovery tests sit at p = 0.005.)
        """
        target = MAX_ODD_FRACTION - 0.05
        res = fit_telegraph_psd(
            _telegraph(rate_up_hz=target / DT, seed=15, n=200_000), DT)
        assert res["p_odd"] == pytest.approx(target, abs=0.02)
        assert res["success"] is True
        assert np.isfinite(res["parity_rate_hz"])
