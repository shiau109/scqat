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
    MIN_PSD_CONTRAST,
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
        assert res["p_high"] == pytest.approx(0.5, abs=0.05)

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
        assert res["p_high"] == pytest.approx(0.8, abs=0.07)

    def test_flat_trace_fails_softly(self):
        res = fit_telegraph_psd(np.zeros(5000), DT)
        assert res["success"] is False
        assert np.isnan(res["parity_rate_hz"])
        assert res["n_transitions"] == 0
        assert res["p_high"] == 0.0
        for key in ("psd_freq_hz", "psd", "psd_fit"):
            assert key in res  # arrays present even on failure

    def test_diagnostics_on_a_tiny_trace(self):
        res = fit_telegraph_psd(np.array([0, 0, 1, 1, 0, 1]), DT)
        assert res["n_transitions"] == 3
        assert res["p_high"] == pytest.approx(0.5)
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


class TestTheKneeGate:
    """What decides a trustworthy fit is the PLATEAU-TO-FLOOR contrast, not the
    switch fraction.

    On uncorrelated data the spectrum is white and ``curve_fit`` still returns a
    finite corner, so the Lorentzian cannot self-diagnose. ``A/B`` can: measured
    ~1e-9 on white data against 1e3-1e6 on real telegraphs — eight orders of
    magnitude of separation.

    This replaced a threshold on ``p_switch``, which was the wrong quantity: any
    shot-to-shot noise drives it toward 0.5 whether or not a knee exists. A real
    chipA run with a clean fit (A/B ~ 7800, corner 0.749 Hz) reported
    p_switch = 0.294 against the old 0.40 ceiling — within 1.4x of a false
    refusal. See ``test_the_real_chipa_run_still_passes``.
    """

    def test_uncorrelated_trace_fails_with_a_nan_rate(self):
        rng = np.random.default_rng(11)
        res = fit_telegraph_psd(rng.integers(0, 2, 60_000).astype(np.int8), DT)
        assert res["success"] is False
        assert np.isnan(res["parity_rate_hz"])
        # refused on CONTRAST: the fitter drives the plateau to nothing
        assert res["psd_contrast"] < MIN_PSD_CONTRAST
        # still diagnosable: the corner and the arrays survive the refusal
        assert np.isfinite(res["psd_corner_hz"])
        assert res["psd_freq_hz"].size > 0

    def test_anticorrelated_trace_is_refused(self):
        rng = np.random.default_rng(12)
        trace = rng.integers(0, 2, 40_000).astype(np.int8)
        trace[::2] = 1 - trace[1::2][:trace[::2].size]  # nudge anti-correlated
        res = fit_telegraph_psd(trace, DT)
        assert res["success"] is False

    def test_p_switch_matches_the_transition_count(self):
        trace = _telegraph(seed=13)
        res = fit_telegraph_psd(trace, DT)
        assert res["p_switch"] == pytest.approx(
            res["n_transitions"] / (trace.size - 1))

    def test_a_resolved_trace_passes_with_high_contrast(self):
        res = fit_telegraph_psd(_telegraph(seed=14), DT)
        assert res["success"] is True
        assert res["psd_contrast"] > 1e3
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)

    def test_a_telegraph_buried_in_noise_still_passes(self):
        """THE case the old p_switch gate got wrong. Flipping 30% of the
        samples drives p_switch to ~0.3 while leaving the knee intact — the fit
        must survive, because the noise is white and the knee is not."""
        rng = np.random.default_rng(16)
        clean = _telegraph(rate_up_hz=2.35, seed=16, n=200_000)
        noisy = (clean ^ (rng.random(200_000) < 0.30)).astype(np.int8)
        res = fit_telegraph_psd(noisy, DT)
        assert res["p_switch"] > 0.25          # would have failed the old gate
        assert res["success"] is True
        assert res["psd_contrast"] > MIN_PSD_CONTRAST

    def test_the_real_chipa_run_still_passes(self):
        """Rebuilt from the run that motivated all of this
        (20260801-191415-048-chipA-qubit_parity_switch-01): rate 2.352 Hz,
        corner 0.7488 Hz, A/B ~ 7800, p_switch 0.294 at dt = 30.379 us. It is a
        good measurement and must never be refused."""
        dt = 3.0379e-05
        rng = np.random.default_rng(17)
        n = 1_000_000
        parity = (np.cumsum(rng.random(n) < 2.352 * dt) % 2).astype(np.int8)
        # ~60% of the parity variance was white in that run
        noisy = (parity ^ (rng.random(n) < 0.147)).astype(np.int8)
        res = fit_telegraph_psd(noisy, dt)
        assert res["success"] is True
        assert res["parity_rate_hz"] == pytest.approx(2.352, rel=0.4)
        assert res["p_switch"] > 0.25          # the old gate's near-miss
        assert res["psd_contrast"] > 1e2


class TestSpectralReach:
    """``psd_freq_min_hz`` is what a record length buys, and the reason the
    experiment is parameterized by record TIME rather than shot count."""

    def test_lowest_bin_tracks_the_record_length(self):
        short = fit_telegraph_psd(_telegraph(n=50_000, seed=18), DT)
        long_ = fit_telegraph_psd(_telegraph(n=400_000, seed=18), DT)
        # 8x the record -> 8x lower reach (Welch defaults to 8 segments)
        assert long_["psd_freq_min_hz"] == pytest.approx(
            short["psd_freq_min_hz"] / 8.0, rel=0.05)

    def test_margins_are_reported(self):
        res = fit_telegraph_psd(_telegraph(seed=19), DT)
        assert res["corner_margin_low"] == pytest.approx(
            res["psd_corner_hz"] / res["psd_freq_min_hz"])
        assert res["psd_freq_max_hz"] == pytest.approx(0.5 / DT, rel=0.01)

    def test_a_thin_plateau_is_reported_but_NOT_refused(self):
        """A corner close to the lowest bin means a longer record would help —
        it does not mean the measurement is invalid, so it must still pass."""
        # ~4 Hz corner with a short record puts the corner near f_min
        res = fit_telegraph_psd(
            _telegraph(rate_up_hz=12.0, n=30_000, seed=20), DT)
        assert res["corner_margin_low"] < 5.0
        assert res["success"] is True


class TestMappingFidelity:
    """``A`` and ``B`` are the reference model's ``4F^2`` and ``(1-F^2)dt``
    terms in different variables, so the SAME three-parameter fit yields the
    sequence mapping fidelity F twice over — no refit, just a change of
    variables. See the tool's "reference parameterization" docstring section.
    """

    def test_the_two_estimates_agree_on_a_clean_telegraph(self):
        res = fit_telegraph_psd(_telegraph(n=400_000, seed=31), DT)
        assert res["mapping_fidelity"] == pytest.approx(1.0, abs=0.05)
        assert res["mapping_fidelity_floor"] == pytest.approx(1.0, abs=0.05)
        assert res["mapping_fidelity_ratio"] == pytest.approx(1.0, abs=0.08)

    @pytest.mark.parametrize("p_err", [0.002, 0.01, 0.05])
    def test_floor_estimate_tracks_F_equals_one_minus_two_eps(self, p_err):
        """On a DIRECTLY sampled telegraph the mapping error really is white,
        and then the floor reads F back essentially exactly."""
        res = fit_telegraph_psd(
            _telegraph(n=400_000, seed=32, p_err=p_err), DT)
        assert res["mapping_fidelity_floor"] == pytest.approx(
            1.0 - 2.0 * p_err, abs=0.01)

    def test_derived_from_the_fit_not_recomputed(self):
        """Both numbers must be exactly the documented functions of (A, f_c, B)
        — if they ever drift into a separate estimate the two would stop being
        an independent cross-check of the same fit."""
        res = fit_telegraph_psd(_telegraph(n=200_000, seed=33), DT)
        a, fc, b = (res["psd_amplitude"], res["psd_corner_hz"],
                    res["psd_white_floor"])
        assert res["mapping_fidelity"] == pytest.approx(
            np.sqrt(2.0 * np.pi * fc * a))
        assert res["mapping_fidelity_floor"] == pytest.approx(
            np.sqrt(1.0 - 2.0 * b / DT))
        # and the rate is the same fit's corner, times pi
        assert res["parity_rate_hz"] == pytest.approx(np.pi * fc)

    def test_power_budget_closes(self):
        """The parameterization is only self-consistent if the two fitted terms
        integrate to the series variance: Lorentzian -> F^2/4, floor ->
        (1-F^2)/4. This is what pins the factor-of-2 conventions (0/1 vs +-1,
        one-sided vs two-sided) that the F formulas depend on."""
        series = _telegraph(n=400_000, seed=34)
        res = fit_telegraph_psd(series, DT)
        lorentz = res["psd_amplitude"] * res["psd_corner_hz"] * np.pi / 2.0
        floor = res["psd_white_floor"] / (2.0 * DT)
        assert lorentz + floor == pytest.approx(float(np.var(series)), rel=0.1)

    def test_a_floor_over_budget_is_NaN_not_a_clamp(self):
        """B > dt/2 exceeds the total noise power the model allows for ANY
        fidelity. That is a broken model, not a low F — say so with NaN rather
        than silently reporting 0."""
        # white noise: no knee, and the fitter puts everything in the floor
        rng = np.random.default_rng(35)
        res = fit_telegraph_psd(
            (rng.random(100_000) < 0.5).astype(np.int8), DT * 4.0)
        assert res["success"] is False
        if res["psd_white_floor"] > 2.0 * DT:
            assert np.isnan(res["mapping_fidelity_floor"])
            assert np.isnan(res["mapping_fidelity_ratio"])

    def test_reported_on_a_failed_fit_too(self):
        """Like every other tier-1 key, present (as NaN) even when the fit is
        refused, so a failure stays diagnosable."""
        res = fit_telegraph_psd(np.zeros(1000, dtype=np.int8), DT)
        for key in ("mapping_fidelity", "mapping_fidelity_floor",
                    "mapping_fidelity_ratio"):
            assert key in res
