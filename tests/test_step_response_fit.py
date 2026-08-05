"""Tests for the multi-exponential step-response fit (cryoscope taps).

Synthetic settled step response ``a_dc + sum_i A_i*exp(-t/tau_i)`` — the shape a
flux line's bias-tee + wiring transients produce. The tool must recover the
components (in the unit of the passed time axis — ns and seconds both exercised,
the tool is unit-agnostic by design), honour ``fixed_taus`` and a pinned
``a_dc``, and degrade to an honest ``success=False`` on degenerate input rather
than raising.

Regime note: the sequential fit is exact once the constant term ``a_dc`` is
known (the estimator that consumes this tool tail-normalizes its input so the
settled level is ~1 and pins it). The rolling-variance tail ESTIMATE of ``a_dc``
is only unbiased when the record is many time-constants long — a smooth,
still-decaying tail reads as "flat" — so the auto-``a_dc`` path is tested in
that settled regime, and the precise-recovery tests pin ``a_dc``.
"""

import numpy as np
import pytest

from scqat.tools.step_response_fit import (
    MIN_SAMPLES,
    fit_step_response,
    mpm_tau_seeds,
    sequential_exp_fit,
)

START_FRACTIONS = [0.5, 0.08]


def _step(t, components, a_dc=1.0, noise=0.0, seed=0):
    y = a_dc + sum(amp * np.exp(-t / tau) for amp, tau in components)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, t.size)
    return y


class TestFitStepResponse:

    def test_two_component_recovery_pinned_dc(self):
        """With the settled level known, both components come back exactly."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(1.0, 401.0)
        res = fit_step_response(
            t, _step(t, comps, noise=1e-4), START_FRACTIONS, a_dc=1.0,
        )
        assert res["success"] is True
        (amp_slow, tau_slow), (amp_fast, tau_fast) = res["components"]
        assert tau_slow == pytest.approx(100.0, rel=0.05)
        assert amp_slow == pytest.approx(0.08, rel=0.05)
        assert tau_fast == pytest.approx(25.0, rel=0.1)
        assert amp_fast == pytest.approx(0.03, rel=0.1)
        assert res["rms"] < 1e-3
        assert np.allclose(res["best_fit"], _step(t, comps), atol=2e-3)

    def test_degenerate_pair_collapses_to_one_component(self):
        """A ``t*exp(-t/tau)``-like response tempts a two-exponential model into
        a giant cancelling pair on near-identical taus (seen at +-22 on real
        ramsey-cryoscope hardware). The guard collapses to ONE bounded
        component instead."""
        t = np.arange(1.0, 101.0)
        y = 1.0 - 0.35 * (t / 8.0) * np.exp(-t / 8.0)  # not a pure exp sum
        res = fit_step_response(t, y, [0.5, 0.01], a_dc=1.0)
        assert len(res["components"]) == 1  # collapsed, not a degenerate pair
        (amp, tau) = res["components"][0]
        assert abs(amp) < 1.0  # no +-22 nonsense
        assert res["success"] is True
        assert len(res["best_fractions"]) == 1

    def test_well_separated_components_are_not_collapsed(self):
        """The collapse must not fire on a genuine two-component response
        (tau ratio 4 — comfortably above degen_tau_ratio)."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(1.0, 401.0)
        res = fit_step_response(
            t, _step(t, comps, noise=1e-4), START_FRACTIONS, a_dc=1.0,
        )
        assert len(res["components"]) == 2
        assert res["success"] is True

    def test_record_starting_far_after_the_transient_no_blowup(self):
        """A record whose first sample is many tau after the transient (the
        +-22866 regression: min_wait 40 ns, tau ~3 ns) must not mint giant
        re-referenced amplitudes — the tau floor keeps exp(t[0]/tau) <= e^2."""
        t = np.arange(40.0, 240.0)  # t[0] = 40, transient tau = 3 (unseen)
        y = _step(t, [(-0.07, 3.0)], noise=2e-3)  # settled + noise, in effect
        res = fit_step_response(t, y, [0.5, 0.05], a_dc=1.0)
        assert all(abs(amp) <= 2.0 for amp, _ in res["components"])

    def test_amp_max_is_respected(self):
        """No component's t=0-referenced amplitude may exceed amp_max."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(1.0, 401.0)
        res = fit_step_response(
            t, _step(t, comps, noise=1e-4), START_FRACTIONS, a_dc=1.0,
            amp_max=0.5,
        )
        assert all(abs(amp) <= 0.5 for amp, _ in res["components"])

    def test_same_sign_close_pair_is_not_treated_as_degenerate(self):
        """The collapse is CANCELLATION-aware: a same-sign close-tau pair
        merely splits one component's amplitude — the criterion must not fire
        on it, and the joint fit represents it honestly (possibly merged into
        one tap carrying the SUMMED amplitude — equivalent for any consumer)."""
        from scqat.tools.step_response_fit import _degenerate_pair

        assert _degenerate_pair([(0.04, 40.0), (0.03, 30.0)], 1.5) is None
        # ... while an opposite-sign cancelling pair at the same ratio fires
        assert _degenerate_pair([(5.0, 40.0), (-4.9, 30.0)], 1.5) is not None

        t = np.arange(1.0, 201.0)
        y = _step(t, [(0.04, 40.0), (0.03, 30.0)], noise=5e-4)  # ratio 1.33
        res = fit_step_response(t, y, [0.5], a_dc=1.0, tau_seeds=[40.0, 30.0])
        assert res["success"] is True
        assert 1 <= len(res["components"]) <= 2
        total = sum(a for a, _ in res["components"])
        assert total == pytest.approx(0.07, rel=0.15)  # summed amplitude kept
        assert all(abs(a) <= 2.0 for a, _ in res["components"])

    def test_single_component_auto_dc_settled_record(self):
        """Over many time-constants the tail estimate is unbiased, so the
        auto-a_dc path recovers a lone component accurately."""
        comps = [(0.08, 20.0)]  # 20 time-constants across the record
        t = np.arange(1.0, 401.0)
        res = fit_step_response(t, _step(t, comps), [0.3])
        assert res["success"] is True
        assert res["a_dc"] == pytest.approx(1.0, abs=1e-3)
        (amp, tau), = res["components"]
        assert tau == pytest.approx(20.0, rel=0.05)
        assert amp == pytest.approx(0.08, rel=0.05)

    def test_unit_agnostic_identical_fit_in_seconds(self):
        """The same samples on a seconds axis return proportional taus and
        identical amplitudes — the fit is scale-invariant (sample-normalized)."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t_ns = np.arange(1.0, 401.0)
        y = _step(t_ns, comps, noise=1e-4)
        res_ns = fit_step_response(t_ns, y, START_FRACTIONS, a_dc=1.0)
        res_s = fit_step_response(t_ns * 1e-9, y, START_FRACTIONS, a_dc=1.0)
        assert res_s["success"] is True
        for (amp_ns, tau_ns), (amp_s, tau_s) in zip(
                res_ns["components"], res_s["components"]):
            assert tau_s == pytest.approx(tau_ns * 1e-9, rel=1e-6)
            assert amp_s == pytest.approx(amp_ns, rel=1e-6)

    def test_amplitudes_referenced_to_t_zero(self):
        """Components reproduce y on the CALLER's axis even when the record
        starts well after t = 0 (the internal fit runs on t - t[0])."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(20.0, 421.0)
        res = fit_step_response(t, _step(t, comps), START_FRACTIONS, a_dc=1.0)
        assert res["success"] is True
        rebuilt = res["a_dc"] + sum(
            amp * np.exp(-t / tau) for amp, tau in res["components"]
        )
        assert np.allclose(rebuilt, _step(t, comps), atol=1e-3)

    def test_fixed_taus_path(self):
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(1.0, 401.0)
        res = fit_step_response(
            t, _step(t, comps, noise=1e-4), START_FRACTIONS,
            fixed_taus=[100.0, 25.0], a_dc=1.0,
        )
        assert res["success"] is True
        (amp_slow, tau_slow), (amp_fast, tau_fast) = res["components"]
        assert tau_slow == 100.0 and tau_fast == 25.0
        assert amp_slow == pytest.approx(0.08, rel=0.05)
        assert amp_fast == pytest.approx(0.03, rel=0.1)

    def test_fixed_taus_length_mismatch_refused(self):
        t = np.arange(1.0, 401.0)
        with pytest.raises(ValueError, match="same length"):
            fit_step_response(
                t, _step(t, [(0.08, 100.0)]), START_FRACTIONS, fixed_taus=[10.0],
            )

    def test_non_descending_fractions_refused(self):
        t = np.arange(1.0, 401.0)
        with pytest.raises(ValueError, match="DESCENDING"):
            fit_step_response(t, _step(t, [(0.08, 100.0)]), [0.08, 0.5])

    def test_degenerate_input_fails_without_raising(self):
        t = np.arange(1.0, float(MIN_SAMPLES))  # one sample short
        res = fit_step_response(t, np.ones_like(t), START_FRACTIONS)
        assert res["success"] is False
        assert res["components"] == []
        assert np.isnan(res["rms"])

    def test_nan_input_fails_without_raising(self):
        t = np.arange(1.0, 401.0)
        y = _step(t, [(0.08, 100.0)])
        y[10] = np.nan
        res = fit_step_response(t, y, START_FRACTIONS)
        assert res["success"] is False


class TestMpmTauSeeds:

    def test_two_component_order_and_taus(self):
        """MPM finds the model order and both taus from the data alone."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        t = np.arange(1.0, 401.0)
        res = mpm_tau_seeds(t, _step(t, comps, noise=5e-4), a_dc=1.0)
        assert len(res["taus"]) == 2
        assert res["taus"][0] == pytest.approx(100.0, rel=0.3)
        assert res["taus"][1] == pytest.approx(25.0, rel=0.3)
        assert res["oscillatory"] is False

    def test_pure_noise_yields_few_seeds(self):
        t = np.arange(1.0, 201.0)
        y = 1.0 + np.random.default_rng(3).normal(0.0, 3e-3, t.size)
        res = mpm_tau_seeds(t, y, a_dc=1.0)
        assert len(res["taus"]) <= 2  # no spurious high-order model

    def test_ringing_sets_the_oscillatory_flag(self):
        """A damped cosine is a complex-pole pair — not representable as real
        decaying exponentials; the flag is the 'consider FIR' signal."""
        t = np.arange(0.0, 200.0)
        y = 1.0 + 0.05 * np.cos(2 * np.pi * 0.05 * t) * np.exp(-t / 40.0)
        res = mpm_tau_seeds(t, y, a_dc=1.0)
        assert res["oscillatory"] is True

    def test_non_uniform_axis_refused_by_name(self):
        t = np.logspace(0, 2, 60)
        with pytest.raises(ValueError, match="UNIFORM"):
            mpm_tau_seeds(t, np.ones_like(t), a_dc=1.0)


class TestJointTauSeededFit:

    def test_fast_plus_slow_pair_recovered(self):
        """The real-data regime the sequential fitter misses: a fast (2 ns)
        undershoot beneath a slow (18 ns) one. MPM seeds + the joint bounded
        fit recover both with sane amplitudes."""
        comps = [(-0.04, 18.0), (-0.06, 2.0)]
        t = np.arange(1.0, 81.0)
        y = _step(t, comps, noise=1e-3)
        seeds = mpm_tau_seeds(t, y, a_dc=1.0)["taus"]
        res = fit_step_response(t, y, [0.5], a_dc=1.0, tau_seeds=seeds)
        assert res["success"] is True
        assert len(res["components"]) == 2
        (a_slow, tau_slow), (a_fast, tau_fast) = res["components"]
        assert tau_slow == pytest.approx(18.0, rel=0.3)
        assert tau_fast == pytest.approx(2.0, rel=0.5)
        assert a_slow == pytest.approx(-0.04, rel=0.3)
        assert res["best_fractions"] == []  # the seeded path uses no fractions

    def test_t_exp_shape_still_collapses(self):
        """A t*exp(-t/tau) response tempts the joint fit into a cancelling
        close pair too — the collapse must fire on the seeded path as well."""
        t = np.arange(1.0, 101.0)
        y = 1.0 - 0.35 * (t / 8.0) * np.exp(-t / 8.0)
        res = fit_step_response(t, y, [0.5], a_dc=1.0, tau_seeds=[9.0, 7.0])
        assert len(res["components"]) == 1
        assert abs(res["components"][0][0]) <= 2.0

    def test_tau_seeds_and_fixed_taus_are_mutually_exclusive(self):
        t = np.arange(0.0, 100.0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            fit_step_response(t, np.ones_like(t), [0.5], a_dc=1.0,
                              fixed_taus=[10.0], tau_seeds=[10.0])

    def test_empty_tau_seeds_refused(self):
        t = np.arange(0.0, 100.0)
        with pytest.raises(ValueError, match="non-empty"):
            fit_step_response(t, np.ones_like(t), [0.5], a_dc=1.0,
                              tau_seeds=[])


class TestSequentialExpFit:

    def test_flat_tail_dc_estimate_settled_record(self):
        comps = [(0.08, 20.0)]
        t = np.arange(1.0, 401.0)
        components, a_dc, residual = sequential_exp_fit(t, _step(t, comps), [0.3])
        assert a_dc == pytest.approx(1.0, abs=1e-3)
        assert len(components) == 1
        assert float(np.sqrt(np.mean(residual ** 2))) < 1e-3

    def test_pinned_a_dc_is_used_verbatim(self):
        t = np.arange(1.0, 401.0)
        _, a_dc, _ = sequential_exp_fit(
            t, _step(t, [(0.08, 100.0)]), START_FRACTIONS, a_dc=1.005,
        )
        assert a_dc == 1.005

    def test_taus_returned_in_caller_units(self):
        """A component fitted on a seconds axis returns its tau in seconds."""
        comps = [(0.08, 100.0)]
        t_ns = np.arange(1.0, 401.0)
        components, _, _ = sequential_exp_fit(
            t_ns * 1e-9, _step(t_ns, comps), [0.3], a_dc=1.0,
        )
        (_, tau_s), = components
        assert tau_s == pytest.approx(100e-9, rel=0.05)
