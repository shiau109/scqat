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
