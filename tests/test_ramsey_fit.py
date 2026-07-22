"""Direct tests of the family-shared per-trace Ramsey fit ``tools.ramsey_fit``.

The FFT-gate + BIC model-selection reduction behind ramsey / charge_gate_ramsey
— tested straight on numpy arrays, no estimator involved (the estimators are
thin wrappers and must agree with these numbers).
"""

import numpy as np
import pytest

from scqat.tools.ramsey_fit import RAMSEY_MODELS, fit_ramsey


def _single(t, f=3.0e6, tau=2.0e-6, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.5 * np.exp(-t / tau) * np.sin(2 * np.pi * f * t) + 0.5
    return y + 0.01 * rng.standard_normal(t.size)


def _beat(t, f1=2.0e6, f2=2.6e6, tau=4.0e-6, seed=0):
    rng = np.random.default_rng(seed)
    env = np.exp(-t / tau)
    y = 0.25 * env * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)) + 0.5
    return y + 0.01 * rng.standard_normal(t.size)


def _relaxation(t, tau=1.5e-6, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(-t / tau) + 0.01 * rng.standard_normal(t.size)


def test_single_model_recovers_frequency():
    t = np.linspace(0, 6e-6, 401)
    r = fit_ramsey(t, _single(t, f=3.0e6))
    assert r["model_type"] == "single"
    assert r["success"]
    assert r["f_1"] == pytest.approx(3.0e6, rel=0.05)
    assert "f_2" not in r  # single model carries no _2 keys
    assert r["best_fit"].shape == t.shape


def test_beat_model_recovers_two_frequencies():
    t = np.linspace(0, 12e-6, 601)
    r = fit_ramsey(t, _beat(t, f1=2.0e6, f2=2.6e6))
    assert r["model_type"] == "beat"
    freqs = sorted([abs(r["f_1"]), abs(r["f_2"])])
    assert freqs[0] == pytest.approx(2.0e6, rel=0.08)
    assert freqs[1] == pytest.approx(2.6e6, rel=0.08)


def test_relaxation_gated_when_no_fringe():
    t = np.linspace(0, 6e-6, 401)
    r = fit_ramsey(t, _relaxation(t))
    assert r["model_type"] == "relaxation"
    assert r["f_1"] == 0.0
    assert r["tau_1"] == pytest.approx(1.5e-6, rel=0.15)


def test_force_model_overrides_selection():
    t = np.linspace(0, 6e-6, 401)
    y = _single(t, f=3.0e6)
    assert fit_ramsey(t, y, force_model="relaxation")["model_type"] == "relaxation"
    assert fit_ramsey(t, y, force_model="single")["model_type"] == "single"


def test_force_model_rejects_unknown():
    t = np.linspace(0, 6e-6, 201)
    with pytest.raises(ValueError, match="force_model"):
        fit_ramsey(t, _single(t), force_model="quadruple")
    assert RAMSEY_MODELS == ("single", "beat", "relaxation")
