"""Tests for tools.local_arch - fringes to delta_f, then the local quadratic."""

import numpy as np
import pytest

from scqat.tools.local_arch import fit_local_arch, fringe_deltas

CURV = 1.52e10  # Hz/V^2, the arch opens downward
RAMP = -4e6


def test_local_arch_tools_recover_a_parabola():
    t = np.linspace(16e-9, 4000e-9, 201)
    xs = np.linspace(-12e-3, 12e-3, 7)
    delta = 20e3 - CURV * (xs - 1.5e-3) ** 2
    rows = 0.5 - 0.45 * np.cos(2 * np.pi * np.abs(RAMP + delta)[:, None] * t)
    d = fringe_deltas(t, rows, RAMP)
    assert d["valid"].all()
    assert d["delta_f_hz"] == pytest.approx(delta, abs=3e3)
    arch = fit_local_arch(xs, d["delta_f_hz"], d["delta_f_stderr_hz"], d["valid"],
                          ramp_detuning_hz=RAMP, span_s=d["span_s"])
    assert arch["fitted"] and arch["apex_not_bracketed"] == 0 and arch["fold_suspected"] == 0
    assert arch["apex_flux"] == pytest.approx(1.5e-3, abs=0.05e-3)
    assert arch["curvature_hz_per_v2"] == pytest.approx(-CURV, rel=0.02)
    short = fit_local_arch(xs, d["delta_f_hz"], d["delta_f_stderr_hz"],
                           np.arange(xs.size) < 4)
    assert short["fitted"] is False and short["apex_not_bracketed"] == 1
    with pytest.raises(ValueError, match="ramp_detuning_hz"):
        fringe_deltas(t, rows, 0.0)
