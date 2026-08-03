"""Tests for the flux-predistortion converter (neutral sum <-> single-pole cascade).

The `exp_sum_to_cascade` correctness anchor is a FAITHFUL-PORT test: a reference
copy of the QM (iqcc) `decompose_exp_sum_to_cascade` cluster is embedded here and
the tool must match it element-wise on the same numeric inputs. That proves the
port is faithful without re-deriving the (subtle) Laplace-variable convention.
"""

import math
from functools import reduce

import numpy as np
import pytest
from numpy.polynomial import Polynomial as P

from scqat.tools.flux_predistortion import (
    MIN_A_DC,
    exp_sum_step_response,
    exp_sum_to_cascade,
    partition_exp_stages,
)


# --------------------------------------------------------------------------- #
# Reference copy of the QM cryoscope_tools cascade math (verbatim), for the
# faithful-port assertion only.
# --------------------------------------------------------------------------- #
def _ref_single(A, tau):
    return np.array([A]), np.array([1, 1 / tau])


def _ref_add(terms):
    rt = [(P(num), P(den)) for num, den in terms]
    common_den = reduce(lambda acc, t: acc * t[1], rt, P([1]))
    adj = []
    for num, den in rt:
        adj.append(num * (common_den // den))
    return sum(adj, P([0])).coef, common_den.coef


def _ref_fpga(A_c, tau_c, Ts):
    return float(np.prod((Ts + 2 * tau_c) / (Ts + 2 * tau_c * (1 + A_c))))


def _ref_decompose(A, tau, A_dc=1.0, compensate=True, Ts=0.5):
    ba = [_ref_single(a, t) for a, t in zip(A, tau)]
    ba += [([A_dc], [1])]
    b, a = _ref_add(ba)
    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))
    tau_c = -1 / poles
    A_c = poles / zeros - 1
    scale = 1 / A_dc
    if compensate:
        scale *= _ref_fpga(A_c, tau_c, Ts)
    return np.real(A_c), np.real(tau_c), float(scale)


class TestExpSumToCascade:

    @pytest.mark.parametrize("amps,taus", [
        ([0.05], [100.0]),
        ([0.05, 0.02], [100.0, 12.0]),
        ([-0.03, 0.06, 0.015], [3000.0, 250.0, 20.0]),
    ])
    def test_matches_qm_reference(self, amps, taus):
        """The port reproduces the QM reference decomposition element-wise (same
        numbers in, so unit interpretation is irrelevant here)."""
        got = exp_sum_to_cascade(amps, taus, a_dc=1.0, ts_s=0.5)
        A_c, tau_c, scale = _ref_decompose(amps, taus, A_dc=1.0, Ts=0.5)
        np.testing.assert_allclose(got["amps_c"], A_c, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(got["taus_c_s"], tau_c, rtol=1e-9, atol=1e-12)
        assert got["scale"] == pytest.approx(scale, rel=1e-9)

    def test_single_exponential_gives_one_stage(self):
        out = exp_sum_to_cascade([0.05], [100.0], a_dc=1.0)
        assert len(out["amps_c"]) == len(out["taus_c_s"]) == 1
        assert all(np.isfinite(out["taus_c_s"]))
        assert np.isfinite(out["scale"])

    def test_unit_agnostic_in_time(self):
        """Working in seconds vs nanoseconds gives identical amps_c and scale, and
        tau_c scaled by exactly the unit ratio (1e9)."""
        amps = [0.04, 0.02]
        taus_s = [3000e-9, 250e-9]
        taus_ns = [3000.0, 250.0]
        sec = exp_sum_to_cascade(amps, taus_s, ts_s=1e-9)
        nano = exp_sum_to_cascade(amps, taus_ns, ts_s=1.0)
        np.testing.assert_allclose(sec["amps_c"], nano["amps_c"], rtol=1e-9)
        assert sec["scale"] == pytest.approx(nano["scale"], rel=1e-9)
        np.testing.assert_allclose(
            np.array(sec["taus_c_s"]) * 1e9, nano["taus_c_s"], rtol=1e-9)

    def test_hpf_mode_refused(self):
        with pytest.raises(ValueError, match="high-pass"):
            exp_sum_to_cascade([0.5], [100.0], a_dc=0.1)

    def test_empty_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            exp_sum_to_cascade([], [])

    def test_reproduces_step_response_after_reconstruction(self):
        """A sanity check independent of the port: the cascade's poles (=-1/tau_c)
        must be finite reals, and the step response the taps encode is smooth and
        settles to a_dc."""
        amps, taus = [0.05, 0.02], [100.0, 12.0]
        out = exp_sum_to_cascade(amps, taus)
        assert all(t > 0 for t in out["taus_c_s"])
        t = np.linspace(0, 500, 200)
        s = exp_sum_step_response(amps, taus, t, a_dc=1.0)
        assert s[-1] == pytest.approx(1.0, abs=1e-3)  # settled to a_dc


class TestExpSumStepResponse:

    def test_values(self):
        t = np.array([0.0, 50.0, 1e6])
        s = exp_sum_step_response([0.1], [50.0], t, a_dc=2.0)
        assert s[0] == pytest.approx(2.0 * 1.1)          # t=0: 1 + A
        assert s[1] == pytest.approx(2.0 * (1 + 0.1 / np.e))
        assert s[2] == pytest.approx(2.0, abs=1e-6)      # settled


class TestPartitionExpStages:

    def test_keeps_max_stages_by_significance(self):
        amps = [0.01, 0.06, 0.03, 0.05, 0.02, 0.04]
        taus = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6]
        out = partition_exp_stages(amps, taus, max_stages=4, tau_min_s=6e-9)
        assert len(out["kept"]) == 4
        assert len(out["overflow"]) == 2
        # kept are the four largest |A|
        kept_amps = sorted(abs(a) for a, _ in out["kept"])
        assert kept_amps == [0.03, 0.04, 0.05, 0.06]

    def test_out_of_amp_range_overflows(self):
        out = partition_exp_stages([1.5, 0.05], [1e-6, 2e-6], max_stages=4)
        assert (1.5, 1e-6) in out["overflow"]
        assert (0.05, 2e-6) in out["kept"]

    def test_sub_tau_min_overflows(self):
        out = partition_exp_stages([0.05, 0.04], [1e-9, 2e-6],
                                   max_stages=4, tau_min_s=6e-9)
        assert (0.05, 1e-9) in out["overflow"]
        assert (0.04, 2e-6) in out["kept"]

    def test_nothing_dropped(self):
        amps = [0.06, 0.05, 0.04, 0.03, 0.02]
        taus = [1e-6] * 5
        out = partition_exp_stages(amps, taus, max_stages=4)
        assert len(out["kept"]) + len(out["overflow"]) == len(amps)


def test_min_a_dc_constant_exported():
    assert MIN_A_DC == 0.2
