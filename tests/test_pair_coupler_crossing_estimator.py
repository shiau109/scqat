"""Synthetic tests for the coupler-crossing estimator.

The data come from the physics the estimator inverts: a hidden coupler arch, a
two-level hybridization of the coupler with each member (the member's level is
pushed by ``delta`` and its drive matrix element shrinks), and a fixed x180 at the
member's idle frequency (the Rabi formula). The numbers are 5Q4C's: members at
5.1446 / 4.8421 GHz, Ec = 0.2 GHz, a 2 mV grid over the default window.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_coupler_crossing import PairCouplerCrossingEstimator
from scqat.estimators.pair_coupler_crossing.estimator import (
    arch_frequency,
    solve_arch,
)

F_HIGH, F_LOW, EC = 5.1446e9, 4.8421e9, 0.2e9
LABELS = ["00", "01", "10", "11"]
PI_NS = 40.0


def _excitation(f_c, f_q, g_hz, prep=0.95):
    """P(excited) after an x180 at ``f_q`` with the coupler at ``f_c``."""
    delta_c = f_c - f_q
    root = np.sqrt(delta_c ** 2 + 4 * g_hz ** 2)
    shift = 0.5 * (delta_c - np.sign(delta_c) * root)   # the member is pushed away
    overlap = np.sqrt(0.5 * (1 + np.abs(delta_c) / root))
    rabi = overlap / (2 * PI_NS * 1e-9)
    omega = np.sqrt(rabi ** 2 + shift ** 2)
    return prep * rabi ** 2 / omega ** 2 * np.sin(np.pi * omega * PI_NS * 1e-9) ** 2


def _dataset(*, f_max=7.0e9, period=0.624, apex=-0.086, kind="apex", g_hz=50e6,
             x=None, measure="both", shots=300, seed=1, low_shift_v=0.0):
    """Joint populations over the coupler flux. ``kind='anti_apex'`` puts the
    arch's MINIMUM at ``apex`` instead (the coupler below both members);
    ``low_shift_v`` moves the low member's features only (a broken center)."""
    x = np.linspace(-0.45, 0.30, 376) if x is None else np.asarray(x, dtype=float)
    top = apex if kind == "apex" else apex + period / 2
    rng = np.random.default_rng(seed)
    p = {}
    for role, f_q, dx in (("high", F_HIGH, 0.0), ("low", F_LOW, low_shift_v)):
        f_c = arch_frequency(x - dx, f_max_hz=f_max, apex_v=top, period_v=period, ec_hz=EC)
        driven = measure in ("both", role)
        pe = _excitation(f_c, f_q, g_hz) if driven else np.zeros_like(x)
        p[role] = np.clip(pe + 0.01, 0, 1)
    p11 = p["high"] * p["low"]
    joint = np.stack([(1 - p["high"]) * (1 - p["low"]), (1 - p["high"]) * p["low"],
                      p["high"] * (1 - p["low"]), p11])
    joint = rng.binomial(shots, joint) / shots
    joint = joint / joint.sum(axis=0, keepdims=True)
    return xr.Dataset({"joint_population": (("joint_state", "coupler_flux_v"), joint)},
                      coords={"joint_state": LABELS, "coupler_flux_v": x})


def _fit(ds, **kwargs):
    kwargs.setdefault("f_high_hz", F_HIGH)
    kwargs.setdefault("f_low_hz", F_LOW)
    kwargs.setdefault("ec_hz", EC)
    return PairCouplerCrossingEstimator().extract_parameters(ds, **kwargs)


@pytest.mark.parametrize("kind", ["apex", "anti_apex"])
def test_solve_arch_inverts_the_model(kind):
    f_max, period = 7.3e9, 0.58
    inv = np.arccos if kind == "apex" else np.arcsin
    d = {f: period / np.pi * inv(((f + EC) / (f_max + EC)) ** 2) for f in (F_HIGH, F_LOW)}
    got = solve_arch(kind, d[F_HIGH], d[F_LOW], F_HIGH, F_LOW, EC)
    assert got == pytest.approx((f_max, period), rel=1e-9)


def test_apex_center_recovers_the_arch():
    r = _fit(_dataset())
    assert r["success"]
    assert r["center_kind"] == "apex" and r["apex_identified"] == 1
    assert r["center_from_idle_v"] == pytest.approx(-0.086, abs=2e-3)
    assert r["apex_from_idle_v"] == pytest.approx(-0.086, abs=2e-3)
    # q1's crossings at 5Q4C's predicted pulse amplitudes, q2's outside them
    assert r["crossing_high_lower_v"] == pytest.approx(-0.282, abs=3e-3)
    assert r["crossing_high_upper_v"] == pytest.approx(0.110, abs=3e-3)
    assert r["half_separation_low_v"] > r["half_separation_high_v"]
    assert not r["arch_unsolved"]
    # the model-dependent pair: within its own error budget (the plan's ~0.25 GHz, ~5 %)
    assert r["f_c_max_hz"] == pytest.approx(7.0e9, abs=max(3 * r["f_c_max_stderr_hz"], 0.3e9))
    assert r["period_v"] == pytest.approx(0.624, rel=0.06)
    assert 0 < r["f_c_max_stderr_hz"] < 1e9
    assert r["f_c_at_idle_hz"] > F_HIGH


def test_anti_apex_center_recovers_the_arch():
    r = _fit(_dataset(f_max=6.0e9, period=0.6, apex=0.02, kind="anti_apex"))
    assert r["success"]
    assert r["center_kind"] == "anti_apex"
    assert r["center_from_idle_v"] == pytest.approx(0.02, abs=2e-3)
    assert r["half_separation_low_v"] < r["half_separation_high_v"]
    assert not r["arch_unsolved"] and r["apex_identified"] == 1
    assert r["period_v"] == pytest.approx(0.6, rel=0.08)
    # the apex image nearest idle: 0.02 - 0.3
    assert r["apex_from_idle_v"] == pytest.approx(0.02 - 0.3, abs=0.05)
    assert r["f_c_at_idle_hz"] < F_LOW


def test_the_order_is_not_a_result(order_free):
    up = order_free(PairCouplerCrossingEstimator(), _dataset(), "coupler_flux_v",
                    f_high_hz=F_HIGH, f_low_hz=F_LOW, ec_hz=EC)
    assert up["center_kind"] == "apex"


def test_one_member_reports_the_symmetry_point_only():
    ds = _dataset(measure="high")
    r = _fit(ds, measure="high")
    assert r["success"]
    assert r["center_kind"] == "unknown" and not r["apex_identified"]
    assert r["arch_unsolved"] == 1
    assert r["center_from_idle_v"] == pytest.approx(-0.086, abs=2e-3)
    assert np.isnan(r["crossing_low_lower_v"]) and r["measured_low"] == 0

    told = _fit(ds, measure="high", coupler_side="above")
    assert told["center_kind"] == "apex" and told["apex_identified"] == 1
    assert told["apex_from_idle_v"] == pytest.approx(-0.086, abs=2e-3)
    assert told["arch_unsolved"] == 1   # one member: no period, no f_c_max


def test_a_window_that_misses_one_side_fails():
    r = _fit(_dataset(x=np.linspace(-0.45, -0.05, 201)))
    assert not r["success"]
    assert r["crossings_not_bracketed_high"] == 1 and r["crossings_not_bracketed_low"] == 1
    assert np.isnan(r["center_from_idle_v"])


def test_disagreeing_centers_fail():
    r = _fit(_dataset(low_shift_v=0.03))
    assert r["center_mismatch"] == 1 and not r["success"]
    assert r["center_kind"] == "unknown" and r["arch_unsolved"] == 1


def test_a_contradicted_side_fails():
    r = _fit(_dataset(), coupler_side="below")
    assert r["side_conflict"] == 1 and not r["success"]
    assert not r["apex_identified"] and r["arch_unsolved"] == 1


def test_unknown_kwargs_raise():
    with pytest.raises(ValueError, match="unknown kwargs"):
        _fit(_dataset(), dip_treshold=0.4)


def test_figures_render_on_a_failed_fit(tmp_path):
    flat = _dataset(x=np.linspace(0.2, 0.3, 51))   # no crossing in the window
    est = PairCouplerCrossingEstimator()
    results, figures = est.analyze(flat, output_dir=str(tmp_path),
                                   f_high_hz=F_HIGH, f_low_hz=F_LOW)
    assert not results["success"]
    assert set(figures) == {"crossings", "arch"}
    assert (tmp_path / "pair_coupler_crossing_metadata.json").exists()
    assert (tmp_path / "pair_coupler_crossing_plotdata.nc").exists()


def test_a_full_run_writes_every_artifact(tmp_path):
    est = PairCouplerCrossingEstimator()
    results, figures = est.analyze(_dataset(), output_dir=str(tmp_path),
                                   f_high_hz=F_HIGH, f_low_hz=F_LOW, ec_hz=EC,
                                   high_name="q1", low_name="q2")
    assert results["success"] and set(figures) == {"crossings", "arch"}
    plot = xr.open_dataset(tmp_path / "pair_coupler_crossing_plotdata.nc")
    assert plot.attrs["center_kind"] == "apex" and plot.attrs["high_name"] == "q1"
    assert np.isfinite(plot["arch_frequency_hz"].values).all()
    plot.close()
