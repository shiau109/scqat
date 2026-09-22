"""Synthetic tests for the Trotter-chain compensation scan.

The dataset is the WHOLE three-qubit chain over (compensation amplitude, N). The
planted physics is the coherent round recursion the estimator's docstring states,

    S <- cos(t1) e^{i dPhi} S
    K <- cos(t2) K - sin(t1) sin(t2) S

so the sink accumulates from every round and the differential phase dPhi decides
whether those contributions add or cancel. That is what makes both discriminators
real in the fixture: at dPhi = 0 the sink peaks late and high, and at dPhi = pi
only the last round survives so it peaks at N = 1.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_trotter_compensation import QcTrotterCompensationEstimator
from scqat.estimators.qc_trotter_compensation.estimator import DENSE_AXIS, FIG_MAP, FIG_SCAN

QUBITS = ["q1", "q2", "q3"]
ROLES = {"source": "q1", "relay": "q2", "sink": "q3",
         "compensation_target": "q1"}

#: the angles measured on 5Q4C (source decay ratio and the N=1 sink rise).
THETA1, THETA2 = 0.62, 0.56

#: the intrinsic per-round phase error, and the Stark coefficient that cancels
#: it: dPhi(a) = PHI_ERR + STARK_K * a^2, zero at a = 0.5.
PHI_ERR, STARK_K = -1.0, 4.0


def _round_trace(dphi: float, n_max: int, decay: float = 0.01):
    """Populations vs N for one differential phase -- the coherent recursion."""
    a = np.cos(THETA1) * np.exp(1j * dphi)
    b = np.cos(THETA2)
    c = -np.sin(THETA1) * np.sin(THETA2)
    s, k = 1.0 + 0j, 0.0 + 0j
    source, sink = [abs(s) ** 2], [abs(k) ** 2]
    for _ in range(n_max):
        s, k = a * s, b * k + c * s        # K uses the OLD S, so assign together
        loss = np.exp(-decay * len(source))
        source.append(abs(s) ** 2 * loss)
        sink.append(abs(k) ** 2 * loss)
    return np.asarray(source), np.asarray(sink)


def _scan_ds(n_amp: int = 21, n_max: int = 12, amp_max: float = 1.0) -> xr.Dataset:
    amps = np.linspace(0.0, amp_max, n_amp)
    rounds = np.arange(0, n_max + 1)
    source = np.empty((n_amp, rounds.size))
    sink = np.empty_like(source)
    for i, amp in enumerate(amps):
        source[i], sink[i] = _round_trace(PHI_ERR + STARK_K * amp ** 2, n_max)
    # the relay is dumped every round, so it sits near zero at every amplitude
    relay = np.full_like(source, 0.009)
    pop = np.stack([source, relay, sink])
    return xr.Dataset(
        {"population": (("qubit", "compensation_amp", "round_count"), pop)},
        coords={"qubit": QUBITS, "compensation_amp": amps, "round_count": rounds},
    )


def test_finds_the_amplitude_that_nulls_the_phase():
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    est._check_data(ds)
    res = est.extract_parameters(ds, **ROLES)

    assert res["success"] is True
    # dPhi = 0 at a = 0.5; the grid step is 0.05
    assert res["best_compensation_amp"] == pytest.approx(0.5, abs=0.05)
    assert res["contrast"] > 2.0


def test_n_at_max_separates_constructive_from_destructive():
    """The second discriminator: cancelling rounds leave only the last one."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, **ROLES)

    amps = np.asarray(res["compensation_amp"])
    n_at_max = np.asarray(res["sink_n_at_max"])
    best = int(np.argmin(np.abs(amps - res["best_compensation_amp"])))
    worst = int(np.argmin(np.abs(amps - res["worst_compensation_amp"])))

    assert n_at_max[best] >= 4
    assert n_at_max[worst] <= 2
    assert res["best_n_at_max"] == n_at_max[best]


def test_reports_the_transport_curve_at_the_optimum():
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, **ROLES)

    curves = res["per_qubit_at_best"]
    assert set(curves) == set(QUBITS)
    assert len(curves["q3"]) == ds.sizes["round_count"]
    # the source starts excited and drains; the sink starts empty and fills
    assert curves["q1"][0] > 0.9 and curves["q1"][-1] < 0.3
    assert curves["q3"][0] == pytest.approx(0.0, abs=1e-9)
    assert max(curves["q3"]) == pytest.approx(res["best_sink_p_max"])


def test_a_missing_sink_is_refused_not_guessed():
    """Without the sink among the targets there is no transport to optimise."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, source="q1", relay="q2", sink="q9")

    assert res["success"] is False
    assert np.isnan(res["best_compensation_amp"])
    assert res["compensation_unresolved"] == 1.0
    assert "not among the chain qubits" in res["refine_reason"]
    # it must still be drawable
    plot_data = est.build_plot_data(ds, res, source="q1", relay="q2", sink="q9")
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}


def test_metadata_drops_the_bulky_maps_but_keeps_the_curves():
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, **ROLES)
    meta = est.extract_metadata(res)

    assert "_sink" not in meta and "_pop" not in meta
    assert "_fit_x" not in meta and "_fit_y" not in meta
    assert len(meta["sink_p_max"]) == ds.sizes["compensation_amp"]
    assert len(meta["sink_round_avg"]) == ds.sizes["compensation_amp"]
    assert meta["compensation_target"] == "q1"
    assert meta["sink"] == "q3"


def test_plot_data_is_netcdf_safe_and_carries_the_raw_map(tmp_path):
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, **ROLES)
    pd = est.build_plot_data(ds, res, **ROLES)

    assert pd["sink"].dims == ("compensation_amp", "round_count")
    assert pd["population"].dims == ("qubit", "compensation_amp", "round_count")
    assert pd["sink_round_avg"].dims == ("compensation_amp",)
    assert pd["round_avg_fit"].dims == (DENSE_AXIS,)
    for name in pd.data_vars:
        assert pd[name].dtype.kind in "fiu", name
    assert isinstance(pd.attrs["success"], int)
    assert isinstance(pd.attrs["compensation_unresolved"], int)
    pd.to_netcdf(tmp_path / "plotdata.nc")          # attrs included: no bool, no None


def test_refined_optimum_reads_the_ridge_centre():
    """The sink averaged over the rounds that interfere peaks on the phase null
    (dPhi = 0 at a = 0.5); the Stark phase is quadratic in the amplitude, so
    the peak leans by a few 1e-3."""
    est = QcTrotterCompensationEstimator()
    res = est.extract_parameters(_scan_ds(n_amp=41), **ROLES)

    assert res["compensation_unresolved"] == 0.0
    assert res["refine_reason"] == ""
    assert res["best_compensation_amp_refined"] == pytest.approx(0.5, abs=0.01)
    assert np.isfinite(res["best_compensation_amp_err"])


def test_one_bright_pixel_moves_the_argmax_but_not_the_refined_optimum():
    """The failure the refinement exists for: on 5Q4C the brightest pixel sat at
    0.367 while the ridge was centred at 0.33."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds(n_amp=41)
    amps = ds.compensation_amp.values
    hot = float(amps[np.argmin(np.abs(amps - 0.65))])
    ds["population"].loc[dict(qubit="q3", compensation_amp=hot, round_count=3)] = 0.99

    res = est.extract_parameters(ds, **ROLES)

    assert res["best_compensation_amp"] == pytest.approx(hot)
    assert res["best_compensation_amp_refined"] == pytest.approx(0.5, abs=0.01)


def test_the_spread_follows_round_to_round_noise_and_is_reproducible():
    est = QcTrotterCompensationEstimator()
    clean = est.extract_parameters(_scan_ds(n_amp=41), **ROLES)
    ds = _scan_ds(n_amp=41)
    noise = np.random.default_rng(3).normal(0.0, 0.03, ds["population"].shape)
    ds["population"] = ds["population"] + noise

    noisy = est.extract_parameters(ds, **ROLES)

    assert noisy["best_compensation_amp_err"] > clean["best_compensation_amp_err"]
    assert noisy["best_compensation_amp_refined"] == pytest.approx(0.5, abs=0.02)
    # the resampling is seeded: re-analysing a run reproduces its numbers
    again = est.extract_parameters(ds, **ROLES)
    assert again["best_compensation_amp_err"] == noisy["best_compensation_amp_err"]


def test_a_scan_that_misses_the_null_refuses_the_refined_optimum():
    """A window below the null has no interior maximum: NaN and the flag, never
    the scan's edge dressed up as an optimum -- and both figures still draw."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds(n_amp=41, amp_max=0.35)

    res = est.extract_parameters(ds, **ROLES)

    assert res["compensation_unresolved"] == 1.0
    assert np.isnan(res["best_compensation_amp_refined"])
    assert np.isnan(res["best_compensation_amp_err"])
    assert res["refine_reason"]
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert plot_data.attrs["compensation_unresolved"] == 1
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}


def test_the_refinement_needs_a_round_that_interferes():
    """Rounds 0 and 1 hold at most one path, so a scan that stops there carries
    no phase to refine."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds(n_max=1)

    res = est.extract_parameters(ds, **ROLES)

    assert res["compensation_unresolved"] == 1.0
    assert "no round >= 2" in res["refine_reason"]
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}


def test_figures_render_on_a_failed_acquisition():
    """An all-NaN scan must still produce BOTH figures, not zero of them."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    ds["population"] = ds["population"] * np.nan

    res = est.extract_parameters(ds, **ROLES)
    assert res["success"] is False
    assert np.isnan(res["best_compensation_amp"])
    assert np.isnan(res["best_compensation_amp_refined"])
    assert res["compensation_unresolved"] == 1.0
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}


def test_figures_render_from_plot_data_alone():
    """The replot path: no results dict, only a saved plot_data."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds()
    res = est.extract_parameters(ds, **ROLES)
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(None, {}, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}


def test_single_round_point_still_draws():
    """max_rounds=0 leaves one column; neither figure may fail on it."""
    est = QcTrotterCompensationEstimator()
    ds = _scan_ds(n_max=0)
    res = est.extract_parameters(ds, **ROLES)
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {FIG_SCAN, FIG_MAP}
