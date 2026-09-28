"""Synthetic-chain tests for the unidirectional-Trotter transport estimator.

The dataset is the WHOLE three-qubit chain (no per-target split), so the tests
pin both projections: the per-qubit transport curves that are always present,
and the joint distribution that only a shot-mode run carries.
"""

import json

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_unidirectional_trotter import QcUnidirectionalTrotterEstimator
from scqat.estimators.qc_unidirectional_trotter.estimator import (
    FIG_JOINT,
    FIG_POPULATIONS,
    ideal_transport,
)

QUBITS = ["q1", "q2", "q3"]
ROLES = {"source": "q1", "relay": "q2", "sink": "q3"}
#: 3-qubit basis labels, digit order = QUBITS order.
LABELS = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]


def _chain_ds(n_max: int = 20, joint: bool = False) -> xr.Dataset:
    """A cascade: the source decays, the sink fills, the relay stays reset."""
    n = np.arange(0, n_max + 1)
    p_source = 0.95 * np.exp(-n / 5.0)
    p_sink = 0.85 * (1.0 - np.exp(-n / 5.0))
    p_relay = np.full(n.shape, 0.04)
    pop = np.stack([p_source, p_relay, p_sink])
    ds = xr.Dataset(
        {"population": (("qubit", "round_count"), pop)},
        coords={"qubit": QUBITS, "round_count": n},
    )
    if joint:
        # A single excitation living on the source or the sink, plus a small
        # ground-state remainder — normalized over the 8 basis states.
        probs = {"100": p_source, "001": p_sink}
        rows = [probs.get(label, np.zeros(n.shape)) for label in LABELS]
        rows[LABELS.index("000")] = np.clip(1.0 - p_source - p_sink, 0.0, 1.0)
        jp = np.stack(rows)
        jp = jp / jp.sum(axis=0, keepdims=True)
        ds["joint_population"] = (("joint_state", "round_count"), jp)
        ds = ds.assign_coords(joint_state=LABELS)
    return ds


def test_summarizes_each_chain_qubit():
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    est._check_data(ds)
    res = est.extract_parameters(ds, **ROLES)

    assert res["qubits"] == QUBITS
    assert res["n_round_count"] == 21 and res["max_round_count"] == 20.0
    assert res["has_joint"] is False
    # the source starts excited and ends near empty; the sink does the reverse
    assert res["per_qubit"]["q1"]["p_initial"] == pytest.approx(0.95)
    assert res["per_qubit"]["q1"]["p_final"] < 0.05
    assert res["per_qubit"]["q3"]["p_initial"] == pytest.approx(0.0, abs=1e-9)
    assert res["per_qubit"]["q3"]["p_final"] > 0.8
    # the relay is emptied every round, so it never carries the excitation
    assert res["per_qubit"]["q2"]["p_max"] < 0.1
    # the transport headline is the sink's peak — what SCQO's min_transfer reads
    assert res["sink_p_max"] == pytest.approx(res["per_qubit"]["q3"]["p_max"])
    assert res["sink_n_at_max"] == 20.0


def test_axis_order_is_irrelevant():
    """Estimators transpose by coordinate NAME, so a caller may pass any order."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    flipped = ds.transpose("round_count", "qubit")
    # compared as JSON: the results carry NaN (no angles given), and NaN != NaN
    assert json.dumps(est.extract_parameters(flipped, **ROLES), sort_keys=True) == (
        json.dumps(est.extract_parameters(ds, **ROLES), sort_keys=True))


def test_joint_is_summarized_and_plotted_only_when_present():
    est = QcUnidirectionalTrotterEstimator()

    marginal_only = est.build_plot_data(_chain_ds(), {}, **ROLES)
    assert "joint_population" not in marginal_only.data_vars
    assert marginal_only.attrs["has_joint"] == 0
    assert set(est.generate_figures(None, None, plot_data=marginal_only)) == {
        FIG_POPULATIONS
    }

    ds = _chain_ds(joint=True)
    est._check_data(ds)
    res = est.extract_parameters(ds, **ROLES)
    assert res["has_joint"] is True
    # the excitation is only ever on the source or the sink
    assert res["joint_p_max"]["100"] > 0.9
    assert res["joint_p_max"]["001"] > 0.8
    assert res["joint_p_max"]["111"] == pytest.approx(0.0, abs=1e-9)

    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert plot_data.attrs["has_joint"] == 1
    assert list(plot_data["joint_state"].values) == LABELS
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


def test_plot_data_roundtrips_through_netcdf(tmp_path):
    """Artifacts must reload with zero re-fit — so no attr may be non-netCDF."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(joint=True)
    plot_data = est.build_plot_data(ds, est.extract_parameters(ds, **ROLES), **ROLES)
    path = tmp_path / "plotdata.nc"
    plot_data.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        assert reloaded.attrs["source"] == "q1"
        figs = est.generate_figures(None, None, plot_data=reloaded.load())
    assert set(figs) == {FIG_POPULATIONS, FIG_JOINT}


def test_figures_render_on_a_failed_acquisition():
    """An all-NaN chain must still produce BOTH figures, not zero of them."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(joint=True)
    ds["population"] = ds["population"] * np.nan
    ds["joint_population"] = ds["joint_population"] * np.nan

    res = est.extract_parameters(ds, **ROLES)
    assert np.isnan(res["per_qubit"]["q1"]["p_max"])
    assert np.isnan(res["sink_p_max"])
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert set(est.generate_figures(ds, res, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


def test_single_round_point_still_draws():
    """max_rounds=0 leaves one column; the joint map must not fail on it."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(n_max=0, joint=True)
    plot_data = est.build_plot_data(ds, est.extract_parameters(ds, **ROLES), **ROLES)
    assert set(est.generate_figures(ds, {}, plot_data=plot_data)) == {
        FIG_POPULATIONS,
        FIG_JOINT,
    }


@pytest.mark.parametrize(
    "mangle, message",
    [
        (lambda ds: ds.drop_vars("population"), "population"),
        (lambda ds: ds.rename({"round_count": "N"}), "round_count"),
        (lambda ds: ds.drop_vars("joint_state"), "joint_state"),
    ],
)
def test_check_data_names_what_is_missing(mangle, message):
    est = QcUnidirectionalTrotterEstimator()
    with pytest.raises(ValueError, match=message):
        est._check_data(mangle(_chain_ds(joint=True)))


#: the 5Q4C 040/060 chain of 2026-09-28: period angles of the two swaps.
ANGLES = {"theta_first_rad": 0.408, "theta_second_rad": 0.597}


def test_ideal_transport_is_the_closed_form():
    """Unequal angles have a geometric closed form; the term-by-term sum the
    estimator uses must reproduce it, and equal angles must not divide by zero."""
    n = np.arange(0, 31)
    t1, t2 = ANGLES.values()
    curves = ideal_transport(n, t1, t2)
    c1, c2 = np.cos(t1), np.cos(t2)
    closed = (np.sin(t1) * np.sin(t2) * (c2**n - c1**n) / (c2 - c1)) ** 2
    np.testing.assert_allclose(curves["sink"], closed, atol=1e-12)
    np.testing.assert_allclose(curves["source"], c1 ** (2 * n))
    # N = 0 is the bare prep: everything on the source
    assert curves["source"][0] == 1.0 and curves["sink"][0] == 0.0

    equal = ideal_transport(n, 0.3, 0.3)["sink"]
    limit = (np.sin(0.3) ** 2 * n * np.cos(0.3) ** (n - 1)) ** 2
    np.testing.assert_allclose(equal, limit, atol=1e-12)
    # the textbook cascaded ceiling 4/e^2 at N = 2/theta^2 for small equal angles
    small = ideal_transport(np.arange(0, 400), 0.1, 0.1)["sink"]
    assert small.max() == pytest.approx(4 / np.e**2, rel=0.02)


def test_ideal_transport_of_an_idle_step():
    """An idle step is angle 0: idling the second swap keeps the sink empty
    while the source still drains; idling the first keeps the source full."""
    n = np.arange(0, 11)
    second_idle = ideal_transport(n, 0.4, 0.0)
    assert np.all(second_idle["sink"] == 0.0)
    assert second_idle["source"][-1] < 0.2
    first_idle = ideal_transport(n, 0.0, 0.4)
    assert np.all(first_idle["source"] == 1.0)
    assert np.all(first_idle["sink"] == 0.0)


def test_unknown_angles_leave_the_ideal_curves_nan():
    """The sink needs both angles, the source only the first."""
    n = np.arange(0, 5)
    only_first = ideal_transport(n, 0.4, float("nan"))
    assert np.all(np.isfinite(only_first["source"]))
    assert np.all(np.isnan(only_first["sink"]))
    assert np.all(np.isnan(ideal_transport(n, float("nan"), 0.4)["source"]))


def test_angles_add_the_ideal_curves_and_their_peak():
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds(n_max=30)
    res = est.extract_parameters(ds, **ROLES, **ANGLES)
    expected = ideal_transport(ds["round_count"].values, *ANGLES.values())["sink"]
    assert res["theta_first_rad"] == ANGLES["theta_first_rad"]
    assert res["ideal_sink_p_max"] == pytest.approx(expected.max())
    assert res["ideal_sink_n_at_max"] == float(np.argmax(expected))

    plot_data = est.build_plot_data(ds, res, **ROLES, **ANGLES)
    ideal = plot_data["ideal_population"]
    np.testing.assert_allclose(ideal.sel(qubit="q3").values, expected)
    assert np.all(np.isfinite(ideal.sel(qubit="q1").values))
    assert np.all(np.isnan(ideal.sel(qubit="q2").values)), "no curve on the relay"
    assert plot_data.attrs["theta_second_rad"] == ANGLES["theta_second_rad"]
    assert set(est.generate_figures(None, None, plot_data=plot_data)) == {
        FIG_POPULATIONS
    }


def test_no_angles_means_no_ideal_curves():
    """The default path: nothing supplied, every ideal value NaN, figure intact."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    res = est.extract_parameters(ds, **ROLES)
    assert np.isnan(res["theta_first_rad"]) and np.isnan(res["ideal_sink_p_max"])
    plot_data = est.build_plot_data(ds, res, **ROLES)
    assert np.all(np.isnan(plot_data["ideal_population"].values))
    assert set(est.generate_figures(None, None, plot_data=plot_data)) == {
        FIG_POPULATIONS
    }


def test_plot_data_written_before_the_overlay_still_replots():
    """A saved plotdata.nc from before the ideal curves has no ideal variable
    and no angle attrs; replotting it must still work."""
    est = QcUnidirectionalTrotterEstimator()
    ds = _chain_ds()
    old = est.build_plot_data(ds, {}, **ROLES).drop_vars("ideal_population")
    for key in ANGLES:
        del old.attrs[key]
    assert set(est.generate_figures(None, None, plot_data=old)) == {FIG_POPULATIONS}


def test_analyze_writes_the_artifacts(tmp_path):
    est = QcUnidirectionalTrotterEstimator()
    results, figures = est.analyze(
        _chain_ds(joint=True), output_dir=str(tmp_path), **ROLES
    )
    assert results["sink_p_max"] > 0.8
    assert set(figures) == {FIG_POPULATIONS, FIG_JOINT}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_unidirectional_trotter_metadata.json" in written
    assert "qc_unidirectional_trotter_plotdata.nc" in written
    # the estimator-named key drops the prefix; the other one keeps it
    assert f"{FIG_POPULATIONS}.png" in written
    assert f"{FIG_POPULATIONS}_{FIG_JOINT}.png" in written
