"""ReadoutTimeOfFlightEstimator — the arrival edge in a raw digitizer trace.

The quantity under test is an ABSOLUTE delay assembled from two halves: the
window origin the probe declared and the arrival it measured inside the trace.
Most of these tests are about the half that is easy to get silently wrong — the
frame — rather than about edge-finding, which is a threshold crossing.
"""

import json

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import ReadoutTimeOfFlightEstimator

FIGURES = {"readout_time_of_flight"}
GRID_NS = 4.0


def _trace(arrival_ns=174.0, window_start_ns=28.0, length_ns=1000.0,
           amplitude=0.18, rise_ns=3.0, noise=2e-3, seed=0, **attrs):
    """One averaged raw trace: noise, a smooth step at ``arrival_ns``, plateau."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, length_ns, 1.0)
    step = 1.0 / (1.0 + np.exp(-(t - arrival_ns) / rise_ns))
    iq = amplitude * step * np.exp(1j * 0.7)
    iq = iq + rng.normal(0, noise, t.size) + 1j * rng.normal(0, noise, t.size)
    base = {"window_start_ns": window_start_ns, "grid_ns": GRID_NS,
            "full_scale_v": 0.5, "old_delay_ns": 300.0}
    base.update(attrs)
    return xr.Dataset({"IQdata": (("readout_time_ns",), iq)},
                      coords={"readout_time_ns": t}, attrs=base)


def test_recovers_the_planted_arrival_and_adds_the_window_origin():
    """The answer is window origin + arrival, not either one alone."""
    res = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(arrival_ns=174.0, window_start_ns=28.0))

    assert res["arrival_ns"] == pytest.approx(174.0, abs=1.0)
    # 28 + 174 = 202, which is NOT on the 4 ns grid the vendor field takes;
    # the rounded answer is what gets written and 202 is what was measured.
    assert res["time_of_flight_ns"] == pytest.approx(204.0, abs=GRID_NS)
    assert res["time_of_flight_ns"] % GRID_NS == 0
    assert res["delta_ns"] == pytest.approx(res["time_of_flight_ns"] - 300.0)
    assert res["success"] is True


def test_the_same_edge_at_a_different_origin_is_the_same_delay():
    """The frame is the point: moving the window must not move the answer."""
    a = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(arrival_ns=200.0, window_start_ns=28.0))
    b = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(arrival_ns=100.0, window_start_ns=128.0))
    assert a["time_of_flight_ns"] == pytest.approx(b["time_of_flight_ns"],
                                                   abs=GRID_NS)


def test_a_missing_window_origin_refuses_the_absolute_but_keeps_the_arrival():
    """A relative arrival with no frame is not a delay. Report one, not both."""
    ds = _trace()
    del ds.attrs["window_start_ns"]
    res = ReadoutTimeOfFlightEstimator().extract_parameters(ds)

    assert np.isfinite(res["arrival_ns"])          # the measurement stands
    assert np.isnan(res["time_of_flight_ns"])      # the claim does not
    assert res["origin_missing"] == 1.0
    assert res["success"] is False


def test_a_window_that_opens_too_late_is_refused_not_rounded():
    """With the pulse already on at sample 0 there is no baseline, so the
    threshold sits inside the plateau's own noise. The old failure mode was a
    confident 0 ns; the flag is the fix."""
    res = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(arrival_ns=-50.0))
    assert res["success"] is False
    assert res["arrival_unresolved"] == 1.0 or res["arrival_at_edge"] == 1.0


def test_an_edge_in_the_end_zone_is_flagged_as_a_bound():
    """An arrival in the last tenth means the window is mis-placed by more than
    the trace can see - the number is a lower bound, not a measurement."""
    res = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(arrival_ns=975.0, length_ns=1000.0))
    assert res["arrival_at_edge"] == 1.0
    assert res["success"] is False


def test_a_saturated_adc_is_flagged_because_it_biases_the_edge_early():
    """Clipping flattens the very edge this estimator measures."""
    clipped = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(amplitude=0.60))            # past the 0.5 V full scale
    assert clipped["adc_saturated"] == 1.0
    assert ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(amplitude=0.18))["adc_saturated"] == 0.0


def test_an_unchecked_saturation_flag_is_nan_not_a_pass():
    """No full scale declared means the check did not run. NaN says so; 0.0
    would read as 'checked, clean'."""
    ds = _trace()
    del ds.attrs["full_scale_v"]
    assert np.isnan(ReadoutTimeOfFlightEstimator().extract_parameters(
        ds)["adc_saturated"])


def test_kwargs_override_the_dataset_attributes():
    """SCQO passes the acquisition-time frame as kwargs; a stale attr must not
    win over what the caller states."""
    res = ReadoutTimeOfFlightEstimator().extract_parameters(
        _trace(window_start_ns=28.0), window_start_ns=128.0)
    assert res["window_start_ns"] == 128.0
    assert res["time_of_flight_ns"] == pytest.approx(304.0, abs=GRID_NS)


def test_figures_render_on_a_failed_fit(tmp_path):
    """A run with no edge at all must still produce its trace figure - that
    figure is how an operator sees WHY (a washed-out step means the phase was
    not reset per shot)."""
    ds = _trace()
    ds["IQdata"].values[:] = 2e-3 * (
        np.random.default_rng(1).standard_normal(ds["IQdata"].shape)
        + 1j * np.random.default_rng(2).standard_normal(ds["IQdata"].shape))
    res, figs = ReadoutTimeOfFlightEstimator().analyze(
        ds, output_dir=str(tmp_path))

    assert res["success"] is False
    assert set(figs) == FIGURES
    written = {p.name for p in tmp_path.iterdir()}
    # the single-figure idiom: a figure keyed with the estimator's own name is
    # saved unstuttered (base_estimator.save_figures)
    assert {"readout_time_of_flight.png",
            "readout_time_of_flight_metadata.json",
            "readout_time_of_flight_plotdata.nc"} <= written


def test_plot_data_always_carries_the_raw_trace(tmp_path):
    """Rule 1 of the raw-data contract: the measured arrays are present whether
    or not the fit worked, and every fit field degrades to NaN, never absent."""
    ds = _trace()
    ds["IQdata"].values[:] = 0.0          # no step, no edge
    est = ReadoutTimeOfFlightEstimator()
    res = est.extract_parameters(ds)
    plot = est.build_plot_data(ds, res)

    assert {"magnitude", "adc_i", "adc_q"} <= set(plot.data_vars)
    assert plot["magnitude"].size == ds["readout_time_ns"].size
    for key in ("arrival_ns", "threshold", "time_of_flight_ns", "delta_ns"):
        assert key in plot.attrs and np.isnan(plot.attrs[key])


def test_artifacts_round_trip(tmp_path):
    est = ReadoutTimeOfFlightEstimator()
    res, _ = est.analyze(_trace(), output_dir=str(tmp_path))
    meta = json.loads(
        (tmp_path / "readout_time_of_flight_metadata.json").read_text())
    assert meta["estimator_name"] == "readout_time_of_flight"
    assert meta["time_of_flight_ns"] == pytest.approx(res["time_of_flight_ns"])
    reloaded = xr.load_dataset(tmp_path / "readout_time_of_flight_plotdata.nc")
    assert reloaded.attrs["time_of_flight_ns"] == pytest.approx(
        res["time_of_flight_ns"])


def test_a_dataset_without_the_time_coordinate_is_refused_by_name():
    ds = _trace().rename({"readout_time_ns": "t"})
    with pytest.raises(ValueError, match="readout_time_ns"):
        ReadoutTimeOfFlightEstimator().analyze(ds)


def test_a_dataset_without_a_trace_is_refused_by_name():
    ds = _trace().drop_vars("IQdata")
    ds["something_else"] = (("readout_time_ns",),
                            np.zeros(ds["readout_time_ns"].size))
    with pytest.raises(ValueError, match="IQdata"):
        ReadoutTimeOfFlightEstimator().analyze(ds)
