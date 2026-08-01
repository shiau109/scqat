"""Tests for the ParitySwitchEstimator (shot-trace -> parity switching rate).

Two input modes, one contract: a per-shot ``state`` variable used verbatim, or
per-shot I/Q discriminated against the stored ``ref_pos_*`` centres (the
``qubit_thermal_population`` pinned-centre path). The shot cadence comes from
the dataset's ``shot_period_s`` (attached by the acquisition layer) or the
``dt_s`` kwarg; the rate scales linearly with it.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators import ParitySwitchEstimator
from scqat.estimators.parity_switch import ParitySwitchEstimator as SubpkgEstimator

GAMMA = 50.0     # planted per-direction switching rate, Hz
DT = 1e-4        # shot period, s
N = 60_000

POS_G = (0.2, -0.4)
POS_E = (4.2, -0.4)


def _trace(n=N, gamma=GAMMA, dt=DT, seed=3):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.random(n) < gamma * dt) % 2).astype(np.int8)


def _state_ds(n=N, seed=3, with_period=True):
    ds = xr.Dataset(
        {"state": ("shot_idx", _trace(n=n, seed=seed))},
        coords={"shot_idx": np.arange(n)},
    )
    if with_period:
        ds["shot_period_s"] = DT
    return ds


def _iq_ds(n=N, noise=0.8, seed=4, with_ref=True):
    trace = _trace(n=n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    i = np.where(trace, POS_E[0], POS_G[0]) + noise * rng.standard_normal(n)
    q = np.where(trace, POS_E[1], POS_G[1]) + noise * rng.standard_normal(n)
    ds = xr.Dataset(
        {"I": ("shot_idx", i), "Q": ("shot_idx", q)},
        coords={"shot_idx": np.arange(n)},
    )
    ds["shot_period_s"] = DT
    if with_ref:
        ds["ref_pos_g_i"], ds["ref_pos_g_q"] = POS_G
        ds["ref_pos_e_i"], ds["ref_pos_e_q"] = POS_E
    return ds


class TestParitySwitchEstimator:

    def test_imports_match(self):
        assert ParitySwitchEstimator is SubpkgEstimator
        assert ParitySwitchEstimator.estimator_name == "parity_switch"

    def test_state_trace_recovery(self):
        res = ParitySwitchEstimator().extract_parameters(_state_ds())
        assert res["success"] is True
        assert res["state_source"] == "state_var"
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)
        assert res["dt_s"] == pytest.approx(DT)

    def test_iq_trace_recovery_via_stored_positions(self):
        res = ParitySwitchEstimator().extract_parameters(_iq_ds())
        assert res["success"] is True
        assert res["state_source"] == "discriminated"
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.25)
        # the pinned centres + outlier diagnostics ride along as provenance
        assert res["pos_e_i"] == pytest.approx(POS_E[0])
        assert 0.0 <= res["outlier_probability"] < 0.1

    def test_user_mean_overrides_missing_reference(self):
        ds = _iq_ds(n=20_000, with_ref=False)
        res = ParitySwitchEstimator().extract_parameters(
            ds, user_mean=[list(POS_G), list(POS_E)])
        assert res["state_source"] == "discriminated"
        assert res["success"] is True

    def test_missing_centres_raise(self):
        with pytest.raises(ValueError, match="ref_pos_"):
            ParitySwitchEstimator().extract_parameters(
                _iq_ds(n=2_000, with_ref=False))

    def test_dt_kwarg_override_scales_the_rate(self):
        ds = _state_ds(seed=5)
        base = ParitySwitchEstimator().extract_parameters(ds)
        halved = ParitySwitchEstimator().extract_parameters(ds, dt_s=2 * DT)
        # same trace, doubled claimed period -> the rate in Hz halves
        assert halved["parity_rate_hz"] == pytest.approx(
            0.5 * base["parity_rate_hz"], rel=0.05)

    def test_dt_from_attr(self):
        ds = _state_ds(with_period=False)
        ds.attrs["shot_period_s"] = DT
        res = ParitySwitchEstimator().extract_parameters(ds)
        assert res["success"] is True

    def test_missing_dt_raises(self):
        with pytest.raises(ValueError, match="shot_period_s"):
            ParitySwitchEstimator().extract_parameters(
                _state_ds(n=2_000, with_period=False))

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="knob"):
            ParitySwitchEstimator().extract_parameters(
                _state_ds(n=2_000), bogus=1)

    def test_check_data_failures(self):
        est = ParitySwitchEstimator()
        with pytest.raises(ValueError):  # no shot_idx coordinate
            est._check_data(xr.Dataset({"state": ("x", [0, 1])},
                                       coords={"x": [0, 1]}))
        with pytest.raises(ValueError):  # neither state nor I/Q
            est._check_data(xr.Dataset(
                {"foo": ("shot_idx", [0.0, 1.0])},
                coords={"shot_idx": [0, 1]}))

    def test_metadata_drops_arrays(self):
        est = ParitySwitchEstimator()
        res = est.extract_parameters(_state_ds())
        meta = est.extract_metadata(res)
        for key in ("trace", "psd_freq_hz", "psd", "psd_fit"):
            assert key not in meta
        assert {"parity_rate_hz", "psd_corner_hz", "n_transitions",
                "p_excited", "success", "dt_s", "state_source"} <= set(meta)

    def test_plot_data_layout(self):
        est = ParitySwitchEstimator()
        ds = _iq_ds()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        assert pd["state"].dims == ("shot_idx",)
        assert pd["psd"].dims == ("psd_freq_hz",)
        assert pd["psd_fit"].dims == ("psd_freq_hz",)
        assert pd.coords["time_s"].dims == ("shot_idx",)
        # IQ subsample capped for the shared panel
        assert pd["iq_i"].dims == ("iq_idx",)
        assert pd["iq_i"].sizes["iq_idx"] <= 4096
        assert pd.attrs["success"] == 1
        assert pd.attrs["state_source"] == "discriminated"
        assert pd.attrs["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.25)

    def test_analyze_roundtrip_state_mode(self, tmp_path):
        est = ParitySwitchEstimator()
        res, figs = est.analyze(_state_ds(), output_dir=str(tmp_path))
        assert (tmp_path / "parity_switch_metadata.json").exists()
        assert (tmp_path / "parity_switch_plotdata.nc").exists()
        assert set(figs) == {"trace", "psd"}  # no IQ cloud in state mode
        assert isinstance(figs["trace"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_iq_mode_and_replot(self, tmp_path):
        est = ParitySwitchEstimator()
        res, figs = est.analyze(_iq_ds(), output_dir=str(tmp_path))
        assert set(figs) == {"trace", "psd", "iq_plane"}
        # replot with zero re-fit, straight from the saved plotdata
        loaded = est.load_plot_data(str(tmp_path))
        refigs = est.generate_figures(None, None, plot_data=loaded)
        assert set(refigs) == {"trace", "psd", "iq_plane"}
        plt.close("all")
