"""Tests for the ParitySwitchDiscreteEstimator (two measurements per cycle ->
parity switching rate; the M1/M2 variant).

Two input modes, one contract: a per-measurement ``state`` variable over
``(shot_idx, meas_idx)`` used verbatim, or per-measurement I/Q discriminated
against the stored ``ref_pos_*`` centres. The cycle period comes from the
dataset's ``shot_period_s`` (attached by the acquisition layer) or the
``dt_s`` kwarg; the rate scales linearly with it.

The fixture builds what the instrument really returns: M1 re-measures the pole
M2 left behind (QND chain, ``m1[i+1] = m2[i]``, ``m1[0] = 0``) and the mapping
block flips the pole iff the parity is odd (``m2[i] = m1[i] XOR p[i]``) — so
``m1`` is the running XOR of the parity while ``m1 XOR m2`` recovers the
planted telegraph exactly.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators import ParitySwitchDiscreteEstimator
from scqat.estimators.parity_switch_discrete import (
    ParitySwitchDiscreteEstimator as SubpkgEstimator,
)

GAMMA = 50.0     # planted per-direction switching rate, Hz
DT = 1e-4        # cycle period, s
N = 60_000

POS_G = (0.2, -0.4)
POS_E = (4.2, -0.4)


def _parity(n=N, gamma=GAMMA, dt=DT, seed=3):
    """The charge parity itself: a telegraph switching at ``gamma``."""
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.random(n) < gamma * dt) % 2).astype(np.int8)


def _m1_m2(n=N, gamma=GAMMA, dt=DT, seed=3):
    """The two measurement traces the instrument returns per cycle."""
    p = _parity(n, gamma, dt, seed)
    m2 = (np.cumsum(p) % 2).astype(np.int8)
    m1 = np.concatenate([[0], m2[:-1]]).astype(np.int8)
    return m1, m2


def _state_ds(n=N, seed=3, with_period=True, m1=None, m2=None):
    if m1 is None:
        m1, m2 = _m1_m2(n=n, seed=seed)
    ds = xr.Dataset(
        {"state": (("shot_idx", "meas_idx"),
                   np.stack([m1, m2], axis=-1))},
        coords={"shot_idx": np.arange(m1.size), "meas_idx": np.arange(2)},
    )
    if with_period:
        ds["shot_period_s"] = DT
    return ds


#: blob noise for the IQ fixtures — same rationale as the continuous tests:
#: separation 4, sigma 0.4 => ~1e-7 discrimination error, so these tests check
#: the IQ PLUMBING, not readout-error robustness.
_IQ_NOISE = 0.4


def _iq_ds(n=N, noise=_IQ_NOISE, seed=4, with_ref=True):
    m1, m2 = _m1_m2(n=n, seed=seed)
    pair = np.stack([m1, m2], axis=-1)          # (n, 2)
    rng = np.random.default_rng(seed + 1)
    i = np.where(pair, POS_E[0], POS_G[0]) + noise * rng.standard_normal(pair.shape)
    q = np.where(pair, POS_E[1], POS_G[1]) + noise * rng.standard_normal(pair.shape)
    ds = xr.Dataset(
        {"I": (("shot_idx", "meas_idx"), i),
         "Q": (("shot_idx", "meas_idx"), q)},
        coords={"shot_idx": np.arange(n), "meas_idx": np.arange(2)},
    )
    ds["shot_period_s"] = DT
    if with_ref:
        ds["ref_pos_g_i"], ds["ref_pos_g_q"] = POS_G
        ds["ref_pos_e_i"], ds["ref_pos_e_q"] = POS_E
    return ds


class TestParitySwitchDiscreteEstimator:

    def test_imports_match(self):
        assert ParitySwitchDiscreteEstimator is SubpkgEstimator
        assert (ParitySwitchDiscreteEstimator.estimator_name
                == "parity_switch_discrete")

    def test_state_recovery_of_the_planted_rate(self):
        res = ParitySwitchDiscreteEstimator().extract_parameters(_state_ds())
        assert res["success"] is True
        assert res["state_source"] == "state_var"
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)
        assert res["dt_s"] == pytest.approx(DT)

    def test_within_cycle_reduction_is_m1_xor_m2(self):
        """The parity is computed WITHIN each cycle — same length as the cycle
        count, equal to the planted telegraph exactly, no cross-cycle chain."""
        est = ParitySwitchDiscreteEstimator()
        ds = _state_ds()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)

        assert pd["parity"].dims == ("shot_idx",)
        m1 = pd["m1"].values
        m2 = pd["m2"].values
        parity = pd["parity"].values
        assert parity.size == m1.size            # per-cycle, NOT n-1
        assert np.array_equal(parity, (m1 != m2).astype(np.int8))
        # and that derived series IS the parity the fixture planted
        assert np.array_equal(parity, _parity())
        # n_transitions counts the PARITY's own switches
        assert int(np.count_nonzero(np.diff(parity))) == res["n_transitions"]

    def test_intercycle_flip_is_zero_on_clean_data_and_counts_breaks(self):
        est = ParitySwitchDiscreteEstimator()
        res = est.extract_parameters(_state_ds())
        assert res["p_intercycle_flip"] == 0.0

        # break the QND chain at k places: flip M1 there — m1[j] != m2[j-1]
        m1, m2 = _m1_m2()
        breaks = (17, 4021, 30_000)
        m1 = m1.copy()
        m1[list(breaks)] ^= 1
        res = est.extract_parameters(_state_ds(m1=m1, m2=m2))
        assert res["p_intercycle_flip"] == pytest.approx(
            len(breaks) / (N - 1))

    def test_one_bad_measurement_flips_exactly_one_parity_sample(self):
        """The discrete variant's error advantage: a bad measurement corrupts
        ONE parity sample (the continuous variant's bad shot corrupts two,
        because adjacent pairs share it)."""
        est = ParitySwitchDiscreteEstimator()
        m1, m2 = _m1_m2()
        j = 12_345
        m2 = m2.copy()
        m2[j] ^= 1                                # one bad M2 readout
        res = est.extract_parameters(_state_ds(m1=m1, m2=m2))
        pd_parity = res["parity"]
        diff = np.flatnonzero(pd_parity != _parity())
        assert diff.tolist() == [j]

    def test_iq_recovery_via_stored_positions(self):
        res = ParitySwitchDiscreteEstimator().extract_parameters(_iq_ds())
        assert res["success"] is True
        assert res["state_source"] == "discriminated"
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.25)
        # the pinned centres + outlier diagnostics ride along as provenance
        assert res["pos_e_i"] == pytest.approx(POS_E[0])
        assert 0.0 <= res["outlier_probability"] < 0.1
        # clean synthetic chain: discrimination error ~1e-7, so no breaks
        assert res["p_intercycle_flip"] < 1e-3

    def test_user_mean_overrides_missing_reference(self):
        ds = _iq_ds(n=20_000, with_ref=False)
        res = ParitySwitchDiscreteEstimator().extract_parameters(
            ds, user_mean=[list(POS_G), list(POS_E)])
        assert res["state_source"] == "discriminated"
        assert res["success"] is True

    def test_missing_centres_raise(self):
        with pytest.raises(ValueError, match="ref_pos_"):
            ParitySwitchDiscreteEstimator().extract_parameters(
                _iq_ds(n=2_000, with_ref=False))

    def test_dt_kwarg_override_scales_the_rate(self):
        ds = _state_ds(seed=5)
        base = ParitySwitchDiscreteEstimator().extract_parameters(ds)
        halved = ParitySwitchDiscreteEstimator().extract_parameters(
            ds, dt_s=2 * DT)
        # same traces, doubled claimed period -> the rate in Hz halves
        assert halved["parity_rate_hz"] == pytest.approx(
            0.5 * base["parity_rate_hz"], rel=0.05)

    def test_dt_from_attr(self):
        ds = _state_ds(with_period=False)
        ds.attrs["shot_period_s"] = DT
        res = ParitySwitchDiscreteEstimator().extract_parameters(ds)
        assert res["success"] is True

    def test_missing_dt_raises(self):
        with pytest.raises(ValueError, match="shot_period_s"):
            ParitySwitchDiscreteEstimator().extract_parameters(
                _state_ds(n=2_000, with_period=False))

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(ValueError, match="knob"):
            ParitySwitchDiscreteEstimator().extract_parameters(
                _state_ds(n=2_000), bogus=1)

    def test_check_data_refusals(self):
        est = ParitySwitchDiscreteEstimator()
        with pytest.raises(ValueError, match="meas_idx"):  # missing meas axis
            est._check_data(xr.Dataset(
                {"state": ("shot_idx", [0, 1])},
                coords={"shot_idx": [0, 1]}))
        with pytest.raises(ValueError, match="exactly 2"):  # wrong meas size
            est._check_data(xr.Dataset(
                {"state": (("shot_idx", "meas_idx"), np.zeros((4, 3)))},
                coords={"shot_idx": np.arange(4), "meas_idx": np.arange(3)}))
        with pytest.raises(ValueError):  # neither state nor I/Q
            est._check_data(xr.Dataset(
                {"foo": (("shot_idx", "meas_idx"), np.zeros((4, 2)))},
                coords={"shot_idx": np.arange(4), "meas_idx": np.arange(2)}))

    def test_metadata_drops_arrays(self):
        est = ParitySwitchDiscreteEstimator()
        res = est.extract_parameters(_state_ds())
        meta = est.extract_metadata(res)
        for key in ("m1", "m2", "parity", "psd_freq_hz", "psd", "psd_fit",
                    "state_psd_freq_hz", "state_psd"):
            assert key not in meta
        assert {"parity_rate_hz", "psd_corner_hz", "n_transitions",
                "p_switch", "p_high", "p_parity_odd", "p_intercycle_flip",
                "p_m1_high", "p_m2_high", "success", "dt_s",
                "state_source"} <= set(meta)

    def test_plot_data_layout(self):
        est = ParitySwitchDiscreteEstimator()
        ds = _iq_ds()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        for var in ("m1", "m2", "parity"):
            assert pd[var].dims == ("shot_idx",)
        assert pd["psd"].dims == ("psd_freq_hz",)
        assert pd["psd_fit"].dims == ("psd_freq_hz",)
        assert pd.coords["time_s"].dims == ("shot_idx",)
        # IQ subsample capped for the shared panel
        assert pd["iq_i"].dims == ("iq_idx",)
        assert pd["iq_i"].sizes["iq_idx"] <= 4096
        assert pd.attrs["success"] == 1
        assert pd.attrs["state_source"] == "discriminated"
        assert pd.attrs["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.25)
        assert "p_intercycle_flip" in pd.attrs
        assert "p_m1_high" in pd.attrs and "p_m2_high" in pd.attrs

    def test_analyze_roundtrip_state_mode(self, tmp_path):
        est = ParitySwitchDiscreteEstimator()
        res, figs = est.analyze(_state_ds(), output_dir=str(tmp_path))
        assert (tmp_path / "parity_switch_discrete_metadata.json").exists()
        assert (tmp_path / "parity_switch_discrete_plotdata.nc").exists()
        # no IQ cloud in state mode
        assert set(figs) == {"timetrace", "psd", "state_psd"}
        assert isinstance(figs["timetrace"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_iq_mode_and_replot(self, tmp_path):
        est = ParitySwitchDiscreteEstimator()
        res, figs = est.analyze(_iq_ds(), output_dir=str(tmp_path))
        assert set(figs) == {"timetrace", "psd", "state_psd", "iq_plane"}
        # replot with zero re-fit, straight from the saved plotdata
        loaded = est.load_plot_data(str(tmp_path))
        refigs = est.generate_figures(None, None, plot_data=loaded)
        assert set(refigs) == {"timetrace", "psd", "state_psd", "iq_plane"}
        plt.close("all")

    def test_figures_render_on_a_failed_fit(self):
        """Identical measurements every cycle => the parity is constant, the
        PSD fit fails (empty spectrum, NaN rate) — but every figure must still
        render, the raw-data panel most of all."""
        est = ParitySwitchDiscreteEstimator()
        zeros = np.zeros(4096, dtype=np.int8)
        ds = _state_ds(m1=zeros, m2=zeros)
        res = est.extract_parameters(ds)
        assert res["success"] is False
        assert np.isnan(res["parity_rate_hz"])
        pd = est.build_plot_data(ds, res)
        figs = est.generate_figures(None, None, plot_data=pd)
        assert {"timetrace", "psd", "state_psd"} <= set(figs)
        plt.close("all")


def test_the_mapping_makes_m1_the_running_xor():
    """The physics identity the fixture (and the real instrument) obeys:
    with the QND chain ``m1[i+1] = m2[i]`` and the mapping
    ``m2[i] = m1[i] XOR p[i]``, m1 is the running XOR of the parity and the
    within-cycle difference recovers p exactly."""
    m1, m2 = _m1_m2(n=10_000)
    p = _parity(n=10_000)
    assert np.array_equal((m1 ^ m2), p)
    assert np.array_equal(m1[1:], m2[:-1])
    # m1 integrates the parity (running XOR, shifted one cycle)
    assert np.array_equal(m1[1:], (np.cumsum(p) % 2)[:-1])
