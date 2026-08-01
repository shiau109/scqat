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


def _parity(n=N, gamma=GAMMA, dt=DT, seed=3):
    """The charge parity itself: a telegraph switching at ``gamma``."""
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.random(n) < gamma * dt) % 2).astype(np.int8)


def _trace(n=N, gamma=GAMMA, dt=DT, seed=3):
    """What the instrument actually returns: the RUNNING XOR of the parity.

    No qubit reset plus a QND readout means each outcome inverts with the pole
    the previous shot left behind, so ``s[i] = s[i-1] XOR parity[i]``. Building
    the fixture this way is the whole point — one that planted the telegraph
    directly into the readout would validate the wrong physics, which is
    exactly how the original bug survived its own tests.
    """
    return (np.cumsum(_parity(n, gamma, dt, seed)) % 2).astype(np.int8)


def _state_ds(n=N, seed=3, with_period=True):
    ds = xr.Dataset(
        {"state": ("shot_idx", _trace(n=n, seed=seed))},
        coords={"shot_idx": np.arange(n)},
    )
    if with_period:
        ds["shot_period_s"] = DT
    return ds


#: blob noise for the IQ fixtures. Deliberately small: the separation is 4, so
#: 0.4 puts the centres 5 sigma apart and the discrimination error is ~1e-7.
#: That matters because these tests check the IQ PLUMBING, and the parity rate
#: is brutally sensitive to readout error (~1 + 2*eps/(Gamma*dt)) — at the old
#: 0.8 the ~0.6% error alone inflated the rate 10x. The sensitivity itself is
#: pinned separately by TestReadoutErrorBias.
_IQ_NOISE = 0.4


def _iq_ds(n=N, noise=_IQ_NOISE, seed=4, with_ref=True):
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
                "p_switch", "p_high", "p_parity_odd", "success", "dt_s",
                "state_source"} <= set(meta)

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

    def test_parity_is_the_consecutive_pair_difference(self):
        """even (0) = the pair agrees, odd (1) = it differs, one shorter than
        the readout trace, and it must equal the PLANTED parity."""
        est = ParitySwitchEstimator()
        ds = _state_ds()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)

        assert pd["parity"].dims == ("pair_idx",)
        assert pd.coords["pair_time_s"].dims == ("pair_idx",)
        state = pd["state"].values
        parity = pd["parity"].values
        assert parity.size == state.size - 1
        assert np.array_equal(parity, (state[:-1] != state[1:]).astype(np.int8))
        # and that derived series IS the parity the fixture planted
        assert np.array_equal(parity, _parity()[1:])
        # n_transitions now counts the PARITY's own switches, not the readout's
        assert int(np.count_nonzero(np.diff(parity))) == res["n_transitions"]
        # each pair is timed BETWEEN its two shots
        assert pd.coords["pair_time_s"].values[0] == pytest.approx(0.5 * DT)

    def test_a_single_shot_has_no_parity_at_all(self):
        """The parity IS the consecutive difference, so one shot carries no
        measurement — refuse by name rather than fit an empty array."""
        est = ParitySwitchEstimator()
        ds = xr.Dataset({"state": ("shot_idx", np.array([1], dtype=np.int8))},
                        coords={"shot_idx": [0]})
        ds["shot_period_s"] = DT
        with pytest.raises(ValueError, match="at least 2 shots"):
            est.extract_parameters(ds)

    def test_analyze_roundtrip_state_mode(self, tmp_path):
        est = ParitySwitchEstimator()
        res, figs = est.analyze(_state_ds(), output_dir=str(tmp_path))
        assert (tmp_path / "parity_switch_metadata.json").exists()
        assert (tmp_path / "parity_switch_plotdata.nc").exists()
        # no IQ cloud in state mode
        assert set(figs) == {"trace", "parity", "psd", "state_psd"}
        assert isinstance(figs["trace"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_iq_mode_and_replot(self, tmp_path):
        est = ParitySwitchEstimator()
        res, figs = est.analyze(_iq_ds(), output_dir=str(tmp_path))
        assert set(figs) == {"trace", "parity", "psd", "state_psd", "iq_plane"}
        # replot with zero re-fit, straight from the saved plotdata
        loaded = est.load_plot_data(str(tmp_path))
        refigs = est.generate_figures(None, None, plot_data=loaded)
        assert set(refigs) == {"trace", "parity", "psd", "state_psd", "iq_plane"}
        plt.close("all")


class TestTheReadoutIsNotTheParity:
    """The bug this experiment shipped with, pinned so it cannot come back.

    The sequence has NO qubit reset and is a unitary, so it maps antipodal
    Bloch vectors to antipodal ones: |0> and |1> can never give the same
    outcome. The measured outcome therefore inverts with the pole the previous
    shot left behind, and the readout is the RUNNING XOR of the parity rather
    than the parity. v0.19.0 fitted the readout and every rate it produced was
    meaningless.
    """

    def test_the_sequence_inverts_with_the_starting_pole(self):
        """The physics the whole analysis rests on, checked from the unitary
        rather than remembered. y90 - Z(+-pi/2) - x90 on |0> and on |1> must
        give OPPOSITE outcomes for the same parity."""
        def rot(axis, ang):
            s = {"x": np.array([[0, 1], [1, 0]]),
                 "y": np.array([[0, -1j], [1j, 0]]),
                 "z": np.array([[1, 0], [0, -1]])}[axis]
            return np.cos(ang / 2) * np.eye(2) - 1j * np.sin(ang / 2) * s

        for phase in (+np.pi / 2, -np.pi / 2):          # even / odd parity
            seq = rot("x", np.pi / 2) @ rot("z", phase) @ rot("y", np.pi / 2)
            p_from_g = abs((seq @ np.array([1, 0], complex))[1]) ** 2
            p_from_e = abs((seq @ np.array([0, 1], complex))[1]) ** 2
            assert p_from_g == pytest.approx(1.0 - p_from_e, abs=1e-12)
            assert abs(p_from_g - p_from_e) == pytest.approx(1.0, abs=1e-12)

    def test_rate_is_recovered_from_a_running_xor_readout(self):
        """THE regression test: feed the estimator what the instrument really
        returns and require the PLANTED rate back. Fitting the readout instead
        of the parity fails this outright."""
        est = ParitySwitchEstimator()
        res = est.extract_parameters(_state_ds())
        assert res["success"] is True
        assert res["parity_rate_hz"] == pytest.approx(GAMMA, rel=0.2)

    def test_a_healthy_run_has_odd_parity_near_one_half(self):
        """p_parity_odd ~ 0.5 is HEALTHY — the chip sits in each parity about
        half the time, and that fraction carries no rate information. Only
        p_switch (how often the parity CHANGES) decides resolvability, and it
        must be small for a well-sampled telegraph."""
        est = ParitySwitchEstimator()
        res = est.extract_parameters(_state_ds())
        assert res["p_parity_odd"] == pytest.approx(0.5, abs=0.1)
        assert res["p_switch"] < 0.05
        assert res["p_switch"] == pytest.approx(GAMMA * DT, rel=0.3)

    def test_the_readout_spectrum_is_kept_but_never_fitted(self):
        est = ParitySwitchEstimator()
        ds = _state_ds()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        assert pd["state_psd"].dims == ("state_psd_freq_hz",)
        assert pd["state_psd"].size > 0
        # no fit curve exists for it, and none may be added
        assert "state_psd_fit" not in pd.data_vars
        # the fitted spectrum is the PARITY's, and it is a different array
        assert pd["psd"].size != pd["state_psd"].size or not np.array_equal(
            pd["psd"].values, pd["state_psd"].values)


class TestReadoutErrorBias:
    """Readout error is NOT benign on this experiment, and the size of the
    effect is worth pinning rather than discovering on hardware.

    The parity is recovered as the difference of consecutive readouts, so ONE
    bad shot flips TWO adjacent parity samples. The corruption is therefore
    lag-1 correlated rather than white, and it biases the fitted corner —
    unlike a directly sampled telegraph, where errors land harmlessly in the
    white floor. The rate inflates as ~1 + 2*eps/(Gamma*dt).
    """

    @staticmethod
    def _rate(eps, seed=7, n=200_000):
        rng = np.random.default_rng(seed)
        parity = (np.cumsum(rng.random(n) < GAMMA * DT) % 2).astype(np.int8)
        state = (np.cumsum(parity) % 2).astype(np.int8)
        if eps:
            state = (state ^ (rng.random(n) < eps)).astype(np.int8)
        ds = xr.Dataset({"state": ("shot_idx", state)},
                        coords={"shot_idx": np.arange(n)})
        ds["shot_period_s"] = DT
        return ParitySwitchEstimator().extract_parameters(ds)["parity_rate_hz"]

    def test_clean_readout_recovers_the_planted_rate(self):
        assert self._rate(0.0) == pytest.approx(GAMMA, rel=0.15)

    def test_error_well_under_gamma_dt_is_tolerable(self):
        # eps = 0.1 * Gamma*dt -> ~1.2x, still the right order
        assert self._rate(0.1 * GAMMA * DT) == pytest.approx(GAMMA, rel=0.35)

    def test_error_comparable_to_gamma_dt_inflates_the_rate_badly(self):
        """The number that sets the fidelity requirement: at eps ~ Gamma*dt the
        reported rate is several times the truth. If this ever starts passing
        as 'close enough', the bias has been corrected and the docs must change."""
        inflated = self._rate(GAMMA * DT)          # eps == Gamma*dt
        assert inflated > 3 * GAMMA
