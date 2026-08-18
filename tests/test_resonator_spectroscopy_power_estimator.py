"""Tests for the ResonatorSpectroscopyPowerEstimator.

Synthesises a resonator-spectroscopy-vs-power map whose dip centre is flat in the
low-power (dispersive) regime and shifts sharply through a transition toward high
power, then checks that the estimator (1) collapses the 2-D (power, detuning) map to
a centre-vs-power trace, (2) picks an optimal readout power from where the centre
stops shifting, and (3) produces a self-sufficient, reconstructable plot_data.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import ResonatorSpectroscopyPowerEstimator
from scqat.estimators.resonator_spectroscopy_power import (
    ResonatorSpectroscopyPowerEstimator as SubpkgEstimator,
)
from scqat.estimators.resonator_spectroscopy_power.estimator import _branch_frequencies
from scqat.estimators.resonator_spectroscopy_power.visualization import plot_power_map


def _make_dataset(n_power=30, n_det=121, noise=0.0, seed=0):
    """Resonator-vs-power IQ map: dip centre flat at low power, then shifts through
    a transition toward high power (a dispersive-shift -> bright-state punch-out)."""
    rng = np.random.default_rng(seed)
    power = np.linspace(-50.0, -25.0, n_power)      # dBm, ascending
    detuning = np.linspace(-3e6, 3e6, n_det)
    lo = 7.0e9
    full_freq = lo + detuning                       # (detuning,)

    shift, p_trans, width = 0.8e6, -34.0, 2.5
    center_det = shift * 0.5 * (1.0 - np.tanh((power - p_trans) / width))  # Hz, per power

    gamma, depth = 0.3e6, 0.85
    iq = np.empty((n_power, n_det), dtype=complex)
    for k in range(n_power):
        p = 1.0 - depth * gamma ** 2 / ((detuning - center_det[k]) ** 2 + gamma ** 2)
        amp = np.sqrt(np.clip(p, 1e-9, None))
        if noise > 0:
            amp = amp + rng.normal(0, noise, size=amp.shape)
        iq[k] = amp.astype(complex)

    ds = xr.Dataset(
        {"IQdata": (("power", "detuning"), iq)},
        coords={
            "power": power,
            "detuning": detuning,
            "full_freq": ("detuning", full_freq),
        },
    )
    return ds, dict(center_det=center_det, power=power, p_trans=p_trans, lo=lo)


class TestResonatorSpectroscopyPower:
    def test_aggregated_and_subpackage_imports_match(self):
        assert ResonatorSpectroscopyPowerEstimator is SubpkgEstimator
        assert ResonatorSpectroscopyPowerEstimator.estimator_name == "resonator_spectroscopy_power"

    def test_results_structure_and_good_points(self):
        ds, _ = _make_dataset()
        results = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        for key in ("power", "detuning", "center_detuning", "good", "amplitude_map",
                    "n_power", "n_good", "optimal_power", "frequency_shift",
                    "resonator_frequency", "optimal_success",
                    "dress_max_power", "bare_min_power", "branch_class"):
            assert key in results
        assert results["n_power"] == 30
        # Most power slices yield a clean dip.
        assert results["n_good"] >= 26
        # The 2-D map is oriented (power, detuning).
        assert results["amplitude_map"].shape == (30, 121)

    def test_center_trace_tracks_truth(self):
        ds, truth = _make_dataset()
        results = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        good = results["good"]
        centre = results["center_detuning"]
        # Fitted centres match the synthetic dip positions on the good points.
        assert np.allclose(centre[good], truth["center_det"][good], atol=0.1e6)

    def test_picks_optimal_power_in_dispersive_regime(self):
        ds, truth = _make_dataset()
        results = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        assert results["optimal_success"] is True
        opt = results["optimal_power"]
        assert np.isfinite(opt)
        # Optimal power lands within the swept range, at/below the transition.
        assert truth["power"].min() <= opt <= truth["power"].max()
        assert opt <= truth["p_trans"] + 1.0
        # Resonator frequency reported on the absolute axis.
        assert np.isfinite(results["resonator_frequency"])
        assert abs(results["resonator_frequency"] - truth["lo"]) < 3e6

    def test_recovers_both_punchout_branches(self):
        """THE punchout physics: the low-power plateau is the DRESSED resonator
        (qubit in |0>), the high-power plateau is the BARE one (qubit saturated),
        and their gap is the Lamb shift. The generator plants dressed = lo+0.8 MHz
        and bare = lo."""
        ds, truth = _make_dataset()
        r = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        assert r["branch_success"] is True
        assert r["f_dress0"] == pytest.approx(truth["lo"] + 0.8e6, abs=60e3)
        assert r["f_bare"] == pytest.approx(truth["lo"], abs=60e3)
        # dressed sits ABOVE bare here (the planted shift is positive)
        assert r["f_dress0"] > r["f_bare"]
        assert r["lamb_shift"] == pytest.approx(0.8e6, abs=100e3)
        assert r["lamb_shift"] == pytest.approx(r["f_dress0"] - r["f_bare"])
        # both plateaus were actually populated, and split about the crossing
        assert r["n_low_plateau"] >= 3 and r["n_high_plateau"] >= 3
        assert truth["power"].min() < r["crossing_power"] < truth["power"].max()
        # the plateau boundary powers bracket the transition, in order
        assert np.isfinite(r["dress_max_power"]) and np.isfinite(r["bare_min_power"])
        assert r["dress_max_power"] < truth["p_trans"] < r["bare_min_power"]
        # branch_class agrees with the boundaries: 1 up to dress_max, 2 from
        # bare_min, 0 strictly between (the transition feeds neither branch)
        power, cls = r["power"], r["branch_class"]
        assert (cls[power <= r["dress_max_power"]] == 1).all()
        assert (cls[power >= r["bare_min_power"]] == 2).all()
        between = (power > r["dress_max_power"]) & (power < r["bare_min_power"])
        assert between.any() and (cls[between] == 0).all()

    def test_missing_high_plateau_reports_only_the_dressed_branch(self):
        """A window that never reaches saturation has no bare branch. The dressed
        one must still be reported — losing both would throw away the half of the
        punchout that DID resolve."""
        ds, truth = _make_dataset()
        # keep only the dispersive side of the transition
        ds = ds.sel(power=ds.coords["power"].values <= truth["p_trans"])
        r = ResonatorSpectroscopyPowerEstimator().extract_parameters(
            ds, branch_min_points=3)
        assert np.isfinite(r["f_dress0"])
        assert not np.isfinite(r["f_bare"])
        assert r["branch_success"] is False
        assert r["n_high_plateau"] < 3
        # the missing branch has no boundary power either
        assert not np.isfinite(r["bare_min_power"])

    def test_branches_need_the_absolute_axis(self):
        """f_bare/f_dress0 are absolute frequencies — a bare/dressed pair means
        nothing as a detuning from an LO that may move. Without `full_freq` they
        are NaN rather than silently reported in the wrong frame."""
        ds, _ = _make_dataset()
        r = ResonatorSpectroscopyPowerEstimator().extract_parameters(
            ds.drop_vars("full_freq"))
        assert not np.isfinite(r["f_dress0"]) and not np.isfinite(r["f_bare"])
        assert r["branch_success"] is False

    def test_figures_render_on_a_failed_fit(self):
        """A run whose fit resolved nothing must still produce its raw map — the
        artifact fallback drops ALL figures on any single plotter exception."""
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyPowerEstimator()
        results = estimator.extract_parameters(ds)
        # degrade every fit-derived scalar/array the plotter might touch
        results = dict(results)
        results["center_detuning"] = np.full_like(results["center_detuning"], np.nan)
        results["center_full_freq"] = np.full_like(results["center_full_freq"], np.nan)
        results["good"] = np.zeros_like(results["good"], dtype=bool)
        for scalar in ("optimal_power", "crossing_power", "frequency_shift",
                       "resonator_frequency", "f_dress0", "f_bare", "lamb_shift",
                       "dress_max_power", "bare_min_power"):
            results[scalar] = float("nan")
        results["optimal_success"] = False
        results["branch_success"] = False
        plot_data = estimator.build_plot_data(ds, results)
        figs = estimator.generate_figures(ds, results, plot_data=plot_data)
        assert "resonator_spectroscopy_power" in figs
        plt.close("all")

    def test_metadata_drops_bulky_arrays(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyPowerEstimator()
        results = estimator.extract_parameters(ds)
        meta = estimator.extract_metadata(results)
        for dropped in ("amplitude_map", "detuning", "full_freq"):
            assert dropped not in meta
        for kept in ("optimal_power", "frequency_shift", "n_good", "center_detuning"):
            assert kept in meta

    def test_plot_data_self_sufficient_and_figure(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyPowerEstimator()
        results = estimator.extract_parameters(ds)
        pd = estimator.build_plot_data(ds, results)

        assert isinstance(pd, xr.Dataset)
        for var in ("amplitude", "center_detuning", "center_full_freq", "good", "outlier"):
            assert var in pd
        for coord in ("power", "detuning", "full_freq"):
            assert coord in pd.coords
        assert pd["amplitude"].dims == ("power", "detuning")
        assert "optimal_power" in pd.attrs

        figs = estimator.generate_figures(None, None, plot_data=pd)
        assert set(figs) == {"resonator_spectroscopy_power"}
        assert isinstance(figs["resonator_spectroscopy_power"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip(self, tmp_path):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyPowerEstimator()
        results, figs = estimator.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "resonator_spectroscopy_power_metadata.json").exists()
        assert (tmp_path / "resonator_spectroscopy_power_plotdata.nc").exists()
        assert isinstance(figs["resonator_spectroscopy_power"], plt.Figure)

        reloaded = estimator.load_plot_data(str(tmp_path))
        fig = plot_power_map(reloaded)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_works_with_I_Q_quadratures(self):
        ds, _ = _make_dataset()
        # Feed I/Q instead of IQdata (the real acquisition path).
        ds_iq = xr.Dataset(
            {"I": ds["IQdata"].real, "Q": ds["IQdata"].imag},
            coords=ds.coords,
        )
        results = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds_iq)
        assert results["n_good"] >= 26
        assert results["optimal_success"] is True

    def test_row_scaled_map_matches_unscaled(self):
        # Real instruments measure |IQ| that grows with the readout drive: each
        # power row arrives scaled by the drive amplitude prefactor 10**(p/20).
        # Per-row dip fits are scale-invariant, so the scaled map must give the
        # same answer as the pre-normalized one.
        ds, truth = _make_dataset()
        scale = xr.DataArray(10.0 ** (ds["power"].values / 20.0), dims="power")
        ds_scaled = ds.assign(IQdata=ds["IQdata"] * scale)

        ref = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        res = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds_scaled)

        assert res["optimal_success"] is True
        step = float(np.diff(truth["power"]).mean())
        assert abs(res["optimal_power"] - ref["optimal_power"]) <= step + 1e-9
        assert res["n_good"] >= ref["n_good"] - 1
        good = res["good"]
        assert np.allclose(res["center_detuning"][good], truth["center_det"][good], atol=0.1e6)


#: Real-hardware regression fixture: the 21 per-power dip centres (Hz) fitted on
#: 5Q4C q2, run 20260818-204626-572-5Q4C-resonator_spectroscopy_power_amp-01
#: (dip_method='circle'). The trace has everything that broke v1: a gradual
#: transition (-22..-16), a bifurcation-regime OVERSHOOT below the bare
#: frequency (-14..-12), and a bare plateau (-6..0) whose linewidth is a third
#: of the dressed one — the old global FWHM gate deleted it, and the old
#: lagging-median plateau growth then built f_bare out of the transition.
_REAL_POWER_DBM = np.arange(-40.0, 0.1, 2.0)
_REAL_CENTER_HZ = np.array([
    6024522156.138736, 6024341272.137275, 6024386854.124103, 6024430636.149333,
    6024323940.045758, 6024337945.974469, 6024272404.223297, 6024213116.111758,
    6024061526.041616, 6023200240.879427, 6023036424.555375, 6021607200.430643,
    6019362896.014840, 6015396547.808883, 6014190724.906467, 6015093425.951558,
    6015866019.103022, 6015985763.472121, 6015945396.984254, 6016036590.185349,
    6016186263.150928,
])


class TestAnchoredBranchClassification:
    """The anchored two-band classifier against the real 5Q4C q2 trace."""

    def test_real_run_20260818_204626_branches(self):
        """The user-reported failure, inverted into the acceptance criteria:
        f_bare comes from the -8..0 dBm plateau (NOT -14..-8), f_dress0 from
        -40..-24 (NOT including -22/-20), and every -22..-10 point lands in
        NEITHER branch — including the -12 dBm overshoot, which sits BELOW the
        bare band and must be rejected by the band, not by luck."""
        good = np.ones(_REAL_POWER_DBM.size, dtype=bool)
        (f_dress0, f_bare, dress_max, bare_min,
         n_low, n_high, cls) = _branch_frequencies(
            _REAL_POWER_DBM, _REAL_CENTER_HZ, good,
            band_frac=0.08, min_points=3, anchor_points=3)

        # dressed branch: exactly the -40..-24 run
        assert n_low == 9
        assert dress_max == pytest.approx(-24.0)
        assert f_dress0 == pytest.approx(6024323940.0, abs=0.2e6)
        # bare branch: the -8..0 plateau — the points the old FWHM gate deleted
        assert n_high == 5
        assert bare_min == pytest.approx(-8.0)
        assert f_bare == pytest.approx(6015985763.5, abs=0.2e6)
        # the transition (-22..-10, overshoot included) feeds neither branch,
        # and single-gap bridging must NOT rescue it: -22/-20 and -10/-12 are
        # consecutive out-of-band pairs, so both runs terminate there
        assert n_low + n_high == 14  # 7 of 21 points rejected as transition
        assert (cls[(_REAL_POWER_DBM >= -22) & (_REAL_POWER_DBM <= -10)] == 0).all()
        # Lamb shift lands at ~8.3 MHz; the v1 answer (transition-contaminated
        # f_bare = 6015.245 GHz) was 9.1 MHz
        assert (f_dress0 - f_bare) == pytest.approx(8.34e6, abs=0.3e6)

    def test_bad_end_slice_excluded_upstream_leaves_the_branch_intact(self):
        """v1 grew each plateau point-by-point from the window end, so one bad
        slice at the very top NaN'd the whole bare branch even AFTER the
        upstream gates excluded it. Classification only sees good points: with
        the corrupt slice not-good (out-of-window / unfittable-width — the
        gates that catch garbage centres in practice), the plateau anchors on
        the remaining points and the branch survives."""
        center = _REAL_CENTER_HZ.copy()
        center[-1] += 30e6  # garbage fit on the topmost slice
        good = np.ones(center.size, dtype=bool)
        good[-1] = False    # ...caught upstream (30 MHz off is out-of-window)
        (_, f_bare, _, bare_min, _, n_high, _cls) = _branch_frequencies(
            _REAL_POWER_DBM, center, good,
            band_frac=0.08, min_points=3, anchor_points=3)
        assert n_high == 4
        assert bare_min == pytest.approx(-8.0)
        assert f_bare == pytest.approx(6015965580.0, abs=0.2e6)

    def test_isolated_glitch_is_bridged_not_amputating(self):
        """One in-window glitch INSIDE a plateau must not amputate it (run
        20260818-205631: a single 2.9 MHz glitch at the 3rd dressed slice cost
        the whole branch under strict contiguity). The run bridges exactly one
        out-of-band point when the next point is back in-band; the glitch
        itself feeds neither the median nor the boundary."""
        center = _REAL_CENTER_HZ.copy()
        center[2] -= 3e6  # -36 dBm slice glitches mid-plateau (in-window)
        (f_dress0, _, dress_max, _, n_low, _, cls) = _branch_frequencies(
            _REAL_POWER_DBM, center, np.ones(center.size, bool),
            band_frac=0.08, min_points=3, anchor_points=3)
        assert n_low == 8                      # 9 members minus the glitch
        assert dress_max == pytest.approx(-24.0)
        assert cls[2] == 0                     # bridged, not a member
        assert np.isfinite(f_dress0)

    def test_two_consecutive_off_points_terminate_the_run(self):
        """The anti-creep limit of bridging: a trace still MOVING at the window
        edge walks out of the band and stays out, so two consecutive
        out-of-band points end the run — a transition cannot be bridged into a
        plateau. Top point 3 MHz off + next point 1.5 MHz off (transition
        shape) leaves the bare run unable to start."""
        center = _REAL_CENTER_HZ.copy()
        center[-1] += 3e6
        center[-2] += 1.5e6
        (_, f_bare, _, bare_min, _, n_high, _cls) = _branch_frequencies(
            _REAL_POWER_DBM, center, np.ones(center.size, bool),
            band_frac=0.08, min_points=3, anchor_points=3)
        assert n_high == 0
        assert not np.isfinite(f_bare) and not np.isfinite(bare_min)

    def test_window_truncated_mid_transition_refuses_the_bare_branch(self):
        """A window whose top ends INSIDE the transition (here: the real trace
        cut at -10 dBm, right in the overshoot region) must not report the
        transition median as f_bare. The top anchor's own scatter is then
        comparable to the band, which fails the credibility gate — without it,
        the scattered anchor would inflate its own tolerance and certify a
        fake plateau."""
        cut = _REAL_POWER_DBM <= -10.0
        (f_dress0, f_bare, _, bare_min, _, n_high, _cls) = _branch_frequencies(
            _REAL_POWER_DBM[cut], _REAL_CENTER_HZ[cut], np.ones(int(cut.sum()), bool),
            band_frac=0.08, min_points=3, anchor_points=3)
        assert np.isfinite(f_dress0)           # the dressed half DID resolve
        assert not np.isfinite(f_bare) and not np.isfinite(bare_min)
        assert n_high == 0

    def test_flat_trace_reports_one_dressed_plateau(self):
        """A window that never left one plateau (anchor separation below the
        noise floor, or no credible second plateau) is one branch, reported as
        dressed, spanning the window."""
        rng = np.random.default_rng(7)
        power = np.linspace(-40.0, 0.0, 21)
        center = 6.02e9 + rng.normal(0, 20e3, power.size)  # flat, 20 kHz noise
        (f_dress0, f_bare, dress_max, bare_min,
         n_low, n_high, cls) = _branch_frequencies(
            power, center, np.ones(power.size, bool),
            band_frac=0.08, min_points=3, anchor_points=3)
        assert np.isfinite(f_dress0) and not np.isfinite(f_bare)
        assert dress_max == pytest.approx(0.0)  # the whole window is one plateau
        assert not np.isfinite(bare_min)
        assert n_low == power.size and n_high == 0
        assert (cls == 1).all()


def _bimodal_width_dataset(n_power=24, n_det=201):
    """A punchout whose BARE dip is 3x narrower than the dressed one — the real
    5Q4C ratio (8 -> 3 MHz). Both plateaus are sharp; only the width changes."""
    power = np.linspace(-45.0, -21.0, n_power)   # 24 pts, step ~1.04 dBm
    detuning = np.linspace(-6e6, 6e6, n_det)
    lo = 7.0e9
    shift, p_trans, width_dbm = 3.0e6, -33.0, 1.0
    center_det = shift * 0.5 * (1.0 - np.tanh((power - p_trans) / width_dbm))
    frac = 0.5 * (1.0 + np.tanh((power - p_trans) / width_dbm))  # 0 dressed -> 1 bare
    gamma = 0.9e6 - 0.6e6 * frac                 # HWHM 0.9 (dressed) -> 0.3 (bare)
    iq = np.empty((n_power, n_det), dtype=complex)
    for k in range(n_power):
        p = 1.0 - 0.85 * gamma[k] ** 2 / ((detuning - center_det[k]) ** 2 + gamma[k] ** 2)
        iq[k] = np.sqrt(np.clip(p, 1e-9, None)).astype(complex)
    return xr.Dataset(
        {"IQdata": (("power", "detuning"), iq)},
        coords={"power": power, "detuning": detuning,
                "full_freq": ("detuning", lo + detuning)},
    ), lo


class TestNoPopulationWidthGate:
    def test_bimodal_linewidth_keeps_every_slice(self):
        """The gate fix, pinned end-to-end: dressed and bare dips with a 3:1
        width ratio are BOTH physics, so nothing may be flagged as an outlier
        and both branches must be recovered. The v1 global median/MAD width
        gate fails this test — it deletes whichever plateau is smaller (on run
        20260818-204626, the entire bare plateau)."""
        ds, lo = _bimodal_width_dataset()
        r = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        assert r["n_outlier"] == 0
        assert r["n_good"] == r["n_power"]
        assert r["branch_success"] is True
        assert r["f_dress0"] == pytest.approx(lo + 3.0e6, abs=100e3)
        assert r["f_bare"] == pytest.approx(lo, abs=100e3)
        assert r["dress_max_power"] < r["bare_min_power"]

    def test_dead_slice_does_not_move_the_branches(self):
        """One dead slice (fit fails, excluded upstream) mid-transition leaves
        both branch frequencies exactly where the clean map put them — the
        classifier only ever sees good points."""
        ds, _ = _bimodal_width_dataset()
        clean = ResonatorSpectroscopyPowerEstimator().extract_parameters(ds)
        iq = ds["IQdata"].values.copy()
        iq[12] = np.nan + 0j  # mid-transition slice dies
        broken = ds.assign(IQdata=(("power", "detuning"), iq))
        r = ResonatorSpectroscopyPowerEstimator().extract_parameters(broken)
        assert r["n_good"] == clean["n_good"] - 1
        assert r["f_dress0"] == pytest.approx(clean["f_dress0"])
        assert r["f_bare"] == pytest.approx(clean["f_bare"])


class TestChainProvenance:
    """Per-power (digital_amp, chain_setting, chain_name) coords — the ONE shared
    provenance form both scqo punchouts emit — pass through to plot_data data_vars
    and draw the amp/chain subplot under the map (two-row figure, shared power
    axis). The v0.1.6-dev scalar form (power_ref/amp_ref/chain_label secondary
    axis, power_offset_dbm axis shift) was removed before release: legacy coords
    are simply ignored."""

    def _stepped_dataset(self):
        ds, truth = _make_dataset()
        n = ds.sizes["power"]
        digital_amp = np.clip(0.5 - 0.01 * np.arange(n), 0.05, 0.5)
        chain_setting = np.repeat(np.arange((n + 1) // 2) * 2.0, 2)[:n]  # even att steps
        return ds.assign_coords(
            digital_amp=("power", digital_amp),
            chain_setting=("power", chain_setting),
            chain_name="output_att (dB)",
        ), digital_amp, chain_setting

    def test_plot_data_carries_per_power_vars(self):
        ds, digital_amp, chain_setting = self._stepped_dataset()
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        assert np.allclose(pd["digital_amp"].values, digital_amp)
        assert np.allclose(pd["chain_setting"].values, chain_setting)
        assert pd.attrs["chain_name"] == "output_att (dB)"

    def test_figure_gains_the_chain_subplot(self):
        ds, _, _ = self._stepped_dataset()
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        fig = plot_power_map(pd)
        ylabels = [a.get_ylabel() for a in fig.axes]
        assert any("digital amp" in lbl for lbl in ylabels)      # bottom-left axis
        assert any("output_att" in lbl for lbl in ylabels)       # bottom-right twin
        # the x-label moved to the bottom row
        xlabels = {a.get_xlabel() for a in fig.axes}
        assert "Readout power (dB)" in xlabels
        plt.close(fig)

    def test_subplot_roundtrips_through_netcdf(self, tmp_path):
        ds, _, _ = self._stepped_dataset()
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        path = tmp_path / "plotdata.nc"
        pd.to_netcdf(path)
        back = xr.load_dataset(path)
        fig = plot_power_map(back)
        assert any("digital amp" in a.get_ylabel() for a in fig.axes)
        plt.close(fig)

    def test_amp_sweep_shape_renders_constant_chain(self):
        # the fast punchout's shape: amp sweeps down, the chain setting is FLAT
        ds, _ = _make_dataset()
        n = ds.sizes["power"]
        power = ds["power"].values
        ds = ds.assign_coords(
            digital_amp=("power", 0.5 * 10.0 ** ((power - power[-1]) / 20.0)),
            chain_setting=("power", np.full(n, 18.0)),
            chain_name="output_att (dB)",
        )
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        fig = plot_power_map(pd)
        assert any("digital amp" in a.get_ylabel() for a in fig.axes)
        assert any("output_att" in a.get_ylabel() for a in fig.axes)
        plt.close(fig)

    def test_absent_coords_leave_figure_unchanged(self):
        ds, _ = _make_dataset()
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        assert "digital_amp" not in pd.data_vars
        fig = plot_power_map(pd)
        assert len(fig.axes) == 2  # map + colorbar only, no subplot row
        assert "\n" not in fig.axes[0].get_title()
        assert fig.axes[0].get_xlabel() == "Readout power (dB)"
        plt.close(fig)

    def test_legacy_scalar_coords_are_ignored(self):
        # pre-release plumbing (power_ref/amp_ref/chain_label/power_offset_dbm) is
        # gone: such coords neither land in attrs nor alter the figure
        ds, _ = _make_dataset()
        ds = ds.assign_coords(power_ref=-25.0, amp_ref=0.47, power_offset_dbm=-6.0,
                              chain_label="output_att=18 dB")
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        for key in ("power_ref", "amp_ref", "chain_label", "power_offset_dbm"):
            assert key not in pd.attrs
        fig = plot_power_map(pd)
        assert not fig.axes[0].child_axes  # no secondary digital-amplitude axis
        assert "output_att=18 dB" not in fig.axes[0].get_title()
        plt.close(fig)


class TestAxisKindAndModeLabel:
    """power_axis_kind labels the x-axis, mode_label tags the mechanism in the
    title — attached by both scqo punchouts (absent -> old rendering)."""

    def test_labels_pass_through_and_render(self):
        ds, _ = _make_dataset()
        ds = ds.assign_coords(power_axis_kind="absolute dBm",
                              mode_label="amplitude sweep (fast)")
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        assert pd.attrs["power_axis_kind"] == "absolute dBm"
        assert pd.attrs["mode_label"] == "amplitude sweep (fast)"

        fig = plot_power_map(pd)
        ax = fig.axes[0]
        assert "absolute dBm" in ax.get_xlabel()
        assert "amplitude sweep (fast)" in ax.get_title()
        # the optimal-marker label carries the absolute unit
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("dBm" in l for l in labels)
        plt.close(fig)

    def test_mode_label_on_chain_stepped_figure(self):
        ds, _ = _make_dataset()
        n = ds.sizes["power"]
        ds = ds.assign_coords(
            digital_amp=("power", np.full(n, 0.5)),
            chain_setting=("power", np.arange(n, dtype=float)),
            chain_name="output_att (dB)",
            power_axis_kind="absolute dBm",
            mode_label="chain-stepped (slow)",
        )
        est = ResonatorSpectroscopyPowerEstimator()
        pd = est.build_plot_data(ds, est.extract_parameters(ds))
        fig = plot_power_map(pd)
        assert "chain-stepped (slow)" in fig.axes[0].get_title()
        assert any("absolute dBm" in a.get_xlabel() for a in fig.axes)  # bottom subplot label
        plt.close(fig)
