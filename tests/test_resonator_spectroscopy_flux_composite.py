"""End-to-end tests for the composite ResonatorSpectroscopyFluxEstimator.

Synthesises a resonator-spectroscopy-vs-flux map whose dip centre follows the
flux-tunable-transmon dispersive model, then checks that the composite chains the
two stages, produces a merged plot_data Dataset, and that the combined figure is
reconstructable from saved plot_data alone (the contract's reconstructability
guarantee).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import ResonatorSpectroscopyFluxEstimator
from scqat.estimators.resonator_spectroscopy_flux import (
    ResonatorSpectroscopyFluxEstimator as SubpkgEstimator,
    fit_flux_trace,
    track_dips,
)
from scqat.estimators.resonator_spectroscopy_flux.visualization import plot_combined


def _flux_dispersion(flux, f_r0, g, phi0, phi_off, f_q_max, ec=0.0):
    f_q = (f_q_max + ec) * np.sqrt(np.abs(np.cos(np.pi * (flux - phi_off) / phi0))) - ec
    return f_r0 + g ** 2 / (f_r0 - f_q)


def _make_dataset(n_flux=21, n_det=121, noise=0.0, seed=0, bg_slope=0.0):
    """Resonator-vs-flux IQ map with a dispersive dip-centre(flux) trace.

    ``bg_slope`` adds a linear ``|IQ|`` background across detuning (fraction of
    the off-resonance level dropped from low to high detuning) — the sloped
    baseline that sends a naive argmin to a dark window edge instead of the dip.
    """
    rng = np.random.default_rng(seed)
    flux = np.linspace(-0.05, 0.05, n_flux)
    detuning = np.linspace(-3e6, 3e6, n_det)
    lo = 7.0e9
    full_freq = lo + detuning  # (detuning,)

    truth = dict(f_r0=7.0e9, g=50e6, phi0=0.1, phi_off=0.0, f_q_max=5.0e9)
    centers_abs = _flux_dispersion(flux, **truth)        # absolute Hz, per flux
    center_det = centers_abs - lo                        # dip detuning, per flux

    ramp = 1.0 - bg_slope * (detuning - detuning[0]) / (detuning[-1] - detuning[0])
    gamma, depth = 0.3e6, 0.85
    iq = np.empty((n_flux, n_det), dtype=complex)
    for k in range(n_flux):
        power = 1.0 - depth * gamma ** 2 / ((detuning - center_det[k]) ** 2 + gamma ** 2)
        amp = np.sqrt(np.clip(power, 1e-9, None)) * ramp
        if noise > 0:
            amp = amp + rng.normal(0, noise, size=amp.shape)
        iq[k] = amp.astype(complex)

    ds = xr.Dataset(
        {"IQdata": (("flux_bias", "detuning"), iq)},
        coords={
            "flux_bias": flux,
            "detuning": detuning,
            "full_freq": ("detuning", full_freq),
        },
    )
    return ds, truth


def _make_complex_dataset(n_flux=21, n_det=121, ql=10_000, qc_abs=20_000):
    """Resonator-vs-flux map with a NOTCH-MODEL complex S21 per slice (real
    phase content), so the ``circle`` dip method can fit — simulated-style
    magnitude-only data (noise Q quadrature) cannot exercise it."""
    flux = np.linspace(-0.05, 0.05, n_flux)
    detuning = np.linspace(-3e6, 3e6, n_det)
    lo = 7.0e9
    full_freq = lo + detuning

    truth = dict(f_r0=7.0e9, g=50e6, phi0=0.1, phi_off=0.0, f_q_max=5.0e9)
    centers_abs = _flux_dispersion(flux, **truth)

    iq = np.empty((n_flux, n_det), dtype=complex)
    for k in range(n_flux):
        fr = centers_abs[k]
        iq[k] = 1.0 - (ql / qc_abs) / (1.0 + 2j * ql * (full_freq / fr - 1.0))

    ds = xr.Dataset(
        {"IQdata": (("flux_bias", "detuning"), iq)},
        coords={
            "flux_bias": flux,
            "detuning": detuning,
            "full_freq": ("detuning", full_freq),
        },
    )
    return ds, truth


def _make_dataset_with_edge_artifacts(n_edge=3, **kw):
    """A clean dispersive map with the first ``n_edge`` flux slices' dips pinned
    near the top detuning edge — the spurious 'centre' a low-SNR slice produces
    when the bounded single-dip fit hits its bound. Returns
    ``(ds, truth, edge_indices)``."""
    ds, truth = _make_dataset(**kw)
    detuning = ds["detuning"].values.astype(float)
    det_lo, det_hi = float(detuning.min()), float(detuning.max())
    # A clean dip ~2 linewidths inside the top edge: recovered without pinning to
    # the bound (so it survives at edge_margin_frac=0), yet inside a wider margin.
    edge_center = det_hi - 0.10 * (det_hi - det_lo)
    gamma, depth = 0.3e6, 0.85
    iq = ds["IQdata"].values.copy()
    for k in range(n_edge):
        power = 1.0 - depth * gamma ** 2 / ((detuning - edge_center) ** 2 + gamma ** 2)
        iq[k] = np.sqrt(np.clip(power, 1e-9, None)).astype(complex)
    ds = ds.copy()
    ds["IQdata"] = (("flux_bias", "detuning"), iq)
    return ds, truth, list(range(n_edge))


class TestResonatorSpectroscopyFluxComposite:
    def test_aggregated_and_subpackage_imports_match(self):
        assert ResonatorSpectroscopyFluxEstimator is SubpkgEstimator
        assert ResonatorSpectroscopyFluxEstimator.estimator_name == "resonator_spectroscopy_flux"

    def test_nested_results_structure(self):
        ds, _ = _make_dataset()
        results = ResonatorSpectroscopyFluxEstimator().extract_parameters(ds)
        assert set(results) == {"vs_flux", "dispersion"}
        # Most flux slices yield a good dip.
        assert results["vs_flux"]["n_good"] >= 18

    def test_recovers_sweet_spot_point(self):
        ds, truth = _make_dataset()
        estimator = ResonatorSpectroscopyFluxEstimator()
        results = estimator.extract_parameters(ds)
        disp = results["dispersion"]
        assert disp["success"] is True
        assert disp["method"] == "dispersive"  # default method
        # The sweet-spot flux is the well-determined
        # (degeneracy-independent) output.
        assert disp["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)
        assert np.isfinite(disp["dv_phi0"]) and disp["dv_phi0"] > 0

    def test_sine_method_runs_and_reports_common_keys(self):
        ds, truth = _make_dataset()
        results = ResonatorSpectroscopyFluxEstimator().extract_parameters(ds, method="sine")
        disp = results["dispersion"]
        assert disp["method"] == "sine"
        assert disp["success"] is True
        # COMMON contract holds for every method...
        for key in ("sweet_spot_flux", "sweet_spot_res",
                    "sweet_spot_low_flux", "sweet_spot_low_res",
                    "dv_phi0", "success"):
            assert key in disp
        # ...and the sine method carries NO dispersive physics.
        assert "f_r0" not in disp and "g" not in disp
        assert disp["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)

    def test_cross_method_agreement_on_max_freq(self):
        """A method changes robustness, not physics: dispersive and sine must
        agree on the sweet-spot point (the COMMON physics)."""
        ds, _ = _make_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        d = est.extract_parameters(ds, method="dispersive")["dispersion"]
        s = est.extract_parameters(ds, method="sine")["dispersion"]
        assert d["sweet_spot_flux"] == pytest.approx(s["sweet_spot_flux"], abs=0.01)
        assert d["sweet_spot_res"] == pytest.approx(s["sweet_spot_res"], rel=2e-4)

    def test_edge_pinned_centres_are_rejected(self):
        """Edge-pinned dip centres (low-SNR slices) must be dropped before the
        flux-model fit, and dropping them must recover the true sweet spot."""
        ds, truth, edge_idx = _make_dataset_with_edge_artifacts(n_edge=3)
        est = ResonatorSpectroscopyFluxEstimator()

        # A margin wider than the artifacts' 10% inset rejects them...
        kept = est.extract_parameters(ds, edge_margin_frac=0.15)
        assert not kept["vs_flux"]["good"][edge_idx].any()
        # ...and dropping them still recovers the true sweet spot.
        assert kept["dispersion"]["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)

        # With the gate disabled the near-edge centres survive as good (knob works).
        leaked = est.extract_parameters(ds, edge_margin_frac=0.0)
        assert leaked["vs_flux"]["good"][edge_idx].all()
        assert leaked["vs_flux"]["n_good"] > kept["vs_flux"]["n_good"]

    def test_stage1_refined_yield_and_depth_gate(self):
        """Argmin-guided local fitting refines every slice of a clean map; a
        dip-less slice fails the depth gate instead of producing a bogus centre."""
        ds, _ = _make_dataset()
        vs = track_dips(ds)
        assert vs["n_success"] == vs["n_flux"]
        assert vs["refined"][vs["success"]].all()
        assert vs["n_refined"] == vs["n_flux"]

        # Kill the dip on one slice -> only that slice fails (depth gate).
        iq = ds["IQdata"].values.copy()
        iq[0, :] = 1.0
        ds2 = ds.copy()
        ds2["IQdata"] = (("flux_bias", "detuning"), iq)
        vs2 = track_dips(ds2)
        assert not vs2["success"][0]
        assert vs2["n_success"] == vs2["n_flux"] - 1

    def test_stage1_tracks_dip_under_sloped_background(self):
        """A strong |IQ| slope across detuning (bright at one edge, dark at the
        other) must not send the dip candidate to the dark window edge — the
        background is detrended before the argmin. Regression for run
        20260722-092830 q1 (3/21 kept, all at the top edge)."""
        # 60% brightness drop across the window — the failing map's regime.
        ds, truth = _make_dataset(bg_slope=0.6)
        vs = track_dips(ds)
        assert vs["n_good"] >= 18
        # Every kept centre is the real dip, near the ~0 detuning arch — NOT the
        # bright/dark window edges (|detuning| up to 3 MHz).
        centers = vs["center_detuning"][vs["good"]]
        assert np.all(np.abs(centers) < 1.5e6)
        # And the flux model still recovers the sweet spot.
        disp = ResonatorSpectroscopyFluxEstimator().extract_parameters(ds)["dispersion"]
        assert disp["success"] is True
        assert disp["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)

    def test_period_multiseeding_defeats_half_period_aliasing(self):
        """The q3-002209 failure mode: with the points around the arch minima
        missing, a single FFT seed locks onto T/2. The multi-seed fit must
        recover the full period for both methods."""
        T, phi_off = 0.6, 0.1
        flux = np.linspace(-0.4, 0.4, 41)
        y = _flux_dispersion(flux, f_r0=5.86e9, g=100e6, phi0=T,
                             phi_off=phi_off, f_q_max=4.4e9)
        # distance to the nearest arch minimum (phi_off + T/2 mod T)
        d = np.abs((flux - (phi_off + T / 2) + T / 2) % T - T / 2)
        good = d > 0.15 * T  # drop the points that pin the minima
        trace = xr.Dataset(
            {"center_freq": ("flux_bias", y), "success": ("flux_bias", good)},
            coords={"flux_bias": flux},
        )
        for method in ("dispersive", "sine"):
            res = fit_flux_trace(trace, method=method)
            assert res["success"] is True
            # The T/2 alias sits at 0.5*T — anything above 0.7*T is the real arch.
            assert res["dv_phi0"] > 0.7 * T, (method, res["dv_phi0"])
            assert res["dv_phi0"] < 1.5 * T, (method, res["dv_phi0"])
            assert res["sweet_spot_flux"] == pytest.approx(phi_off, abs=0.05)

    def test_dip_method_circle_threads_and_stamps(self):
        """dip_method is a plain, flat kwarg: circle fits every slice of a
        complex notch map, its provenance is stamped everywhere, and its slices
        (no dip amplitude — a lorentzian-owned extra) are exempt from the
        amplitude MAD gate."""
        ds, truth = _make_complex_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        res = est.extract_parameters(ds, dip_method="circle")
        vs, disp = res["vs_flux"], res["dispersion"]
        assert vs["dip_method"] == "circle"
        assert vs["n_refined"] >= vs["n_flux"] - 2  # circle really fit the slices
        assert np.isnan(vs["dip_amplitude"][vs["refined"]]).all()
        assert disp["success"] is True
        assert disp["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)
        # Provenance in all three artifacts.
        assert est.extract_metadata(res)["dip_method"] == "circle"
        assert est.build_plot_data(ds, res).attrs["dip_method"] == "circle"

    def test_unknown_kwargs_fail_loudly_before_fitting(self):
        """The flat surface rejects wrong names with actionable errors: a bogus
        dip method, a typo'd knob, and the historical mistake of passing a DIP
        method as the flux-model method."""
        ds, _ = _make_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        with pytest.raises(ValueError, match="dip method"):
            est.extract_parameters(ds, dip_method="bogus")
        with pytest.raises(ValueError, match="basline_order"):
            est.extract_parameters(ds, basline_order=2)  # typo'd knob
        with pytest.raises(ValueError, match="dispersive"):
            est.extract_parameters(ds, method="circle")  # dip method != flux model

    def test_lower_sweet_spot_reported(self):
        """Both methods report the arch bottom (lower sweet spot) mapped into
        the swept range, half a period from the upper spot."""
        ds, truth = _make_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        for method in ("dispersive", "sine"):
            disp = est.extract_parameters(ds, method=method)["dispersion"]
            low_f, low_r = disp["sweet_spot_low_flux"], disp["sweet_spot_low_res"]
            # truth: phi_off=0, phi0=0.1, sweep ±0.05 -> minima at both edges.
            assert abs(abs(low_f) - truth["phi0"] / 2) < 0.012, (method, low_f)
            assert low_r < disp["sweet_spot_res"], method
            assert np.isfinite(low_r)

    def test_metadata_projection(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyFluxEstimator()
        results = estimator.extract_parameters(ds)
        meta = estimator.extract_metadata(results)
        # Flat, JSON-friendly scalars only — no nested stage dicts.
        for key in ("n_flux", "n_good", "method", "sweet_spot_flux", "sweet_spot_res",
                    "dv_phi0", "f_r0", "g", "ec", "dispersion_success"):
            assert key in meta
        assert "vs_flux" not in meta and "dispersion" not in meta

    def test_ec_corrected_arch_recovers_g_within_physical_bound(self):
        """With f_q_max and E_C supplied (the control repo sources both), the
        E_C-corrected arch recovers the true coupling — and the fit honours the
        physical 0 < g < f_r0/10 bound. Realistic small detuning + large g, where
        the E_C arch-shape correction actually bites (unlike the wide-detuning
        _make_dataset)."""
        f_r0, f_q_max, ec, g_true = 6.0e9, 5.2e9, 0.2e9, 90e6
        T, phi_off = 0.6, 0.1
        flux = np.linspace(-0.4, 0.4, 61)
        y = _flux_dispersion(flux, f_r0=f_r0, g=g_true, phi0=T,
                             phi_off=phi_off, f_q_max=f_q_max, ec=ec)
        trace = xr.Dataset({"center_freq": ("flux_bias", y)},
                           coords={"flux_bias": flux})
        res = fit_flux_trace(trace, method="dispersive", f_q_max=f_q_max, ec=ec)
        assert res["success"] is True
        assert res["ec"] == pytest.approx(ec)           # held fixed, echoed back
        assert res["g"] == pytest.approx(g_true, rel=0.05)
        assert 0.0 < res["g"] < f_r0 / 10.0             # the physical bound

    def test_wrong_ec_biases_g(self):
        """E_C is not fitted, so an assumed value the data can't correct shows up
        as a biased g — the motivation for sourcing it. Fitting the same
        E_C-corrected data with ec=0 recovers a g measurably off the truth."""
        f_r0, f_q_max, ec, g_true = 6.0e9, 5.2e9, 0.2e9, 90e6
        flux = np.linspace(-0.4, 0.4, 61)
        y = _flux_dispersion(flux, f_r0=f_r0, g=g_true, phi0=0.6,
                             phi_off=0.1, f_q_max=f_q_max, ec=ec)
        trace = xr.Dataset({"center_freq": ("flux_bias", y)},
                           coords={"flux_bias": flux})
        right = fit_flux_trace(trace, method="dispersive", f_q_max=f_q_max, ec=ec)
        wrong = fit_flux_trace(trace, method="dispersive", f_q_max=f_q_max, ec=0.0)
        assert abs(right["g"] - g_true) < abs(wrong["g"] - g_true)

    def test_pinned_f_r0_breaks_the_degeneracy(self):
        """f_r0 and g trade off along a valley (the trace fixes only the pull
        g^2/(f_r0-f_q)). Pinning f_r0 at the TRUE bare frequency must return it
        untouched and recover the true g."""
        f_r0, f_q_max, ec, g_true = 6.0e9, 5.2e9, 0.2e9, 90e6
        flux = np.linspace(-0.4, 0.4, 61)
        y = _flux_dispersion(flux, f_r0=f_r0, g=g_true, phi0=0.6,
                             phi_off=0.1, f_q_max=f_q_max, ec=ec)
        trace = xr.Dataset({"center_freq": ("flux_bias", y)},
                           coords={"flux_bias": flux})
        res = fit_flux_trace(trace, method="dispersive",
                             f_q_max=f_q_max, ec=ec, f_r0=f_r0)
        assert res["success"] is True
        assert res["f_r0_fixed"] is True
        assert res["f_r0"] == pytest.approx(f_r0)        # held, not fitted
        assert res["g"] == pytest.approx(g_true, rel=0.02)

    def test_f_r0_free_by_default_and_seedable(self):
        """Default = today's behaviour (f_r0 free, seeded from the trace).
        Passing fit_f_r0=True supplies a mere STARTING guess — the right choice
        for a nominal value, since an error in a pinned f_r0 lands in g."""
        ds, _ = _make_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        default = est.extract_parameters(ds)["dispersion"]
        assert default["f_r0_fixed"] is False

        # a deliberately-off seed, left free, must still converge near the truth
        seeded = est.extract_parameters(
            ds, f_r0=6.99e9, fit_f_r0=True)["dispersion"]
        assert seeded["f_r0_fixed"] is False
        assert seeded["success"] is True
        assert seeded["sweet_spot_flux"] == pytest.approx(0.0, abs=0.012)

    def test_ec_and_g_init_kwargs_thread_without_error(self):
        """ec / g_init are recognised flux-model kwargs (not misrouted to the dip
        stage), and ec is stamped through metadata and plot_data."""
        ds, _ = _make_dataset()
        est = ResonatorSpectroscopyFluxEstimator()
        res = est.extract_parameters(ds, ec=0.25e9, g_init=40e6)
        assert res["dispersion"]["ec"] == pytest.approx(0.25e9)
        assert est.extract_metadata(res)["ec"] == pytest.approx(0.25e9)
        assert est.build_plot_data(ds, res).attrs["ec"] == pytest.approx(0.25e9)

    def test_plot_data_is_self_sufficient(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyFluxEstimator()
        results = estimator.extract_parameters(ds)
        pd = estimator.build_plot_data(ds, results)

        assert isinstance(pd, xr.Dataset)
        for var in ("amplitude", "center_full_freq", "good", "outlier", "fit_freq"):
            assert var in pd
        for coord in ("flux_bias", "detuning", "fit_flux", "full_freq"):
            assert coord in pd.coords
        assert "sweet_spot_flux" in pd.attrs and "method" in pd.attrs
        # The 2-D map is oriented (flux_bias, detuning).
        assert pd["amplitude"].dims == ("flux_bias", "detuning")

    def test_generate_figures_from_plot_data_only(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyFluxEstimator()
        results = estimator.extract_parameters(ds)
        pd = estimator.build_plot_data(ds, results)

        figs = estimator.generate_figures(None, None, plot_data=pd)
        assert set(figs) == {"resonator_spectroscopy_flux"}
        assert isinstance(figs["resonator_spectroscopy_flux"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_and_reconstructable_figure(self, tmp_path):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyFluxEstimator()
        results, figs = estimator.analyze(ds, output_dir=str(tmp_path))

        # Artifacts written with the estimator_name prefix.
        assert (tmp_path / "resonator_spectroscopy_flux_metadata.json").exists()
        assert (tmp_path / "resonator_spectroscopy_flux_plotdata.nc").exists()
        assert isinstance(figs["resonator_spectroscopy_flux"], plt.Figure)

        # Reconstruct the figure from the reloaded plot_data alone (no re-analysis).
        reloaded = estimator.load_plot_data(str(tmp_path))
        fig = plot_combined(reloaded)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
