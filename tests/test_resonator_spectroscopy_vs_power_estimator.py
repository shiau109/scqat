"""Tests for the ResonatorSpectroscopyVsPowerEstimator.

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

from scqat.estimators import ResonatorSpectroscopyVsPowerEstimator
from scqat.estimators.resonator_spectroscopy_vs_power import (
    ResonatorSpectroscopyVsPowerEstimator as SubpkgEstimator,
)
from scqat.estimators.resonator_spectroscopy_vs_power.visualization import plot_power_map


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


class TestResonatorSpectroscopyVsPower:
    def test_aggregated_and_subpackage_imports_match(self):
        assert ResonatorSpectroscopyVsPowerEstimator is SubpkgEstimator
        assert ResonatorSpectroscopyVsPowerEstimator.estimator_name == "resonator_spectroscopy_vs_power"

    def test_results_structure_and_good_points(self):
        ds, _ = _make_dataset()
        results = ResonatorSpectroscopyVsPowerEstimator().extract_parameters(ds)
        for key in ("power", "detuning", "center_detuning", "good", "amplitude_map",
                    "n_power", "n_good", "optimal_power", "frequency_shift",
                    "resonator_frequency", "optimal_success"):
            assert key in results
        assert results["n_power"] == 30
        # Most power slices yield a clean dip.
        assert results["n_good"] >= 26
        # The 2-D map is oriented (power, detuning).
        assert results["amplitude_map"].shape == (30, 121)

    def test_center_trace_tracks_truth(self):
        ds, truth = _make_dataset()
        results = ResonatorSpectroscopyVsPowerEstimator().extract_parameters(ds)
        good = results["good"]
        centre = results["center_detuning"]
        # Fitted centres match the synthetic dip positions on the good points.
        assert np.allclose(centre[good], truth["center_det"][good], atol=0.1e6)

    def test_picks_optimal_power_in_dispersive_regime(self):
        ds, truth = _make_dataset()
        results = ResonatorSpectroscopyVsPowerEstimator().extract_parameters(ds)
        assert results["optimal_success"] is True
        opt = results["optimal_power"]
        assert np.isfinite(opt)
        # Optimal power lands within the swept range, at/below the transition.
        assert truth["power"].min() <= opt <= truth["power"].max()
        assert opt <= truth["p_trans"] + 1.0
        # Resonator frequency reported on the absolute axis.
        assert np.isfinite(results["resonator_frequency"])
        assert abs(results["resonator_frequency"] - truth["lo"]) < 3e6

    def test_metadata_drops_bulky_arrays(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyVsPowerEstimator()
        results = estimator.extract_parameters(ds)
        meta = estimator.extract_metadata(results)
        for dropped in ("amplitude_map", "detuning", "full_freq"):
            assert dropped not in meta
        for kept in ("optimal_power", "frequency_shift", "n_good", "center_detuning"):
            assert kept in meta

    def test_plot_data_self_sufficient_and_figure(self):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyVsPowerEstimator()
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
        assert set(figs) == {"resonator_spectroscopy_vs_power"}
        assert isinstance(figs["resonator_spectroscopy_vs_power"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip(self, tmp_path):
        ds, _ = _make_dataset()
        estimator = ResonatorSpectroscopyVsPowerEstimator()
        results, figs = estimator.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "resonator_spectroscopy_vs_power_metadata.json").exists()
        assert (tmp_path / "resonator_spectroscopy_vs_power_plotdata.nc").exists()
        assert isinstance(figs["resonator_spectroscopy_vs_power"], plt.Figure)

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
        results = ResonatorSpectroscopyVsPowerEstimator().extract_parameters(ds_iq)
        assert results["n_good"] >= 26
        assert results["optimal_success"] is True
