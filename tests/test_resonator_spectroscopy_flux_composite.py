"""End-to-end tests for the composite ResonatorSpectroscopyFluxAnalyzer.

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

from scqat.protocols import ResonatorSpectroscopyFluxAnalyzer
from scqat.protocols.resonator_spectroscopy_flux import (
    ResonatorSpectroscopyFluxAnalyzer as SubpkgAnalyzer,
)
from scqat.protocols.resonator_spectroscopy_flux.visualization import plot_combined


def _flux_dispersion(flux, f_r0, g, phi0, phi_off, f_q_max):
    f_q = f_q_max * np.sqrt(np.abs(np.cos(np.pi * (flux - phi_off) / phi0)))
    return f_r0 + g ** 2 / (f_r0 - f_q)


def _make_dataset(n_flux=21, n_det=121, noise=0.0, seed=0):
    """Resonator-vs-flux IQ map with a dispersive dip-centre(flux) trace."""
    rng = np.random.default_rng(seed)
    flux = np.linspace(-0.05, 0.05, n_flux)
    detuning = np.linspace(-3e6, 3e6, n_det)
    lo = 7.0e9
    full_freq = lo + detuning  # (detuning,)

    truth = dict(f_r0=7.0e9, g=50e6, phi0=0.1, phi_off=0.0, f_q_max=5.0e9)
    centers_abs = _flux_dispersion(flux, **truth)        # absolute Hz, per flux
    center_det = centers_abs - lo                        # dip detuning, per flux

    gamma, depth = 0.3e6, 0.85
    iq = np.empty((n_flux, n_det), dtype=complex)
    for k in range(n_flux):
        power = 1.0 - depth * gamma ** 2 / ((detuning - center_det[k]) ** 2 + gamma ** 2)
        amp = np.sqrt(np.clip(power, 1e-9, None))
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


class TestResonatorSpectroscopyFluxComposite:
    def test_aggregated_and_subpackage_imports_match(self):
        assert ResonatorSpectroscopyFluxAnalyzer is SubpkgAnalyzer
        assert ResonatorSpectroscopyFluxAnalyzer.protocol_name == "resonator_spectroscopy_flux"

    def test_nested_results_structure(self):
        ds, _ = _make_dataset()
        results = ResonatorSpectroscopyFluxAnalyzer().extract_parameters(ds)
        assert set(results) == {"vs_flux", "dispersion"}
        # Most flux slices yield a good dip.
        assert results["vs_flux"]["n_good"] >= 18

    def test_recovers_sweet_spot(self):
        ds, truth = _make_dataset()
        analyzer = ResonatorSpectroscopyFluxAnalyzer()
        results = analyzer.extract_parameters(ds)
        disp = results["dispersion"]
        assert disp["success"] is True
        # Sweet spot is the well-determined (degeneracy-independent) output.
        assert disp["sweet_spot_flux"] == pytest.approx(truth["phi_off"], abs=0.012)
        assert np.isfinite(disp["dv_phi0"]) and disp["dv_phi0"] > 0

    def test_metadata_projection(self):
        ds, _ = _make_dataset()
        analyzer = ResonatorSpectroscopyFluxAnalyzer()
        results = analyzer.extract_parameters(ds)
        meta = analyzer.extract_metadata(results)
        # Flat, JSON-friendly scalars only — no nested stage dicts.
        for key in ("n_flux", "n_good", "sweet_spot_flux", "sweet_spot_freq",
                    "dv_phi0", "f_r0", "g", "dispersion_success"):
            assert key in meta
        assert "vs_flux" not in meta and "dispersion" not in meta

    def test_plot_data_is_self_sufficient(self):
        ds, _ = _make_dataset()
        analyzer = ResonatorSpectroscopyFluxAnalyzer()
        results = analyzer.extract_parameters(ds)
        pd = analyzer.build_plot_data(ds, results)

        assert isinstance(pd, xr.Dataset)
        for var in ("amplitude", "center_full_freq", "good", "outlier", "fit_freq"):
            assert var in pd
        for coord in ("flux_bias", "detuning", "fit_flux", "full_freq"):
            assert coord in pd.coords
        assert "sweet_spot_flux" in pd.attrs
        # The 2-D map is oriented (flux_bias, detuning).
        assert pd["amplitude"].dims == ("flux_bias", "detuning")

    def test_generate_figures_from_plot_data_only(self):
        ds, _ = _make_dataset()
        analyzer = ResonatorSpectroscopyFluxAnalyzer()
        results = analyzer.extract_parameters(ds)
        pd = analyzer.build_plot_data(ds, results)

        figs = analyzer.generate_figures(None, None, plot_data=pd)
        assert set(figs) == {"resonator_spectroscopy_flux"}
        assert isinstance(figs["resonator_spectroscopy_flux"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_and_reconstructable_figure(self, tmp_path):
        ds, _ = _make_dataset()
        analyzer = ResonatorSpectroscopyFluxAnalyzer()
        results, figs = analyzer.analyze(ds, output_dir=str(tmp_path))

        # Artifacts written with the protocol_name prefix.
        assert (tmp_path / "resonator_spectroscopy_flux_metadata.json").exists()
        assert (tmp_path / "resonator_spectroscopy_flux_plotdata.nc").exists()
        assert isinstance(figs["resonator_spectroscopy_flux"], plt.Figure)

        # Reconstruct the figure from the reloaded plot_data alone (no re-analysis).
        reloaded = analyzer.load_plot_data(str(tmp_path))
        fig = plot_combined(reloaded)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
