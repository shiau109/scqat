"""QubitFluxArchEstimator: branch selection + transmon arch recovery on a
synthetic f01(flux) map with a spurious weaker line."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_flux_arch import QubitFluxArchEstimator

EC = 0.2  # GHz
EJ_SUM = 25.0  # GHz
OFFSET = 0.05  # V
PERIOD = 0.8  # V


def _f01_ghz(flux):
    q = (flux - OFFSET) / PERIOD
    ej_eff = EJ_SUM * np.abs(np.cos(np.pi * q))
    return np.sqrt(8.0 * EC * ej_eff) - EC


def _make_map(n_flux=25, n_freq=201, noise_std=0.02, with_spurious=True):
    flux = np.linspace(-0.23, 0.33, n_flux)  # q in [-0.35, 0.35] around the arch top
    full = np.linspace(4.4e9, 6.6e9, n_freq)
    lo = 5.5e9
    detuning = full - lo
    rng = np.random.default_rng(3)

    amp = np.zeros((n_flux, n_freq))
    fwhm = 12e6
    for k, fb in enumerate(flux):
        f01 = _f01_ghz(fb) * 1e9
        amp[k] += 1.0 / (1.0 + ((full - f01) / (fwhm / 2)) ** 2)
        if with_spurious:  # two-photon 0-2/2 line: Ec/2 below, weaker
            f02 = f01 - EC / 2 * 1e9
            amp[k] += 0.35 / (1.0 + ((full - f02) / (fwhm / 2)) ** 2)
    amp += rng.normal(0, noise_std, size=amp.shape)

    return xr.Dataset(
        {
            "I": (("flux_bias", "detuning"), amp),
            "Q": (("flux_bias", "detuning"), np.zeros_like(amp)),
        },
        coords={
            "flux_bias": flux,
            "detuning": detuning,
            "full_freq": ("detuning", full),
        },
    )


class TestQubitFluxArchEstimator:
    def test_arch_recovery_with_spurious_line(self):
        results, _ = QubitFluxArchEstimator().analyze(_make_map(), skip_figures=True)
        arch = results["arch"]
        assert arch["success"]
        assert arch["sweet_spot_flux"] == pytest.approx(OFFSET, abs=0.02)
        assert arch["ej_sum_ghz"] == pytest.approx(EJ_SUM, rel=0.1)
        assert arch["f01_max_hz"] == pytest.approx((np.sqrt(8 * EC * EJ_SUM) - EC) * 1e9, rel=0.02)
        assert arch["n_used"] >= 15

    def test_metadata_is_scalar_only(self):
        est = QubitFluxArchEstimator()
        results = est.extract_parameters(_make_map(n_flux=15))
        meta = est.extract_metadata(results)
        assert {"sweet_spot_flux", "ej_sum_ghz", "f01_max_hz", "arch_success"} <= set(meta)
        assert all(np.isscalar(v) or isinstance(v, (str, bool)) for v in meta.values())

    def test_artifacts_and_figure_name(self, tmp_path):
        QubitFluxArchEstimator().analyze(_make_map(n_flux=15), output_dir=str(tmp_path))
        names = {p.name for p in tmp_path.iterdir()}
        assert "qubit_flux_arch_metadata.json" in names
        assert "qubit_flux_arch_plotdata.nc" in names
        assert "qubit_flux_arch.png" in names

    def test_requires_full_freq(self):
        ds = _make_map(n_flux=8).drop_vars("full_freq")
        with pytest.raises(ValueError, match="full_freq"):
            QubitFluxArchEstimator().analyze(ds)
