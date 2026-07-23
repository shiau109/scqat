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
        # complex input -> the shared IQ-plane panel rides along
        assert "qubit_flux_arch_iq_plane.png" in names

    def test_requires_full_freq(self):
        ds = _make_map(n_flux=8).drop_vars("full_freq")
        with pytest.raises(ValueError, match="full_freq"):
            QubitFluxArchEstimator().analyze(ds)

    def test_gentle_top_recovers_when_period_far_exceeds_span(self):
        """The bring-up window case (modeled on the real 5Q4C run): only the
        GENTLE TOP of a period~1 V arch inside a +-0.1 V / 200 MHz window. The
        curvature-based period seed must land the right basin — the old
        period=span seed converged to a wrong basin with an off-window top."""
        ec, ej, offset, period = 0.2, 17.8, 0.013, 0.96
        flux = np.linspace(-0.1, 0.1, 11)
        f01_now = 5.1364e9
        full = np.linspace(f01_now - 100e6, f01_now + 100e6, 201)
        rng = np.random.default_rng(11)
        amp = np.zeros((flux.size, full.size))
        for k, fb in enumerate(flux):
            q = (fb - offset) / period
            f01 = (np.sqrt(8 * ec * ej * abs(np.cos(np.pi * q))) - ec) * 1e9
            amp[k] += 1.0 / (1.0 + ((full - f01) / (6e6 / 2)) ** 2)
        amp += rng.normal(0, 0.02, size=amp.shape)
        ds = xr.Dataset(
            {"I": (("flux_bias", "detuning"), amp),
             "Q": (("flux_bias", "detuning"), np.zeros_like(amp))},
            coords={"flux_bias": flux, "detuning": full - f01_now,
                    "full_freq": ("detuning", full)},
        )
        arch = QubitFluxArchEstimator().extract_parameters(ds)["arch"]
        assert arch["success"]
        assert arch["sweet_spot_flux"] == pytest.approx(offset, abs=0.01)
        assert arch["ej_sum_ghz"] == pytest.approx(ej, rel=0.05)
        # the top was observed inside the window and the curve tracks the points
        assert full.min() <= arch["f01_max_hz"] <= full.max()
        assert arch["rms_residual_hz"] < 0.25 * (full.max() - full.min())

    def test_off_window_top_fails_the_gate(self):
        """Flank-only data: the arch top was never observed (it sits above the
        swept frequency window), so however well the optimizer converges the
        result must be gated FAILED — extrapolated Ej/f01_max are untrustworthy."""
        flux = np.linspace(0.13, 0.33, 11)  # one descending flank of the arch
        full = np.linspace(4.6e9, 5.5e9, 201)  # window far below f01_max ~ 6.125 GHz
        lo = 5.0e9
        rng = np.random.default_rng(5)
        amp = np.zeros((flux.size, full.size))
        for k, fb in enumerate(flux):
            f01 = _f01_ghz(fb) * 1e9
            amp[k] += 1.0 / (1.0 + ((full - f01) / (12e6 / 2)) ** 2)
        amp += rng.normal(0, 0.02, size=amp.shape)
        ds = xr.Dataset(
            {"I": (("flux_bias", "detuning"), amp),
             "Q": (("flux_bias", "detuning"), np.zeros_like(amp))},
            coords={"flux_bias": flux, "detuning": full - lo,
                    "full_freq": ("detuning", full)},
        )
        est = QubitFluxArchEstimator()
        results = est.extract_parameters(ds)
        arch = results["arch"]
        assert arch["n_selected"] >= 5  # the flank line itself was found
        assert not arch["success"]  # ...but the top was never measured
        meta = est.extract_metadata(results)
        for key in ("rms_residual_hz", "freq_window_lo_hz", "freq_window_hi_hz"):
            assert key in meta
