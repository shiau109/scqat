"""Tests for the SpectroscopyCryoscopeEstimator (long-time flux step response).

The forward model mirrors the probe: a flux pulse settles as a (distorted) step
``S(t)`` toward the parked amplitude, so the qubit's detuning from the sweet spot
is ``f_park * S(t)**2``. The drive is parked at ``f_park``, so the spectroscopy
peak sits at ``center(t) = f_park * (S(t)**2 - 1)`` relative to the drive. The
estimator inverts this: Lorentzian peak center per wait -> add back the parked
offset -> ``sqrt(|.|)`` -> tail-normalize -> ``S(t)``, then a multi-exponential
fit.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import SpectroscopyCryoscopeEstimator
from scqat.estimators.spectroscopy_cryoscope import (
    SpectroscopyCryoscopeEstimator as SubpkgEstimator,
)

F_PARK = 300e6  # parked detuning the drive is centered on, Hz


def _forward(components, *, n_wait=44, wmin_ns=16, wmax_ns=20000, f_park=F_PARK,
             ndet=81, span_hz=200e6, fwhm_hz=2e6, noise=0.005, seed=1,
             attach_offset=True):
    """Synthesize a spectroscopy-cryoscope map and return ``(dataset, S)`` — S the
    true normalized step response over wait time. ``components`` are ``(amp,
    tau_ns)`` pairs."""
    wait_ns = np.unique(
        np.logspace(np.log10(wmin_ns), np.log10(wmax_ns), n_wait).astype(int)
    ).astype(float)
    wait_s = wait_ns * 1e-9
    detuning = np.linspace(-span_hz / 2, span_hz / 2, ndet)
    S = 1.0 + sum(amp * np.exp(-wait_ns / tau) for amp, tau in components)
    center = f_park * (S ** 2 - 1.0)  # peak position relative to the parked drive
    gamma = fwhm_hz / 2
    sig = 1.0 / (1.0 + ((detuning[None, :] - center[:, None]) / gamma) ** 2)
    rng = np.random.default_rng(seed)
    if noise:
        sig = sig + rng.normal(0, noise, sig.shape)
    data = {"signal": (("wait_time", "detuning"), sig)}
    if attach_offset:
        data["center_offset_hz"] = f_park
    ds = xr.Dataset(data, coords={"wait_time": wait_s, "detuning": detuning})
    return ds, S


class TestSpectroscopyCryoscopeEstimator:

    def test_step_response_reconstruction_is_faithful(self):
        """The peak-track + sqrt-arch + tail-normalize chain reproduces the planted
        step response (the estimator's core job)."""
        ds, S = _forward([(0.05, 5000.0), (0.03, 400.0)])
        r = SpectroscopyCryoscopeEstimator().extract_parameters(ds)
        assert r["n_peaks_found"] == len(r["wait_time_s"])  # every slice found a peak
        # the reconstructed step response tracks the truth well away from the edges
        assert np.nanmax(np.abs(r["step_response"] - S)) < 0.02
        assert r["step_response"][-5:].mean() == pytest.approx(1.0, abs=0.02)

    def test_single_component_recovery(self):
        """A single exponential (tau within the record) comes back accurately."""
        ds, _ = _forward([(0.06, 3000.0)])
        r = SpectroscopyCryoscopeEstimator().extract_parameters(ds, start_fractions=(0.3,))
        assert r["success"] is True
        assert r["n_components"] == 1
        (amp,), (tau,) = r["component_amps"], r["component_taus_s"]
        assert tau == pytest.approx(3000e-9, rel=0.15)
        assert amp == pytest.approx(0.06, rel=0.2)

    def test_two_component_fit_succeeds_with_dominant_slow(self):
        ds, _ = _forward([(0.05, 5000.0), (0.03, 400.0)])
        r = SpectroscopyCryoscopeEstimator().extract_parameters(
            ds, start_fractions=(0.5, 0.05))
        assert r["success"] is True
        assert r["n_components"] == 2
        amp_slow, tau_slow = r["component_amps"][0], r["component_taus_s"][0]
        assert 3000e-9 < tau_slow < 8000e-9
        assert amp_slow == pytest.approx(0.05, rel=0.4)

    def test_missing_center_offset_defaults_to_zero(self):
        """Without the parked-offset variable the estimator still runs (offset 0),
        exercising the fallback path."""
        ds, _ = _forward([(0.06, 3000.0)], attach_offset=False)
        r = SpectroscopyCryoscopeEstimator().extract_parameters(ds, start_fractions=(0.3,))
        assert r["center_offset_hz"] == 0.0
        # offset 0 makes the reconstruction wrong, but it must not crash
        assert "step_response" in r

    def test_no_peaks_fails_without_raising(self):
        """A flat, peak-less map yields an honest failed fit, not an exception."""
        wait_s = np.logspace(np.log10(16e-9), np.log10(20000e-9), 30)
        detuning = np.linspace(-100e6, 100e6, 41)
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 0.01, (wait_s.size, detuning.size))  # pure noise
        ds = xr.Dataset({"signal": (("wait_time", "detuning"), sig),
                         "center_offset_hz": F_PARK},
                        coords={"wait_time": wait_s, "detuning": detuning})
        r = SpectroscopyCryoscopeEstimator().extract_parameters(ds)
        assert r["success"] is False

    def test_figures_render_on_a_failed_fit(self, tmp_path):
        """The raw spectrogram must ALWAYS render — a failed fit (all-NaN step
        response) annotates the fit panel instead of crashing, and neither figure
        is dropped. Direct regression guard for the missing-figures bug."""
        wait_s = np.logspace(np.log10(16e-9), np.log10(20000e-9), 30)
        detuning = np.linspace(-100e6, 100e6, 41)
        rng = np.random.default_rng(0)
        sig = rng.normal(0, 0.01, (wait_s.size, detuning.size))  # pure noise -> no peaks
        ds = xr.Dataset({"signal": (("wait_time", "detuning"), sig),
                         "center_offset_hz": F_PARK},
                        coords={"wait_time": wait_s, "detuning": detuning})
        est = SpectroscopyCryoscopeEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert results["success"] is False
        assert set(figs) == {"spectroscopy_cryoscope", "spectrogram"}
        assert (tmp_path / "spectroscopy_cryoscope.png").exists()
        assert (tmp_path / "spectroscopy_cryoscope_spectrogram.png").exists()

    def test_analyze_roundtrips_artifacts(self, tmp_path):
        ds, _ = _forward([(0.06, 3000.0)])
        est = SpectroscopyCryoscopeEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path), start_fractions=(0.3,))
        assert set(figs) == {"spectroscopy_cryoscope", "spectrogram"}
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        meta = est.load_metadata(str(tmp_path))
        assert meta["estimator_name"] == "spectroscopy_cryoscope"
        assert "component_taus_s" in meta and "step_response" not in meta
        pd = est.load_plot_data(str(tmp_path))
        assert set(pd["signal"].dims) == {"wait_time", "detuning"}
        assert "best_fit" in pd and pd.attrs["success"] == 1

    def test_figures_redraw_from_plotdata_alone(self, tmp_path):
        ds, _ = _forward([(0.06, 3000.0)])
        est = SpectroscopyCryoscopeEstimator()
        est.analyze(ds, output_dir=str(tmp_path), start_fractions=(0.3,))
        pd = est.load_plot_data(str(tmp_path))
        figs = est.generate_figures(None, None, plot_data=pd)
        assert set(figs) == {"spectroscopy_cryoscope", "spectrogram"}

    def test_check_data_requires_signal_and_coords(self):
        est = SpectroscopyCryoscopeEstimator()
        wait_s = np.logspace(np.log10(16e-9), np.log10(20000e-9), 20)
        detuning = np.linspace(-100e6, 100e6, 41)
        no_signal = xr.Dataset(coords={"wait_time": wait_s, "detuning": detuning})
        with pytest.raises(ValueError, match="signal"):
            est._check_data(no_signal)
        no_wait = xr.Dataset({"signal": ("detuning", np.ones(detuning.size))},
                             coords={"detuning": detuning})
        with pytest.raises(ValueError, match="wait_time"):
            est._check_data(no_wait)

    def test_aggregate_and_subpackage_export_same_class(self):
        assert SpectroscopyCryoscopeEstimator is SubpkgEstimator
