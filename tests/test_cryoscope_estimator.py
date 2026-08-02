"""Tests for the CryoscopeEstimator (flux-line step-response reconstruction).

The forward model mirrors the probe: a flux pulse of swept DURATION delivers a
(distorted) flux ``S(t)`` — the normalized step response, settling to 1 — which
detunes the qubit by ``freq(t) = f_scale * S(t)**2`` (quadratic arch). The Ramsey
phase accumulates as ``phi(t) = 2*pi * cumsum(freq) * dt`` and the closing frame
sweep reads it out as a fringe ``0.5 + 0.5*cos(2*pi*frame - phi)``.

The estimator inverts this: lock-in phase per duration -> unwrap -> Savitzky-Golay
derivative -> ``sqrt(|freq|)`` -> tail-normalize -> ``S(t)``, then a
multi-exponential fit. The tests check the reconstruction fidelity (the
estimator's own value-add), single-exponential recovery through the whole chain,
that the complex-IQ path is blind to the readout rotation, and that ``analyze``
round-trips its artifacts.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import CryoscopeEstimator
from scqat.estimators.cryoscope import CryoscopeEstimator as SubpkgEstimator

F_SCALE = 100e6  # settled detuning, Hz — sets fringes/turn (dphi/step < pi)


def _forward(components, tmax_ns=400, nframe=16, f_scale=F_SCALE, *, iq=False,
             theta=0.0, noise=0.0, seed=1):
    """Synthesize a cryoscope dataset for a planted step response and return
    ``(dataset, S)`` — S being the true normalized step response over duration."""
    t_ns = np.arange(1.0, tmax_ns + 1.0)
    t_s = t_ns * 1e-9
    frame = np.linspace(0.0, 1.0, nframe, endpoint=False)
    S = 1.0 + sum(amp * np.exp(-t_ns / tau) for amp, tau in components)
    freq = f_scale * S ** 2
    dt = t_s[1] - t_s[0]
    phase = 2 * np.pi * np.cumsum(freq) * dt
    pop = 0.5 + 0.5 * np.cos(2 * np.pi * frame[None, :] - phase[:, None])
    rng = np.random.default_rng(seed)
    if iq:
        # blobs move along the readout axis at rotation theta as pop varies
        cplx = complex(0.3, -0.2) + 1.0 * pop * np.exp(1j * theta)
        if noise:
            cplx = cplx + rng.normal(0, noise, cplx.shape) + 1j * rng.normal(0, noise, cplx.shape)
        ds = xr.Dataset(
            {"I": (("duration", "frame"), cplx.real),
             "Q": (("duration", "frame"), cplx.imag)},
            coords={"duration": t_s, "frame": frame},
        )
    else:
        if noise:
            pop = pop + rng.normal(0, noise, pop.shape)
        ds = xr.Dataset(
            {"signal": (("duration", "frame"), pop)},
            coords={"duration": t_s, "frame": frame},
        )
    return ds, S


class TestCryoscopeEstimator:

    def test_step_response_reconstruction_is_faithful(self):
        """The lock-in + savgol chain reproduces the planted step response
        (the estimator's core job) — tight, even for a two-component S."""
        ds, S = _forward([(0.08, 100.0), (0.03, 25.0)])
        r = CryoscopeEstimator().extract_parameters(ds)
        # skip the first few samples (savgol edge)
        assert np.max(np.abs(r["step_response"][3:] - S[3:])) < 0.01
        assert r["step_response"][-10:].mean() == pytest.approx(1.0, abs=0.02)

    def test_single_component_recovery(self):
        """A well-conditioned single exponential (tau well within the record)
        comes back accurately through the full chain."""
        ds, _ = _forward([(0.08, 50.0)])
        r = CryoscopeEstimator().extract_parameters(ds, start_fractions=(0.3,))
        assert r["success"] is True
        assert r["n_components"] == 1
        (amp,), (tau,) = r["component_amps"], r["component_taus_s"]
        assert tau == pytest.approx(50e-9, rel=0.1)
        assert amp == pytest.approx(0.08, rel=0.1)

    def test_two_component_fit_succeeds_with_dominant_slow(self):
        """The two-component fit runs and returns the slow component dominant and
        in the right ballpark (the ill-conditioned decomposition is the tool's
        concern; here we only require a sane, successful fit)."""
        ds, _ = _forward([(0.08, 100.0), (0.03, 25.0)])
        r = CryoscopeEstimator().extract_parameters(ds, start_fractions=(0.5, 0.01))
        assert r["success"] is True
        assert r["n_components"] == 2
        amp_slow, tau_slow = r["component_amps"][0], r["component_taus_s"][0]
        assert 60e-9 < tau_slow < 140e-9
        assert amp_slow == pytest.approx(0.08, rel=0.3)

    def test_iq_path_is_blind_to_readout_rotation(self):
        """The complex-IQ lock-in returns the same components regardless of the
        readout rotation, and matches the pre-discriminated real path."""
        comps = [(0.08, 100.0), (0.03, 25.0)]
        est = CryoscopeEstimator()
        base = est.extract_parameters(_forward(comps)[0])
        for theta in (0.0, 1.1, -2.3):
            ds_iq, _ = _forward(comps, iq=True, theta=theta)
            r = est.extract_parameters(ds_iq)
            assert r["component_amps"] == pytest.approx(base["component_amps"], rel=1e-6)
            assert r["component_taus_s"] == pytest.approx(base["component_taus_s"], rel=1e-6)

    def test_analyze_roundtrips_artifacts(self, tmp_path):
        ds, _ = _forward([(0.08, 60.0)])
        est = CryoscopeEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path), start_fractions=(0.3,))
        assert set(figs) == {"cryoscope", "phase_freq"}
        for fig in figs.values():
            assert isinstance(fig, plt.Figure)
        # metadata JSON self-identifies and drops the bulky arrays
        meta = est.load_metadata(str(tmp_path))
        assert meta["estimator_name"] == "cryoscope"
        assert "component_taus_s" in meta and "step_response" not in meta
        # plot_data netCDF reloads with the raw fringe map + fitted trace
        pd = est.load_plot_data(str(tmp_path))
        assert set(pd["signal"].dims) == {"duration", "frame"}
        assert "best_fit" in pd and pd.attrs["success"] == 1
        for name in ("cryoscope", "phase_freq"):
            assert (tmp_path / f"cryoscope_{name}.png").exists() or (tmp_path / "cryoscope.png").exists()

    def test_figures_redraw_from_plotdata_alone(self, tmp_path):
        """generate_figures must draw from a reloaded plot_data with no results."""
        ds, _ = _forward([(0.08, 60.0)])
        est = CryoscopeEstimator()
        est.analyze(ds, output_dir=str(tmp_path), start_fractions=(0.3,))
        pd = est.load_plot_data(str(tmp_path))
        figs = est.generate_figures(None, None, plot_data=pd)
        assert set(figs) == {"cryoscope", "phase_freq"}

    def test_short_record_fails_without_raising(self):
        """A record too short for the multi-exponential fit yields an honest
        failed fit (the tool's MIN_SAMPLES guard), not an exception."""
        t_s = np.arange(1.0, 13.0) * 1e-9  # below step_response_fit.MIN_SAMPLES
        frame = np.linspace(0.0, 1.0, 16, endpoint=False)
        rng = np.random.default_rng(0)
        sig = 0.5 + 0.5 * np.cos(2 * np.pi * frame[None, :]) + rng.normal(0, 0.01, (t_s.size, frame.size))
        ds = xr.Dataset({"signal": (("duration", "frame"), sig)},
                        coords={"duration": t_s, "frame": frame})
        r = CryoscopeEstimator().extract_parameters(ds)
        assert r["success"] is False

    def test_check_data_requires_signal_and_coords(self):
        est = CryoscopeEstimator()
        frame = np.linspace(0.0, 1.0, 16, endpoint=False)
        t_s = np.arange(1.0, 21.0) * 1e-9
        no_signal = xr.Dataset(coords={"duration": t_s, "frame": frame})
        with pytest.raises(ValueError, match="signal"):
            est._check_data(no_signal)
        no_frame = xr.Dataset({"signal": ("duration", np.ones(t_s.size))},
                              coords={"duration": t_s})
        with pytest.raises(ValueError, match="frame"):
            est._check_data(no_frame)

    def test_aggregate_and_subpackage_export_same_class(self):
        assert CryoscopeEstimator is SubpkgEstimator
