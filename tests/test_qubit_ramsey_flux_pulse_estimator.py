"""Tests for the qubit_ramsey_flux_pulse estimator (Ramsey fringe vs flux).

Synthetic maps from the model the estimator fits: a local transmon arch
``delta_f(x) = -k (x - x0)^2 + h`` seen through a Ramsey fringe at
``|D + delta_f|`` (the qubit_ramsey sign convention). Checks the apex and park
readings, the flux_side choice, the refusals (out of window, not bracketed), the
fold check, the IQ path, and that NEITHER axis order changes any result.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators import QubitRamseyFluxPulseEstimator

CURV = -1.52e10  # Hz/V^2 (5Q4C q1: -0.0152 MHz/mV^2)
DRIVE = 5.1446e9


def _map(x0=0.8e-3, h=15e3, ramp=4e6, xs=None, noise=0.01, seed=0, iq=False):
    xs = np.linspace(-12e-3, 12e-3, 7) if xs is None else np.asarray(xs, dtype=float)
    t = np.linspace(16e-9, 4000e-9, 201)
    rng = np.random.default_rng(seed)
    delta = CURV * (xs - x0) ** 2 + h
    fringe = np.abs(ramp + delta)
    pop = 0.5 - 0.45 * np.exp(-t / 10e-6) * np.cos(2 * np.pi * fringe[:, None] * t[None, :])
    pop = pop + noise * rng.standard_normal(pop.shape)
    coords = {"flux_bias": xs, "idle_time": t}
    if not iq:
        return xr.Dataset({"signal": (("flux_bias", "idle_time"), pop)}, coords=coords)
    g, e = 1.0 + 0.5j, 2.0 + 1.5j
    z = g + (e - g) * pop
    return xr.Dataset({"I": (("flux_bias", "idle_time"), z.real),
                       "Q": (("flux_bias", "idle_time"), z.imag),
                       "ref_pos_g_i": g.real, "ref_pos_g_q": g.imag,
                       "ref_pos_e_i": e.real, "ref_pos_e_q": e.imag}, coords=coords)


def _fit(ds, **kw):
    kw.setdefault("ramp_detuning_hz", 4e6)
    kw.setdefault("drive_freq_hz", DRIVE)
    return QubitRamseyFluxPulseEstimator().extract_parameters(ds, **kw)


def test_apex_recovered():
    r = _fit(_map())
    assert r["success"] and r["question"] == "apex"
    assert r["apex_flux"] == pytest.approx(0.8e-3, abs=0.05e-3)
    assert r["apex_delta_f_hz"] == pytest.approx(15e3, abs=5e3)
    assert r["apex_f01_hz"] == pytest.approx(DRIVE + 15e3, abs=5e3)
    assert r["curvature_hz_per_v2"] == pytest.approx(CURV, rel=0.05)
    assert r["apex_not_bracketed"] == 0 and r["fold_suspected"] == 0
    assert r["n_valid_points"] == 7


def test_negative_ramp_reads_the_same_physics():
    """sign(D) is the probe's choice; delta_f must not depend on it."""
    pos = _fit(_map(ramp=4e6), ramp_detuning_hz=4e6)
    neg = _fit(_map(ramp=-4e6), ramp_detuning_hz=-4e6)
    assert neg["apex_flux"] == pytest.approx(pos["apex_flux"], abs=0.05e-3)
    assert neg["apex_delta_f_hz"] == pytest.approx(pos["apex_delta_f_hz"], abs=5e3)


def test_sweep_direction_cannot_change_the_answer(order_free):
    results = order_free(QubitRamseyFluxPulseEstimator(), _map(), ("flux_bias", "idle_time"),
                         ramp_detuning_hz=4e6, drive_freq_hz=DRIVE)
    assert results["apex_flux"] == pytest.approx(0.8e-3, abs=0.05e-3)


def test_non_monotone_point_list_gives_the_same_answer():
    """A ping-pong walk (centre, -1, +1, ...) is an explicit point list, not a reversal."""
    ds = _map()
    ref = _fit(ds)
    perm = np.random.default_rng(2).permutation(ds.sizes["flux_bias"])
    r = _fit(ds.isel(flux_bias=perm))
    for key in ("apex_flux", "apex_flux_stderr", "apex_delta_f_hz", "curvature_hz_per_v2"):
        assert r[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-15), key
    assert r["flux_bias"] == sorted(r["flux_bias"])


def test_apex_outside_window_is_not_bracketed():
    r = _fit(_map(x0=20e-3))
    assert r["success"] is False
    assert r["apex_not_bracketed"] == 1


@pytest.mark.parametrize("side,expected", [("lower", 0.8e-3 - 5.64e-3),
                                           ("upper", 0.8e-3 + 5.64e-3)])
def test_park_root_by_side(side, expected):
    # 0.5 MHz below the apex: roots at x0 -+ sqrt(0.5e6 / 1.52e10) = x0 -+ 5.74 mV
    target = DRIVE + 15e3 - 0.5e6
    r = _fit(_map(), park_frequency_hz=target, flux_side=side)
    assert r["success"] and r["question"] == "park" and r["park_out_of_window"] == 0
    assert len(r["park_roots"]) == 2
    root = 0.8e-3 + (-1 if side == "lower" else 1) * np.sqrt(0.5e6 / -CURV)
    assert r["park_flux"] == pytest.approx(root, abs=0.1e-3)
    assert np.isfinite(r["park_flux_stderr"])
    assert (r["park_slope_hz_per_v"] > 0) == (side == "lower")


def test_park_nearest_uses_nearest_to():
    target = DRIVE + 15e3 - 0.5e6
    r = _fit(_map(), park_frequency_hz=target, nearest_to=5e-3)
    assert r["park_flux"] > 0.8e-3


def test_park_outside_window_is_refused_not_extrapolated():
    target = DRIVE - 20e6  # far below anything the +-12 mV window reaches
    r = _fit(_map(), park_frequency_hz=target)
    assert r["success"] is False
    assert r["park_out_of_window"] == 1
    assert np.isnan(r["park_flux"])


def test_park_needs_drive_frequency():
    with pytest.raises(ValueError, match="drive_freq_hz"):
        QubitRamseyFluxPulseEstimator().extract_parameters(
            _map(), ramp_detuning_hz=4e6, park_frequency_hz=DRIVE)


def test_bad_kwargs_raise():
    with pytest.raises(ValueError, match="ramp_detuning_hz"):
        QubitRamseyFluxPulseEstimator().extract_parameters(_map())
    with pytest.raises(ValueError, match="flux_side"):
        _fit(_map(), flux_side="left")
    with pytest.raises(ValueError):
        _fit(_map(), not_a_knob=1)


def test_fold_is_flagged():
    """A 1 MHz detuning over a 2.2 MHz swing folds the fringe through zero."""
    r = _fit(_map(ramp=1e6), ramp_detuning_hz=1e6)
    assert r["fold_suspected"] == 1 or r["success"] is False


def test_iq_input_with_stored_positions():
    r = _fit(_map(iq=True))
    assert r["reduction_method"] == "positions"
    assert r["apex_flux"] == pytest.approx(0.8e-3, abs=0.05e-3)


def test_figures_render_on_a_failed_fit(tmp_path):
    rng = np.random.default_rng(4)
    t = np.linspace(16e-9, 4000e-9, 201)
    xs = np.linspace(-12e-3, 12e-3, 7)
    ds = xr.Dataset({"signal": (("flux_bias", "idle_time"),
                                0.5 + 0.01 * rng.standard_normal((xs.size, t.size)))},
                    coords={"flux_bias": xs, "idle_time": t})
    est = QubitRamseyFluxPulseEstimator()
    results, figs = est.analyze(ds, output_dir=str(tmp_path), ramp_detuning_hz=4e6)
    assert results["success"] is False
    assert set(figs) == {"fringe_map", "flux_curve"}
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_analyze_roundtrips_artifacts(tmp_path):
    est = QubitRamseyFluxPulseEstimator()
    results, figs = est.analyze(_map(), output_dir=str(tmp_path),
                                ramp_detuning_hz=4e6, drive_freq_hz=DRIVE)
    meta = est.load_metadata(str(tmp_path))
    assert "signal" not in meta
    assert meta["apex_flux"] == pytest.approx(results["apex_flux"])
    json.dumps(meta)
    plot = est.load_plot_data(str(tmp_path))
    assert set(plot.data_vars) >= {"signal", "delta_f_hz", "fit_delta_f_hz", "valid"}
    redrawn = est.generate_figures(None, None, plot_data=plot)
    assert set(redrawn) == {"fringe_map", "flux_curve"}
    plt.close("all")
