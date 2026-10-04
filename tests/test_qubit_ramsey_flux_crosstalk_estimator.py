"""Tests for the qubit_ramsey_flux_crosstalk estimator.

Synthetic maps from the model the estimator fits: with the target's own line at
``a`` and the source line at ``b``, ``delta_f(a, b) = h(b) - k (a - a0 + m b)^2``,
seen through a Ramsey fringe at ``|D + delta_f|``. Checks that the SIGNED ``m`` is
recovered, that an apex-height shift (which is not crosstalk) leaves it alone, the
refusals (too few source points, apex outside the window), the nonlinearity flag,
the IQ path, and that NO axis order changes any result.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators import QubitRamseyFluxCrosstalkEstimator

CURV = 1.52e10  # Hz/V^2 (5Q4C q1: 0.0152 MHz/mV^2), the arch opens downward
RAMP = -4e6     # the apex case: the qubit only drops below its drive
DIMS = ("source_flux", "flux_bias", "idle_time")
FIGURES = {"fringe_maps", "flux_curves", "apex_vs_source"}


def _map(m=0.05, a0=0.6e-3, height=None, bs=None, xs=None, noise=0.01, seed=0,
         iq=False, bend=0.0):
    """delta_f(a, b) = h(b) - CURV (a - a0 + m b + bend b^2)^2 as a fringe map."""
    bs = np.linspace(-0.05, 0.05, 5) if bs is None else np.asarray(bs, dtype=float)
    xs = np.linspace(-0.02, 0.02, 7) if xs is None else np.asarray(xs, dtype=float)
    t = np.linspace(16e-9, 2000e-9, 101)
    rng = np.random.default_rng(seed)
    h = np.zeros_like(bs) if height is None else np.asarray(height, dtype=float)
    shift = a0 - m * bs - bend * bs ** 2
    delta = h[:, None] - CURV * (xs[None, :] - shift[:, None]) ** 2
    fringe = np.abs(RAMP + delta)
    pop = 0.5 - 0.45 * np.exp(-t / 10e-6) * np.cos(2 * np.pi * fringe[..., None] * t)
    pop = pop + noise * rng.standard_normal(pop.shape)
    coords = {"source_flux": bs, "flux_bias": xs, "idle_time": t}
    if not iq:
        return xr.Dataset({"signal": (DIMS, pop)}, coords=coords)
    g, e = 1.0 + 0.5j, 2.0 + 1.5j
    z = g + (e - g) * pop
    return xr.Dataset({"I": (DIMS, z.real), "Q": (DIMS, z.imag),
                       "ref_pos_g_i": g.real, "ref_pos_g_q": g.imag,
                       "ref_pos_e_i": e.real, "ref_pos_e_q": e.imag}, coords=coords)


def _fit(ds, **kw):
    kw.setdefault("ramp_detuning_hz", RAMP)
    return QubitRamseyFluxCrosstalkEstimator().extract_parameters(ds, **kw)


@pytest.mark.parametrize("m", [0.05, -0.05, 0.01, -0.07])
def test_signed_crosstalk_recovered(m):
    r = _fit(_map(m=m))
    assert r["success"]
    assert r["flux_crosstalk"] == pytest.approx(m, abs=5e-4)
    assert abs(r["flux_crosstalk"] - m) < 5 * r["flux_crosstalk_stderr"] + 2e-4
    assert r["apex_flux_at_zero_source"] == pytest.approx(0.6e-3, abs=0.05e-3)
    assert r["curvature_hz_per_v2"] == pytest.approx(-CURV, rel=0.05)
    assert r["n_valid_source_points"] == 5 and r["n_apex_not_bracketed"] == 0
    assert r["nonlinear_suspected"] == 0 and r["fold_suspected"] == 0


def test_an_apex_height_shift_is_not_crosstalk():
    """A coupler's dispersive shift moves the apex HEIGHT with the source amplitude.
    It must come out as ``apex_height_span_hz`` and leave ``m`` alone."""
    bs = np.linspace(-0.05, 0.05, 5)
    lamb = 0.5e6 * (bs / 0.05) ** 2 + 0.3e6 * (bs / 0.05)
    plain = _fit(_map(m=0.03))
    shifted = _fit(_map(m=0.03, height=lamb))
    assert shifted["flux_crosstalk"] == pytest.approx(plain["flux_crosstalk"], abs=2e-4)
    assert shifted["apex_height_span_hz"] == pytest.approx(np.ptp(lamb), rel=0.05)
    assert plain["apex_height_span_hz"] < 20e3


def test_no_axis_order_can_change_the_answer(order_free):
    results = order_free(QubitRamseyFluxCrosstalkEstimator(), _map(), DIMS,
                         ramp_detuning_hz=RAMP)
    assert results["flux_crosstalk"] == pytest.approx(0.05, abs=5e-4)


def test_a_shuffled_source_list_gives_the_same_answer():
    ds = _map()
    ref = _fit(ds)
    perm = np.random.default_rng(2).permutation(ds.sizes["source_flux"])
    r = _fit(ds.isel(source_flux=perm))
    for key in ("flux_crosstalk", "flux_crosstalk_stderr", "apex_flux_at_zero_source"):
        assert r[key] == pytest.approx(ref[key], rel=1e-9, abs=1e-15), key
    assert r["source_flux"] == sorted(r["source_flux"])


def test_an_apex_pushed_out_of_the_window_drops_that_source_point():
    """m * b = 30 mV at the widest source amplitude - past the +-20 mV own window."""
    r = _fit(_map(m=0.3, bs=np.linspace(-0.1, 0.1, 7)))
    assert r["n_apex_not_bracketed"] >= 2
    assert r["n_valid_source_points"] == 7 - r["n_apex_not_bracketed"]
    if r["success"]:
        assert r["flux_crosstalk"] == pytest.approx(0.3, abs=5e-3)


def test_too_few_source_points_is_no_fit():
    r = _fit(_map(bs=[-0.05, 0.05]))
    assert r["success"] is False
    assert np.isnan(r["flux_crosstalk"])
    assert r["n_valid_source_points"] == 2


def test_a_bent_apex_track_is_flagged():
    """An apex that does not follow a line (a resonance near the source, a second
    mechanism) must not pass as a clean coefficient."""
    r = _fit(_map(m=0.02, bend=1.0))  # 2.5 mV of bow over +-50 mV
    assert r["nonlinear_suspected"] == 1
    assert r["line_max_residual_v"] > r["linearity_tol_v"]


def test_iq_input_with_stored_positions():
    r = _fit(_map(iq=True))
    assert r["reduction_method"] == "positions"
    assert r["flux_crosstalk"] == pytest.approx(0.05, abs=5e-4)


def test_bad_kwargs_raise():
    with pytest.raises(ValueError, match="ramp_detuning_hz"):
        QubitRamseyFluxCrosstalkEstimator().extract_parameters(_map())
    with pytest.raises(ValueError, match="min_source_points"):
        _fit(_map(), min_source_points=2)
    with pytest.raises(ValueError):
        _fit(_map(), not_a_knob=1)


def test_figures_render_on_a_failed_fit(tmp_path):
    rng = np.random.default_rng(4)
    ds = _map()
    flat = xr.Dataset({"signal": (DIMS, 0.5 + 0.01 * rng.standard_normal(ds["signal"].shape))},
                      coords=ds.coords)
    est = QubitRamseyFluxCrosstalkEstimator()
    results, figs = est.analyze(flat, output_dir=str(tmp_path), ramp_detuning_hz=RAMP)
    assert results["success"] is False
    assert set(figs) == FIGURES
    for fig in figs.values():
        assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_analyze_roundtrips_artifacts(tmp_path):
    est = QubitRamseyFluxCrosstalkEstimator()
    results, figs = est.analyze(_map(), output_dir=str(tmp_path), ramp_detuning_hz=RAMP,
                                drive_freq_hz=5.1446e9)
    meta = est.load_metadata(str(tmp_path))
    assert "signal" not in meta
    assert meta["flux_crosstalk"] == pytest.approx(results["flux_crosstalk"])
    json.dumps(meta)
    plot = est.load_plot_data(str(tmp_path))
    assert set(plot.data_vars) >= {"signal", "delta_f_hz", "apex_flux", "fit_apex_flux"}
    redrawn = est.generate_figures(None, None, plot_data=plot)
    assert set(redrawn) == FIGURES
    plt.close("all")
