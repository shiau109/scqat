"""Synthetic-grid tests for the swap-flux-map estimator.

The fixed-time sibling of the chevron: same ``joint_population`` form, drawn over
``(qubit_flux_v, coupler_flux_v)``. The shared summary / plot-data / figure code
is exercised in depth by ``test_pair_swap_chevron_estimator``; here we pin the
flux-map's own axis names, the artifact writeout, and the per-coupler-column
coupling fit this estimator adds on top.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_swap_flux_map import PairSwapFluxMapEstimator

VQ0, VC0 = 0.1, 0.0  # where the (synthetic) swap spot sits
LABELS = ["00", "01", "10", "11"]

#: the fixed duration the coupling-fit fixtures are played at, and the coupling
#: that makes exactly one full swap there (theta = pi/2).
T_NS = 100.0
J_FULL_HZ = 1.0 / (4 * T_NS * 1e-9)


def _flux_map_ds() -> xr.Dataset:
    vq = np.linspace(0.0, 0.2, 7)
    vc = np.linspace(-0.15, 0.15, 9)
    spot = np.exp(-((vq[:, None] - VQ0) ** 2) / 0.004) * np.exp(-((vc[None, :] - VC0) ** 2) / 0.01)
    p11 = np.full((vq.size, vc.size), 0.01)
    p10 = 0.84 * spot                       # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - spot)               # driven (low) member depletes at the spot
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])
    return xr.Dataset(
        {"joint_population": (("joint_state", "qubit_flux_v", "coupler_flux_v"), jp)},
        coords={"joint_state": LABELS, "qubit_flux_v": vq, "coupler_flux_v": vc},
    )


def _swap_map_ds(j_slope_hz_per_v=9e6, coupler_off_v=-0.06, nq=31, nc=21,
                 noise=0.008, seed=0, prep=0.97, t1_ns=800.0):
    """A physical fixed-time swap map: the exchange kernel with J tuned by vc.

    ``J(vc) = |j_slope * (vc - coupler_off)|`` crosses zero at ``coupler_off``,
    which is the decouple point the estimator must find, and the resonance line
    bends quadratically with the coupler bias the way the SCQO simulator draws
    it. Returns ``(dataset, coupler_axis, true_J)``.
    """
    vq = np.linspace(0.04, 0.16, nq)
    vc = np.linspace(-0.15, 0.15, nc)
    step = float(vq[1] - vq[0])
    kappa = 2 * 2e6 / (2 * step)                      # Hz per V, grid-relative
    j_hz = np.abs(j_slope_hz_per_v * (vc - coupler_off_v))
    v0 = 0.10 + 0.15 * vc ** 2                        # the bending resonance line
    delta = kappa * (vq[:, None] - v0[None, :])
    omega = np.sqrt(delta ** 2 + (2 * j_hz[None, :]) ** 2)
    swap = (2 * j_hz[None, :]) ** 2 / omega ** 2 * np.sin(np.pi * omega * T_NS * 1e-9) ** 2
    decay = np.exp(-T_NS / t1_ns)                     # common to both members
    rng = np.random.default_rng(seed)
    p11 = np.full((nq, nc), 0.012)
    p10 = np.clip(prep * swap * decay + rng.normal(0, noise, (nq, nc)), 0, 1)
    p01 = np.clip(prep * (1 - swap) * decay + rng.normal(0, noise, (nq, nc)), 0, 1)
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    ds = xr.Dataset(
        {"joint_population": (("joint_state", "qubit_flux_v", "coupler_flux_v"),
                              np.stack([p00, p01, p10, p11]))},
        coords={"joint_state": LABELS, "qubit_flux_v": vq, "coupler_flux_v": vc},
    )
    return ds, vc, j_hz


def test_summarizes_spot_on_flux_axes():
    ds = _flux_map_ds()
    est = PairSwapFluxMapEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"
    assert res["best_qubit_flux_v"] == pytest.approx(VQ0, abs=0.02)
    assert res["best_coupler_flux_v"] == pytest.approx(VC0, abs=0.04)
    assert res["n_qubit_flux_v"] == 7 and res["n_coupler_flux_v"] == 9


def test_plot_data_uses_flux_axes_and_joint_basis():
    ds = _flux_map_ds()
    est = PairSwapFluxMapEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert {"p00", "p01", "p10", "p11"} <= set(pd.data_vars)
    assert set(pd.coords) == {"qubit_flux_v", "coupler_flux_v"}
    assert pd.attrs["axis0"] == "qubit_flux_v"
    assert pd.attrs["axis1"] == "coupler_flux_v"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    ds = _flux_map_ds().rename({"coupler_flux_v": "flux"})
    with pytest.raises(ValueError, match="coupler_flux_v"):
        PairSwapFluxMapEstimator()._check_data(ds)


def test_rejects_an_unknown_fit_knob():
    """Knobs are validated ONCE, before the column loop — a typo must not be
    swallowed by the per-column degrade-don't-raise path."""
    with pytest.raises(ValueError, match="min_contast"):
        PairSwapFluxMapEstimator().extract_parameters(
            _flux_map_ds(), drive_side="low", min_contast=0.1)


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _flux_map_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = PairSwapFluxMapEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert res["n_j_ok"] == 0
    assert set(figs) == {"pair_swap_flux_map", "coupling", "transfer_fit"}
    written = {p.name for p in tmp_path.iterdir()}
    assert {"pair_swap_flux_map.png", "pair_swap_flux_map_coupling.png",
            "pair_swap_flux_map_transfer_fit.png"} <= written


def test_analyze_writes_artifacts(tmp_path):
    est = PairSwapFluxMapEstimator()
    _res, figs = est.analyze(_flux_map_ds(), output_dir=str(tmp_path),
                             skip_figures=False, drive_side="low")
    assert set(figs) == {"pair_swap_flux_map", "coupling", "transfer_fit"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "pair_swap_flux_map.png" in written
    assert "pair_swap_flux_map_plotdata.nc" in written
    assert "pair_swap_flux_map_metadata.json" in written


# --------------------------------------------------------------- coupling fit

def test_recovers_the_coupling_curve_and_the_decouple_point():
    ds, vc, j_true = _swap_map_ds(j_slope_hz_per_v=9e6, coupler_off_v=-0.06)
    res = PairSwapFluxMapEstimator().extract_parameters(
        ds, drive_side="low", swap_time_ns=T_NS)

    assert res["n_j_ok"] >= 12
    # the polynomial through J^2 brackets the zero crossing to inside one grid step
    assert res["off_is_interpolated"] == 1
    assert res["coupler_off_v"] == pytest.approx(-0.06, abs=float(vc[1] - vc[0]))
    assert res["j_at_off_hz"] < 0.2e6
    assert res["poly_r_squared"] > 0.9

    j_fit = np.asarray(res["j_hz"], dtype=float)
    quotable = (np.asarray(res["fit_success"], dtype=bool)
                & ~np.asarray(res["branch_warn"], dtype=bool))
    error = np.abs(j_fit[quotable] - j_true[quotable]) / np.maximum(j_true[quotable], 1e-9)
    assert np.median(error) < 0.12


def test_flags_columns_past_a_full_swap_and_keeps_them_out_of_the_polynomial():
    """A map run near a full swap folds at its strong-coupling end."""
    ds, vc, j_true = _swap_map_ds(j_slope_hz_per_v=20e6, coupler_off_v=-0.06)
    assert j_true.max() > J_FULL_HZ            # the fixture really does fold
    res = PairSwapFluxMapEstimator().extract_parameters(
        ds, drive_side="low", swap_time_ns=T_NS)

    assert res["n_branch_warn"] > 0
    assert res["n_poly_rows"] == res["n_j_ok"] - res["n_branch_warn"]
    # every flagged column really is one the fit under-reports
    warn = np.asarray(res["branch_warn"], dtype=bool)
    assert (j_true[warn] > 0.8 * J_FULL_HZ).all()
    # and the decouple point survives, because the folded columns were excluded
    assert res["coupler_off_v"] == pytest.approx(-0.06, abs=float(vc[1] - vc[0]))


def test_angle_is_reported_without_a_duration_but_hz_is_not():
    ds, _vc, _j = _swap_map_ds()
    res = PairSwapFluxMapEstimator().extract_parameters(ds, drive_side="low")
    assert np.isnan(res["swap_time_ns"])
    assert np.isfinite(np.asarray(res["theta_rad"], dtype=float)).any()
    assert np.isnan(np.asarray(res["j_hz"], dtype=float)).all()
    assert np.isnan(res["j_max_hz"]) and np.isnan(res["j_at_off_hz"])


def test_decouple_point_degrades_to_the_grid_when_not_bracketed():
    """An off point outside the swept window is reported as the grid edge, not
    extrapolated to somewhere the sweep never went."""
    ds, vc, _j = _swap_map_ds(j_slope_hz_per_v=6e6, coupler_off_v=-0.40)
    res = PairSwapFluxMapEstimator().extract_parameters(
        ds, drive_side="low", swap_time_ns=T_NS)
    assert res["off_is_interpolated"] == 0
    assert float(vc.min()) <= res["coupler_off_v"] <= float(vc.max())


def test_plot_data_carries_every_curve_and_replots_from_netcdf(tmp_path):
    ds, _vc, _j = _swap_map_ds()
    est = PairSwapFluxMapEstimator()
    est.analyze(ds, output_dir=str(tmp_path), skip_figures=False,
                drive_side="low", flux_side="low", high_name="q1", low_name="q2",
                swap_time_ns=T_NS)
    plot_data = xr.load_dataset(tmp_path / "pair_swap_flux_map_plotdata.nc")
    assert {"transfer_norm", "transfer_fit", "theta_rad", "j_hz", "peak_transfer",
            "resonance_qubit_flux_v", "hwhm_qubit_flux_v", "fit_success",
            "branch_warn", "j_poly_curve"} <= set(plot_data.data_vars)
    assert plot_data.attrs["swap_time_ns"] == pytest.approx(T_NS)
    # the whole point of the split: redraw with NO dataset and NO results
    figs = est.generate_figures(None, None, plot_data=plot_data)
    assert set(figs) == {"pair_swap_flux_map", "coupling", "transfer_fit"}


def test_metadata_drops_the_bulky_maps_but_keeps_the_curves():
    ds, vc, _j = _swap_map_ds()
    est = PairSwapFluxMapEstimator()
    res = est.extract_parameters(ds, drive_side="low", swap_time_ns=T_NS)
    meta = est.extract_metadata(res)
    assert not any(key.startswith("_") for key in meta)
    for key in ("theta_rad", "j_hz", "fit_success", "branch_warn"):
        assert len(meta[key]) == vc.size
