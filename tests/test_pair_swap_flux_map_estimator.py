"""Synthetic-grid tests for the swap-flux-map raw-population estimator.

The fixed-time sibling of the chevron: same ``joint_population`` form, drawn
over ``(qubit_flux_v, coupler_flux_v)``. The shared summary / plot-data / figure
code is exercised in depth by ``test_pair_swap_chevron_estimator``; here we pin
the flux-map's own axis names and the artifact writeout.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_swap_flux_map import PairSwapFluxMapEstimator

VQ0, VC0 = 0.1, 0.0  # where the (synthetic) swap spot sits
LABELS = ["00", "01", "10", "11"]


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
    assert set(pd.data_vars) == {"p00", "p01", "p10", "p11"}
    assert set(pd.coords) == {"qubit_flux_v", "coupler_flux_v"}
    assert pd.attrs["axis0"] == "qubit_flux_v"
    assert pd.attrs["axis1"] == "coupler_flux_v"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    ds = _flux_map_ds().rename({"coupler_flux_v": "flux"})
    with pytest.raises(ValueError, match="coupler_flux_v"):
        PairSwapFluxMapEstimator()._check_data(ds)


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _flux_map_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = PairSwapFluxMapEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == {"pair_swap_flux_map"}
    assert "pair_swap_flux_map.png" in {p.name for p in tmp_path.iterdir()}


def test_analyze_writes_artifacts(tmp_path):
    est = PairSwapFluxMapEstimator()
    _res, figs = est.analyze(_flux_map_ds(), output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert set(figs) == {"pair_swap_flux_map"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "pair_swap_flux_map.png" in written
    assert "pair_swap_flux_map_plotdata.nc" in written
    assert "pair_swap_flux_map_metadata.json" in written
