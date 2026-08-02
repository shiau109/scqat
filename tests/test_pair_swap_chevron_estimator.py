"""Synthetic-grid tests for the swap-chevron raw-population estimator.

Dataset contract under test: joint-population vars ``p_high`` / ``p_low`` /
``p_ee`` (``p_gg`` optional) on coords ``(flux_amp_v, swap_time_ns)``. The
estimator only draws the four populations and reports where the transfer peaks —
there is no fit and no ``min_transfer`` verdict here (that stays in SCQO).
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_swap_chevron import PairSwapChevronEstimator

V0, T0 = 0.18, 60.0  # where the (synthetic) transfer peaks


def _chevron_ds(with_p_gg: bool = False) -> xr.Dataset:
    v = np.linspace(0.0, 0.3, 9)
    t = np.linspace(1.0, 100.0, 11)
    bump = np.exp(-((v[:, None] - V0) ** 2) / 0.004) * np.exp(-((t[None, :] - T0) ** 2) / 400.0)
    p_high = 0.85 * bump                                   # transfer onto the undriven (high) member
    p_low = 0.9 * (1.0 - bump)                             # driven member depletes as it transfers
    p_ee = np.full((v.size, t.size), 0.01)
    data = {
        "p_high": (("flux_amp_v", "swap_time_ns"), p_high),
        "p_low": (("flux_amp_v", "swap_time_ns"), p_low),
        "p_ee": (("flux_amp_v", "swap_time_ns"), p_ee),
    }
    if with_p_gg:
        data["p_gg"] = (("flux_amp_v", "swap_time_ns"), np.clip(1.0 - (p_high + p_low) + p_ee, 0.0, 1.0))
    return xr.Dataset(data, coords={"flux_amp_v": v, "swap_time_ns": t})


def test_summarizes_transfer_peak():
    ds = _chevron_ds()
    est = PairSwapChevronEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"                     # drive low -> transfer onto high
    assert res["best_flux_amp_v"] == pytest.approx(V0, abs=0.02)
    assert res["best_swap_time_ns"] == pytest.approx(T0, abs=5.0)
    assert res["n_flux_amp_v"] == 9 and res["n_swap_time_ns"] == 11
    assert 0.0 <= res["best_transfer"] <= 1.0


def test_drive_side_high_selects_low_partner():
    res = PairSwapChevronEstimator().extract_parameters(_chevron_ds(), drive_side="high")
    assert res["partner"] == "p_low"


def test_plot_data_has_joint_basis_axes_and_attrs():
    ds = _chevron_ds()
    est = PairSwapChevronEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert set(pd.data_vars) == {"p00", "p01", "p10", "p11"}
    assert set(pd.coords) == {"flux_amp_v", "swap_time_ns"}
    for name in ("p00", "p01", "p10", "p11"):
        assert pd[name].dims == ("flux_amp_v", "swap_time_ns")
    assert pd.attrs["axis0"] == "flux_amp_v"
    assert pd.attrs["axis1"] == "swap_time_ns"
    # drive low prepares |ge> (p01) and transfers the excitation to |eg> (p10)
    assert pd.attrs["prepared_state"] == "p01"
    assert pd.attrs["transfer_state"] == "p10"


def test_joint_basis_partitions_unity():
    pd = PairSwapChevronEstimator().build_plot_data(_chevron_ds(), {}, drive_side="low")
    total = sum(pd[n].values for n in ("p00", "p01", "p10", "p11"))
    assert abs(float(total.mean()) - 1.0) < 0.05


def test_joint_basis_reconstructed_from_marginals():
    ds = _chevron_ds(with_p_gg=True)
    pd = PairSwapChevronEstimator().build_plot_data(ds, {}, drive_side="low")
    assert np.allclose(pd["p00"].values, ds["p_gg"].values)          # P00 = p_gg
    assert np.allclose(pd["p11"].values, ds["p_ee"].values)          # P11 = p_ee
    # single-excitation joints remove the double-excitation overlap
    assert np.allclose(pd["p10"].values, np.clip(ds["p_high"].values - ds["p_ee"].values, 0, 1))
    assert np.allclose(pd["p01"].values, np.clip(ds["p_low"].values - ds["p_ee"].values, 0, 1))


def test_drive_side_flips_prepared_and_transfer():
    est = PairSwapChevronEstimator()
    lo = est.build_plot_data(_chevron_ds(), {}, drive_side="low").attrs
    hi = est.build_plot_data(_chevron_ds(), {}, drive_side="high").attrs
    assert (lo["prepared_state"], lo["transfer_state"]) == ("p01", "p10")
    assert (hi["prepared_state"], hi["transfer_state"]) == ("p10", "p01")


def test_drive_flux_roles_and_qubit_names_recorded():
    # the figure names the actual member qubits, so the drive/flux roles AND the
    # high/low -> qubit-name mapping must reach plot_data.attrs and the metadata.
    ds = _chevron_ds()
    est = PairSwapChevronEstimator()
    pd = est.build_plot_data(ds, {}, drive_side="low", flux_side="high",
                             high_name="q1", low_name="q0")
    assert pd.attrs["drive_side"] == "low" and pd.attrs["flux_side"] == "high"
    assert pd.attrs["high_name"] == "q1" and pd.attrs["low_name"] == "q0"
    meta = est.extract_parameters(ds, drive_side="low", flux_side="high",
                                  high_name="q1", low_name="q0")
    assert meta["high_name"] == "q1" and meta["low_name"] == "q0"


def test_all_nan_map_is_unsuccessful_not_a_crash():
    ds = _chevron_ds()
    ds["p_high"].values[:] = np.nan
    res = PairSwapChevronEstimator().extract_parameters(ds, drive_side="low")
    assert res["success"] is False
    assert np.isnan(res["best_transfer"])


def test_rejects_missing_population():
    ds = _chevron_ds().drop_vars("p_ee")
    with pytest.raises(ValueError, match="p_ee"):
        PairSwapChevronEstimator()._check_data(ds)


def test_rejects_missing_coordinate():
    ds = _chevron_ds().rename({"swap_time_ns": "time"})
    with pytest.raises(ValueError, match="swap_time_ns"):
        PairSwapChevronEstimator()._check_data(ds)


def test_analyze_writes_artifacts(tmp_path):
    est = PairSwapChevronEstimator()
    res, figs = est.analyze(_chevron_ds(), output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert set(figs) == {"pair_swap_chevron"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "pair_swap_chevron.png" in written
    assert "pair_swap_chevron_plotdata.nc" in written
    assert "pair_swap_chevron_metadata.json" in written
    # the saved plotdata redraws the figure with no re-fit
    pd = xr.load_dataset(tmp_path / "pair_swap_chevron_plotdata.nc")
    assert set(est.generate_figures(None, None, plot_data=pd)) == {"pair_swap_chevron"}
