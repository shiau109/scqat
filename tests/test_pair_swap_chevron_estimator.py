"""Synthetic-grid tests for the swap-chevron raw-population estimator.

Dataset contract under test (the unified readout schema's joint form): one
``joint_population`` variable over a ``joint_state`` label coordinate
(``"00"/"01"/"10"/"11"``, leftmost digit = the HIGH member) on coords
``(flux_amp_v, swap_time_ns)``. The estimator only draws the four populations
and reports where the transfer peaks — there is no fit and no ``min_transfer``
verdict here (that stays in SCQO).
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_swap_chevron import PairSwapChevronEstimator

V0, T0 = 0.18, 60.0  # where the (synthetic) transfer peaks
LABELS = ["00", "01", "10", "11"]


def _chevron_ds() -> xr.Dataset:
    v = np.linspace(0.0, 0.3, 9)
    t = np.linspace(1.0, 100.0, 11)
    bump = np.exp(-((v[:, None] - V0) ** 2) / 0.004) * np.exp(-((t[None, :] - T0) ** 2) / 400.0)
    p11 = np.full((v.size, t.size), 0.01)
    p10 = 0.84 * bump                       # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - bump)               # driven (low) member depletes as it transfers
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])     # (joint_state, flux_amp_v, swap_time_ns)
    return xr.Dataset(
        {"joint_population": (("joint_state", "flux_amp_v", "swap_time_ns"), jp)},
        coords={"joint_state": LABELS, "flux_amp_v": v, "swap_time_ns": t},
    )


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


def test_marginals_are_partial_traces():
    # the summary's p_high/p_low ranges are the member marginals traced out of
    # the joint distribution: P(high=e) = P10 + P11, P(low=e) = P01 + P11.
    ds = _chevron_ds()
    res = PairSwapChevronEstimator().extract_parameters(ds, drive_side="low")
    jp = ds["joint_population"]
    p_high = (jp.sel(joint_state="10") + jp.sel(joint_state="11")).values
    p_low = (jp.sel(joint_state="01") + jp.sel(joint_state="11")).values
    assert res["p_high_max"] == pytest.approx(float(np.nanmax(p_high)))
    assert res["p_high_min"] == pytest.approx(float(np.nanmin(p_high)))
    assert res["p_low_max"] == pytest.approx(float(np.nanmax(p_low)))
    assert res["best_transfer"] == pytest.approx(float(np.nanmax(p_high)))
    assert res["p_ee_max"] == pytest.approx(float(jp.sel(joint_state="11").max()))


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


def test_plot_data_carries_joint_population_verbatim():
    ds = _chevron_ds()
    pd = PairSwapChevronEstimator().build_plot_data(ds, {}, drive_side="low")
    for label in LABELS:
        assert np.allclose(pd[f"p{label}"].values,
                           ds["joint_population"].sel(joint_state=label).values)


def test_axis_order_invariance():
    # the estimator transposes by coordinate NAME, so a probe may emit the
    # joint_state / sweep axes in any order.
    ds = _chevron_ds().transpose("swap_time_ns", "joint_state", "flux_amp_v")
    res = PairSwapChevronEstimator().extract_parameters(ds, drive_side="low")
    assert res["best_flux_amp_v"] == pytest.approx(V0, abs=0.02)
    assert res["best_swap_time_ns"] == pytest.approx(T0, abs=5.0)


def test_joint_basis_partitions_unity():
    pd = PairSwapChevronEstimator().build_plot_data(_chevron_ds(), {}, drive_side="low")
    total = sum(pd[n].values for n in ("p00", "p01", "p10", "p11"))
    assert abs(float(total.mean()) - 1.0) < 0.05


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
    ds["joint_population"].values[:] = np.nan
    res = PairSwapChevronEstimator().extract_parameters(ds, drive_side="low")
    assert res["success"] is False
    assert np.isnan(res["best_transfer"])


def test_figures_render_on_a_failed_fit(tmp_path):
    # the raw-data figure must survive an all-NaN (failed) acquisition: the run
    # still writes the PNG, with success=False in the metadata.
    ds = _chevron_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = PairSwapChevronEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == {"pair_swap_chevron"}
    assert "pair_swap_chevron.png" in {p.name for p in tmp_path.iterdir()}


def test_rejects_missing_joint_population():
    ds = _chevron_ds().rename({"joint_population": "populations"})
    with pytest.raises(ValueError, match="joint_population"):
        PairSwapChevronEstimator()._check_data(ds)


def test_rejects_missing_basis_label():
    ds = _chevron_ds().sel(joint_state=["00", "01", "10"])
    with pytest.raises(ValueError, match="11"):
        PairSwapChevronEstimator().extract_parameters(ds, drive_side="low")


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
