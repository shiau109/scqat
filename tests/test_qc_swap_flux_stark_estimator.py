"""Synthetic-grid tests for the fixed-N flux x stark raw-population estimator.

The two-amplitude member of the pair-swap map family: same ``joint_population``
form as ``qc_n_swap_amp``, but BOTH axes are amplitudes (the swap count is frozen
to a scalar and never reaches the dataset). The shared summary / plot-data /
figure code is exercised in depth by ``test_pair_swap_chevron_estimator``; here we
pin this estimator's own axis names and the artifact writeout.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_swap_flux_stark import QcSwapFluxStarkEstimator

VA0, A0 = 0.15, 0.4  # where the (synthetic) transfer peaks: resonance V, compensating factor
LABELS = ["00", "01", "10", "11"]


def _flux_stark_ds() -> xr.Dataset:
    va = np.linspace(0.10, 0.20, 11)          # control flux amplitude (V)
    a = np.linspace(0.0, 1.0, 9)              # stark amplitude factor
    spot = (np.exp(-((va[:, None] - VA0) ** 2) / 5e-5)
            * np.exp(-((a[None, :] - A0) ** 2) / 0.05))
    p11 = np.full((va.size, a.size), 0.01)
    p10 = 0.84 * spot                        # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - spot)                # driven (low) member depletes at the peak
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])
    return xr.Dataset(
        {"joint_population": (("joint_state", "flux_amp_v", "stark_amp"), jp)},
        coords={"joint_state": LABELS, "flux_amp_v": va, "stark_amp": a},
    )


def test_summarizes_spot_on_both_amplitude_axes():
    ds = _flux_stark_ds()
    est = QcSwapFluxStarkEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"
    assert res["best_flux_amp_v"] == pytest.approx(VA0, abs=0.01)
    assert res["best_stark_amp"] == pytest.approx(A0, abs=0.13)
    assert res["n_flux_amp_v"] == 11 and res["n_stark_amp"] == 9


def test_plot_data_uses_flux_stark_axes_and_joint_basis():
    ds = _flux_stark_ds()
    est = QcSwapFluxStarkEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert set(pd.data_vars) == {"p00", "p01", "p10", "p11"}
    assert set(pd.coords) == {"flux_amp_v", "stark_amp"}
    assert pd.attrs["axis0"] == "flux_amp_v"
    assert pd.attrs["axis1"] == "stark_amp"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    """The stark axis is this estimator's own; a dataset carrying the sibling's
    count axis instead must be refused by name, not silently analysed."""
    ds = _flux_stark_ds().rename({"stark_amp": "swap_count"})
    with pytest.raises(ValueError, match="stark_amp"):
        QcSwapFluxStarkEstimator()._check_data(ds)


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _flux_stark_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = QcSwapFluxStarkEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == {"qc_swap_flux_stark"}
    assert "qc_swap_flux_stark.png" in {p.name for p in tmp_path.iterdir()}


def test_analyze_writes_artifacts(tmp_path):
    est = QcSwapFluxStarkEstimator()
    _res, figs = est.analyze(_flux_stark_ds(), output_dir=str(tmp_path),
                             skip_figures=False, drive_side="low")
    assert set(figs) == {"qc_swap_flux_stark"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_swap_flux_stark.png" in written
    assert "qc_swap_flux_stark_plotdata.nc" in written
    assert "qc_swap_flux_stark_metadata.json" in written
