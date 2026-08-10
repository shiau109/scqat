"""Synthetic-grid tests for the N-swap AC-Stark amplitude raw-population estimator.

The AC-Stark sibling of ``qc_n_swap_amp``: same ``joint_population`` form, drawn
over ``(stark_amp, swap_count)`` with an INTEGER swap-count axis. The shared
summary / plot-data / figure code is exercised in depth by
``test_pair_swap_chevron_estimator``; here we pin this estimator's own axis
names (including the integer count axis) and the artifact writeout.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_n_stark_amp import QcNStarkAmpEstimator

SA0, N0 = 0.0, 7  # where the (synthetic) transfer peaks: ideal stark amplitude, swap count
LABELS = ["00", "01", "10", "11"]


def _n_stark_amp_ds() -> xr.Dataset:
    sa = np.linspace(-0.1, 0.1, 11)          # AC-Stark amplitude (prefactor)
    n = np.arange(0, 11)                      # swap count N (integer)
    spot = np.exp(-((sa[:, None] - SA0) ** 2) / 0.002) * np.exp(-((n[None, :] - N0) ** 2) / 8.0)
    p11 = np.full((sa.size, n.size), 0.01)
    p10 = 0.84 * spot                        # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - spot)                # driven (low) member depletes at the peak
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])
    return xr.Dataset(
        {"joint_population": (("joint_state", "stark_amp", "swap_count"), jp)},
        coords={"joint_state": LABELS, "stark_amp": sa, "swap_count": n},
    )


def test_summarizes_spot_on_amp_and_count_axes():
    ds = _n_stark_amp_ds()
    est = QcNStarkAmpEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"
    assert res["best_stark_amp"] == pytest.approx(SA0, abs=0.02)
    assert res["best_swap_count"] == pytest.approx(N0, abs=1)
    assert res["n_stark_amp"] == 11 and res["n_swap_count"] == 11


def test_plot_data_uses_amp_count_axes_and_joint_basis():
    ds = _n_stark_amp_ds()
    est = QcNStarkAmpEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert set(pd.data_vars) == {"p00", "p01", "p10", "p11"}
    assert set(pd.coords) == {"stark_amp", "swap_count"}
    assert pd.attrs["axis0"] == "stark_amp"
    assert pd.attrs["axis1"] == "swap_count"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    ds = _n_stark_amp_ds().rename({"swap_count": "rounds"})
    with pytest.raises(ValueError, match="swap_count"):
        QcNStarkAmpEstimator()._check_data(ds)


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _n_stark_amp_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = QcNStarkAmpEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == {"qc_n_stark_amp"}
    assert "qc_n_stark_amp.png" in {p.name for p in tmp_path.iterdir()}


def test_analyze_writes_artifacts(tmp_path):
    est = QcNStarkAmpEstimator()
    _res, figs = est.analyze(_n_stark_amp_ds(), output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert set(figs) == {"qc_n_stark_amp"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_n_stark_amp.png" in written
    assert "qc_n_stark_amp_plotdata.nc" in written
    assert "qc_n_stark_amp_metadata.json" in written
