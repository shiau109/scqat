"""Synthetic-grid tests for the T1-vs-flux estimator.

Dataset contract under test: data var ``signal`` on coords
``(flux_bias, wait_time)`` — the axis renamed from the old ``flux_amp``.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qubit_relaxation_flux import QubitRelaxationFluxEstimator


def _t1_for(flux: float) -> float:
    return 25e-6 * (1.0 - 20.0 * flux * flux)


def _synthetic_dataset() -> tuple[np.ndarray, xr.Dataset]:
    flux = np.linspace(-0.08, 0.08, 7)
    wait = np.linspace(16e-9, 120e-6, 61)
    signal = np.stack([np.exp(-wait / _t1_for(f)) for f in flux])
    ds = xr.Dataset(
        {"signal": (("flux_bias", "wait_time"), signal)},
        coords={"flux_bias": flux, "wait_time": wait},
    )
    return flux, ds


def test_recovers_t1_spectrum():
    flux, ds = _synthetic_dataset()
    est = QubitRelaxationFluxEstimator()
    est._check_data(ds)
    results = est.extract_parameters(ds)
    assert results["success"]
    assert np.allclose(results["flux_bias"], flux)
    assert np.allclose(results["t1"], [_t1_for(f) for f in flux], rtol=0.05)


def test_plot_data_uses_flux_bias_axis():
    _flux, ds = _synthetic_dataset()
    est = QubitRelaxationFluxEstimator()
    results = est.extract_parameters(ds)
    plot_data = est.build_plot_data(ds, results)
    assert "flux_bias" in plot_data.coords
    assert plot_data["t1"].dims == ("flux_bias",)


def test_rejects_old_flux_amp_coordinate():
    flux, ds = _synthetic_dataset()
    stale = ds.rename({"flux_bias": "flux_amp"})
    with pytest.raises(ValueError, match="flux_bias"):
        QubitRelaxationFluxEstimator()._check_data(stale)
