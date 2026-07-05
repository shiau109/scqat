"""
T2 Echo Estimator
=================
Single-exponential fit of the Hahn-echo signal (X90 - tau/2 - X - tau/2 - X90).

The echo refocuses quasi-static dephasing, so on resonance the envelope is a
pure decay with time constant T2_echo (no fringe) — the same model as T1, on a
different physical quantity.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - idle_time : 1-D float array - total echo idle time tau (s).
Data variables:
    - signal    : (idle_time,) - echo signal decaying toward its offset.

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_exp_decay import FitExponentialDecay

from scqat.estimators.t2_echo.visualization import plot_decay


class T2EchoEstimator(BaseEstimator):
    """Fit ``signal = a * exp(-t / t2_echo) + c`` and report T2_echo (seconds)."""

    estimator_name = "t2_echo"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset.data_vars:
            raise ValueError("T2 echo estimator requires a 'signal' data variable")
        if "idle_time" not in dataset.coords:
            raise ValueError("T2 echo estimator requires an 'idle_time' coordinate (seconds)")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        da = dataset["signal"].squeeze().rename({"idle_time": "x"})
        fit_result = FitExponentialDecay(da).fit()
        t2e = float(fit_result.params["tau"].value)
        t_span = float(da["x"].values[-1] - da["x"].values[0])
        return {
            "t2_echo": t2e,
            "t2_echo_stderr": float(fit_result.params["tau"].stderr or np.nan),
            "amplitude": float(fit_result.params["a"].value),
            "offset": float(fit_result.params["c"].value),
            "redchi": float(fit_result.redchi),
            # physical: converged, positive, and not absurdly beyond the swept window
            "success": bool(fit_result.success) and np.isfinite(t2e) and 0 < t2e < 10 * t_span,
            "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "best_fit"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        idle = np.asarray(dataset["idle_time"].values, dtype=float)
        return xr.Dataset(
            {
                "signal": ("idle_time", np.asarray(dataset["signal"].values, dtype=float)),
                "best_fit": ("idle_time", results["best_fit"]),
            },
            coords={"idle_time": idle},
            attrs={
                "t2_echo": results["t2_echo"],
                "amplitude": results["amplitude"],
                "offset": results["offset"],
                "success": int(bool(results["success"])),
            },
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        # single-figure idiom: key == estimator_name -> saved as t2_echo.png
        return {"t2_echo": plot_decay(plot_data)}
