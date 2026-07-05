"""
T1 Relaxation Estimator
=======================
Single-exponential fit of the excited-state signal after a pi pulse.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - wait_time : 1-D float array - delay after the pi pulse (s).
Data variables:
    - signal    : (wait_time,) - excited-state signal (e.g. rotated I quadrature
                  or population), decaying toward its offset.

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_exp_decay import FitExponentialDecay

from scqat.estimators.t1_relaxation.visualization import plot_decay


class T1RelaxationEstimator(BaseEstimator):
    """Fit ``signal = a * exp(-t / t1) + c`` and report T1 (seconds)."""

    estimator_name = "t1_relaxation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset.data_vars:
            raise ValueError("T1 estimator requires a 'signal' data variable")
        if "wait_time" not in dataset.coords:
            raise ValueError("T1 estimator requires a 'wait_time' coordinate (seconds)")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        da = dataset["signal"].squeeze().rename({"wait_time": "x"})
        fit_result = FitExponentialDecay(da).fit()
        t1 = float(fit_result.params["tau"].value)
        t_span = float(da["x"].values[-1] - da["x"].values[0])
        return {
            "t1": t1,
            "t1_stderr": float(fit_result.params["tau"].stderr or np.nan),
            "amplitude": float(fit_result.params["a"].value),
            "offset": float(fit_result.params["c"].value),
            "redchi": float(fit_result.redchi),
            # physical: converged, positive, and not absurdly beyond the swept window
            "success": bool(fit_result.success) and np.isfinite(t1) and 0 < t1 < 10 * t_span,
            "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "best_fit"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        wait = np.asarray(dataset["wait_time"].values, dtype=float)
        return xr.Dataset(
            {
                "signal": ("wait_time", np.asarray(dataset["signal"].values, dtype=float)),
                "best_fit": ("wait_time", results["best_fit"]),
            },
            coords={"wait_time": wait},
            attrs={
                "t1": results["t1"],
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
        # single-figure idiom: key == estimator_name -> saved as t1_relaxation.png
        return {"t1_relaxation": plot_decay(plot_data)}
