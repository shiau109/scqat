"""DRAG Parameter Calibration (Alternating Pulse Method) Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_drag_alternating.visualization import plot_drag_alternating


class QubitDragAlternatingEstimator(BaseEstimator):
    """Estimate optimal DRAG by finding flat signal response over pulse count."""

    estimator_name = "qubit_drag_alternating"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset.data_vars:
            raise ValueError("DRAG alternating estimator requires a 'signal' data variable")
        if "beta" not in dataset.coords:
            raise ValueError("DRAG alternating estimator requires a 'beta' coordinate")
        if "nb_of_pulses" not in dataset.coords:
            raise ValueError("DRAG alternating estimator requires a 'nb_of_pulses' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        beta = dataset["beta"].values
        npi = dataset["nb_of_pulses"].values
        signal = dataset["signal"].values  # shape (nb_of_pulses, beta)
        
        # Calculate signal variance over the pulse sweep axis for each beta point
        # The optimal beta point minimizes the error accumulation and remains flat
        variances = np.var(signal, axis=0)
        try:
            p = np.polyfit(beta, variances, 2)
            if p[0] > 0:  # U-shaped parabola with a minimum
                vertex = -p[1] / (2 * p[0])
                if beta[0] <= vertex <= beta[-1]:
                    opt_beta = float(vertex)
                else:
                    opt_beta = float(beta[np.argmin(variances)])
            else:
                opt_beta = float(beta[np.argmin(variances)])
        except Exception:
            opt_beta = float(beta[np.argmin(variances)])
        
        return {
            "beta": beta.tolist(),
            "nb_of_pulses": npi.tolist(),
            "opt_beta": opt_beta,
            "success": True,
            "signal": signal,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "signal"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        ds = xr.Dataset(
            {
                "signal": (("nb_of_pulses", "beta"), np.asarray(results["signal"], dtype=float)),
            },
            coords={
                "beta": np.asarray(results["beta"], dtype=float),
                "nb_of_pulses": np.asarray(results["nb_of_pulses"], dtype=float),
            },
        )
        ds.attrs["opt_beta"] = results["opt_beta"]
        return ds

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"qubit_drag_alternating": plot_drag_alternating(plot_data)}
