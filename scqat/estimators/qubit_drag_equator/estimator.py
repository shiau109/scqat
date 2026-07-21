"""DRAG Parameter Calibration (3-Line Equator Method) Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_drag_equator.visualization import plot_drag_equator


class QubitDragEquatorEstimator(BaseEstimator):
    """Estimate optimal DRAG beta by finding intersection of the sequences."""

    estimator_name = "qubit_drag_equator"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset.data_vars:
            raise ValueError("DRAG equator estimator requires a 'signal' data variable")
        if "beta" not in dataset.coords:
            raise ValueError("DRAG equator estimator requires a 'beta' coordinate")
        if "seq_idx" not in dataset.coords:
            raise ValueError("DRAG equator estimator requires a 'seq_idx' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        beta = dataset["beta"].values
        # signal has shape (seq_idx, beta)
        signal = dataset["signal"].values
        
        y0 = signal[0, :]
        y1 = signal[1, :]
        
        # Fit lines to y0 (Rx(pi)-Ry(pi/2)) and y1 (Ry(pi)-Rx(pi/2)) to find their intersection
        p0 = np.polyfit(beta, y0, 1)
        p1 = np.polyfit(beta, y1, 1)
        
        slope_diff = p0[0] - p1[0]
        if np.abs(slope_diff) > 1e-8:
            opt_beta = (p1[1] - p0[1]) / slope_diff
            success = bool(beta[0] <= opt_beta <= beta[-1])
        else:
            opt_beta = float((beta[0] + beta[-1]) / 2)
            success = False
            
        return {
            "beta": beta.tolist(),
            "opt_beta": float(opt_beta) if success else float((p1[1] - p0[1]) / slope_diff) if abs(slope_diff) > 1e-8 else None,
            "success": True if abs(slope_diff) > 1e-8 else False,
            "seq0": y0.tolist(),
            "seq1": y1.tolist(),
            "fit_seq0": (p0[0] * beta + p0[1]).tolist(),
            "fit_seq1": (p1[0] * beta + p1[1]).tolist(),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k not in ("seq0", "seq1", "fit_seq0", "fit_seq1")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        ds = xr.Dataset(
            {
                "seq0": ("beta", np.asarray(results["seq0"], dtype=float)),
                "seq1": ("beta", np.asarray(results["seq1"], dtype=float)),
                "fit_seq0": ("beta", np.asarray(results["fit_seq0"], dtype=float)),
                "fit_seq1": ("beta", np.asarray(results["fit_seq1"], dtype=float)),
            },
            coords={"beta": np.asarray(results["beta"], dtype=float)},
        )
        if results.get("opt_beta") is not None:
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
        return {"qubit_drag_equator": plot_drag_equator(plot_data)}
