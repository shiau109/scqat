"""Qubit Single Qubit Randomized Benchmarking (SQRB) Estimator.

Fits a base power-law model:
    y = A * (alpha ** depth) + C
where alpha is the depolarizing parameter.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_powerlaw_base import FitBasePowerLaw


class QubitSQRBEstimator(BaseEstimator):
    """Fit SQRB population decay and extract gate fidelity."""

    estimator_name = "qubit_sqrb"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "I" not in dataset.data_vars:
            raise ValueError("SQRB estimator requires an 'I' data variable")
        if "depth" not in dataset.coords:
            raise ValueError("SQRB estimator requires a 'depth' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        # Average over sequence_idx to get the decay curve
        if "sequence_idx" in dataset.dims:
            da = dataset["I"].mean(dim="sequence_idx").squeeze().rename({"depth": "x"})
        else:
            da = dataset["I"].squeeze().rename({"depth": "x"})
            
        fit_result = FitBasePowerLaw(da).fit()
        alpha = float(fit_result.params["base"].value)
        
        # Error per Clifford: r_c = 0.5 * (1 - alpha)
        error_per_clifford = 0.5 * (1.0 - alpha)
        
        # Average number of gates per Clifford: 1.875
        average_gate_per_clifford = 1.875
        error_per_gate = error_per_clifford / average_gate_per_clifford
        gate_fidelity = 1.0 - error_per_gate
        
        return {
            "alpha": alpha,
            "alpha_stderr": float(fit_result.params["base"].stderr or np.nan),
            "amplitude": float(fit_result.params["a"].value),
            "offset": float(fit_result.params["c"].value),
            "error_per_clifford": error_per_clifford,
            "error_per_gate": error_per_gate,
            "gate_fidelity": gate_fidelity,
            "success": bool(fit_result.success) and (0.0 < alpha <= 1.0),
            "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "best_fit"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        depths = np.asarray(dataset["depth"].values, dtype=float)
        if "sequence_idx" in dataset.dims:
            y_avg = np.asarray(dataset["I"].mean(dim="sequence_idx").values, dtype=float)
        else:
            y_avg = np.asarray(dataset["I"].values, dtype=float)
            
        return xr.Dataset(
            {
                "signal": ("depth", y_avg),
                "best_fit": ("depth", results["best_fit"]),
            },
            coords={"depth": depths},
            attrs={
                "gate_fidelity": results["gate_fidelity"],
                "error_per_gate": results["error_per_gate"],
                "alpha": results["alpha"],
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
            
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(plot_data["depth"], plot_data["signal"], "o-", label="Data")
        ax.plot(plot_data["depth"], plot_data["best_fit"], "-", label="Fit")
        ax.set_xlabel("Clifford Depth")
        ax.set_ylabel("Ground State Population")
        ax.set_title(f"Single Qubit RB (Fidelity: {plot_data.attrs['gate_fidelity']*100:.3f}%)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        return {"qubit_sqrb": fig}
