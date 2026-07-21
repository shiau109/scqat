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
        if not any(var in dataset.data_vars for var in ("I", "state", "signal")):
            raise ValueError("SQRB estimator requires an 'I', 'state', or 'signal' data variable")
        if "depth" not in dataset.coords:
            raise ValueError("SQRB estimator requires a 'depth' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        var_name = "state" if "state" in dataset.data_vars else ("I" if "I" in dataset.data_vars else "signal")
        # Average over sequence_idx to get the decay curve
        if "sequence_idx" in dataset.dims:
            da = dataset[var_name].mean(dim="sequence_idx").squeeze().rename({"depth": "x"})
        else:
            da = dataset[var_name].squeeze().rename({"depth": "x"})
            
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
        var_name = "state" if "state" in dataset.data_vars else ("I" if "I" in dataset.data_vars else "signal")
        depths = np.asarray(dataset["depth"].values, dtype=float)
        if "sequence_idx" in dataset.dims:
            y_avg = np.asarray(dataset[var_name].mean(dim="sequence_idx").values, dtype=float)
        else:
            y_avg = np.asarray(dataset[var_name].values, dtype=float)
            
        return xr.Dataset(
            {
                "signal": ("depth", y_avg),
                "best_fit": ("depth", results["best_fit"]),
            },
            coords={"depth": depths},
            attrs={
                "gate_fidelity": results["gate_fidelity"],
                "error_per_gate": results["error_per_gate"],
                "error_per_clifford": results["error_per_clifford"],
                "alpha": results["alpha"],
                "alpha_stderr": results.get("alpha_stderr", np.nan),
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
            
        fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
        ax.plot(plot_data["depth"], plot_data["signal"], "o", color="#1f77b4", label="Data (sequence avg)", markersize=5)
        ax.plot(plot_data["depth"], plot_data["best_fit"], "-", color="#d62728", linewidth=2, label="Power-law fit")

        ax.set_xscale("log")
        ax.set_xlabel("Clifford Depth (log scale)", fontsize=11)
        ax.set_ylabel("Readout Signal (a.u.)", fontsize=11)

        fidelity = plot_data.attrs.get("gate_fidelity", np.nan) * 100.0
        epg = plot_data.attrs.get("error_per_gate", np.nan)
        epc = plot_data.attrs.get("error_per_clifford", np.nan)
        alpha = plot_data.attrs.get("alpha", np.nan)
        alpha_err = plot_data.attrs.get("alpha_stderr", np.nan)

        ax.set_title(f"Single Qubit Randomized Benchmarking (SQRB)\nGate Fidelity: {fidelity:.3f}%", fontsize=12, fontweight="bold")

        if np.isfinite(alpha_err):
            info_text = (
                f"Gate Fidelity: {fidelity:.3f}%\n"
                f"Error / Gate (r_g): {epg:.4e}\n"
                f"Error / Clifford (r_c): {epc:.4e}\n"
                f"Depolarizing (alpha): {alpha:.6f} ± {alpha_err:.6f}"
            )
        else:
            info_text = (
                f"Gate Fidelity: {fidelity:.3f}%\n"
                f"Error / Gate (r_g): {epg:.4e}\n"
                f"Error / Clifford (r_c): {epc:.4e}\n"
                f"Depolarizing (alpha): {alpha:.6f}"
            )

        ax.text(
            0.05, 0.05, info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"),
        )

        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()

        return {"qubit_sqrb": fig}
