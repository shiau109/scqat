"""
Qubit Tomography Estimator
==========================
Analyzes state tomography data using GMM-trained classifiers.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - basis           : 1-D string array - measurement bases (e.g. ['x', 'y', 'z'])
    - sym             : 1-D string array - readout symmetry (e.g. ['reg', 'inv'] or ['reg'])
    - gate_count      : 1-D int array - number of target gates applied
    - shot_idx        : 1-D int array - shot indices for tomography
    - prepared_state  : 1-D int array - prepared training states (e.g. [0, 1])
    - train_shot_idx  : 1-D int array - shot indices for training GMM

Data variables:
    - I_tomo          : (basis, sym, gate_count, shot_idx) - raw I quadrature for tomography
    - Q_tomo          : (basis, sym, gate_count, shot_idx) - raw Q quadrature for tomography
    - I_train         : (prepared_state, train_shot_idx) - raw I quadrature for GMM training
    - Q_train         : (prepared_state, train_shot_idx) - raw Q quadrature for GMM training
"""

import json
from typing import Any, Dict, Optional
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.state_discrimination import StateDiscriminationEstimator


class QubitTomographyEstimator(BaseEstimator):
    """Classify tomography shots using GMM-trained centers and calculate basis populations."""

    estimator_name = "qubit_tomography"

    def _check_data(self, dataset: xr.Dataset) -> None:
        for var in ("I_tomo", "Q_tomo", "I_train", "Q_train"):
            if var not in dataset:
                raise ValueError(f"QubitTomographyEstimator requires variable '{var}'")
        for coord in ("basis", "sym", "gate_count", "shot_idx", "prepared_state", "train_shot_idx"):
            if coord not in dataset.coords:
                raise ValueError(f"QubitTomographyEstimator requires coordinate '{coord}'")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        # 1. Train GMM classifier on the training data
        sd = StateDiscriminationEstimator()
        
        # Prepare training dataset for StateDiscriminationEstimator
        train_ds = xr.Dataset(
            {
                "I": dataset["I_train"],
                "Q": dataset["Q_train"]
            }
        ).rename({"train_shot_idx": "shot_idx"})
        
        sd_res = sd.extract_parameters(train_ds, **kwargs)
        centers = sd_res["trained_paras"]["mean"]  # shape (2, 2)
        counts = sd_res["direct_counts"]
        
        # Resolve center mapping
        if counts.shape == (2, 2) and (counts[0, 0] + counts[1, 1] < counts[0, 1] + counts[1, 0]):
            mean_0 = centers[1]
            mean_1 = centers[0]
            fidelity = 0.5 * (counts[0, 1] + counts[1, 0])
        else:
            mean_0 = centers[0]
            mean_1 = centers[1]
            fidelity = 0.5 * (counts[0, 0] + counts[1, 1])

        # 2. Vectorized Euclidean distance classification for tomography shots
        I_tomo = dataset["I_tomo"].values
        Q_tomo = dataset["Q_tomo"].values
        
        dist0 = np.sqrt((I_tomo - mean_0[0])**2 + (Q_tomo - mean_0[1])**2)
        dist1 = np.sqrt((I_tomo - mean_1[0])**2 + (Q_tomo - mean_1[1])**2)
        
        # Classify as 1 if closer to mean_1, else 0
        labels = (dist1 < dist0).astype(float)
        
        # Average over shot_idx (the last dimension) to get population of state 1
        pop_sym = np.mean(labels, axis=-1)
        
        bases = [b.lower() for b in dataset.coords["basis"].values]
        syms = list(dataset.coords["sym"].values)
        gate_counts = dataset.coords["gate_count"].values
        
        # Calculate final population for each basis (handling sym: reg and inv)
        pop_final = {}
        for b_idx, basis_name in enumerate(bases):
            if "inv" in syms:
                reg_idx = syms.index("reg")
                inv_idx = syms.index("inv")
                p_reg = pop_sym[b_idx, reg_idx, :]
                p_inv = pop_sym[b_idx, inv_idx, :]
                # Symmetrized readout mitigation
                p_final = (p_reg + (1.0 - p_inv)) / 2.0
            else:
                reg_idx = syms.index("reg")
                p_final = pop_sym[b_idx, reg_idx, :]
            pop_final[basis_name] = p_final

        return {
            "centers": {"0": mean_0.tolist(), "1": mean_1.tolist()},
            "readout_fidelity": float(fidelity),
            "confusion_matrix": counts.tolist(),
            "gate_counts": gate_counts.tolist(),
            "population_x": pop_final.get("x", np.zeros_like(gate_counts)).tolist(),
            "population_y": pop_final.get("y", np.zeros_like(gate_counts)).tolist(),
            "population_z": pop_final.get("z", np.zeros_like(gate_counts)).tolist(),
            "success": bool(np.isfinite(fidelity) and 0.5 < fidelity <= 1.0)
        }


    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        return xr.Dataset(
            {
                "population_x": ("gate_count", np.array(results["population_x"])),
                "population_y": ("gate_count", np.array(results["population_y"])),
                "population_z": ("gate_count", np.array(results["population_z"])),
            },
            coords={"gate_count": np.array(results["gate_counts"])},
            attrs={
                "centers": json.dumps(results["centers"]),
                "readout_fidelity": results["readout_fidelity"]
            }
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
            
        gate_counts = plot_data.coords["gate_count"].values
        pop_x = plot_data["population_x"].values
        pop_y = plot_data["population_y"].values
        pop_z = plot_data["population_z"].values
        
        # 1. 2D Populations Plot
        fig_2d, axs = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (basis, data) in enumerate(zip(["X", "Y", "Z"], [pop_x, pop_y, pop_z])):
            ax = axs[idx]
            ax.scatter(gate_counts, data, label="Measured Data", color="#1f77b4", alpha=0.8, edgecolors='k')
            ax.set_title(f"{basis} Basis", fontsize=12, fontweight='bold')
            ax.set_xlabel("Gate Count")
            ax.set_ylabel("|1> Population")
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
        fig_2d.suptitle("Tomography Populations vs. Gate Count", fontsize=14, fontweight='bold')
        fig_2d.tight_layout()
        
        # 2. 3D Trajectory Plot
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        scatter = ax_3d.scatter(pop_x, pop_y, pop_z, c=gate_counts, cmap='viridis', s=60, alpha=0.9, edgecolors='k')
        fig_3d.colorbar(scatter, ax=ax_3d, label="Gate Count")
        
        # Draw Bloch sphere wireframe centered at (0.5, 0.5, 0.5)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = 0.5 + 0.5 * np.cos(u) * np.sin(v)
        sphere_y = 0.5 + 0.5 * np.sin(u) * np.sin(v)
        sphere_z = 0.5 + 0.5 * np.cos(v)
        ax_3d.plot_wireframe(sphere_x, sphere_y, sphere_z, color="gray", alpha=0.2)
        
        ax_3d.set_xlabel("X Axis")
        ax_3d.set_ylabel("Y Axis")
        ax_3d.set_zlabel("Z Axis")
        ax_3d.set_title("3D Visualization of Tomography Gate Error", fontsize=14, fontweight='bold')
        
        # 3. Vector Length vs Gate Count
        distances = 2 * np.sqrt((pop_x - 0.5) ** 2 + (pop_y - 0.5) ** 2 + (pop_z - 0.5) ** 2)
        fig_dist = plt.figure(figsize=(8, 6))
        ax_dist = fig_dist.add_subplot(111)
        ax_dist.scatter(gate_counts, distances, color='#d62728', s=50, alpha=0.8, edgecolors='k', label="Vector Length")
        ax_dist.set_xlabel("Gate Count")
        ax_dist.set_ylabel("Vector Length")
        ax_dist.set_title("Vector Length vs. Gate Count", fontsize=14, fontweight='bold')
        ax_dist.set_ylim(-0.05, 1.05)
        ax_dist.legend()
        ax_dist.grid(True, linestyle='--', alpha=0.5)
        
        return {
            "qubit_tomography_2d": fig_2d,
            "qubit_tomography_3d": fig_3d,
            "qubit_tomography_dist": fig_dist,
        }
