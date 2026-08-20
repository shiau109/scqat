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
from scqat.tools.discriminate import discriminate_states, validate_discriminate_kwargs


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
        if dataset.sizes.get("prepared_state", 0) < 2:
            raise ValueError(
                "QubitTomographyEstimator requires at least the two prepared "
                "training states 0 and 1."
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Kwargs — flat and fully owned; unknown names raise:
        ``user_mean`` / ``user_std`` / ``outlier_sigma``, the knobs of
        :func:`scqat.tools.discriminate.discriminate_states`."""
        validate_discriminate_kwargs(kwargs)

        # 1. Train GMM classifier on the training shots, one row per prepared
        # state (the ``(prepared_state, train_shot_idx)`` variables are already
        # array-shaped — no coordinate-rename dance needed for a pure reduction).
        states = dataset.coords["prepared_state"].values
        I_train = np.stack([dataset["I_train"].sel(prepared_state=s).values.ravel() for s in states])
        Q_train = np.stack([dataset["Q_train"].sel(prepared_state=s).values.ravel() for s in states])

        # noise_mode spectator: the probe wrote all-zero dummy I/Q (no
        # measurement was played), so there is nothing to discriminate.
        # Populations degrade to the maximally-ignorant 0.5 and success stays
        # False — a spectator record must never pass for a real tomography fit.
        if not I_train.any() and not Q_train.any():
            gate_counts = dataset.coords["gate_count"].values
            n_gc = len(gate_counts)
            return {
                "centers": {"0": [0.0, 0.0], "1": [0.0, 0.0]},
                "readout_fidelity": 0.5,
                "confusion_matrix": [[0.5, 0.5], [0.5, 0.5]],
                "gate_counts": gate_counts.tolist(),
                "population_x": [0.5] * n_gc,
                "population_y": [0.5] * n_gc,
                "population_z": [0.5] * n_gc,
                "baseline_population_x": [0.5] * n_gc,
                "baseline_population_y": [0.5] * n_gc,
                "baseline_population_z": [0.5] * n_gc,
                "delta_population_x": [0.0] * n_gc,
                "delta_population_y": [0.0] * n_gc,
                "delta_population_z": [0.0] * n_gc,
                "differential_drift": [0.0] * n_gc,
                "interleaved_noise": False,
                "noise_mode": 1.0,
                "success": 0.0,
            }

        sd_res = discriminate_states(I_train, Q_train, **kwargs)
        centers = sd_res["trained_paras"]["mean"]  # shape (2, 2)
        counts = sd_res["direct_counts"]           # fixed (n_state, n_center)

        # Resolve center mapping. direct_counts has a guaranteed
        # (n_state, n_center) shape — a centre that captured no shot is a zero
        # column, so the diagonal reads below never index out of range.
        if counts[0, 0] + counts[1, 1] < counts[0, 1] + counts[1, 0]:
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

        has_nc = "noise_condition" in dataset.coords
        nc_list = [str(nc).lower() for nc in dataset.coords["noise_condition"].values] if has_nc else []
        bases = [b.lower() for b in dataset.coords["basis"].values]
        syms = list(dataset.coords["sym"].values)
        gate_counts = dataset.coords["gate_count"].values

        def _calc_pop_dict(pop_slice: np.ndarray) -> Dict[str, np.ndarray]:
            res = {}
            for b_idx, basis_name in enumerate(bases):
                if "inv" in syms:
                    reg_idx = syms.index("reg")
                    inv_idx = syms.index("inv")
                    p_reg = pop_slice[b_idx, reg_idx, :]
                    p_inv = pop_slice[b_idx, inv_idx, :]
                    p_final = (p_reg + (1.0 - p_inv)) / 2.0
                else:
                    reg_idx = syms.index("reg")
                    p_final = pop_slice[b_idx, reg_idx, :]
                res[basis_name] = p_final
            return res

        has_interleaved = has_nc and "off" in nc_list and "on" in nc_list

        if has_nc:
            pop_by_nc = {}
            for nc_idx, nc_name in enumerate(nc_list):
                pop_by_nc[nc_name] = _calc_pop_dict(pop_sym[nc_idx])

            if has_interleaved:
                pop_on = pop_by_nc["on"]
                pop_off = pop_by_nc["off"]
                pop_x = pop_on.get("x", np.zeros_like(gate_counts))
                pop_y = pop_on.get("y", np.zeros_like(gate_counts))
                pop_z = pop_on.get("z", np.zeros_like(gate_counts))
                base_x = pop_off.get("x", np.zeros_like(gate_counts))
                base_y = pop_off.get("y", np.zeros_like(gate_counts))
                base_z = pop_off.get("z", np.zeros_like(gate_counts))
            else:
                primary = nc_list[0]
                pop_x = pop_by_nc[primary].get("x", np.zeros_like(gate_counts))
                pop_y = pop_by_nc[primary].get("y", np.zeros_like(gate_counts))
                pop_z = pop_by_nc[primary].get("z", np.zeros_like(gate_counts))
                base_x, base_y, base_z = pop_x, pop_y, pop_z
        else:
            pop_dict = _calc_pop_dict(pop_sym)
            pop_x = pop_dict.get("x", np.zeros_like(gate_counts))
            pop_y = pop_dict.get("y", np.zeros_like(gate_counts))
            pop_z = pop_dict.get("z", np.zeros_like(gate_counts))
            base_x, base_y, base_z = pop_x, pop_y, pop_z

        delta_x = pop_x - base_x
        delta_y = pop_y - base_y
        delta_z = pop_z - base_z
        diff_drift = 2.0 * np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        return {
            "centers": {"0": mean_0.tolist(), "1": mean_1.tolist()},
            "readout_fidelity": float(fidelity),
            "confusion_matrix": counts.tolist(),
            "gate_counts": gate_counts.tolist(),
            "population_x": pop_x.tolist(),
            "population_y": pop_y.tolist(),
            "population_z": pop_z.tolist(),
            "baseline_population_x": base_x.tolist(),
            "baseline_population_y": base_y.tolist(),
            "baseline_population_z": base_z.tolist(),
            "delta_population_x": delta_x.tolist(),
            "delta_population_y": delta_y.tolist(),
            "delta_population_z": delta_z.tolist(),
            "differential_drift": diff_drift.tolist(),
            "interleaved_noise": bool(has_interleaved),
            "success": bool(np.isfinite(fidelity) and 0.5 < fidelity <= 1.0)
        }


    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        pop_x = np.array(results["population_x"])
        pop_y = np.array(results["population_y"])
        pop_z = np.array(results["population_z"])
        base_x = np.array(results["baseline_population_x"])
        base_y = np.array(results["baseline_population_y"])
        base_z = np.array(results["baseline_population_z"])

        dist_noise = 2 * np.sqrt((pop_x - 0.5) ** 2 + (pop_y - 0.5) ** 2 + (pop_z - 0.5) ** 2)
        dist_base = 2 * np.sqrt((base_x - 0.5) ** 2 + (base_y - 0.5) ** 2 + (base_z - 0.5) ** 2)

        return xr.Dataset(
            {
                "population_x": ("gate_count", pop_x),
                "population_y": ("gate_count", pop_y),
                "population_z": ("gate_count", pop_z),
                "baseline_population_x": ("gate_count", base_x),
                "baseline_population_y": ("gate_count", base_y),
                "baseline_population_z": ("gate_count", base_z),
                "vector_length": ("gate_count", dist_noise),
                "baseline_vector_length": ("gate_count", dist_base),
                "differential_drift": ("gate_count", np.array(results["differential_drift"])),
            },
            coords={"gate_count": np.array(results["gate_counts"])},
            attrs={
                "centers": json.dumps(results["centers"]),
                "readout_fidelity": results["readout_fidelity"],
                "interleaved_noise": int(results.get("interleaved_noise", False)),
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
        base_x = plot_data["baseline_population_x"].values
        base_y = plot_data["baseline_population_y"].values
        base_z = plot_data["baseline_population_z"].values
        dist_noise = plot_data["vector_length"].values
        dist_base = plot_data["baseline_vector_length"].values
        diff_drift = plot_data["differential_drift"].values
        has_interleaved = bool(plot_data.attrs.get("interleaved_noise", 0))

        # 1. 2D Populations Plot (Pure scatter, no connecting lines)
        fig_2d, axs = plt.subplots(1, 3, figsize=(15, 5))
        for idx, (basis, data, base) in enumerate(zip(["X", "Y", "Z"], [pop_x, pop_y, pop_z], [base_x, base_y, base_z])):
            ax = axs[idx]
            if has_interleaved:
                ax.scatter(gate_counts, base, marker="o", facecolors="none", edgecolors="#64748b", s=55, lw=1.5, label="Baseline (Noise OFF)", alpha=0.9)
                ax.scatter(gate_counts, data, marker="o", facecolors="#dc2626", edgecolors="k", s=55, lw=0.8, label="Noise ON", alpha=0.9)
            else:
                ax.scatter(gate_counts, data, marker="o", facecolors="#1f77b4", edgecolors="k", s=55, alpha=0.85, label="Measured Data")
            ax.set_title(f"{basis} Basis", fontsize=12, fontweight="bold")
            ax.set_xlabel("Gate Count")
            ax.set_ylabel("|1> Population")
            ax.set_ylim(-0.05, 1.05)
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
        title_prefix = "Interleaved Crosstalk Tomography" if has_interleaved else "Tomography Populations"
        fig_2d.suptitle(f"{title_prefix} vs. Gate Count", fontsize=14, fontweight="bold")
        fig_2d.tight_layout()

        # 2. 3D Trajectory Plot (Scatter with colormap gradient for gate count)
        fig_3d = plt.figure(figsize=(8, 8))
        ax_3d = fig_3d.add_subplot(111, projection="3d")
        if has_interleaved:
            ax_3d.scatter(base_x, base_y, base_z, c=gate_counts, cmap="cool", marker="^", s=70, edgecolors="k", lw=0.8, alpha=0.85, label="Baseline (Noise OFF)")
            sc_noise = ax_3d.scatter(pop_x, pop_y, pop_z, c=gate_counts, cmap="viridis", marker="o", s=80, edgecolors="k", lw=0.8, alpha=0.95, label="Noise ON")
            fig_3d.colorbar(sc_noise, ax=ax_3d, label="Gate Count", shrink=0.7, pad=0.1)
        else:
            sc = ax_3d.scatter(pop_x, pop_y, pop_z, c=gate_counts, cmap="viridis", marker="o", s=80, edgecolors="k", lw=0.8, alpha=0.95, label="Measured Data")
            fig_3d.colorbar(sc, ax=ax_3d, label="Gate Count", shrink=0.7, pad=0.1)

        # Draw Bloch sphere wireframe centered at (0.5, 0.5, 0.5)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = 0.5 + 0.5 * np.cos(u) * np.sin(v)
        sphere_y = 0.5 + 0.5 * np.sin(u) * np.sin(v)
        sphere_z = 0.5 + 0.5 * np.cos(v)
        ax_3d.plot_wireframe(sphere_x, sphere_y, sphere_z, color="gray", alpha=0.18)

        ax_3d.set_xlabel("X Axis")
        ax_3d.set_ylabel("Y Axis")
        ax_3d.set_zlabel("Z Axis")
        ax_3d.set_title("3D Visualization of Tomography Gate Error", fontsize=14, fontweight="bold")
        ax_3d.legend()

        # 3. Vector Length vs Gate Count (Pure scatter)
        fig_dist = plt.figure(figsize=(8, 6))
        ax_dist = fig_dist.add_subplot(111)
        if has_interleaved:
            ax_dist.scatter(gate_counts, dist_base, marker="o", facecolors="none", edgecolors="#64748b", s=60, lw=1.5, label="Baseline (Noise OFF)", alpha=0.9)
            ax_dist.scatter(gate_counts, dist_noise, marker="o", facecolors="#dc2626", edgecolors="k", s=60, lw=0.8, label="Noise ON", alpha=0.9)
        else:
            ax_dist.scatter(gate_counts, dist_noise, marker="o", facecolors="#1f77b4", edgecolors="k", s=60, alpha=0.85, label="Vector Length")
        ax_dist.set_xlabel("Gate Count")
        ax_dist.set_ylabel("Vector Length |r| (Purity)")
        ax_dist.set_title("Bloch Vector Length vs. Gate Count", fontsize=14, fontweight="bold")
        ax_dist.set_ylim(-0.05, 1.05)
        ax_dist.legend()
        ax_dist.grid(True, linestyle="--", alpha=0.5)

        figs = {
            "qubit_tomography_2d": fig_2d,
            "qubit_tomography_3d": fig_3d,
            "qubit_tomography_dist": fig_dist,
        }

        # 4. Differential Drift Plot (Pure scatter)
        if has_interleaved:
            fig_diff = plt.figure(figsize=(8, 6))
            ax_diff = fig_diff.add_subplot(111)
            ax_diff.scatter(gate_counts, diff_drift, marker="o", facecolors="#dc2626", edgecolors="k", s=70, lw=1.0, label="Pure Crosstalk Drift |Δr|", alpha=0.9)
            ax_diff.set_xlabel("Gate Count")
            ax_diff.set_ylabel("Differential Drift |Δr| (Bloch Units)")
            ax_diff.set_title("Crosstalk Differential Drift (Common-Mode Drift Subtracted)", fontsize=14, fontweight="bold")
            ax_diff.set_ylim(-0.05, max(1.05, float(np.max(diff_drift) * 1.15) if len(diff_drift) else 1.05))
            ax_diff.legend()
            ax_diff.grid(True, linestyle="--", alpha=0.5)
            figs["qubit_tomography_diff"] = fig_diff

        return figs
