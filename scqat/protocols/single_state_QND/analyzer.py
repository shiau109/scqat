from typing import Any, Dict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_gaussian2d import FitGaussian2D
from scqat.protocols.single_state_QND.visualization import (
    plot_2d_histogram_single,
    plot_outliers_single,
    plot_distance_vs_shot,
)


class SingleStateOutlierAnalyzer(BaseAnalyzer):
    """
    Analyzes single-state I/Q plane data using a single 2D Gaussian fit.
    Expects a dataset with 'I' and 'Q' variables indexed by 'shot_idx' only
    (no 'prepared_state' coordinate).
    """

    protocol_name = "single_state_QND"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "shot_idx" not in dataset.coords:
            raise ValueError("single_state_QND requires 'shot_idx' coordinate in the dataset.")
        for var in ["I", "Q"]:
            if var not in dataset.data_vars:
                raise ValueError(f"single_state_QND requires '{var}' data variable in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fits a single 2D Gaussian to the I/Q histogram and detects outliers.

        Kwargs:
            user_std (float): Optional initial guess for Gaussian std dev.
            outlier_sigma (float): Threshold for outlier detection (default: 3).
        """
        user_std = kwargs.get("user_std", None)
        outlier_sigma = kwargs.get("outlier_sigma", 3)

        # 1. Preprocess into 2D histogram
        hist_dataset, std_init = self._preprocess_data(dataset, user_std)

        # 2. Fit single 2D Gaussian
        fitted_paras, fit_residue, norm_res = self._fit_gaussian(hist_dataset)

        # 3. Outlier detection
        I_vals = dataset["I"].values.ravel()
        Q_vals = dataset["Q"].values.ravel()
        mean = fitted_paras["mean"][0]
        distances = np.sqrt((I_vals - mean[0]) ** 2 + (Q_vals - mean[1]) ** 2)
        outlier_threshold = outlier_sigma * fitted_paras["std"]
        outlier_mask = distances > outlier_threshold
        outlier_indices = np.flatnonzero(outlier_mask)
        outlier_probability = np.count_nonzero(outlier_mask) / len(distances)

        return {
            "fitted_paras": fitted_paras,
            "outlier_mask": outlier_mask,
            "outlier_indices": outlier_indices,
            "outlier_probability": outlier_probability,
            "norm_res": norm_res,
            "fit_residue": fit_residue,
            "hist_dataset": hist_dataset,
        }

    def generate_figures(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Dict[str, plt.Figure]:
        figs = {}

        fig_hist, ax_hist = plot_2d_histogram_single(
            results["hist_dataset"], analysis_result=results
        )
        fig_outliers, ax_outliers = plot_outliers_single(
            dataset, results["outlier_mask"], analysis_result=results
        )

        fig_dist, ax_dist = plot_distance_vs_shot(dataset, results)

        figs["2DHist"] = fig_hist
        figs["outliers"] = fig_outliers
        figs["distance_vs_shot"] = fig_dist
        return figs

    # ==========================================
    # STATELESS HELPER METHODS
    # ==========================================

    def _preprocess_data(self, dataset: xr.Dataset, user_std=None):
        I_vals = dataset["I"].values.ravel()
        Q_vals = dataset["Q"].values.ravel()

        std_I = np.std(I_vals)
        std_Q = np.std(Q_vals)
        std_init = min(std_I, std_Q)

        step = (user_std if user_std else std_init) / 3

        xedges = np.arange(I_vals.min(), I_vals.max() + step, step)
        yedges = np.arange(Q_vals.min(), Q_vals.max() + step, step)
        if len(xedges) < 2:
            xedges = np.linspace(I_vals.min(), I_vals.max(), 2)
        if len(yedges) < 2:
            yedges = np.linspace(Q_vals.min(), Q_vals.max(), 2)

        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

        H, _, _ = np.histogram2d(I_vals, Q_vals, bins=[xedges, yedges], density=True)
        density = H.T  # shape (len(y), len(x))

        hist_dataset = xr.Dataset(
            {"density": (["y", "x"], density)},
            coords={"x": xcenters, "y": ycenters},
        )
        return hist_dataset, std_init

    def _fit_gaussian(self, hist_dataset):
        density = hist_dataset["density"].values
        x = hist_dataset["x"].values
        y = hist_dataset["y"].values

        fitter = FitGaussian2D(density, x, y)
        fitter.params['sigma_y'].set(expr='sigma_x')
        fit_result = fitter.fit()

        p = fit_result.params
        fitted_paras = {
            "mean": np.array([[p["x0"].value, p["y0"].value]]),
            "sigma_x": p["sigma_x"].value,
            "sigma_y": p["sigma_y"].value,
            "std": (p["sigma_x"].value + p["sigma_y"].value) / 2,
            "amp": p["amp"].value,
            "offset": p["offset"].value,
        }

        best_fit = fit_result.best_fit.reshape(density.shape)
        fit_residue = density - best_fit
        total_density = np.nansum(density)
        norm_res = np.nansum(fit_residue) / total_density if total_density != 0 else np.nan

        return fitted_paras, fit_residue, norm_res
