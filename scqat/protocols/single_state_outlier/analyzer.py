from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_gaussian2d import FitGaussian2D
from scqat.protocols.single_state_outlier.visualization import (
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
            user_mean (array-like of length 2): Optional initial guess for [I, Q] center.
            user_std (float): Optional initial guess for Gaussian std dev.
            outlier_sigma (float): Threshold for outlier detection (default: 3).
            fixed_mean (array-like of length 2): Fixed [I, Q] center — skips fitting.
            fixed_std (float): Fixed Gaussian std — skips fitting.
                When both fixed_mean and fixed_std are given, the analyzer runs
                in "outlier-only" mode: no histogram, no fit, just distance-based
                outlier detection using the provided parameters.
        """
        user_mean = kwargs.get("user_mean", None)
        user_std = kwargs.get("user_std", None)
        outlier_sigma = kwargs.get("outlier_sigma", 3)
        fixed_mean = kwargs.get("fixed_mean", None)
        fixed_std = kwargs.get("fixed_std", None)

        # Outlier-only mode: skip fitting entirely
        if fixed_mean is not None and fixed_std is not None:
            fixed_mean = np.asarray(fixed_mean)
            fitted_paras = {
                "mean": fixed_mean.reshape(1, 2),
                "sigma_x": float(fixed_std),
                "sigma_y": float(fixed_std),
                "std": float(fixed_std),
                "amp": np.nan,
                "offset": np.nan,
            }
            hist_dataset = None
            fit_residue = None
            norm_res = np.nan
        else:
            # 1. Preprocess into 2D histogram
            hist_dataset, std_init = self._preprocess_data(dataset, user_std)

            # 2. Fit single 2D Gaussian
            fitted_paras, fit_residue, norm_res = self._fit_gaussian(
                hist_dataset, user_mean=user_mean, user_std=user_std
            )

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

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters and outlier summary; drop the per-shot mask,
        residue array, and binned histogram."""
        return {
            "fitted_paras": results["fitted_paras"],
            "outlier_probability": results["outlier_probability"],
            "outlier_indices": results["outlier_indices"],
            "norm_res": results["norm_res"],
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the 2-D histogram density (over ``x``/``Q``-``y``) with the fitted
        Gaussian centre, std, and outlier probability in ``.attrs`` so the 2-D
        histogram figure redraws without refitting.

        Returns ``None`` in outlier-only mode (no histogram was built).
        """
        hist = results.get("hist_dataset")
        if hist is None:
            return None

        fp = results["fitted_paras"]
        mean = fp["mean"][0]
        return xr.Dataset(
            {"density": (["y", "x"], hist["density"].values)},
            coords={"x": hist["x"].values, "y": hist["y"].values},
            attrs={
                "mean_I": float(mean[0]),
                "mean_Q": float(mean[1]),
                "std": float(fp["std"]),
                "outlier_probability": float(results["outlier_probability"]),
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
        if plot_data is None:
            return {}

        fig_hist, _ = plot_2d_histogram_single(plot_data)
        return {"2DHist": fig_hist}

    # ==========================================
    # STATELESS HELPER METHODS
    # ==========================================

    def _preprocess_data(self, dataset: xr.Dataset, user_std=None):
        I_vals = dataset["I"].values.ravel()
        Q_vals = dataset["Q"].values.ravel()

        # Filter out NaN/Inf values
        valid = np.isfinite(I_vals) & np.isfinite(Q_vals)
        I_vals = I_vals[valid]
        Q_vals = Q_vals[valid]
        if len(I_vals) == 0:
            raise ValueError("No finite I/Q data remaining after filtering NaN/Inf.")

        std_I = np.std(I_vals)
        std_Q = np.std(Q_vals)
        std_init = min(std_I, std_Q)
        if std_init == 0:
            std_init = max(std_I, std_Q)
        if std_init == 0:
            std_init = 1.0  # fallback for completely degenerate data

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

    def _fit_gaussian(self, hist_dataset, user_mean=None, user_std=None):
        density = hist_dataset["density"].values
        x = hist_dataset["x"].values
        y = hist_dataset["y"].values

        fitter = FitGaussian2D(density, x, y)
        fitter.params['sigma_y'].set(expr='sigma_x')
        fitter.params['offset'].set(value=0, vary=False)
        if user_mean is not None:
            fitter.params['x0'].set(value=user_mean[0])
            fitter.params['y0'].set(value=user_mean[1])
        if user_std is not None:
            fitter.params['sigma_x'].set(value=user_std)
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
