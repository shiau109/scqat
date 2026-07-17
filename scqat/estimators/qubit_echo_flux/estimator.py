"""Qubit Echo vs Flux (T2 Echo Spectrum) Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_exp_decay import FitExponentialDecay
from scqat.estimators.qubit_echo_flux.visualization import plot_echo_flux


class QubitEchoFluxEstimator(BaseEstimator):
    """Fit T2 echo decay curve at each flux point to extract the T2 echo spectrum."""

    estimator_name = "qubit_echo_flux"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset.data_vars:
            raise ValueError("T2 echo vs flux estimator requires a 'signal' data variable")
        if "wait_time" not in dataset.coords:
            raise ValueError("T2 echo vs flux estimator requires a 'wait_time' coordinate (seconds)")
        if "flux_amp" not in dataset.coords:
            raise ValueError("T2 echo vs flux estimator requires a 'flux_amp' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        flux_amps = dataset["flux_amp"].values
        wait_times = dataset["wait_time"].values
        signal = dataset["signal"].values
        
        n_flux = len(flux_amps)
        t2_arr = np.zeros(n_flux)
        t2_stderr_arr = np.zeros(n_flux)
        amp_arr = np.zeros(n_flux)
        offset_arr = np.zeros(n_flux)
        success_arr = np.zeros(n_flux, dtype=bool)
        
        t_span = float(wait_times[-1] - wait_times[0])
        best_fit = np.zeros((n_flux, len(wait_times)))
        
        for idx in range(n_flux):
            y = signal[idx, :]
            fit_result = FitExponentialDecay(x=wait_times, data=y).fit()
            t2 = float(fit_result.params["tau"].value)
            t2_arr[idx] = t2
            t2_stderr_arr[idx] = float(fit_result.params["tau"].stderr or np.nan)
            amp_arr[idx] = float(fit_result.params["a"].value)
            offset_arr[idx] = float(fit_result.params["c"].value)
            success_arr[idx] = bool(fit_result.success) and np.isfinite(t2) and (0 < t2 < 10 * t_span)
            best_fit[idx, :] = np.asarray(fit_result.best_fit, dtype=float)
            
        return {
            "flux_amp": flux_amps.tolist(),
            "t2_echo": t2_arr.tolist(),
            "t2_echo_stderr": t2_stderr_arr.tolist(),
            "amplitude": amp_arr.tolist(),
            "offset": offset_arr.tolist(),
            "success": bool(np.any(success_arr)),
            "best_fit": best_fit,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k != "best_fit"}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        return xr.Dataset(
            {
                "signal": (("flux_amp", "wait_time"), np.asarray(dataset["signal"].values, dtype=float)),
                "best_fit": (("flux_amp", "wait_time"), np.asarray(results["best_fit"], dtype=float)),
                "t2_echo": ("flux_amp", np.asarray(results["t2_echo"], dtype=float)),
            },
            coords={
                "flux_amp": np.asarray(dataset["flux_amp"].values, dtype=float),
                "wait_time": np.asarray(dataset["wait_time"].values, dtype=float),
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
        return {"qubit_echo_flux": plot_echo_flux(plot_data)}
