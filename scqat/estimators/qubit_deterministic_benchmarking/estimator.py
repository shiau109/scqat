"""Deterministic Benchmarking Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_deterministic_benchmarking.visualization import plot_deterministic_benchmarking


def damped_cosine_zero_phase(n, A, gamma, omega, C):
    return A * np.exp(-gamma * n) * np.cos(omega * n) + C


class QubitDeterministicBenchmarkingEstimator(BaseEstimator):
    """Estimate pulse amplitude scaling factor using zero-phase damped cosine fit."""

    estimator_name = "qubit_deterministic_benchmarking"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "repetition" not in dataset.coords:
            raise ValueError("Deterministic benchmarking estimator requires a 'repetition' coordinate")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        reps = np.asarray(dataset["repetition"].values, dtype=float)
        if "amp_factor" in dataset.coords:
            amp_factors = np.asarray(dataset["amp_factor"].values, dtype=float)
        else:
            amp_factors = np.array([1.0])

        if "state" in dataset.data_vars:
            raw_data = dataset["state"].values
            unit_str = "P0"

            def to_pz(arr):
                m = np.mean(arr)
                if m < 0.0 or m > 1.0:
                    return np.clip(1.0 - arr, 0.0, 1.0)
                return np.clip(arr, 0.0, 1.0)
        else:
            var_name = "I" if "I" in dataset.data_vars else list(dataset.data_vars.keys())[0]
            raw_data = dataset[var_name].values
            unit_str = var_name

            def to_pz(arr):
                d_min, d_max = np.min(arr), np.max(arr)
                return (arr - d_min) / (d_max - d_min) if d_max > d_min else np.full_like(arr, 0.5)

        if raw_data.ndim == 1:
            pz_data = to_pz(raw_data)[np.newaxis, :]
        else:
            pz_data = np.array([to_pz(raw_data[i]) for i in range(len(amp_factors))])

        num_amps = len(amp_factors)
        omegas, signed_omegas, gammas, fit_curves = [], [], [], []
        reps_fine = np.linspace(reps.min(), reps.max(), 200)

        for i_a, a_val in enumerate(amp_factors):
            pz_curve = pz_data[i_a]
            p0_guess = [0.45, 0.01, 0.05, 0.5]
            try:
                popt, _ = curve_fit(
                    damped_cosine_zero_phase,
                    reps,
                    pz_curve,
                    p0=p0_guess,
                    bounds=([0.0, 0.0, 0.0, 0.0], [0.6, 0.5, np.pi, 1.0]),
                    maxfev=2000,
                )
                omega_fit, gamma_fit = popt[2], popt[1]
                curve_fine = damped_cosine_zero_phase(reps_fine, *popt)

                w_val = float(omega_fit)
                s_w = w_val if a_val >= 1.0 else -w_val
                omegas.append(w_val)
                signed_omegas.append(s_w)
                gammas.append(float(gamma_fit))
                fit_curves.append([float(x) for x in curve_fine])
            except Exception:
                omegas.append(0.0)
                signed_omegas.append(0.0)
                gammas.append(0.0)
                fit_curves.append([0.5] * len(reps_fine))

        if num_amps > 1 and len(set(amp_factors)) > 1:
            try:
                poly = np.polyfit(amp_factors, signed_omegas, 1)
                k_slope, b_intercept = poly[0], poly[1]
                a_opt = float(-b_intercept / k_slope) if abs(k_slope) > 1e-6 else float(amp_factors[np.argmin(omegas)])
            except Exception:
                a_opt = 1.0
        else:
            a_opt = 1.0

        a_opt = float(np.clip(a_opt, 0.5, 1.5))

        return {
            "opt_factor": a_opt,
            "amp_factors": amp_factors.tolist(),
            "repetitions": reps.tolist(),
            "reps_fine": reps_fine.tolist(),
            "omegas": omegas,
            "gammas": gammas,
            "pz_data": pz_data.tolist(),
            "fit_curves": fit_curves,
            "unit": unit_str,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "opt_factor": results["opt_factor"],
            "unit": results["unit"],
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        amp_factors = np.asarray(results["amp_factors"], dtype=float)
        reps_fine = np.asarray(results["reps_fine"], dtype=float)
        reps = np.asarray(results["repetitions"], dtype=float)

        pz_data = np.asarray(results["pz_data"], dtype=float)
        fit_curves = np.asarray(results["fit_curves"], dtype=float)

        ds = xr.Dataset(
            {
                "pz": (("amp_factor", "repetition"), pz_data),
                "fit_pz": (("amp_factor", "repetition_fine"), fit_curves),
                "omega": ("amp_factor", np.asarray(results["omegas"], dtype=float)),
            },
            coords={
                "amp_factor": amp_factors,
                "repetition": reps,
                "repetition_fine": reps_fine,
            },
            attrs={"unit": results["unit"], "opt_factor": results["opt_factor"]},
        )

        if len(amp_factors) > 1:
            a_opt = results["opt_factor"]
            a_fine = np.linspace(min(amp_factors.min(), a_opt - 0.02), max(amp_factors.max(), a_opt + 0.02), 100)
            try:
                signed_omegas = [w if a >= 1.0 else -w for w, a in zip(results["omegas"], amp_factors)]
                poly = np.polyfit(amp_factors, signed_omegas, 1)
                fit_w = np.abs(poly[0] * a_fine + poly[1])
                ds["fit_omega_fine"] = ("amp_factor_fine", fit_w)
                ds.coords["amp_factor_fine"] = a_fine
            except Exception:
                pass

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
        return {"qubit_deterministic_benchmarking": plot_deterministic_benchmarking(plot_data)}
