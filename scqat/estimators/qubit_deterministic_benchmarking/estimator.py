"""Deterministic Benchmarking Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, reduced_signal
from scqat.tools.iq_reduce import AXIAL_KNOBS, validate_iq_reduce_kwargs
from scqat.estimators.qubit_deterministic_benchmarking.visualization import plot_deterministic_benchmarking


def damped_cosine_zero_phase(n, A, gamma, omega, C):
    return A * np.exp(-gamma * n) * np.cos(omega * n) + C


def _oriented_unit(trace: np.ndarray) -> np.ndarray:
    """One trace rescaled to [0, 1] and oriented so N=0 sits at the MAXIMUM.

    The sequence starts in |0>, so the zero-repetition point is an extremum by
    construction. :func:`damped_cosine_zero_phase` fixes ``phi = 0`` and the fit
    bounds ``A >= 0``, so the model can only describe a trace that starts HIGH —
    orienting here is what stops the fit collapsing onto the flat ``A ~ 0`` bound.
    (``PowerRabiEstimator`` hits the same ``a >= 0`` trap and solves it by fitting
    both phase seeds; here the physics pins the phase, so orientation is the fix.)
    """
    lo, hi = float(np.min(trace)), float(np.max(trace))
    span = hi - lo
    if span <= 1e-12:
        return np.full_like(trace, 0.5, dtype=float)
    unit = (trace - lo) / span
    return unit if unit[0] >= 0.5 else 1.0 - unit


class QubitDeterministicBenchmarkingEstimator(BaseEstimator):
    """Estimate pulse amplitude scaling factor using zero-phase damped cosine fit.

    Expects an xarray.Dataset with:
        - Variables: complex ``IQdata`` (or both ``I`` and ``Q``) — reduced to the
          signed axial projection onto the |0>-|1> axis — OR a pre-reduced real
          ``signal`` (an already-discriminated state/population).
        - Coordinate: ``repetition`` (gate repetition count N), optionally
          ``amp_factor`` (amplitude scaling sweep).
    """

    estimator_name = "qubit_deterministic_benchmarking"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "repetition" not in dataset.coords:
            raise ValueError("Deterministic benchmarking estimator requires a 'repetition' coordinate")
        has_iq = "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Deterministic benchmarking analysis requires a 'signal' variable, or "
                "complex 'IQdata', or both 'I' and 'Q'."
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Fit the repetition trace at each amplitude factor.

        Kwargs — flat and fully owned; unknown names raise:
            angle, positions, pca_sign
                IQ->1-D axial-reduction knobs (see :func:`scqat.tools.iq_reduce.axial`);
                ignored when the dataset already carries a real ``signal``.
        """
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)
        reps = np.asarray(dataset["repetition"].values, dtype=float)

        # Raw I is NOT the signal: the readout blobs sit at an arbitrary rotation in
        # the IQ plane, so reducing onto the |0>-|1> axis is what makes the trace mean
        # "population" at all. `reduced_signal` passes a pre-reduced `signal` through
        # untouched and axially reduces I/Q otherwise.
        #
        # It is 1-D by construction (it takes `iq.dims[0]` as THE sweep axis), and this
        # is the first estimator with a 2-D sweep — so reduce ONE repetition trace at a
        # time instead of widening shared core. `_oriented_unit` rescales each trace
        # independently anyway, so a per-trace reduction axis costs nothing here.
        if "amp_factor" in dataset.dims:
            amp_factors = np.asarray(dataset["amp_factor"].values, dtype=float).reshape(-1)
            slices = [dataset.isel(amp_factor=j) for j in range(amp_factors.size)]
        else:
            # amp_factor absent, or already squeezed to a scalar coord
            amp_factors = (
                np.atleast_1d(np.asarray(dataset["amp_factor"].values, dtype=float))
                if "amp_factor" in dataset.coords
                else np.array([1.0])
            )
            slices = [dataset]

        unit_str = "P0"
        pz_data = np.array([
            _oriented_unit(np.asarray(reduced_signal(s, **kwargs).values, dtype=float))
            for s in slices
        ])

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
