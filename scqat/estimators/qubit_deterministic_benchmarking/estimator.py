"""Deterministic Benchmarking Estimator."""

from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, reduced_signal
from scqat.tools.iq_reduce import AXIAL_KNOBS, validate_iq_reduce_kwargs
from scqat.estimators._twin_axis import twin_at, twin_values
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
          ``amp_prefactor`` (amplitude scaling sweep).
    """

    estimator_name = "qubit_deterministic_benchmarking"

    #: optional companion scale for the amplitude axis — a coordinate over the same
    #: points plus its label (see :mod:`scqat.estimators._twin_axis`).
    twin_coord: Optional[str] = None
    twin_label: Optional[str] = None

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
            twin_coord, twin_label
                optional companion scale for the amplitude axis (see
                :mod:`scqat.estimators._twin_axis`). The single-amplitude mode has no
                axis to draw, so it is simply not drawn there.
        """
        # popped BEFORE the reduction check: these are this estimator's own knobs, and
        # AXIAL_KNOBS is shared by five families
        twin_coord = kwargs.pop("twin_coord", self.twin_coord)
        twin_label = kwargs.pop("twin_label", self.twin_label)
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
        if "amp_prefactor" in dataset.dims:
            amp_prefactors = np.asarray(dataset["amp_prefactor"].values, dtype=float).reshape(-1)
            slices = [dataset.isel(amp_prefactor=j) for j in range(amp_prefactors.size)]
        else:
            # amp_prefactor absent, or already squeezed to a scalar coord
            amp_prefactors = (
                np.atleast_1d(np.asarray(dataset["amp_prefactor"].values, dtype=float))
                if "amp_prefactor" in dataset.coords
                else np.array([1.0])
            )
            slices = [dataset]

        unit_str = "P0"
        pz_data = np.array([
            _oriented_unit(np.asarray(reduced_signal(s, **kwargs).values, dtype=float))
            for s in slices
        ])

        num_amps = len(amp_prefactors)
        omegas, signed_omegas, gammas, fit_curves = [], [], [], []
        reps_fine = np.linspace(reps.min(), reps.max(), 200)

        max_omega_bound = 0.45
        for i_a, a_val in enumerate(amp_prefactors):
            pz_curve = pz_data[i_a]

            # Extract dominant frequency candidate via FFT
            n_pts = len(reps)
            dt = float(np.mean(np.diff(reps))) if n_pts > 1 else 1.0
            fft_vals = np.abs(np.fft.rfft(pz_curve - np.mean(pz_curve)))
            freqs = np.fft.rfftfreq(n_pts, d=dt)
            valid_mask = (freqs > 0) & (freqs * 2 * np.pi <= max_omega_bound)
            w_fft = float(2 * np.pi * freqs[valid_mask][np.argmax(fft_vals[valid_mask])]) if np.any(valid_mask) else 0.05

            w_candidates = sorted(list(set([w_fft, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2])))
            best_popt = None
            best_cost = float("inf")

            for w_g in w_candidates:
                if w_g > max_omega_bound:
                    continue
                p0_guess = [0.45, 0.01, w_g, 0.5]
                try:
                    popt, _ = curve_fit(
                        damped_cosine_zero_phase,
                        reps,
                        pz_curve,
                        p0=p0_guess,
                        bounds=([0.0, 0.0, 0.0, 0.0], [0.6, 0.5, max_omega_bound, 1.0]),
                        maxfev=2000,
                    )
                    pred = damped_cosine_zero_phase(reps, *popt)
                    cost = np.sum((pz_curve - pred) ** 2)
                    if cost < best_cost:
                        best_cost = cost
                        best_popt = popt
                except Exception:
                    continue

            if best_popt is not None:
                omega_fit, gamma_fit = best_popt[2], best_popt[1]
                curve_fine = damped_cosine_zero_phase(reps_fine, *best_popt)

                w_val = float(omega_fit)
                s_w = w_val if a_val >= 1.0 else -w_val
                omegas.append(w_val)
                signed_omegas.append(s_w)
                gammas.append(float(gamma_fit))
                fit_curves.append([float(x) for x in curve_fine])
            else:
                omegas.append(0.0)
                signed_omegas.append(0.0)
                gammas.append(0.0)
                fit_curves.append([0.5] * len(reps_fine))

        if num_amps > 1 and len(set(amp_prefactors)) > 1:
            try:
                poly = np.polyfit(amp_prefactors, signed_omegas, 1)
                k_slope, b_intercept = poly[0], poly[1]
                a_opt = float(-b_intercept / k_slope) if abs(k_slope) > 1e-6 else float(amp_prefactors[np.argmin(omegas)])
            except Exception:
                a_opt = 1.0
        else:
            a_opt = 1.0

        a_opt = float(np.clip(a_opt, 0.5, 1.5))

        results = {
            "opt_factor": a_opt,
            "amp_prefactors": amp_prefactors.tolist(),
            "repetitions": reps.tolist(),
            "reps_fine": reps_fine.tolist(),
            "omegas": omegas,
            "gammas": gammas,
            "pz_data": pz_data.tolist(),
            "fit_curves": fit_curves,
            "unit": unit_str,
        }
        # the optional companion scale — absent entirely when undrawable, which
        # includes the single-amplitude mode (one point is not an axis)
        twin = twin_values(dataset, "amp_prefactor", twin_coord)
        if twin is not None:
            results["twin_values"] = twin
            results["twin_label"] = str(twin_label or twin_coord)
            results["opt_twin_value"] = twin_at(amp_prefactors, twin, a_opt)
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        metadata = {
            "opt_factor": results["opt_factor"],
            "unit": results["unit"],
        }
        for key in ("twin_label", "opt_twin_value"):
            if key in results:
                metadata[key] = results[key]
        return metadata

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        amp_prefactors = np.asarray(results["amp_prefactors"], dtype=float)
        reps_fine = np.asarray(results["reps_fine"], dtype=float)
        reps = np.asarray(results["repetitions"], dtype=float)

        pz_data = np.asarray(results["pz_data"], dtype=float)
        fit_curves = np.asarray(results["fit_curves"], dtype=float)

        ds = xr.Dataset(
            {
                "pz": (("amp_prefactor", "repetition"), pz_data),
                "fit_pz": (("amp_prefactor", "repetition_fine"), fit_curves),
                "omega": ("amp_prefactor", np.asarray(results["omegas"], dtype=float)),
            },
            coords={
                "amp_prefactor": amp_prefactors,
                "repetition": reps,
                "repetition_fine": reps_fine,
            },
            attrs={"unit": results["unit"], "opt_factor": results["opt_factor"]},
        )

        # the companion scale + its label, so the figure draws the secondary axis
        # from plot_data ALONE (the self-enforcing rule)
        if results.get("twin_values") is not None:
            ds["twin"] = ("amp_prefactor", np.asarray(results["twin_values"], dtype=float))
            ds.attrs["twin_label"] = str(results.get("twin_label", ""))
            ds.attrs["opt_twin_value"] = float(
                results.get("opt_twin_value", float("nan")))

        if len(amp_prefactors) > 1:
            a_opt = results["opt_factor"]
            a_fine = np.linspace(min(amp_prefactors.min(), a_opt - 0.02), max(amp_prefactors.max(), a_opt + 0.02), 100)
            try:
                signed_omegas = [w if a >= 1.0 else -w for w, a in zip(results["omegas"], amp_prefactors)]
                poly = np.polyfit(amp_prefactors, signed_omegas, 1)
                fit_w = np.abs(poly[0] * a_fine + poly[1])
                ds["fit_omega_fine"] = ("amp_prefactor_fine", fit_w)
                ds.coords["amp_prefactor_fine"] = a_fine
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
