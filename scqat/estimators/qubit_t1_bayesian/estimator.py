"""Adaptive Bayesian T1 tracking (Berritta et al., arXiv:2506.09576).

Each block runs N adaptive single-shot probes: the wait is tau = c * T1_est
from the current posterior, and an on-FPGA method-of-moments update tracks the
posterior over the relaxation rate Gamma1 ~ Gamma(shape=k, rate=theta) in the
u = 1/k parametrization (k itself never materializes in the QUA fixed-point
range). The streams carry the final (T1, u) per block; this estimator
reconstructs k, draws the posterior credible interval, characterizes the T1
trace's stability (Welch PSD + Allan deviation), and — when the probe
interleaved non-adaptive validation shots — fits the classical decay curve as
an independent cross-check.

Expected xarray.Dataset contract (``target`` dimension already removed)
-----------------------------------------------------------------------
Coordinates:
    - block_idx : 1-D int — one adaptive estimation block per entry.
Data variables:
    - estimated_t1_s : (block_idx,) — final posterior T1 per block, seconds.
    - u_final        : (block_idx,) — final u = 1/k per block (dimensionless).
Optional variables:
    - state        : (block_idx, probe_idx) — per-probe outcomes (level > 0
                     counts as excited).
    - tau_s        : (block_idx, probe_idx) — adaptive waits used, seconds.
    - state_lin    : (block_idx, probe_idx) — interleaved NON-adaptive
                     validation outcomes on the linear wait grid.
    - lin_wait_s   : (probe_idx,) — the validation wait grid, seconds.
    - u_evol       : (probe_idx,) — u before each probe of the LAST block.
    - t1_evol_s    : (probe_idx,) — T1 before each probe of the LAST block.
    - block_time_s : (block_idx,) — elapsed lab time per block, seconds.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import invgamma

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qubit_t1_bayesian.visualization import (
    plot_allan,
    plot_posterior_evolution,
    plot_psd,
    plot_t1_trace,
    plot_validation,
)
from scqat.tools.allan import overlapping_allan_deviation
from scqat.tools.fit_exp_decay import FitExponentialDecay
from scqat.tools.timeseries_psd import timeseries_psd, validate_timeseries_psd_kwargs

#: adaptive-vs-validation ratio beyond which the run is flagged: a prior far
#: from the truth pins the adaptive estimate at a rail while the interleaved
#: classical decay still reads the real T1 (the reference node warned at 3x).
_VALIDATION_RATIO_MAX = 3.0

#: minimum finite trace points for a meaningful fluctuation spectrum.
_MIN_TRACE_FOR_PSD = 8

#: T1-grid points of the posterior-evolution map.
_POSTERIOR_GRID = 400


class QubitT1BayesianEstimator(BaseEstimator):
    """T1 trace + posterior credible interval from the adaptive Bayes streams."""

    estimator_name = "qubit_t1_bayesian"

    def _check_data(self, dataset: xr.Dataset) -> None:
        for var in ("estimated_t1_s", "u_final"):
            if var not in dataset.data_vars:
                raise ValueError(
                    f"Bayesian T1 analysis requires a '{var}' data variable "
                    "(the per-block FPGA stream)."
                )
        if "block_idx" not in dataset.coords:
            raise ValueError(
                "Bayesian T1 analysis requires a 'block_idx' coordinate."
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Reconstruct k, the credible interval, and the trace statistics.

        Kwargs — flat and fully owned; unknown names raise:
            ci (float): credible-interval level for the posterior band
                (default 0.90).
            nperseg, window, detrend:
                PSD knobs (see :func:`scqat.tools.timeseries_psd.timeseries_psd`).
        """
        ci = float(kwargs.pop("ci", 0.90))
        if not 0.0 < ci < 1.0:
            raise ValueError(f"ci must be in (0, 1), got {ci!r}")
        validate_timeseries_psd_kwargs(kwargs)  # everything left is a PSD knob

        t1_s = np.asarray(
            dataset["estimated_t1_s"].transpose("block_idx").values, dtype=float
        )
        u = np.asarray(dataset["u_final"].transpose("block_idx").values, dtype=float)
        n_blocks = t1_s.size

        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(np.isfinite(u) & (u > 0), 1.0 / u, np.nan)

        # Posterior over Gamma1 is Gamma(shape=k, rate=theta) with
        # theta = k * T1, so T1 = 1/Gamma1 is inverse-gamma(shape=k,
        # scale=theta) — the credible interval comes from its ppf.
        t1_lo = np.full(n_blocks, np.nan)
        t1_hi = np.full(n_blocks, np.nan)
        post = np.isfinite(k) & (k > 0) & np.isfinite(t1_s) & (t1_s > 0)
        if np.any(post):
            theta = k[post] * t1_s[post]
            t1_lo[post] = invgamma.ppf((1.0 - ci) / 2.0, a=k[post], scale=theta)
            t1_hi[post] = invgamma.ppf(1.0 - (1.0 - ci) / 2.0, a=k[post], scale=theta)

        results: Dict[str, Any] = {
            "ci": ci,
            "n_blocks": int(n_blocks),
            "t1_s_trace": t1_s,
            "k_trace": k,
            "t1_ci_low_s": t1_lo,
            "t1_ci_high_s": t1_hi,
            "has_lab_time": False,
            "has_evolution": False,
            "has_validation": False,
        }

        finite = np.isfinite(t1_s)
        results["t1_median_s"] = float(np.nanmedian(t1_s)) if np.any(finite) else float("nan")
        results["k_final_median"] = float(np.nanmedian(k)) if np.any(np.isfinite(k)) else float("nan")
        results["success"] = bool(np.sum(finite) >= 2)

        # Lab-time axis + fluctuation statistics of the trace.
        lab_time_s = np.full(n_blocks, np.nan)
        if "block_time_s" in dataset.data_vars:
            lab_time_s = np.asarray(
                dataset["block_time_s"].transpose("block_idx").values, dtype=float
            )
            results["has_lab_time"] = bool(np.any(np.isfinite(lab_time_s)))
        results["lab_time_s"] = lab_time_s

        psd_freq = psd = np.array([])
        allan_tau = allan_dev = np.array([])
        psd_dt = float("nan")
        if results["has_lab_time"] and np.sum(finite) >= _MIN_TRACE_FOR_PSD:
            diffs = np.diff(lab_time_s[np.isfinite(lab_time_s)])
            if diffs.size and np.all(diffs > 0):
                psd_dt = float(np.median(diffs))
                # a few rail/NaN blocks must not kill the spectrum: fill them
                # with the finite median (diagnostic spectra, documented)
                trace = np.where(finite, t1_s, np.nanmedian(t1_s))
                psd_freq, psd = timeseries_psd(trace, psd_dt, **kwargs)
                allan_tau, allan_dev = overlapping_allan_deviation(trace, psd_dt)
        results.update(
            psd_freq_hz=psd_freq, psd=psd, psd_dt_s=psd_dt,
            allan_tau_s=allan_tau, allan_dev=allan_dev,
        )

        # Posterior sharpening over the LAST block, rendered as pdf maps.
        if "u_evol" in dataset.data_vars and "t1_evol_s" in dataset.data_vars:
            u_ev = np.asarray(
                dataset["u_evol"].transpose("probe_idx").values, dtype=float
            )
            t1_ev = np.asarray(
                dataset["t1_evol_s"].transpose("probe_idx").values, dtype=float
            )
            ok = np.isfinite(u_ev) & (u_ev > 0) & np.isfinite(t1_ev) & (t1_ev > 0)
            if np.any(ok):
                k_ev = np.where(ok, 1.0 / u_ev, np.nan)
                lo = 0.25 * float(np.nanmin(t1_ev[ok]))
                hi = 2.0 * float(np.nanmax(t1_ev[ok]))
                grid = np.linspace(max(lo, 1e-9), hi, _POSTERIOR_GRID)
                pdf = np.full((t1_ev.size, grid.size), np.nan)
                for p in np.nonzero(ok)[0]:
                    pdf[p] = invgamma.pdf(grid, a=k_ev[p], scale=k_ev[p] * t1_ev[p])
                results.update(
                    has_evolution=True,
                    t1_evol_s=t1_ev, k_evol=k_ev,
                    t1_grid_s=grid, posterior_pdf=pdf,
                )

        # Interleaved non-adaptive validation: classical decay, classical fit.
        if "state_lin" in dataset.data_vars and "lin_wait_s" in dataset.data_vars:
            state_lin = dataset["state_lin"].transpose("block_idx", "probe_idx")
            p_lin = (np.asarray(state_lin.values) > 0).astype(float).mean(axis=0)
            lin_wait = np.asarray(
                dataset["lin_wait_s"].transpose("probe_idx").values, dtype=float
            )
            results.update(has_validation=True, p_lin=p_lin, lin_wait_s=lin_wait)
            t1_lin = float("nan")
            best_fit = np.full(p_lin.size, np.nan)
            try:
                fit = FitExponentialDecay(x=lin_wait, data=p_lin).fit()
                if bool(fit.success):
                    t1_lin = float(fit.params["tau"].value)
                    best_fit = np.asarray(fit.best_fit, dtype=float)
            except Exception:
                pass  # validation is a cross-check; its failure must not sink the trace
            results["t1_lin_s"] = t1_lin
            results["lin_best_fit"] = best_fit
            ratio = float("nan")
            if np.isfinite(t1_lin) and t1_lin > 0 and np.isfinite(results["t1_median_s"]):
                ratio = results["t1_median_s"] / t1_lin
            results["t1_lin_ratio"] = ratio
            # a big gap means the PRIOR was wrong, not the chip — rerun with
            # t1_prior near the validation fit
            results["validation_disagrees"] = bool(
                np.isfinite(ratio)
                and not (1.0 / _VALIDATION_RATIO_MAX < ratio < _VALIDATION_RATIO_MAX)
            )
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the summary scalars; the traces/maps live in plot_data."""
        drop = {"t1_s_trace", "k_trace", "t1_ci_low_s", "t1_ci_high_s",
                "lab_time_s", "psd_freq_hz", "psd", "allan_tau_s", "allan_dev",
                "t1_evol_s", "k_evol", "t1_grid_s", "posterior_pdf",
                "p_lin", "lin_wait_s", "lin_best_fit"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        n_blocks = int(results["n_blocks"])
        data_vars: Dict[str, Any] = {
            "t1_s": ("block_idx", np.asarray(results["t1_s_trace"], dtype=float)),
            "k": ("block_idx", np.asarray(results["k_trace"], dtype=float)),
            "t1_ci_low_s": ("block_idx",
                            np.asarray(results["t1_ci_low_s"], dtype=float)),
            "t1_ci_high_s": ("block_idx",
                             np.asarray(results["t1_ci_high_s"], dtype=float)),
        }
        coords: Dict[str, Any] = {
            "block_idx": np.arange(n_blocks),
            "lab_time_s": ("block_idx",
                           np.asarray(results["lab_time_s"], dtype=float)),
        }
        if np.asarray(results["psd"]).size:
            coords["psd_freq_hz"] = np.asarray(results["psd_freq_hz"], dtype=float)
            data_vars["psd"] = ("psd_freq_hz", np.asarray(results["psd"], dtype=float))
        if np.asarray(results["allan_dev"]).size:
            coords["allan_tau_s"] = np.asarray(results["allan_tau_s"], dtype=float)
            data_vars["allan_dev"] = ("allan_tau_s",
                                      np.asarray(results["allan_dev"], dtype=float))
        if results.get("has_evolution"):
            coords["evol_probe_idx"] = np.arange(
                np.asarray(results["t1_evol_s"]).size
            )
            coords["t1_grid_s"] = np.asarray(results["t1_grid_s"], dtype=float)
            data_vars["t1_evol_s"] = ("evol_probe_idx",
                                      np.asarray(results["t1_evol_s"], dtype=float))
            data_vars["k_evol"] = ("evol_probe_idx",
                                   np.asarray(results["k_evol"], dtype=float))
            data_vars["posterior_pdf"] = (("evol_probe_idx", "t1_grid_s"),
                                          np.asarray(results["posterior_pdf"],
                                                     dtype=float))
        if results.get("has_validation"):
            coords["lin_wait_s"] = np.asarray(results["lin_wait_s"], dtype=float)
            data_vars["p_lin"] = ("lin_wait_s",
                                  np.asarray(results["p_lin"], dtype=float))
            data_vars["lin_best_fit"] = ("lin_wait_s",
                                         np.asarray(results["lin_best_fit"],
                                                    dtype=float))

        attrs = {
            k: results[k]
            for k in ("t1_median_s", "k_final_median", "ci", "n_blocks",
                      "psd_dt_s", "t1_lin_s", "t1_lin_ratio")
            if k in results
        }
        for flag in ("success", "has_lab_time", "has_evolution",
                     "has_validation", "validation_disagrees"):
            if flag in results:
                attrs[flag] = int(bool(results[flag]))
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Trace + credible band, posterior-evolution map, fluctuation PSD,
        Allan deviation and the interleaved validation fit. Draws only from
        ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        pd = plot_data
        builders = {
            # single-figure idiom: key == estimator_name -> qubit_t1_bayesian.png
            "qubit_t1_bayesian": lambda: plot_t1_trace(pd),
        }
        if "posterior_pdf" in pd.data_vars:
            builders["posterior_evolution"] = lambda: plot_posterior_evolution(pd)
        if "psd" in pd.data_vars:
            builders["psd"] = lambda: plot_psd(pd)
        if "allan_dev" in pd.data_vars:
            builders["allan"] = lambda: plot_allan(pd)
        if "p_lin" in pd.data_vars:
            builders["validation"] = lambda: plot_validation(pd)
        return render_figures(builders, label=self.estimator_name)
