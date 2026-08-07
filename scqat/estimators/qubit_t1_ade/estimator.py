"""T1-vs-lab-time tracking via Analytical Decay Estimation (ADE).

Reference: arXiv:2602.11912. Each block measures P(|1>) at the three delays
t0 / t0+dt / t0+3*dt and the FPGA reports the closed-form decay rate plus its
analytic shot-noise sigma (:mod:`scqat.tools.ade_decay` holds the math and the
validity domain). This estimator converts the rate trace to T1, recomputes the
closed form offline from the raw shots when they were streamed — flagging the
blocks the fixed-point implementation had to clip — and optionally bootstraps
an independent sigma from the same shots.

Expected xarray.Dataset contract (``target`` dimension already removed)
-----------------------------------------------------------------------
Coordinates:
    - block_idx : 1-D int — one T1 estimation block per entry.
Data variables:
    - estimated_gamma : (block_idx,) — FPGA decay-rate estimate, 1/s.
    - sigma_gamma     : (block_idx,) — FPGA analytic sigma of gamma, 1/s.
    - dt_s            : (block_idx,) — delay spacing dt used, seconds
                        (varies per block when the probe adapted it).
Optional variables:
    - state        : (block_idx, delay_idx, shot_idx) — per-shot integer
                     levels at the three delays (delay_idx order t0, t0+dt,
                     t0+3dt; level > 0 counts as excited). Enables the offline
                     recompute, the clip mask and the bootstrap sigma.
    - block_time_s : (block_idx,) — elapsed lab time of each block, seconds
                     (hardware timestamps). Absent -> figures use the index.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qubit_t1_ade.visualization import plot_dt_trace, plot_t1_trace
from scqat.tools.ade_decay import ade_bootstrap_sigma_t1, ade_gamma, ade_sigma_gamma

#: relative gamma deviation (FPGA vs offline float recompute of the SAME shots)
#: above which a block counts as a fixed-point mismatch. Rounding in the QUA
#: ``fixed`` [-8, 8) pipeline sits far below this; a mismatch means a clip
#: floor engaged (or the wrong shots were streamed).
_FPGA_MISMATCH_REL = 0.05


class QubitT1AdeEstimator(BaseEstimator):
    """T1 trace + analytic/bootstrap sigmas from the 3-delay ADE streams."""

    estimator_name = "qubit_t1_ade"

    def _check_data(self, dataset: xr.Dataset) -> None:
        for var in ("estimated_gamma", "sigma_gamma", "dt_s"):
            if var not in dataset.data_vars:
                raise ValueError(
                    f"ADE analysis requires a '{var}' data variable "
                    "(the per-block FPGA stream)."
                )
        if "block_idx" not in dataset.coords:
            raise ValueError("ADE analysis requires a 'block_idx' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Convert the rate trace to T1 and cross-check it against the shots.

        Kwargs — flat and fully owned; unknown names raise:
            n_bootstrap (int): bootstrap resamples per block (default 0 = off);
                needs the per-shot ``state`` variable.
            bootstrap_seed (int): RNG seed for the bootstrap (default 0).
        """
        n_bootstrap = int(kwargs.pop("n_bootstrap", 0))
        bootstrap_seed = int(kwargs.pop("bootstrap_seed", 0))
        if kwargs:
            raise ValueError(
                f"Unknown qubit_t1_ade kwarg(s) {sorted(kwargs)}; "
                "valid: ['n_bootstrap', 'bootstrap_seed']"
            )

        gamma = np.asarray(
            dataset["estimated_gamma"].transpose("block_idx").values, dtype=float
        )
        sigma_gamma = np.asarray(
            dataset["sigma_gamma"].transpose("block_idx").values, dtype=float
        )
        dt_s = np.asarray(dataset["dt_s"].transpose("block_idx").values, dtype=float)
        n_blocks = gamma.size

        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.isfinite(gamma) & (gamma > 0)
            t1_s = np.where(positive, 1.0 / gamma, np.nan)
            t1_sigma_s = np.where(positive, sigma_gamma / gamma**2, np.nan)

        results: Dict[str, Any] = {
            "n_blocks": int(n_blocks),
            "n_bootstrap": int(n_bootstrap),
            "has_shots": False,
            "has_lab_time": False,
        }

        # Offline recompute from the raw shots: exact float math on the same
        # data. Blocks OUTSIDE the ADE validity domain are the ones the FPGA
        # clip floors silently turned into plausible-looking numbers — mask
        # them out of every statistic.
        clipped = np.zeros(n_blocks, dtype=bool)
        gamma_offline = np.full(n_blocks, np.nan)
        t1_boot_sigma_s = np.full(n_blocks, np.nan)
        n_fpga_mismatch = 0
        if "state" in dataset.data_vars:
            results["has_shots"] = True
            state = dataset["state"].transpose("block_idx", "delay_idx", "shot_idx")
            excited = (np.asarray(state.values) > 0).astype(float)
            n_avg = excited.shape[2]
            pops = excited.mean(axis=2)
            gamma_offline, valid = ade_gamma(pops[:, 0], pops[:, 1], pops[:, 2], dt_s)
            clipped = ~valid
            sigma_offline = ade_sigma_gamma(
                pops[:, 0], pops[:, 1], pops[:, 2], dt_s, n_avg
            )
            results["sigma_gamma_offline_median"] = float(
                np.nanmedian(sigma_offline)
            ) if np.any(np.isfinite(sigma_offline)) else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(gamma - gamma_offline) / np.abs(gamma_offline)
            n_fpga_mismatch = int(np.sum(valid & (rel > _FPGA_MISMATCH_REL)))
            if n_bootstrap > 0:
                t1_boot_sigma_s = ade_bootstrap_sigma_t1(
                    excited[:, 0], excited[:, 1], excited[:, 2], dt_s,
                    n_bootstrap, seed=bootstrap_seed,
                )

        lab_time_s = np.full(n_blocks, np.nan)
        if "block_time_s" in dataset.data_vars:
            lab_time_s = np.asarray(
                dataset["block_time_s"].transpose("block_idx").values, dtype=float
            )
            results["has_lab_time"] = bool(np.any(np.isfinite(lab_time_s)))

        valid_t1 = np.isfinite(t1_s) & ~clipped
        n_valid = int(np.sum(valid_t1))
        results.update(
            t1_s=t1_s,
            t1_sigma_s=t1_sigma_s,
            t1_boot_sigma_s=t1_boot_sigma_s,
            gamma_fpga=gamma,
            gamma_offline=gamma_offline,
            sigma_gamma_fpga=sigma_gamma,
            dt_s=dt_s,
            clipped=clipped,
            lab_time_s=lab_time_s,
            n_valid=n_valid,
            n_clipped=int(np.sum(clipped)),
            n_fpga_mismatch=n_fpga_mismatch,
            t1_median_s=float(np.nanmedian(t1_s[valid_t1])) if n_valid else float("nan"),
            t1_sigma_median_s=float(np.nanmedian(t1_sigma_s[valid_t1])) if n_valid else float("nan"),
            t1_boot_sigma_median_s=(
                float(np.nanmedian(t1_boot_sigma_s))
                if np.any(np.isfinite(t1_boot_sigma_s)) else float("nan")
            ),
            # two finite unclipped estimates make a trace; below that the run
            # produced no usable tracking data
            success=bool(n_valid >= 2),
        )
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the summary scalars; the traces live in plot_data."""
        drop = {"t1_s", "t1_sigma_s", "t1_boot_sigma_s", "gamma_fpga",
                "gamma_offline", "sigma_gamma_fpga", "dt_s", "clipped",
                "lab_time_s"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        n_blocks = int(results["n_blocks"])
        data_vars = {
            "t1_s": ("block_idx", np.asarray(results["t1_s"], dtype=float)),
            "t1_sigma_s": ("block_idx", np.asarray(results["t1_sigma_s"], dtype=float)),
            "t1_boot_sigma_s": ("block_idx",
                                np.asarray(results["t1_boot_sigma_s"], dtype=float)),
            "dt_s": ("block_idx", np.asarray(results["dt_s"], dtype=float)),
            "clipped": ("block_idx",
                        np.asarray(results["clipped"], dtype=np.int8)),
            "gamma_fpga": ("block_idx",
                           np.asarray(results["gamma_fpga"], dtype=float)),
            "gamma_offline": ("block_idx",
                              np.asarray(results["gamma_offline"], dtype=float)),
        }
        coords = {
            "block_idx": np.arange(n_blocks),
            "lab_time_s": ("block_idx",
                           np.asarray(results["lab_time_s"], dtype=float)),
        }
        attrs = {
            k: results[k]
            for k in ("t1_median_s", "t1_sigma_median_s", "t1_boot_sigma_median_s",
                      "n_blocks", "n_valid", "n_clipped", "n_fpga_mismatch",
                      "n_bootstrap")
            if k in results
        }
        attrs["success"] = int(bool(results["success"]))
        attrs["has_shots"] = int(bool(results["has_shots"]))
        attrs["has_lab_time"] = int(bool(results["has_lab_time"]))
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """The T1 trace with its sigma bands, plus the dt trace. Draws only
        from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        pd = plot_data
        builders = {
            # single-figure idiom: key == estimator_name -> qubit_t1_ade.png
            "qubit_t1_ade": lambda: plot_t1_trace(pd),
            "dt_trace": lambda: plot_dt_trace(pd),
        }
        return render_figures(builders, label=self.estimator_name)
