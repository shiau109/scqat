from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_cosine import FitCosine
from scqat.estimators.swap_oscillation.visualization import plot_rounds_fit


class SwapOscillationEstimator(BaseEstimator):
    """
    Analyzes a swap-chain (N-swap) sweep to extract the swap-oscillation frequency.

    Expects an xarray.Dataset with:
        - Variable: 'signal'
        - Coordinate: 'round'  (integer number of swaps applied; N=0 — the no-swap
          baseline — is allowed)

    A coherent (partial) swap exchanges population between the pair swap by swap, so
    the measured population follows a non-decaying cosine in N. Fits
    ``a*cos(2*pi*f*x + phi) + c`` (via :class:`FitCosine`) and reports ``f`` — the
    oscillation frequency in cycles per swap — and ``swap_period = 1/f`` — the number
    of swaps per full population cycle.

    Note on the full-swap limit: a *full* iSWAP oscillates at exactly the Nyquist
    limit of the integer-N sweep (f = 0.5 for a step of 1), where ``a`` and ``phi``
    are individually degenerate (``cos(pi*N + phi) = cos(pi*N)*cos(phi)`` at integer
    N); ``f`` itself is still well determined.
    """

    estimator_name = "swap_oscillation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset:
            raise ValueError("Swap-oscillation analysis requires a 'signal' variable in the dataset.")
        if "round" not in dataset.coords:
            raise ValueError("Swap-oscillation analysis requires a 'round' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the swap oscillation and extract its frequency (cycles per swap).

        Returns a dict with:
            a, f, phi, c, swap_period, r_squared, success, best_fit,
            round_dense, best_fit_dense, fit_report.
        """
        # Prepare a DataArray with an 'x' coordinate for the FitCosine fitter.
        fit_data = dataset["signal"].rename({"round": "x"}).squeeze()

        # FitCosine bounds the amplitude a >= 0, so a single phi=0 seed can get trapped
        # at a flat (a~0) fit for a qubit whose population *rises* from zero (the swap
        # target, which needs phi ~ pi). Fit with both phase seeds and keep the
        # lower-residual result so the extraction is robust to which qubit is measured.
        fit_result = None
        for phi_seed in (0.0, np.pi):
            fitter = FitCosine(fit_data)
            fitter.guess()
            fitter.params["phi"].set(value=phi_seed)
            res = fitter.fit()
            if fit_result is None or res.chisqr < fit_result.chisqr:
                fit_result = res
        p = {k: v.value for k, v in fit_result.params.items()}  # a, f, phi, c

        rounds = np.asarray(dataset.coords["round"].values, dtype=float)
        nyquist = 0.5 / float(np.min(np.diff(rounds))) if rounds.size > 1 else float("nan")
        swap_period = 1.0 / p["f"] if p["f"] > 0 else float("nan")

        # Reject a degenerate (flat) fit: a real swap oscillation has a ~ half the
        # signal peak-to-peak, so guard against a ~ 0 falsely reporting success.
        ptp = float(np.ptp(np.asarray(fit_data.values, dtype=float)))
        contrast_ok = ptp > 0 and p["a"] > 0.05 * ptp

        # The contrast guard is relative to the signal's own peak-to-peak, so a
        # pure-noise (no-op swap macro) curve passes it trivially. Require the cosine
        # to also beat a constant model: R^2 of the fit vs the mean. Noise-fitting
        # gives R^2 ~ 0.2; a real oscillation gives ~ 0.99 — 0.5 separates cleanly
        # and is scale-free (works for population and raw-I signals alike).
        y = np.asarray(fit_data.values, dtype=float)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(fit_result.chisqr) / ss_tot if ss_tot > 0 else float("nan")
        quality_ok = np.isfinite(r_squared) and r_squared > 0.5

        success = bool(
            bool(fit_result.success)
            and np.isfinite(nyquist)
            and 0 < p["f"] <= nyquist
            and contrast_ok
            and quality_ok
        )

        # Dense fit curve for plotting: the sweep has only a handful of integer-N
        # points, so the best-fit sampled there draws as a jagged polyline. Evaluate
        # the fitted cosine on a fine grid to render a smooth line.
        round_dense = np.linspace(rounds.min(), rounds.max(), 501)
        best_fit_dense = np.asarray(fit_result.eval(x=round_dense), dtype=float)

        return {
            "a": p["a"],
            "f": p["f"],
            "phi": p["phi"],
            "c": p["c"],
            "swap_period": float(swap_period),
            "r_squared": float(r_squared),
            "success": success,
            "best_fit": fit_result.best_fit,
            "round_dense": round_dense,
            "best_fit_dense": best_fit_dense,
            "fit_report": fit_result.fit_report(),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters and swap period; drop the diagnostic arrays."""
        drop = {"best_fit", "round_dense", "best_fit_dense", "fit_report"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the raw signal + best-fit curve over ``round``; the fit parameters and
        ``swap_period`` live in ``.attrs`` so the figure needs no recomputation.
        """
        rounds = np.asarray(dataset.coords["round"].values, dtype=float)
        signal = np.asarray(dataset["signal"].squeeze().values, dtype=float)
        best_fit = np.asarray(results["best_fit"], dtype=float)

        attrs = {
            "a": float(results["a"]),
            "f": float(results["f"]),
            "phi": float(results["phi"]),
            "c": float(results["c"]),
            "swap_period": float(results["swap_period"]),
            "success": int(bool(results["success"])),
        }

        return xr.Dataset(
            {
                "signal": ("round", signal),
                "best_fit": ("round", best_fit),
                "best_fit_dense": ("round_dense", np.asarray(results["best_fit_dense"], dtype=float)),
            },
            coords={
                "round": rounds,
                "round_dense": np.asarray(results["round_dense"], dtype=float),
            },
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the rounds-fit plot, drawing strictly from ``plot_data`` so the
        figure stays reconstructable downstream; rebuild it only when called outside
        ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"rounds": plot_rounds_fit(plot_data)}
