from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_cosine import FitCosine
from scqat.estimators.power_rabi.visualization import plot_amplitude_fit


class PowerRabiEstimator(BaseEstimator):
    """
    Analyzes power-Rabi data to extract the drive-amplitude prefactor of a pi pulse.

    Expects an xarray.Dataset with:
        - Variable: 'signal'
        - Coordinate: 'amp_prefactor'  (dimensionless multiplier of the current pulse
          amplitude, i.e. the QUA ``amplitude_scale`` sweep)

    Fits a non-decaying cosine ``a*cos(2*pi*f*x + phi) + c`` (via :class:`FitCosine`)
    and reports ``opt_amp_prefactor`` — the prefactor of the first oscillation extremum,
    which is the multiplier on the **current** pulse amplitude that yields a pi pulse.
    The estimator stays device-agnostic: the absolute amplitude is applied by the caller
    (the QM node multiplies this prefactor by the operation's current amplitude).
    """

    estimator_name = "power_rabi"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "signal" not in dataset:
            raise ValueError("Power-Rabi analysis requires a 'signal' variable in the dataset.")
        if "amp_prefactor" not in dataset.coords:
            raise ValueError("Power-Rabi analysis requires an 'amp_prefactor' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the power-Rabi oscillation and extract the pi-pulse amplitude prefactor.

        Returns a dict with:
            a, f, phi, c, opt_amp_prefactor, success, best_fit, fit_report.
        """
        # Prepare a DataArray with an 'x' coordinate for the FitCosine fitter.
        fit_data = dataset["signal"].rename({"amp_prefactor": "x"}).squeeze()

        # FitCosine bounds the amplitude a >= 0, so a single phi=0 seed can get trapped
        # at a flat (a~0) fit for qubits whose readout *rises* from zero amplitude (which
        # needs phi ~ pi). Fit with both phase seeds and keep the lower-residual result so
        # the extraction is robust to the per-qubit readout sign.
        fit_result = None
        for phi_seed in (0.0, np.pi):
            fitter = FitCosine(fit_data)
            fitter.guess()
            fitter.params["phi"].set(value=phi_seed)
            res = fitter.fit()
            if fit_result is None or res.chisqr < fit_result.chisqr:
                fit_result = res
        p = {k: v.value for k, v in fit_result.params.items()}  # a, f, phi, c

        # pi pulse = the amplitude where the fitted readout deviates most from its
        # zero-amplitude (ground-state) value. The excited-state population rises
        # monotonically from 0 at x=0 to 1 at the pi pulse, so |signal - signal(0)| peaks
        # there — sign-independent (works for either readout sign) and correct even though
        # the sweep starts at an extremum.
        best_fit = np.asarray(fit_result.best_fit, dtype=float)
        amp = np.asarray(dataset.coords["amp_prefactor"].values, dtype=float)
        if p["f"] > 0 and best_fit.size == amp.size:
            opt_amp_prefactor = float(amp[int(np.argmax(np.abs(best_fit - best_fit[0])))])
        else:
            opt_amp_prefactor = float("nan")

        # Reject a degenerate (flat) fit: a real Rabi oscillation has a ~ half the signal
        # peak-to-peak, so guard against a ~ 0 falsely reporting success.
        ptp = float(np.ptp(np.asarray(fit_data.values, dtype=float)))
        contrast_ok = ptp > 0 and p["a"] > 0.05 * ptp
        success = bool(
            bool(fit_result.success)
            and np.isfinite(opt_amp_prefactor)
            and 0 < opt_amp_prefactor < 2
            and contrast_ok
        )

        return {
            "a": p["a"],
            "f": p["f"],
            "phi": p["phi"],
            "c": p["c"],
            "opt_amp_prefactor": float(opt_amp_prefactor),
            "success": success,
            "best_fit": fit_result.best_fit,
            "fit_report": fit_result.fit_report(),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters and optimal prefactor; drop the diagnostic arrays."""
        drop = {"best_fit", "fit_report"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the raw signal + best-fit curve over ``amp_prefactor``; the fit
        parameters and ``opt_amp_prefactor`` live in ``.attrs`` so the figure needs
        no recomputation.
        """
        amp_prefactor = np.asarray(dataset.coords["amp_prefactor"].values, dtype=float)
        signal = np.asarray(dataset["signal"].squeeze().values, dtype=float)
        best_fit = np.asarray(results["best_fit"], dtype=float)

        attrs = {
            "a": float(results["a"]),
            "f": float(results["f"]),
            "phi": float(results["phi"]),
            "c": float(results["c"]),
            "opt_amp_prefactor": float(results["opt_amp_prefactor"]),
            "success": int(bool(results["success"])),
        }

        return xr.Dataset(
            {
                "signal": ("amp_prefactor", signal),
                "best_fit": ("amp_prefactor", best_fit),
            },
            coords={"amp_prefactor": amp_prefactor},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the amplitude-fit plot, drawing strictly from ``plot_data`` so the
        figure stays reconstructable downstream; rebuild it only when called outside
        ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"amplitude": plot_amplitude_fit(plot_data)}
