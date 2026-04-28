"""
Qubit Decoherence Protocol
==========================
Fits time-resolved density-matrix elements (rho_11, rho_10) to extract
decoherence parameters Gamma (relaxation rate) and Lambda (spectral width)
from the non-Markovian amplitude-damping model:

    d = sqrt(Lambda * (Lambda - 2*Gamma))
    G(t) = exp(-Lambda*t/2) [cosh(d*t/2) + (Lambda/d) sinh(d*t/2)]

    rho_11(t) = |G(t)|^2  rho_11(0)
    rho_10(t) =  G(t)     rho_10(0)

Expected xarray.Dataset contract:
    Coordinates:
        - time : 1-D array of measurement times
    Data variables (at least one required):
        - rho_11 : 1-D array – excited-state population vs time
        - rho_10 : 1-D array – off-diagonal coherence vs time
"""

from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_qubit_decoherence import (
    FitQubitDecoherence,
    decoherence_G as _decoherence_G,
    rho11_model as _rho11_model,
    rho10_model as _rho10_model,
)


class QubitDecoherenceAnalyzer(BaseAnalyzer):
    """
    Fits qubit decoherence data (rho_11 and/or rho_10 vs time) to the
    non-Markovian amplitude-damping model parameterised by Gamma and Lambda.
    """

    protocol_name = "qubit_decoherence"

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if "time" not in dataset.coords:
            raise ValueError(
                "QubitDecoherenceAnalyzer requires a 'time' coordinate."
            )
        if "rho_11" not in dataset.data_vars and "rho_10" not in dataset.data_vars:
            raise ValueError(
                "Dataset must contain at least one of 'rho_11' or 'rho_10' "
                "data variables."
            )

    # ------------------------------------------------------------------
    # BaseAnalyzer interface
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit rho_11(t) and/or rho_10(t) to the decoherence model.

        Returns
        -------
        dict with a sub-dict for each fitted variable containing:
            Gamma, Gamma_err, Lambda, Lambda_err, d, rho_0, rho_0_err,
            fit_curve, residuals, regime
        """
        t = dataset.coords["time"].values.astype(float)
        results: Dict[str, Any] = {}

        for var_name in ("rho_11", "rho_10"):
            if var_name not in dataset.data_vars:
                continue

            y_data = dataset[var_name].values.astype(float)
            da = xr.DataArray(y_data, coords={"x": t}, dims="x")

            fitter = FitQubitDecoherence(da, component=var_name)
            result = fitter.fit()

            p = result.params
            Gamma_fit = float(p["Gamma"].value)
            Lambda_fit = float(p["Lambda"].value)
            rho0_fit = float(p["rho_0"].value)
            Gamma_err = float(p["Gamma"].stderr) if p["Gamma"].stderr is not None else float("nan")
            Lambda_err = float(p["Lambda"].stderr) if p["Lambda"].stderr is not None else float("nan")
            rho0_err = float(p["rho_0"].stderr) if p["rho_0"].stderr is not None else float("nan")

            model_fn = _rho11_model if var_name == "rho_11" else _rho10_model
            y_fit = model_fn(t, Gamma_fit, Lambda_fit, rho0_fit)

            d_sq = Lambda_fit * (Lambda_fit - 2 * Gamma_fit)
            if d_sq > 1e-20:
                regime = "overdamped"
            elif d_sq < -1e-20:
                regime = "underdamped"
            else:
                regime = "critical"

            results[var_name] = {
                "Gamma": Gamma_fit,
                "Gamma_err": Gamma_err,
                "Lambda": Lambda_fit,
                "Lambda_err": Lambda_err,
                "d": complex(np.sqrt(np.complex128(d_sq))),
                "rho_0": rho0_fit,
                "rho_0_err": rho0_err,
                "fit_curve": y_fit,
                "residuals": y_data - y_fit,
                "regime": regime,
            }

        return results

    def generate_figures(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Dict[str, plt.Figure]:
        """One figure per fitted variable: data + fit overlay with residual subplot."""
        t = dataset.coords["time"].values
        figs: Dict[str, plt.Figure] = {}

        for var_name in ("rho_11", "rho_10"):
            if var_name not in results:
                continue
            res = results[var_name]
            y_data = dataset[var_name].values
            y_fit = res["fit_curve"]
            residuals = res["residuals"]
            label = r"$\rho_{11}$" if var_name == "rho_11" else r"$\rho_{10}$"

            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]},
            )

            ax_top.plot(t, y_data, "o", ms=3, alpha=0.6, label=f"{label} data")
            ax_top.plot(
                t, y_fit, "-",
                label=(
                    f"fit ("
                    f"\u0393={res['Gamma']:.4g}, "
                    f"\u039b={res['Lambda']:.4g})"
                ),
            )
            ax_top.set_ylabel(label)
            ax_top.legend()
            ax_top.set_title(f"{label}(t) decoherence fit  [{res['regime']}]")

            ax_bot.plot(t, residuals, ".-", ms=2)
            ax_bot.axhline(0, color="k", lw=0.5)
            ax_bot.set_xlabel("Time")
            ax_bot.set_ylabel("Residual")

            fig.tight_layout()
            figs[var_name] = fig

        return figs
