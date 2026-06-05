"""
Qubit Decoherence Protocol
==========================
Fits time-resolved density-matrix elements (rho_11, rho_10) to extract
the Lindbladian decoherence rates (gamma, lambda_, Delta) from the
non-Markovian amplitude-damping model:

    Lambda = gamma / 2
    a      = Lambda - 1j*Delta
    d      = sqrt(a**2 - 2*Gamma*Lambda) = sqrt(a**2 - 4*lambda_**2)
    G(t)   = exp(-a*t/2) [cosh(d*t/2) + (a/d) sinh(d*t/2)]

    rho_11(t) = |G(t)|^2  rho_11(0)

Relation to the older (Gamma, Lambda) parameterisation:
    gamma   = 2 * Lambda                  (relaxation rate)
    lambda_ = sqrt(Gamma * Lambda / 2)    (coupling strength)

Expected xarray.Dataset contract:
    Coordinates:
        - time : 1-D array of measurement times
    Data variables (at least one required):
        - rho_11 : 1-D array – excited-state population vs time
        - rho_10 : 1-D array – off-diagonal coherence vs time
"""

from typing import Any, Dict, Optional

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
    non-Markovian amplitude-damping model parameterised by (gamma, lambda_).
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

        Parameters
        ----------
        fix_delta : bool, optional
            If True, Delta is fixed to 0 during fitting (default False).

        Returns
        -------
        dict with a sub-dict for each fitted variable containing:
            gamma, gamma_err, lambda_, lambda_err, Delta, Delta_err,
            d, rho_0, rho_0_err, fit_curve, residuals, regime
        """
        t = dataset.coords["time"].values.astype(float)
        fix_delta: bool = bool(kwargs.get("fix_delta", False))
        results: Dict[str, Any] = {}

        for var_name in ("rho_11", "rho_10"):
            if var_name not in dataset.data_vars:
                continue

            y_data = dataset[var_name].values.astype(float)
            da = xr.DataArray(y_data, coords={"x": t}, dims="x")

            fitter = FitQubitDecoherence(da, component=var_name, fix_delta=fix_delta)
            result = fitter.fit()

            p = result.params
            gamma_fit = float(p["gamma"].value)
            lambda_fit = float(p["lambda_"].value)
            Delta_fit = float(p["Delta"].value)
            rho0_fit = float(p["rho_0"].value)
            gamma_err = float(p["gamma"].stderr) if p["gamma"].stderr is not None else float("nan")
            lambda_err = float(p["lambda_"].stderr) if p["lambda_"].stderr is not None else float("nan")
            Delta_err = float(p["Delta"].stderr) if p["Delta"].stderr is not None else float("nan")
            rho0_err = float(p["rho_0"].stderr) if p["rho_0"].stderr is not None else float("nan")

            model_fn = _rho11_model if var_name == "rho_11" else _rho10_model
            y_fit = model_fn(t, gamma_fit, lambda_fit, Delta_fit, rho0_fit)

            # d^2 = (Lambda - 1j*Delta)^2 - 4*lambda_^2
            a = gamma_fit / 2.0 - 1j * Delta_fit
            d_complex = complex(np.sqrt(np.complex128(a * a - 4.0 * lambda_fit ** 2)))
            d_sq_real = (gamma_fit / 2.0) ** 2 - 4.0 * lambda_fit ** 2
            if d_sq_real > 1e-20:
                regime = "overdamped"
            elif d_sq_real < -1e-20:
                regime = "underdamped"
            else:
                regime = "critical"

            results[var_name] = {
                "gamma": gamma_fit,
                "gamma_err": gamma_err,
                "lambda_": lambda_fit,
                "lambda_err": lambda_err,
                "Delta": Delta_fit,
                "Delta_err": Delta_err,
                "d": d_complex,
                "rho_0": rho0_fit,
                "rho_0_err": rho0_err,
                "fit_curve": y_fit,
                "residuals": y_data - y_fit,
                "regime": regime,
            }

        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters per variable; drop the per-time arrays."""
        drop = {"fit_curve", "residuals"}
        return {
            var: {k: v for k, v in res.items() if k not in drop}
            for var, res in results.items()
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle, per fitted variable, the data / fit / residual curves (over
        ``time``) plus the key fit parameters (in ``.attrs``) so each figure is
        reconstructable without refitting.
        """
        t = dataset.coords["time"].values.astype(float)
        data_vars: Dict[str, Any] = {}
        attrs: Dict[str, Any] = {}
        present = []
        for var_name in ("rho_11", "rho_10"):
            if var_name not in results:
                continue
            res = results[var_name]
            present.append(var_name)
            data_vars[f"{var_name}_data"] = ("time", dataset[var_name].values.astype(float))
            data_vars[f"{var_name}_fit"] = ("time", np.asarray(res["fit_curve"], dtype=float))
            data_vars[f"{var_name}_residual"] = ("time", np.asarray(res["residuals"], dtype=float))
            attrs[f"{var_name}_gamma"] = float(res["gamma"])
            attrs[f"{var_name}_lambda"] = float(res["lambda_"])
            attrs[f"{var_name}_Delta"] = float(res["Delta"])
            attrs[f"{var_name}_regime"] = res["regime"]
        attrs["variables"] = ",".join(present)
        return xr.Dataset(data_vars, coords={"time": t}, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """One figure per fitted variable: data + fit overlay with residual subplot."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        t = plot_data.coords["time"].values
        figs: Dict[str, plt.Figure] = {}

        for var_name in ("rho_11", "rho_10"):
            if f"{var_name}_data" not in plot_data:
                continue
            y_data = plot_data[f"{var_name}_data"].values
            y_fit = plot_data[f"{var_name}_fit"].values
            residuals = plot_data[f"{var_name}_residual"].values
            label = r"$\rho_{11}$" if var_name == "rho_11" else r"$\rho_{10}$"

            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]},
            )

            ax_top.plot(t, y_data, "o", ms=3, alpha=0.6, label=f"{label} data")
            ax_top.plot(
                t, y_fit, "-",
                label=(
                    f"fit ("
                    f"\u03b3={plot_data.attrs[f'{var_name}_gamma']:.4g}, "
                    f"\u03bb={plot_data.attrs[f'{var_name}_lambda']:.4g}, "
                    f"\u0394={plot_data.attrs[f'{var_name}_Delta']:.4g})"
                ),
            )
            ax_top.set_ylabel(label)
            ax_top.legend()
            ax_top.set_title(
                f"{label}(t) decoherence fit  [{plot_data.attrs[f'{var_name}_regime']}]"
            )

            ax_bot.plot(t, residuals, ".-", ms=2)
            ax_bot.axhline(0, color="k", lw=0.5)
            ax_bot.set_xlabel("Time")
            ax_bot.set_ylabel("Residual")

            fig.tight_layout()
            figs[var_name] = fig

        return figs
