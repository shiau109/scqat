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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_analyzer import BaseAnalyzer


# ------------------------------------------------------------------
# Model functions (module-level so they can be imported for testing)
# ------------------------------------------------------------------

def _decoherence_G(t, Gamma, Lambda):
    """
    Decoherence function G(t).

    Parameters
    ----------
    t : array-like
        Time values.
    Gamma : float
        Qubit relaxation rate (1/T1).
    Lambda : float
        Spectral-width parameter of the environment.

    Returns
    -------
    G : ndarray (real)
    """
    d_sq = Lambda * (Lambda - 2 * Gamma)
    d = np.sqrt(np.complex128(d_sq))
    if np.abs(d) < 1e-15:
        # Critical damping: L'Hopital limit of (Lambda/d)*sinh(d*t/2) -> Lambda*t/2
        G = np.exp(-Lambda * t / 2) * (1.0 + Lambda * t / 2)
    else:
        arg = d * t / 2
        G = np.exp(-Lambda * t / 2) * (
            np.cosh(arg) + (Lambda / d) * np.sinh(arg)
        )
    return np.real(G).astype(float)


def _rho11_model(t, Gamma, Lambda, rho11_0):
    """Model for rho_11(t) = |G(t)|^2 * rho_11(0)."""
    G = _decoherence_G(t, Gamma, Lambda)
    return np.abs(G) ** 2 * rho11_0


def _rho10_model(t, Gamma, Lambda, rho10_0):
    """Model for rho_10(t) = G(t) * rho_10(0)."""
    G = _decoherence_G(t, Gamma, Lambda)
    return G * rho10_0


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
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _initial_guess_Lambda(t, y):
        """Rough estimate of Lambda from the envelope decay."""
        y_abs = np.abs(y)
        mask = y_abs > y_abs.max() * 0.01
        if mask.sum() < 2:
            return 1.0
        t_span = t[mask][-1] - t[mask][0]
        ratio = y_abs[mask][-1] / y_abs[mask][0]
        if ratio <= 0 or ratio >= 1 or t_span <= 0:
            return 1.0 / max(t_span, 1.0)
        return -np.log(ratio) / t_span

    @staticmethod
    def _fit_with_retry(model_fn, t, y_data, p0, bounds):
        """Try curve_fit; on failure, retry with swapped Gamma/Lambda guess."""
        try:
            popt, pcov = curve_fit(
                model_fn, t, y_data, p0=p0, bounds=bounds, maxfev=10000,
            )
            return popt, np.sqrt(np.diag(pcov))
        except RuntimeError:
            p0_alt = list(p0)
            p0_alt[0], p0_alt[1] = p0_alt[1], p0_alt[0]
            popt, pcov = curve_fit(
                model_fn, t, y_data, p0=p0_alt, bounds=bounds, maxfev=10000,
            )
            return popt, np.sqrt(np.diag(pcov))

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

        var_models = [("rho_11", _rho11_model), ("rho_10", _rho10_model)]
        for var_name, model_fn in var_models:
            if var_name not in dataset.data_vars:
                continue

            y_data = dataset[var_name].values.astype(float)

            Lambda_guess = self._initial_guess_Lambda(t, y_data)
            Gamma_guess = Lambda_guess * 0.5
            rho0_guess = float(y_data[0])
            p0 = [Gamma_guess, Lambda_guess, rho0_guess]
            bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])

            popt, perr = self._fit_with_retry(model_fn, t, y_data, p0, bounds)
            Gamma_fit, Lambda_fit, rho0_fit = popt
            Gamma_err, Lambda_err, rho0_err = perr

            y_fit = model_fn(t, *popt)

            d_sq = Lambda_fit * (Lambda_fit - 2 * Gamma_fit)
            if d_sq > 1e-20:
                regime = "overdamped"
            elif d_sq < -1e-20:
                regime = "underdamped"
            else:
                regime = "critical"

            results[var_name] = {
                "Gamma": float(Gamma_fit),
                "Gamma_err": float(Gamma_err),
                "Lambda": float(Lambda_fit),
                "Lambda_err": float(Lambda_err),
                "d": complex(np.sqrt(np.complex128(d_sq))),
                "rho_0": float(rho0_fit),
                "rho_0_err": float(rho0_err),
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
