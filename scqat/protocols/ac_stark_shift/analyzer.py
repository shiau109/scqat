from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from lmfit import Model

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.qubit_spectroscopy import QubitSpectroscopyAnalyzer
from scqat.protocols.ac_stark_shift.visualization import (
    plot_raw_2d_with_f01,
    plot_stark_fit,
    plot_photon_number,
)


def _starkshift_p(x, A, fa, X_eff):
    """AC-Stark model: f01(P) = fa - (2*A*P + 1)*X_eff, with photon number n = A*P."""
    return fa - (2.0 * x * A + 1.0) * X_eff


class AcStarkShiftAnalyzer(BaseAnalyzer):
    """
    AC-Stark shift: qubit frequency vs readout power.

    Expects an xarray.Dataset with:
        - Variable:    a real spectroscopy signal (default ``I``; override with
          the ``signal_var`` kwarg/class attr) with dims (<power_coord>, detuning)
        - Coordinate:  ``detuning`` — drive-frequency detuning
        - Coordinate:  the readout-power axis named by :attr:`power_coord`
          (default ``readout_amp_ratio``)

    For each power point the qubit line is located by reusing
    :class:`QubitSpectroscopyAnalyzer` (peak detuning → ``f01``). ``f01`` is then
    fit against power ``P = amp**2`` with the AC-Stark model
    ``f01(P) = fa - (2*A*P + 1)*X_eff``, giving the bare frequency ``fa`` and the
    power→photon-number coefficient ``A`` (so ``n = A*P``). ``X_eff`` (the
    dispersive shift, Hz) must be supplied as a kwarg; without it only the
    ``f01`` vs power curve is produced.

    Ported from qcat ``ac_stark_shift`` — replaces the NCU ``QS_fit_analysis``
    per-slice fit with scqat's ``QubitSpectroscopyAnalyzer`` and keeps the same
    Stark model. The wiring-attenuation / photon-number calibration block from
    qcat (largely commented out there) is not ported.
    """

    protocol_name = "ac_stark_shift"
    power_coord: str = "readout_amp_ratio"
    signal_var: str = "I"

    # ------------------------------------------------------------------
    def _resolve(self, kwargs: Dict[str, Any]):
        return (
            kwargs.get("power_coord", self.power_coord),
            kwargs.get("signal_var", self.signal_var),
        )

    def _check_data(self, dataset: xr.Dataset) -> None:
        if self.power_coord not in dataset.coords:
            raise ValueError(f"AC-Stark shift requires a '{self.power_coord}' coordinate.")
        if "detuning" not in dataset.coords:
            raise ValueError("AC-Stark shift requires a 'detuning' coordinate.")
        if not any(v in dataset for v in (self.signal_var, "I", "signal")):
            raise ValueError(
                f"AC-Stark shift requires a signal variable "
                f"('{self.signal_var}', 'I', or 'signal')."
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Locate f01 per power slice and fit the AC-Stark model.

        Kwargs:
            X_eff (float): Dispersive shift (Hz). Required for the Stark fit.
            power_coord (str), signal_var (str): Override the defaults.
            fit_window (tuple[int, int]): Restrict the Stark fit to a slice
                ``[lo:hi]`` of the (sorted) power sweep.
            Any other kwargs are forwarded to QubitSpectroscopyAnalyzer.

        Returns a dict: power_coord, power_values, P, f01, X_eff, has_fit,
        coeff_A, f01_bare, photon_number, redchi.
        """
        power_coord, signal_var = self._resolve(kwargs)
        X_eff = kwargs.get("X_eff", None)
        fit_window = kwargs.get("fit_window", None)
        qs_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("ref", "prominence", "fit_window_factor")}

        power_values = np.asarray(dataset.coords[power_coord].values, dtype=float)
        P = power_values ** 2

        qs = QubitSpectroscopyAnalyzer()
        f01 = np.full(power_values.shape, np.nan)
        for i, val in enumerate(power_values):
            sub = dataset.sel({power_coord: val})
            try:
                res = qs.extract_parameters(sub, signal_var=signal_var, max_peaks=1, **qs_kwargs)
                peaks = res.get("peaks", [])
                if peaks:
                    f01[i] = float(peaks[0]["detuning"])
            except Exception:
                pass

        results: Dict[str, Any] = {
            "power_coord": power_coord,
            "power_values": power_values,
            "P": P,
            "f01": f01,
            "X_eff": (float(X_eff) if X_eff is not None else None),
            "has_fit": False,
            "coeff_A": np.nan,
            "f01_bare": np.nan,
            "photon_number": np.full(power_values.shape, np.nan),
            "redchi": np.nan,
        }

        if X_eff is None:
            return results

        # ---- Stark-model fit of f01 vs P ----
        mask = np.isfinite(f01) & np.isfinite(P)
        if fit_window is not None:
            lo, hi = fit_window
            window = np.zeros_like(mask)
            window[lo:hi] = True
            mask = mask & window

        if np.count_nonzero(mask) >= 3:
            x, y = P[mask], f01[mask]
            # Data-driven guesses (unit-agnostic in P):
            dP = x[-1] - x[0]
            A_guess = -(y[-1] - y[0]) / (2.0 * X_eff * dP) if (X_eff * dP) != 0 else 1.0
            fa_guess = y[0] + X_eff
            model = Model(_starkshift_p, independent_vars=["x"])
            params = model.make_params(A=A_guess, fa=fa_guess, X_eff=float(X_eff))
            params["X_eff"].set(vary=False)  # X_eff is a given factor, not fitted
            try:
                fit = model.fit(y, params, x=x)
            except Exception:
                fit = None
            if fit is not None and fit.success:
                A = float(fit.best_values["A"])
                fa = float(fit.best_values["fa"])
                results.update(
                    has_fit=True,
                    coeff_A=A,
                    f01_bare=fa,
                    photon_number=A * P,
                    redchi=float(fit.redchi),
                )

        return results

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Bundle the raw 2D spectroscopy map, the per-power f01 and photon
        number, and a pre-evaluated Stark fit curve into one self-sufficient
        Dataset so the figures redraw from the saved ``*_plotdata.nc`` alone."""
        power_coord, signal_var = self._resolve(kwargs)
        if signal_var not in dataset:
            signal_var = "I" if "I" in dataset else "signal"

        power_values = np.asarray(results["power_values"], dtype=float)
        detuning = np.asarray(dataset.coords["detuning"].values, dtype=float)
        raw = dataset[signal_var].transpose(power_coord, "detuning").values

        data_vars: Dict[str, Any] = {
            "raw_signal": ([power_coord, "detuning"], np.asarray(raw, dtype=float)),
            "f01": (power_coord, np.asarray(results["f01"], dtype=float)),
            "P": (power_coord, np.asarray(results["P"], dtype=float)),
            "photon_number": (power_coord, np.asarray(results["photon_number"], dtype=float)),
        }
        coords: Dict[str, Any] = {power_coord: power_values, "detuning": detuning}
        attrs: Dict[str, Any] = {
            "power_coord": power_coord,
            "has_fit": int(bool(results["has_fit"])),
            "X_eff": float(results["X_eff"]) if results["X_eff"] is not None else np.nan,
            "coeff_A": float(results["coeff_A"]),
            "f01_bare": float(results["f01_bare"]),
        }

        if results["has_fit"]:
            P = np.asarray(results["P"], dtype=float)
            P_fine = np.linspace(float(P.min()), float(P.max()), 20 * len(P))
            curve = _starkshift_p(P_fine, results["coeff_A"], results["f01_bare"], results["X_eff"])
            coords["P_fine"] = P_fine
            data_vars["stark_fit"] = ("P_fine", curve)

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the AC-Stark figures, drawing only from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)

        figs: Dict[str, plt.Figure] = {
            "raw_2d": plot_raw_2d_with_f01(plot_data),
            "stark_fit": plot_stark_fit(plot_data),
        }
        if plot_data.attrs.get("has_fit", 0):
            figs["photon_number"] = plot_photon_number(plot_data)
        return figs
