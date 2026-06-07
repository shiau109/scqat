"""
Resonator Flux Dispersion Estimator
==================================
Fit the resonator centre frequency as a function of flux with the **full
flux-tunable-transmon dispersive model** — the downstream step that consumes the
1-D ``center_frequency(flux)`` trace produced by
:class:`~scqat.estimators.resonator_spectroscopy_vs_flux.ResonatorSpectroscopyVsFluxEstimator`.

Model
-----
The transmon frequency is periodic in flux (symmetric junction)::

    f_q(phi)  = f_q_max * sqrt(|cos(pi * (phi - phi_off) / phi0)|)

and the dispersively coupled resonator is pulled by ``g^2 / (f_r0 - f_q)``::

    f_r(phi)  = f_r0 + g**2 / (f_r0 - f_q(phi))

Parameters: ``f_r0`` (bare resonator), ``g`` (coupling), ``phi0`` (flux period in
volts, i.e. ``dv_phi0``), ``phi_off`` (sweet-spot flux), ``f_q_max`` (qubit
frequency at the sweet spot).

Degeneracy (important)
----------------------
In the dispersive limit ``f_r ≈ f_r0 + (g^2 / f_r0**2) * f_q(phi)``, so the trace
amplitude only fixes the **product** ``g^2 * f_q_max`` — ``g`` and ``f_q_max`` are
**not separable** from resonator data alone.  Therefore ``f_q_max`` is held
**fixed** at a nominal value by default (override via the ``f_q_max`` kwarg, e.g.
from qubit spectroscopy / the QUAM state, to make ``g`` physical).  The quantities
that *are* well determined independently of this choice — ``phi0``, ``phi_off``,
``f_r0`` and the sweet-spot resonator frequency — are reported as the primary
outputs; ``g`` is reported as *conditional* on the assumed ``f_q_max``.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
Data variables:
    - center_freq : (flux_bias,) – fitted resonator centre frequency (Hz).
    - success     : (flux_bias,) bool – optional per-point validity mask.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from lmfit import Model

from scqat.core.base_estimator import BaseEstimator


# Default assumed sweet-spot detuning (Hz) used to fix f_q_max = f_r0 - this,
# when no f_q_max is supplied. Only affects the (conditional) reported g.
_DEFAULT_SWEET_SPOT_DETUNING = 1.5e9


def flux_dispersion(flux, f_r0, g, phi0, phi_off, f_q_max):
    """Full-transmon dispersive resonator pull (see module docstring)."""
    f_q = f_q_max * np.sqrt(np.abs(np.cos(np.pi * (flux - phi_off) / phi0)))
    return f_r0 + g ** 2 / (f_r0 - f_q)


def _estimate_period(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the flux period of the trace from its dominant FFT component.
    Falls back to twice the swept span when fewer than ~one period is visible."""
    n = len(x)
    span = float(x.max() - x.min()) or 1.0
    if n < 4:
        return 2.0 * span
    dx = float(np.median(np.diff(np.sort(x))))
    if dx <= 0:
        return 2.0 * span
    yd = (y - np.mean(y)) * np.hanning(n)
    amp = np.abs(np.fft.rfft(yd))
    freqs = np.fft.rfftfreq(n, d=dx)
    amp[0] = 0.0  # drop DC
    k = int(np.argmax(amp))
    f_peak = freqs[k]
    if f_peak <= 0:
        return 2.0 * span
    return 1.0 / f_peak


class ResonatorFluxDispersionEstimator(BaseEstimator):
    """Fit ``center_frequency(flux)`` with the full-transmon dispersive model."""

    estimator_name = "resonator_flux_dispersion"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if "flux_bias" not in dataset.coords:
            raise ValueError("ResonatorFluxDispersionEstimator requires a 'flux_bias' coordinate.")
        if "center_freq" not in dataset:
            raise ValueError("ResonatorFluxDispersionEstimator requires a 'center_freq' variable.")

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the dispersive model to the centre-frequency-vs-flux trace.

        Keyword arguments
        -----------------
        f_q_max : float, optional
            Qubit frequency at the sweet spot (Hz). When given it is held fixed
            and ``g`` becomes physical; otherwise ``f_q_max`` defaults to
            ``f_r0_guess - 1.5 GHz`` (fixed) and ``g`` is conditional.
        fit_f_q_max : bool, optional
            If True, let ``f_q_max`` vary too (ill-conditioned without a prior;
            default False).

        Returns
        -------
        dict
            ``{flux_bias, center_freq, fit_flux, fit_freq, f_r0, g, phi0,
            phi_off, f_q_max, *_err, sweet_spot_flux, sweet_spot_freq,
            dv_phi0, max_pull, f_q_max_fixed, success}``
        """
        flux_all = dataset.coords["flux_bias"].values.astype(float)
        y_all = np.asarray(dataset["center_freq"].values, dtype=float)

        mask = np.isfinite(flux_all) & np.isfinite(y_all)
        if "success" in dataset:
            mask &= np.asarray(dataset["success"].values, dtype=bool)
        flux = flux_all[mask]
        y = y_all[mask]

        # --- Seeds (robust, FFT period + extremum) -----------------------
        order = np.argsort(flux)
        flux, y = flux[order], y[order]
        n_valid = len(flux)
        f_r0_guess = float(np.min(y)) if n_valid else np.nan
        span = float(flux.max() - flux.min()) if n_valid else 1.0
        amp = float(np.max(y) - np.min(y)) if n_valid else 0.0

        f_q_max = kwargs.get("f_q_max", None)
        fit_f_q_max = bool(kwargs.get("fit_f_q_max", False))
        f_q_max_fixed = f_q_max is None and not fit_f_q_max
        if f_q_max is None:
            f_q_max = f_r0_guess - _DEFAULT_SWEET_SPOT_DETUNING

        nan = float("nan")
        fail = {
            "flux_bias": flux_all, "center_freq": y_all,
            "fit_flux": flux_all, "fit_freq": np.full_like(flux_all, nan),
            "f_r0": nan, "g": nan, "phi0": nan, "phi_off": nan, "f_q_max": float(f_q_max),
            "f_r0_err": nan, "g_err": nan, "phi0_err": nan, "phi_off_err": nan,
            "sweet_spot_flux": nan, "sweet_spot_freq": nan, "dv_phi0": nan, "max_pull": nan,
            "f_q_max_fixed": bool(f_q_max_fixed), "n_points": int(n_valid), "success": False,
        }
        if n_valid < 5:
            return fail

        phi0_guess = _estimate_period(flux, y)
        phi_off_guess = float(flux[int(np.argmax(y))])  # sweet spot = max pull
        detuning_guess = max(f_r0_guess - f_q_max, 1e6)
        g_guess = float(np.sqrt(max(amp, 1.0) * detuning_guess))
        dx = float(np.median(np.diff(flux)))

        model = Model(flux_dispersion, independent_vars=["flux"])
        params = model.make_params(
            f_r0=f_r0_guess, g=g_guess, phi0=phi0_guess, phi_off=phi_off_guess, f_q_max=f_q_max,
        )
        params["f_r0"].set(min=f_q_max + 1e6, max=float(np.max(y)) + 5 * amp + 1.0)
        params["g"].set(min=0.0, max=10 * g_guess + 1.0)
        params["phi0"].set(min=2 * abs(dx) + 1e-12, max=1e4 * span + 1.0)
        params["phi_off"].set(min=flux.min() - phi0_guess, max=flux.max() + phi0_guess)
        params["f_q_max"].set(vary=fit_f_q_max, max=params["f_r0"].max)

        try:
            result = model.fit(y, params, flux=flux)
            converged = bool(result.success)
        except Exception:
            return fail

        p = result.params

        def _v(name):
            return float(p[name].value)

        def _e(name):
            return float(p[name].stderr) if p[name].stderr is not None else nan

        f_r0, g, phi0, phi_off, f_q_max_fit = (_v("f_r0"), _v("g"), _v("phi0"),
                                               _v("phi_off"), _v("f_q_max"))
        dense = np.linspace(flux_all[np.isfinite(flux_all)].min(),
                            flux_all[np.isfinite(flux_all)].max(), 400)
        fit_freq = flux_dispersion(dense, f_r0, g, phi0, phi_off, f_q_max_fit)
        sweet_spot_freq = float(flux_dispersion(np.array([phi_off]), f_r0, g, phi0, phi_off, f_q_max_fit)[0])
        max_pull = float(g ** 2 / (f_r0 - f_q_max_fit))

        in_range = flux.min() - phi0 <= phi_off <= flux.max() + phi0
        success = bool(converged and np.isfinite(phi0) and phi0 > 0 and in_range)

        return {
            "flux_bias": flux_all,
            "center_freq": y_all,
            "fit_flux": dense,
            "fit_freq": fit_freq,
            "f_r0": f_r0, "g": g, "phi0": phi0, "phi_off": phi_off, "f_q_max": f_q_max_fit,
            "f_r0_err": _e("f_r0"), "g_err": _e("g"), "phi0_err": _e("phi0"),
            "phi_off_err": _e("phi_off"),
            "sweet_spot_flux": float(phi_off),
            "sweet_spot_freq": sweet_spot_freq,
            "dv_phi0": float(phi0),
            "max_pull": max_pull,
            "f_q_max_fixed": bool(f_q_max_fixed),
            "n_points": int(n_valid),
            "success": success,
        }

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the scalar fit parameters; drop the trace/fit arrays."""
        drop = {"flux_bias", "center_freq", "fit_flux", "fit_freq"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the data points and the dense fit curve so the figure redraws
        without refitting (separate ``flux_bias`` / ``fit_flux`` dims)."""
        data_vars = {
            "center_freq": ("flux_bias", np.asarray(results["center_freq"], float)),
            "fit_freq": ("fit_flux", np.asarray(results["fit_freq"], float)),
        }
        coords = {
            "flux_bias": np.asarray(results["flux_bias"], float),
            "fit_flux": np.asarray(results["fit_flux"], float),
        }
        attrs = {
            "f_r0": float(results["f_r0"]),
            "g": float(results["g"]),
            "dv_phi0": float(results["dv_phi0"]),
            "sweet_spot_flux": float(results["sweet_spot_flux"]),
            "sweet_spot_freq": float(results["sweet_spot_freq"]),
            "max_pull": float(results["max_pull"]),
            "f_q_max": float(results["f_q_max"]),
            "f_q_max_fixed": int(bool(results["f_q_max_fixed"])),
            "success": int(bool(results["success"])),
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Centre-frequency-vs-flux data points with the dispersive fit overlaid
        and the sweet spot marked, drawn from plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        flux = plot_data.coords["flux_bias"].values.astype(float)
        center = plot_data["center_freq"].values.astype(float)
        fit_flux = plot_data.coords["fit_flux"].values.astype(float)
        fit_freq = plot_data["fit_freq"].values.astype(float)
        ss_flux = float(plot_data.attrs.get("sweet_spot_flux", np.nan))
        ss_freq = float(plot_data.attrs.get("sweet_spot_freq", np.nan))
        dv_phi0 = float(plot_data.attrs.get("dv_phi0", np.nan))

        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.plot(flux, center / 1e9, "o", ms=4, color="C0", label="fitted centre (data)")
        if np.isfinite(fit_freq).any():
            ax.plot(fit_flux, fit_freq / 1e9, "-", lw=1.8, color="C1",
                    label=f"dispersive fit (dv_phi0={dv_phi0:.4f} V)")
        if np.isfinite(ss_flux):
            ax.axvline(ss_flux, color="C3", ls=":", lw=1.0,
                       label=f"sweet spot @ {ss_flux:.4f} V, {ss_freq / 1e9:.4f} GHz")

        ax.set_xlabel("Flux bias (V)")
        ax.set_ylabel("Resonator centre frequency (GHz)")
        cond = "" if plot_data.attrs.get("f_q_max_fixed", 1) == 0 else " (g conditional on assumed f_q_max)"
        ax.set_title("Resonator flux dispersion" + cond)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return {"resonator_flux_dispersion": fig}
