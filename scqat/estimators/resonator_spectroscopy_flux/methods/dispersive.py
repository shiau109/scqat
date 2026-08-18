"""Dispersive method — full flux-tunable-transmon resonator pull.

The transmon frequency is periodic in flux (symmetric junction). The SQUID
Josephson energy is ``E_J(phi) = E_JSigma |cos(pi (phi - phi_off) / phi0)|`` and
``f_q = sqrt(8 E_C E_J) - E_C``, which factorizes (pinning the sweet-spot value to
``f_q_max``) to::

    f_q(phi)  = (f_q_max + E_C) * sqrt(|cos(pi * (phi - phi_off) / phi0)|) - E_C

``E_C`` is the charging energy, held fixed (``ec`` argument); ``E_C -> 0`` recovers
the bare ``f_q_max * sqrt(|cos|)``, and keeping it corrects the arch shape near the
low sweet spot. The dispersively coupled resonator is pulled by ``g^2 / (f_r0 - f_q)``::

    f_r(phi)  = f_r0 + g**2 / (f_r0 - f_q(phi))

Degeneracy (important)
----------------------
In the dispersive limit ``f_r ~ f_r0 + (g^2 / f_r0**2) * f_q(phi)``, so the trace
amplitude only fixes the **product** ``g^2 * f_q_max`` — ``g`` and ``f_q_max`` are
**not separable** from resonator data alone. Therefore ``f_q_max`` is held
**fixed** at a nominal value by default (override via the ``f_q_max`` kwarg, e.g.
from qubit spectroscopy / the QUAM state, to make ``g`` physical).

``f_r0`` carries a SECOND degeneracy with ``g``: the trace constrains only the pull
``g^2 / (f_r0 - f_q)``, so lowering ``f_r0`` and raising ``g`` traces a valley of
near-equal fits. Supplying a MEASURED ``f_r0`` (kwarg, pinned unless
``fit_f_r0=True``) breaks it and makes ``g`` quantitative; supplying a merely
nominal one is worse than fitting free, because the whole flux-dependent signal is
only the few-MHz pull and an error in ``f_r0`` lands almost entirely in ``g``.

The quantities that *are* well determined independently of both choices —
``dv_phi0``, ``sweet_spot_flux`` and ``sweet_spot_res`` — are the primary outputs;
``g`` is *conditional* on the assumed ``f_q_max`` and ``f_r0``.
"""

from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from lmfit import Model

from .base import (
    FluxModel,
    dense_flux,
    fail_result,
    locate_extremum,
    period_candidates,
    robust_peak_flux,
    robust_trough_flux,
)

# Default assumed sweet-spot detuning (Hz) used to fix f_q_max = f_r0 - this,
# when no f_q_max is supplied. Only affects the (conditional) reported g.
_DEFAULT_SWEET_SPOT_DETUNING = 1.5e9

# Charging energy E_C (Hz) held fixed in the arch. The exact SQUID relation is
# E_J(phi) = E_JSigma |cos(pi phi / phi0)| and f_q = sqrt(8 E_C E_J) - E_C, which
# factorizes (pinning the sweet-spot value to f_q_max) to
# f_q = (f_q_max + E_C) sqrt(|cos|) - E_C. The E_C -> 0 limit is the bare
# f_q_max sqrt(|cos|); keeping E_C corrects the arch shape near the LOW sweet
# spot. The control repo sources it (fact -> design -> this default); 0.2 GHz is
# a typical transmon E_C and equals the codebase's ec_ghz convention.
_DEFAULT_EC_HZ = 0.2e9

# Physical initial guess for the coupling g (Hz) when the caller supplies none.
# g is fitted (the arch amplitude fixes it once f_q_max is held), so this is only
# a seed; the control repo overrides it with a stored/design g_hz when known.
_DEFAULT_G_INIT_HZ = 50e6


def flux_dispersion(flux, f_r0, g, phi0, phi_off, f_q_max, ec):
    """Full-transmon dispersive resonator pull (see module docstring).

    ``ec`` is the charging energy E_C (Hz), held fixed; the arch is the exact
    SQUID form ``f_q = (f_q_max + ec) sqrt(|cos|) - ec`` (``ec=0`` recovers the
    bare ``f_q_max sqrt(|cos|)``).
    """
    f_q = (f_q_max + ec) * np.sqrt(np.abs(np.cos(np.pi * (flux - phi_off) / phi0))) - ec
    return f_r0 + g ** 2 / (f_r0 - f_q)


class DispersiveMethod(FluxModel):
    """Fit ``center_frequency(flux)`` with the full-transmon dispersive model."""

    name = "dispersive"

    def extract(self, flux, center, flux_all, center_all, **kwargs) -> Dict[str, Any]:
        n_valid = len(flux)
        f_r0_guess = float(np.min(center)) if n_valid else np.nan
        amp = float(np.max(center) - np.min(center)) if n_valid else 0.0

        f_q_max = kwargs.get("f_q_max", None)
        fit_f_q_max = bool(kwargs.get("fit_f_q_max", False))
        f_q_max_fixed = f_q_max is None and not fit_f_q_max
        if f_q_max is None:
            f_q_max = f_r0_guess - _DEFAULT_SWEET_SPOT_DETUNING
        ec = float(kwargs.get("ec", _DEFAULT_EC_HZ))
        g_init = float(kwargs.get("g_init", _DEFAULT_G_INIT_HZ))

        # The BARE resonator frequency. Supplying it PINS the fit (vary=False),
        # which breaks the f_r0/g degeneracy: the trace constrains the pull
        # g^2/(f_r0 - f_q), so f_r0 and g trade off along a valley and only an
        # independently MEASURED f_r0 makes g quantitative. Pass fit_f_r0=True to
        # supply it as a mere starting guess (the right choice for a value that
        # is only nominal — an error here goes straight into g). None keeps the
        # historical behaviour: seed from min(center), fit it free.
        f_r0_in = kwargs.get("f_r0", None)
        fit_f_r0 = bool(kwargs.get("fit_f_r0", f_r0_in is None))
        f_r0_fixed = f_r0_in is not None and not fit_f_r0
        f_r0_seed = f_r0_guess if f_r0_in is None else float(f_r0_in)

        extra_defaults = {
            "f_r0": float("nan"), "g": float("nan"), "phi_off": float("nan"),
            "f_q_max": float(f_q_max), "ec": float(ec), "f_r0_err": float("nan"),
            "g_err": float("nan"), "phi_off_err": float("nan"), "max_pull": float("nan"),
            "f_q_max_fixed": bool(f_q_max_fixed), "f_r0_fixed": bool(f_r0_fixed),
        }
        if n_valid < 5:
            return fail_result(flux_all, center_all, n_valid, extra_defaults)

        # ROBUST seed: median-smooth before argmax so one spurious high point
        # (e.g. an edge-pinned dip on a low-SNR slice) can't capture the phase.
        phi_off_guess = robust_peak_flux(flux, center)
        dx = float(np.median(np.diff(flux)))
        span = float(flux.max() - flux.min())

        # Multi-seed period: the FFT estimate alone locks onto T/2 on sparse
        # traces — fit every candidate and keep the lowest-residual solution.
        model = Model(flux_dispersion, independent_vars=["flux"])
        result = None
        for phi0_guess in period_candidates(flux, center):
            params = model.make_params(
                f_r0=f_r0_seed, g=g_init, phi0=phi0_guess,
                phi_off=phi_off_guess, f_q_max=f_q_max, ec=ec,
            )
            # A PINNED f_r0 must not be clipped by the free-fit bounds (a
            # measured bare resonator may legitimately sit below min(center)).
            if f_r0_fixed:
                params["f_r0"].set(vary=False)
            else:
                params["f_r0"].set(
                    min=f_q_max + 1e6, max=float(np.max(center)) + 5 * amp + 1.0)
            # Physical coupling bound: 0 < g < f_r0/10 (a resonator is never
            # coupled harder than ~a tenth of its frequency), independent of the
            # data-derived seed.
            params["g"].set(min=0.0, max=f_r0_seed / 10.0)
            params["phi0"].set(min=2 * abs(dx) + 1e-12, max=1e4 * span + 1.0)
            params["phi_off"].set(min=flux.min() - phi0_guess, max=flux.max() + phi0_guess)
            # The arch top must stay BELOW the bare resonator (the pull
            # g^2/(f_r0 - f_q) diverges as f_q -> f_r0). A pinned f_r0 has no
            # upper bound of its own, so derive the ceiling from its value.
            params["f_q_max"].set(
                vary=fit_f_q_max,
                max=(f_r0_seed - 1e6) if f_r0_fixed else params["f_r0"].max)
            params["ec"].set(vary=False)
            try:
                cand = model.fit(center, params, flux=flux)
            except Exception:
                continue
            # Prefer converged solutions; among equals, the lowest residual.
            if (
                result is None
                or (cand.success and not result.success)
                or (cand.success == result.success and cand.redchi < result.redchi)
            ):
                result = cand
        if result is None:
            return fail_result(flux_all, center_all, n_valid, extra_defaults)
        converged = bool(result.success)

        p = result.params

        def _v(name):
            return float(p[name].value)

        def _e(name):
            return float(p[name].stderr) if p[name].stderr is not None else float("nan")

        f_r0, g, phi0, phi_off, f_q_max_fit, ec_fit = (
            _v("f_r0"), _v("g"), _v("phi0"), _v("phi_off"), _v("f_q_max"), _v("ec")
        )
        dense = dense_flux(flux_all)
        fit_freq = flux_dispersion(dense, f_r0, g, phi0, phi_off, f_q_max_fit, ec_fit)

        # Report both sweet spots mapped into the swept range (kills the
        # periodic-image ambiguity of a bare phi_off): the UPPER spot (arch top,
        # max qubit frequency) at phi_off images, the LOWER spot (arch bottom)
        # half a period away.
        model_at = lambda ff: flux_dispersion(ff, f_r0, g, phi0, phi_off, f_q_max_fit, ec_fit)  # noqa: E731
        # Extrema are mapped into the FULL swept flux range (flux_all), not the
        # good-point subset — an arch bottom often sits where per-slice fits were
        # rejected, and clamping it to the good range would misreport it.
        finite_all = flux_all[np.isfinite(flux_all)]
        lo, hi = float(finite_all.min()), float(finite_all.max())
        sweet_spot_flux, sweet_spot_res = locate_extremum(
            phi_off, phi0, lo, hi, robust_peak_flux(flux, center), model_at)
        sweet_spot_low_flux, sweet_spot_low_res = locate_extremum(
            phi_off + phi0 / 2.0, phi0, lo, hi, robust_trough_flux(flux, center), model_at)
        max_pull = float(g ** 2 / (f_r0 - f_q_max_fit))

        in_range = flux.min() - phi0 <= phi_off <= flux.max() + phi0
        success = bool(converged and np.isfinite(phi0) and phi0 > 0 and in_range)

        return {
            "flux_bias": flux_all,
            "center_freq": center_all,
            "fit_flux": dense,
            "fit_freq": fit_freq,
            "sweet_spot_flux": float(sweet_spot_flux),
            "sweet_spot_res": float(sweet_spot_res),
            "sweet_spot_low_flux": float(sweet_spot_low_flux),
            "sweet_spot_low_res": float(sweet_spot_low_res),
            "dv_phi0": float(phi0),
            "sweet_spot_flux_err": _e("phi_off"),
            "dv_phi0_err": _e("phi0"),
            "f_r0": f_r0, "g": g, "phi_off": phi_off, "f_q_max": f_q_max_fit,
            "ec": ec_fit,
            "f_r0_err": _e("f_r0"), "g_err": _e("g"), "phi_off_err": _e("phi_off"),
            "max_pull": max_pull,
            "f_q_max_fixed": bool(f_q_max_fixed),
            "f_r0_fixed": bool(f_r0_fixed),
            "n_points": int(n_valid),
            "success": success,
        }

    def build_plot_data(self, results: Dict[str, Any]) -> xr.Dataset:
        data_vars = {
            "center_freq": ("flux_bias", np.asarray(results["center_freq"], float)),
            "fit_freq": ("fit_flux", np.asarray(results["fit_freq"], float)),
        }
        coords = {
            "flux_bias": np.asarray(results["flux_bias"], float),
            "fit_flux": np.asarray(results["fit_flux"], float),
        }
        attrs = {
            "method": self.name,
            "sweet_spot_flux": float(results["sweet_spot_flux"]),
            "sweet_spot_res": float(results["sweet_spot_res"]),
            "sweet_spot_low_flux": float(results["sweet_spot_low_flux"]),
            "sweet_spot_low_res": float(results["sweet_spot_low_res"]),
            "dv_phi0": float(results["dv_phi0"]),
            "f_r0": float(results["f_r0"]),
            "g": float(results["g"]),
            "max_pull": float(results["max_pull"]),
            "f_q_max": float(results["f_q_max"]),
            "ec": float(results["ec"]),
            "f_q_max_fixed": int(bool(results["f_q_max_fixed"])),
            "f_r0_fixed": int(bool(results["f_r0_fixed"])),
            "success": int(bool(results["success"])),
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        from ..visualization import plot_flux_model
        return plot_flux_model(plot_data)
