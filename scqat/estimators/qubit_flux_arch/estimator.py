"""
Qubit Spectroscopy vs Flux — Full Analysis (composite)
======================================================
Two-stage qubit-flux analysis that chains the flux stage functions and an
existing fitter, and owns the canonical combined figure:

  1. :func:`scqat.estimators.qubit_spectroscopy_flux.track_flux_peaks` turns
     the 2-D ``(flux_bias, detuning)`` map into a peak **point-cloud**
     (several transitions may coexist per flux slice), then
  2. one peak per flux slice is assigned to the 0-1 branch (``branch`` strategy),
     and the branch is fitted with :class:`~scqat.tools.fit_transmon_freq_flux.
     FitTransmonFrequencyFlux` (``f = sqrt(8*Ec*Ej_eff) - Ec``) — with one robust
     residual-rejection refit — to obtain the sweet spot, flux period/offset and
     ``Ej_sum``.

This is the Phase-3 feeder: ``f01(flux)`` + ``Ej_sum``/``Ec`` are the inputs the
device-level EJ/EC inference consumes. Turning the arch outputs into instrument
state (flux offsets in V vs Phi0, QUAM writes) is out of scope — the calling repo
owns that.

``results`` is nested as ``{"point_cloud": <stage-1 results>, "arch": <stage-2
results>}`` so the two stages' keys stay unambiguous.

Expected xarray.Dataset contract
--------------------------------
Identical to stage 1 (the ``qubit`` dimension already removed), with one
addition: the **absolute** frequency axis is required, because the arch model is
absolute-scale.

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – drive-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute drive frequency (Hz). REQUIRED here.
Data variables:
    - IQdata : (flux_bias, detuning) complex, **or** I and Q, **or** a real
      variable named by ``signal_var``.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.qubit_spectroscopy_flux import (
    check_flux_dataset,
    flux_cloud_plotdata,
    track_flux_peaks,
)
from scqat.tools.fit_transmon_freq_flux import FitTransmonFrequencyFlux

from scqat.estimators.qubit_flux_arch.visualization import plot_arch

#: kwargs consumed by the arch (stage-2) fit; everything else flows to stage 1.
_ARCH_KWARGS = ("branch", "ec_ghz", "fit_d", "arch_n_sigma")


class QubitFluxArchEstimator(BaseEstimator):
    """Composite qubit-vs-flux estimator: peak point-cloud + transmon arch fit."""

    estimator_name = "qubit_flux_arch"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        check_flux_dataset(dataset)
        if "full_freq" not in dataset.coords:
            raise ValueError(
                "QubitFluxArchEstimator requires the absolute 'full_freq' coordinate — "
                "the transmon arch model is absolute-frequency."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the point-cloud fit, select the 0-1 branch, fit the transmon arch.

        Keyword arguments
        -----------------
        branch : {"strongest", "highest"}, optional
            Which good peak represents the 0-1 line at each flux: the largest
            ``|amplitude|`` (default — two-photon/neighbour lines are weaker under
            weak drive) or the highest absolute frequency (the 0-2/2 line sits
            ~Ec/2 below 0-1).
        ec_ghz : float, optional
            Charging energy in GHz, held fixed in the arch model (default 0.2).
        fit_d : bool, optional
            Free the junction-asymmetry parameter ``d`` (default False).
        arch_n_sigma : float, optional
            Robust residual-rejection threshold between the two arch fits
            (default 3.0).
        signal_var, prominence, max_peaks, n_sigma, ...
            Forwarded to the stage-1 tracker
            :func:`~scqat.estimators.qubit_spectroscopy_flux.track_flux_peaks`
            (unknown names raise before any per-slice fit).
        """
        branch = str(kwargs.pop("branch", "strongest"))
        ec_ghz = float(kwargs.pop("ec_ghz", 0.2))
        fit_d = bool(kwargs.pop("fit_d", False))
        arch_n_sigma = float(kwargs.pop("arch_n_sigma", 3.0))

        cloud = track_flux_peaks(dataset, **kwargs)

        # ---- branch selection: one good peak per flux slice --------------
        good = np.asarray(cloud["good"], dtype=bool)
        flux_index = np.asarray(cloud["peak_flux_index"], dtype=int)
        amp = np.abs(np.asarray(cloud["peak_amplitude"], dtype=float))
        freq_hz = np.asarray(cloud["peak_full_freq"], dtype=float)

        sel: list[int] = []
        for k in np.unique(flux_index[good]):
            candidates = np.flatnonzero(good & (flux_index == k) & np.isfinite(freq_hz))
            if candidates.size == 0:
                continue
            score = amp[candidates] if branch == "strongest" else freq_hz[candidates]
            sel.append(int(candidates[int(np.argmax(score))]))
        sel_arr = np.asarray(sel, dtype=int)

        arch: Dict[str, Any] = {
            "branch": branch, "ec_ghz": ec_ghz, "n_selected": int(sel_arr.size),
            "n_used": 0, "success": False,
        }
        if sel_arr.size >= 5:
            x = np.asarray(cloud["peak_flux"], dtype=float)[sel_arr]
            y_ghz = freq_hz[sel_arr] * 1e-9
            used = np.ones(x.size, dtype=bool)

            def _fit(mask: np.ndarray):
                fitter = FitTransmonFrequencyFlux(data=y_ghz[mask], x=x[mask], Ec_design=ec_ghz)
                fitter.guess()
                if fit_d:
                    fitter.params["d"].set(vary=True, min=0.0, max=1.0)
                return fitter, fitter.fit()

            fitter, result = _fit(used)
            # one robust refit: drop MAD-outlier residuals (spurious lines that
            # survived selection), then fit again on the kept points
            resid = y_ghz - fitter.model.eval(result.params, x=x)
            med = float(np.median(resid[used]))
            mad = float(np.median(np.abs(resid[used] - med)))
            if mad > 0:
                keep = np.abs(resid - med) <= arch_n_sigma * 1.4826 * mad
                if keep.sum() >= 5 and keep.sum() < used.sum():
                    used = keep
                    fitter, result = _fit(used)

            offset = float(result.params["offset"].value)
            period = float(result.params["period"].value)
            ej_sum = float(result.params["Ej_sum"].value)
            d = float(result.params["d"].value)
            f01_max_ghz = float(np.sqrt(8.0 * ec_ghz * ej_sum) - ec_ghz)
            flux_lo = float(np.min(cloud["flux_bias"]))
            flux_hi = float(np.max(cloud["flux_bias"]))
            span = flux_hi - flux_lo
            arch.update(
                {
                    "sweet_spot_flux": offset,
                    "flux_period": period,
                    "ej_sum_ghz": ej_sum,
                    "asymmetry_d": d,
                    "f01_max_hz": f01_max_ghz * 1e9,
                    "offset_stderr": float(result.params["offset"].stderr or np.nan),
                    "period_stderr": float(result.params["period"].stderr or np.nan),
                    "ej_sum_stderr_ghz": float(result.params["Ej_sum"].stderr or np.nan),
                    "redchi": float(result.redchi),
                    "n_used": int(used.sum()),
                    "sel_flux": x, "sel_freq_hz": freq_hz[sel_arr], "sel_used": used,
                    # physical gates: converged, positive scales, sweet spot not
                    # absurdly far outside the swept window
                    "success": bool(result.success)
                    and period > 0
                    and ej_sum > 0
                    and (flux_lo - span) < offset < (flux_hi + span),
                }
            )
            fit_flux = np.linspace(flux_lo, flux_hi, 201)
            arch["fit_flux"] = fit_flux
            arch["fit_freq_hz"] = fitter.model.eval(result.params, x=fit_flux) * 1e9

        return {"point_cloud": cloud, "arch": arch}

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        cloud, arch = results["point_cloud"], results["arch"]
        meta = {
            "n_flux": int(cloud["n_flux"]),
            "n_peaks": int(cloud["n_peaks"]),
            "n_good": int(cloud["n_good"]),
            "branch": arch["branch"],
            "ec_ghz": arch["ec_ghz"],
            "n_selected": int(arch["n_selected"]),
            "n_used": int(arch["n_used"]),
            "arch_success": bool(arch["success"]),
        }
        for key in ("sweet_spot_flux", "flux_period", "ej_sum_ghz", "asymmetry_d",
                    "f01_max_hz", "offset_stderr", "period_stderr",
                    "ej_sum_stderr_ghz", "redchi"):
            if key in arch:
                meta[key] = float(arch[key])
        return meta

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Stage-1 plot data (map + point-cloud) merged with the selected branch
        points and the dense arch curve; arch scalars as attrs."""
        cloud, arch = results["point_cloud"], results["arch"]
        merged = flux_cloud_plotdata(cloud)
        merged.attrs["arch_success"] = int(bool(arch["success"]))
        merged.attrs["branch"] = arch["branch"]
        if arch["n_used"]:
            merged = merged.assign_coords(
                sel=("sel", np.arange(len(arch["sel_flux"]))),
                fit_flux=("fit_flux", np.asarray(arch["fit_flux"], dtype=float)),
            )
            merged["sel_flux"] = ("sel", np.asarray(arch["sel_flux"], dtype=float))
            merged["sel_freq_hz"] = ("sel", np.asarray(arch["sel_freq_hz"], dtype=float))
            merged["sel_used"] = ("sel", np.asarray(arch["sel_used"], dtype=bool))
            merged["fit_freq_hz"] = ("fit_flux", np.asarray(arch["fit_freq_hz"], dtype=float))
            for key in ("sweet_spot_flux", "flux_period", "ej_sum_ghz", "f01_max_hz"):
                merged.attrs[key] = float(arch[key])
        return merged

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
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"qubit_flux_arch": plot_arch(plot_data)}
