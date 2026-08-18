"""
Resonator Spectroscopy vs Flux — Full Analysis
==============================================
The ONE estimator for the resonator-vs-flux experiment. Two internal stages
(each also importable as a plain function by control repos):

  1. :func:`.dips.track_dips` collapses the 2-D ``(flux_bias, detuning)`` map
     into a 1-D resonator ``center_frequency(flux)`` trace (per-slice dip fit
     via the family-shared :func:`scqat.tools.dip_fit.fit_dip` — ``dip_method``
     selects ``lorentzian``/``circle`` — with edge-margin + MAD acceptance),
     then
  2. :func:`.trace_fit.fit_flux_trace` fits that trace with the selected
     flux-dependence model (``method="dispersive"`` — the full
     flux-tunable-transmon model — or the model-light ``"sine"``; strategy
     objects in ``methods/``) to obtain the sweet-spot point, flux period
     (``dv_phi0``), and, for the dispersive method, the bare resonator ``f_r0``
     and (conditional) coupling ``g``.

The estimator owns the **canonical combined figure**, so every consumer (e.g.
QM / QBLOX control repos) reconstructs an *identical* plot from the saved plot
data instead of re-implementing plotting. Turning the fit outputs into
instrument/state quantities (idle offset, min-frequency point, ``phi0`` in
current, QUAM state writes) is **out of scope** — that belongs to the calling
repo, which also owns the raw-file→Dataset adapter.

``results`` is nested as ``{"vs_flux": <stage-1 results>, "dispersion":
<stage-2 results>}`` so the two stages' keys (e.g. the per-flux ``success`` array
vs. the flux-model scalar ``success``) stay unambiguous.

Expected xarray.Dataset contract
--------------------------------
See :mod:`.dips` (the ``qubit`` dimension already removed):

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute readout frequency (Hz). Optional; when
                  present the centre trace and flux-model fit are reported in
                  absolute frequency.
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator

from .dips import check_dataset, dips_plot_data, track_dips
from .methods import METHODS
from .trace_fit import fit_flux_trace
from .visualization import plot_combined


# kwargs consumed by the flux-model (stage-2) fit; everything else (n_sigma,
# edge_margin_frac, per-slice dip kwargs) flows to stage 1.
_DISPERSION_KWARGS = ("method", "f_q_max", "fit_f_q_max", "ec", "g_init",
                      "f_r0", "fit_f_r0")


class ResonatorSpectroscopyFluxEstimator(BaseEstimator):
    """Composite resonator-vs-flux estimator: dip-vs-flux trace + flux-model fit.

    The combined figure (raw ``|IQ|`` map + per-flux fitted centres + flux-model
    fit curve + sweet-spot point) is built **only** from the merged
    ``plot_data``, so it is reconstructable by any consumer without rerunning the
    analysis.
    """

    estimator_name = "resonator_spectroscopy_flux"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        check_dataset(dataset)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the vs-flux dip fit, then the selected flux-dependence fit
        (dispersive or sine).

        Keyword arguments (the estimator's FLAT, fully-owned surface)
        -------------------------------------------------------------
        method, f_q_max, fit_f_q_max
            The flux-model (stage-2) fit — ``method`` selects ``"dispersive"``
            (default) or ``"sine"`` (see :mod:`.trace_fit`).
        dip_method, baseline_order, delay
            The per-slice dip fit (see :func:`scqat.tools.dip_fit.fit_dip`):
            ``dip_method`` selects ``"lorentzian"`` (default) or ``"circle"``;
            the rest are that method's knobs.
        n_sigma, edge_margin_frac, min_dip_depth, local_window_frac
            Stage-1 acceptance tunables (see :func:`.dips.track_dips`).

        Unknown kwargs raise ValueError before any fitting.

        Returns
        -------
        dict
            ``{"vs_flux": <stage-1 results>, "dispersion": <stage-2 results>}``.
        """
        disp_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _DISPERSION_KWARGS}
        # Fail loudly up front on a bad flux-model method — before the (slower)
        # stage-1 slice loop runs.
        model = disp_kwargs.get("method", "dispersive")
        if model not in METHODS:
            raise ValueError(f"Unknown method {model!r}; available: {sorted(METHODS)}")

        vs = track_dips(dataset, **kwargs)

        # Build the centre-frequency(flux) trace for the flux-model fit: prefer
        # the absolute frequency when available, and feed only the kept (good)
        # points.
        center = np.asarray(vs.get("center_full_freq", vs["center_detuning"]), dtype=float)
        trace = xr.Dataset(
            {
                "center_freq": ("flux_bias", center),
                "success": ("flux_bias", np.asarray(vs["good"], dtype=bool)),
            },
            coords={"flux_bias": np.asarray(vs["flux_bias"], dtype=float)},
        )
        disp = fit_flux_trace(trace, **disp_kwargs)

        return {"vs_flux": vs, "dispersion": disp}

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Project the key scalars from both stages for the persisted metadata.

        The COMMON flux-model keys are always present; the dispersive-only extras
        (``f_r0``/``g``/``f_q_max``) are included only when the dispersive method
        produced them (the sine method has no such physics)."""
        vs = results["vs_flux"]
        disp = results["dispersion"]
        meta: Dict[str, Any] = {
            "n_flux": int(vs["n_flux"]),
            "n_good": int(vs["n_good"]),
            "n_outlier": int(vs["n_outlier"]),
            "method": disp.get("method", "dispersive"),
            "dip_method": str(vs["dip_method"]),
            "sweet_spot_flux": float(disp["sweet_spot_flux"]),
            "sweet_spot_res": float(disp["sweet_spot_res"]),
            "sweet_spot_low_flux": float(disp["sweet_spot_low_flux"]),
            "sweet_spot_low_res": float(disp["sweet_spot_low_res"]),
            "dv_phi0": float(disp["dv_phi0"]),
            "dispersion_success": bool(disp["success"]),
        }
        for k in ("f_r0", "g", "f_q_max", "ec"):
            if k in disp:
                meta[k] = float(disp[k])
        for k in ("f_q_max_fixed", "f_r0_fixed"):
            if k in disp:
                meta[k] = bool(disp[k])
        return meta

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Merge the two stages' plot data into one self-sufficient Dataset.

        Stage-1 plot data (2-D map + per-flux centres + good/outlier masks +
        ``full_freq``) is the base; the flux-model fit curve is added on its own
        dense ``fit_flux`` axis and the model scalars are stored as attrs. The
        stage helpers (:func:`.dips.dips_plot_data` / each method's
        ``build_plot_data``) keep the plot-data schema in one place.
        """
        vs = results["vs_flux"]
        disp = results["dispersion"]

        vs_pd = dips_plot_data(vs)
        disp_pd = METHODS[disp["method"]].build_plot_data(disp)

        merged = vs_pd.copy()
        # Flux-model fit curve on its own dense flux axis (independent dim).
        merged = merged.assign_coords(
            fit_flux=("fit_flux", disp_pd.coords["fit_flux"].values.astype(float))
        )
        merged["fit_freq"] = ("fit_flux", disp_pd["fit_freq"].values.astype(float))
        # Flux-model scalars as attrs; rename 'success' to disambiguate from the
        # per-flux 'success' variable carried over from stage 1.
        for k, v in disp_pd.attrs.items():
            merged.attrs["dispersion_success" if k == "success" else k] = v
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
        """One combined figure, drawn entirely from the merged plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"resonator_spectroscopy_flux": plot_combined(plot_data)}
