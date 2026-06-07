"""
Resonator Spectroscopy vs Flux — Full Analysis (composite)
==========================================================
Two-stage resonator-flux analysis that chains two existing analyzers and owns the
**canonical combined figure**, so every consumer (e.g. QM / QBLOX control repos)
reconstructs an *identical* plot from the saved plot data instead of
re-implementing plotting:

  1. :class:`~scqat.protocols.resonator_spectroscopy_vs_flux.ResonatorSpectroscopyVsFluxAnalyzer`
     collapses the 2-D ``(flux_bias, detuning)`` map into a 1-D resonator
     ``center_frequency(flux)`` trace (single inverted-Lorentzian dip fit per flux
     slice), then
  2. :class:`~scqat.protocols.resonator_flux_dispersion.ResonatorFluxDispersionAnalyzer`
     fits that trace with the full flux-tunable-transmon dispersive model to obtain
     the sweet spot, flux period (``dv_phi0``), bare resonator ``f_r0`` and the
     (conditional) coupling ``g``.

Turning the dispersive outputs into instrument/state quantities (idle offset,
min-frequency point, ``phi0`` in current, QUAM state writes) is **out of scope** —
that belongs to the calling repo, which also owns the raw-file→Dataset adapter.

``results`` is nested as ``{"vs_flux": <stage-1 results>, "dispersion":
<stage-2 results>}`` so the two stages' keys (e.g. the per-flux ``success`` array
vs. the dispersive scalar ``success``) stay unambiguous.

Expected xarray.Dataset contract
--------------------------------
Identical to the stage-1 analyzer (the ``qubit`` dimension already removed):

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute readout frequency (Hz). Optional; when
                  present the centre trace and dispersive fit are reported in
                  absolute frequency.
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.resonator_spectroscopy_vs_flux import ResonatorSpectroscopyVsFluxAnalyzer
from scqat.protocols.resonator_flux_dispersion import ResonatorFluxDispersionAnalyzer

from .visualization import plot_combined


# kwargs consumed by the dispersive (stage-2) fit; everything else flows to stage 1.
_DISPERSION_KWARGS = ("f_q_max", "fit_f_q_max")


class ResonatorSpectroscopyFluxAnalyzer(BaseAnalyzer):
    """Composite resonator-vs-flux analyzer: dip-vs-flux trace + dispersive fit.

    The combined figure (raw ``|IQ|`` map + per-flux fitted centres + dispersive
    fit curve + sweet spot) is built **only** from the merged ``plot_data``, so it
    is reconstructable by any consumer without rerunning the analysis.
    """

    protocol_name = "resonator_spectroscopy_flux"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        # The composite's input contract is exactly stage 1's.
        ResonatorSpectroscopyVsFluxAnalyzer()._check_data(dataset)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the vs-flux dip fit, then the dispersive flux-dependence fit.

        Keyword arguments
        -----------------
        f_q_max, fit_f_q_max
            Forwarded to the dispersive (stage-2) fit (see
            :class:`ResonatorFluxDispersionAnalyzer`).
        n_sigma, baseline_order, baseline_quantile, ...
            Forwarded to the vs-flux (stage-1) fit.

        Returns
        -------
        dict
            ``{"vs_flux": <stage-1 results>, "dispersion": <stage-2 results>}``.
        """
        disp_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in _DISPERSION_KWARGS}

        vs = ResonatorSpectroscopyVsFluxAnalyzer().extract_parameters(dataset, **kwargs)

        # Build the centre-frequency(flux) trace for the dispersive fit: prefer the
        # absolute frequency when available, and feed only the kept (good) points.
        center = np.asarray(vs.get("center_full_freq", vs["center_detuning"]), dtype=float)
        trace = xr.Dataset(
            {
                "center_freq": ("flux_bias", center),
                "success": ("flux_bias", np.asarray(vs["good"], dtype=bool)),
            },
            coords={"flux_bias": np.asarray(vs["flux_bias"], dtype=float)},
        )
        disp = ResonatorFluxDispersionAnalyzer().extract_parameters(trace, **disp_kwargs)

        return {"vs_flux": vs, "dispersion": disp}

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Project the key scalars from both stages for the persisted metadata."""
        vs = results["vs_flux"]
        disp = results["dispersion"]
        return {
            "n_flux": int(vs["n_flux"]),
            "n_good": int(vs["n_good"]),
            "n_outlier": int(vs["n_outlier"]),
            "sweet_spot_flux": float(disp["sweet_spot_flux"]),
            "sweet_spot_freq": float(disp["sweet_spot_freq"]),
            "dv_phi0": float(disp["dv_phi0"]),
            "f_r0": float(disp["f_r0"]),
            "g": float(disp["g"]),
            "f_q_max": float(disp["f_q_max"]),
            "f_q_max_fixed": bool(disp["f_q_max_fixed"]),
            "dispersion_success": bool(disp["success"]),
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Merge the two sub-analyzers' plot data into one self-sufficient Dataset.

        Stage-1 plot data (2-D map + per-flux centres + good/outlier masks +
        ``full_freq``) is the base; the dispersive fit curve is added on its own
        dense ``fit_flux`` axis and the dispersive scalars are stored as attrs.
        Reusing each analyzer's ``build_plot_data`` keeps their plot-data schema the
        single source of truth.
        """
        vs = results["vs_flux"]
        disp = results["dispersion"]

        vs_pd = ResonatorSpectroscopyVsFluxAnalyzer().build_plot_data(dataset, vs)
        disp_pd = ResonatorFluxDispersionAnalyzer().build_plot_data(None, disp)

        merged = vs_pd.copy()
        # Dispersive fit curve on its own dense flux axis (independent dim).
        merged = merged.assign_coords(
            fit_flux=("fit_flux", disp_pd.coords["fit_flux"].values.astype(float))
        )
        merged["fit_freq"] = ("fit_flux", disp_pd["fit_freq"].values.astype(float))
        # Dispersive scalars as attrs; rename 'success' to disambiguate from the
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
