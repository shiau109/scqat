"""
Qubit Spectroscopy vs Flux Estimator
===================================
Extract the qubit transition peaks from a 2-D qubit-spectroscopy-vs-flux map by
fitting **flux-by-flux**, keeping **all** peaks found at each flux (there can be
two or more transitions at the same flux — e.g. the 0->1 line and the two-photon
0->2/2 line, or, with a single xy drive source, another qubit's line showing up).

The heavy lifting is the stage function :func:`.peaks.track_flux_peaks` (the
family-shared per-trace fit :func:`scqat.tools.peak_fit.fit_peaks`, driven
per-flux by the generic map tracker :func:`scqat.tools.peak_map.track_peaks`);
this estimator only forwards its flat kwarg surface and owns the artifacts.
The result is a **point-cloud** of peaks ``(flux, frequency, fwhm, amplitude)``
rather than a single ``frequency(flux)`` value — assigning points to individual
transition branches belongs to the downstream flux-dependence fit.

Cleaning: a peak is kept (``good``) when its centre lies strictly inside the
swept detuning window and its ``fwhm`` / ``|amplitude|`` are not robust
(median/MAD) outliers across the pooled set of detected peaks.

Expected xarray.Dataset contract — see :mod:`.peaks` (the ``qubit`` dimension
already removed).
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators.qubit_spectroscopy_flux.peaks import (
    check_flux_dataset,
    flux_cloud_plotdata,
    track_flux_peaks,
)
from scqat.estimators.qubit_spectroscopy_flux.visualization import plot_flux_map


class QubitSpectroscopyFluxEstimator(BaseEstimator):
    """Fit the qubit peak(s) at every flux bias and report them as a point-cloud.

    The result dict reports, per detected peak, the ``peak_flux`` it was found at,
    its ``peak_detuning`` (and absolute ``peak_full_freq`` when available),
    ``peak_fwhm``, ``peak_amplitude``, the strict ``in_window`` mask, the
    ``outlier`` mask (robust width/amplitude rejection) and the surviving ``good``
    mask, alongside the 2-D signal ``amplitude_map`` kept for plotting.
    """

    estimator_name = "qubit_spectroscopy_flux"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        check_flux_dataset(dataset)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit and collect every qubit peak in every ``flux_bias`` slice.

        Thin wrapper over :func:`.peaks.track_flux_peaks` — see there for the
        full flat kwarg surface (``n_sigma``, ``signal_var``, ``ref_scope``
        [per_slice | global radial reference], and the
        :func:`scqat.tools.peak_fit.fit_peaks` knobs such as ``prominence`` and
        ``max_peaks``; unknown names raise before any per-slice fit) and the
        result contract. By default each flux slice is capped to its 4 most
        prominent peaks; pass ``max_peaks=None`` to keep every peak above the
        prominence threshold.
        """
        return track_flux_peaks(dataset, **kwargs)

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the peak point-cloud + stats; drop the bulky 2-D maps and axes."""
        drop = {"amplitude_map", "reduced_map", "iq_i_map", "iq_q_map",
                "detuning", "full_freq"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 2-D signal map and the peak point-cloud (with good/outlier
        masks) into one self-sufficient Dataset."""
        return flux_cloud_plotdata(results)

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
        """The 2-D signal map over (flux, frequency) with every kept qubit peak
        overlaid and outliers marked, plus the shared IQ-plane panel (raw cloud
        + per-slice references) when the input carried complex IQ."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        figs = {"qubit_spectroscopy_flux": plot_flux_map(plot_data)}
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
