"""Raw joint-state-population maps for the fixed-time swap flux map.

The fixed-duration sibling of ``pair_swap_chevron``: the same four joint
populations (``p_high`` / ``p_low`` / ``p_ee`` / ``p_gg``), drawn here over the
coupler-flux x member-flux grid so the swap **spot** and how the coupler bias
moves it are visible. Record-only, extracting only where the transfer peaks; the
SUCCESS / ``min_transfer`` verdict stays in SCQO.

Dataset contract:
  vars   : ``p_high`` / ``p_low`` / ``p_ee``  (``p_gg`` optional; derived if absent)
  coords : ``qubit_flux_v`` (V) / ``coupler_flux_v`` (V)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) — selects the transfer partner
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators._pair_swap_maps import pair_swap_plot_data, summarize_pair_swap
from scqat.estimators.pair_swap_flux_map.visualization import plot_pair_swap_flux_map

AXIS0 = "qubit_flux_v"
AXIS1 = "coupler_flux_v"


class PairSwapFluxMapEstimator(BaseEstimator):
    """Draw the joint state populations of the swap flux map (record-only)."""

    estimator_name = "pair_swap_flux_map"

    def _check_data(self, dataset: xr.Dataset) -> None:
        missing = [v for v in ("p_high", "p_low", "p_ee") if v not in dataset.data_vars]
        if missing:
            raise ValueError(
                f"pair_swap_flux_map estimator requires joint-population variables "
                f"p_high, p_low, p_ee (missing: {missing})"
            )
        for axis in (AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"pair_swap_flux_map estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        return summarize_pair_swap(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None, **kwargs
    ) -> Optional[xr.Dataset]:
        return pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(
                dataset, results,
                drive_side=kwargs.get("drive_side", "low"),
                flux_side=kwargs.get("flux_side"),
                high_name=kwargs.get("high_name"),
                low_name=kwargs.get("low_name"),
            )
        return {"pair_swap_flux_map": plot_pair_swap_flux_map(plot_data)}
