"""Raw joint-state-population maps for the N-swap x flux-amplitude sweep.

Record-only: this estimator does NOT fit the swap. It draws the four joint
two-qubit populations over the flux-amplitude x swap-count grid so an operator
can see how a small swap-amplitude miscalibration amplifies with the number of
swaps, and it extracts a small self-describing summary of where the transfer
peaks. The SUCCESS / ``min_transfer`` verdict stays in SCQO.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` — dims ``(joint_state, flux_amp_v, swap_count)``
           in any order; ``joint_state`` labels ``"00"/"01"/"10"/"11"``
           (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``flux_amp_v`` (V) / ``swap_count`` (dimensionless N)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) — selects the transfer partner
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import pair_swap_plot_data, summarize_pair_swap
from scqat.estimators.qc_n_swap_amp.visualization import plot_qc_n_swap_amp

AXIS0 = "flux_amp_v"
AXIS1 = "swap_count"


class QcNSwapAmpEstimator(BaseEstimator):
    """Draw the joint state populations of the N-swap amplitude map (record-only)."""

    estimator_name = "qc_n_swap_amp"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "qc_n_swap_amp estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_n_swap_amp estimator requires a {axis!r} coordinate"
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
        return render_figures(
            {"qc_n_swap_amp": lambda: plot_qc_n_swap_amp(plot_data)},
            label=self.estimator_name,
        )
