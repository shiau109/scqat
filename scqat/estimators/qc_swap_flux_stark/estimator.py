"""Raw joint-state-population maps for the fixed-N flux x AC-Stark sweep.

Record-only: this estimator does NOT fit the swap. It draws the four joint
two-qubit populations over the flux-amplitude x stark-amplitude grid, at a FIXED
number of swaps, so an operator can see the two knobs together — the control
flux brings the members onto resonance while the Stark tone nulls the phase the
members accumulate between swaps, and those two are coupled (the phase drags the
transfer peak off zero detuning). A one-knob-at-a-time scan lands on a ridge of
that 2-D surface; this map shows the surface.

The swap count is NOT an axis here, which is the whole difference from
``qc_n_swap_amp`` (flux x N) and ``qc_n_stark_amp`` (stark x N): with N frozen
there is nothing to fit along, so the reading is the map itself plus a small
self-describing summary of where the transfer peaks. The SUCCESS /
``min_transfer`` verdict stays in SCQO.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` — dims ``(joint_state, flux_amp_v, stark_amp)``
           in any order; ``joint_state`` labels ``"00"/"01"/"10"/"11"``
           (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``flux_amp_v`` (V) / ``stark_amp`` (dimensionless
           factor of the stark operation's baked amplitude)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) — selects the transfer partner
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import pair_swap_plot_data, summarize_pair_swap
from scqat.estimators.qc_swap_flux_stark.visualization import plot_qc_swap_flux_stark

AXIS0 = "flux_amp_v"
AXIS1 = "stark_amp"


class QcSwapFluxStarkEstimator(BaseEstimator):
    """Draw the joint state populations of the fixed-N flux x stark map (record-only)."""

    estimator_name = "qc_swap_flux_stark"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "qc_swap_flux_stark estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_swap_flux_stark estimator requires a {axis!r} coordinate"
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
            {"qc_swap_flux_stark": lambda: plot_qc_swap_flux_stark(plot_data)},
            label=self.estimator_name,
        )
