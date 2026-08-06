"""Figure for the ``qc_n_swap_amp`` raw-population maps.

The N-swap amplitude map's figure is the shared 2x2 joint-population map; the
alias keeps the per-estimator import path stable (and a natural place to diverge
later) while the drawing lives once in ``_pair_swap_maps``. Draws from
``plot_data`` only.
"""

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_qc_n_swap_amp"]

plot_qc_n_swap_amp = plot_pair_swap_map
