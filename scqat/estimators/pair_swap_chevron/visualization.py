"""Figure for the ``pair_swap_chevron`` raw-population maps.

The chevron's figure is the shared 2x2 joint-population map; the alias keeps the
per-estimator import path stable (and a natural place to diverge later) while the
drawing lives once in ``_pair_swap_maps``. Draws from ``plot_data`` only.
"""

from scqat.estimators._pair_swap_maps import plot_pair_swap_map

__all__ = ["plot_pair_swap_chevron"]

plot_pair_swap_chevron = plot_pair_swap_map
