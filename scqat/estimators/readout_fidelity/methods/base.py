"""Method strategy base for the readout_fidelity estimator.

A method owns everything that differs between two ways of turning ONE sweep
point's prepared-state I/Q block into a summary: which knobs it accepts, what
per-slice keys it produces, which curves it derives from them, and which curve
the best sweep point maximises. The estimator owns the loop, the stacking, the
selection mechanics and the artifacts — those are identical for every method.

The two-tier result contract (repo rule "Multi-method estimators"): the COMMON
keys every method produces are ``sweep_coord`` / ``sweep_values`` / ``mean`` /
``separation`` / ``failed`` / ``method`` / ``metric`` / ``best_index`` /
``best_sweep_value`` / ``best_metric`` / ``success``; everything a method adds
through :attr:`slice_keys` or :meth:`derive` is method-owned and consumed
downstream only via ``if key in results``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, Tuple

import numpy as np


class ReadoutFidelityMethod(ABC):
    """One approach to summarising a sweep point's prepared-state I/Q clouds."""

    #: registry key; stamped into results, metadata and ``plot_data.attrs``
    name: str = ""
    #: True when the method needs every shot (a ``shot_idx`` axis); False when a
    #: single averaged point per prepared state is enough
    requires_shots: bool = True
    #: the per-sweep results key whose MAXIMUM is the answer
    metric: str = ""
    #: per-slice keys :meth:`reduce` returns; the estimator stacks each over the
    #: sweep (failed slices filled with NaN of the right shape)
    slice_keys: Tuple[str, ...] = ()
    #: the kwargs this method accepts — validated ONCE, before the slice loop,
    #: so a typo'd knob dies loudly instead of inside a per-slice try/except
    knobs: FrozenSet[str] = frozenset()

    @abstractmethod
    def reduce(self, I: np.ndarray, Q: np.ndarray, **knobs) -> Dict[str, Any]:
        """Summarise one sweep point. ``I``/``Q`` are ``(n_prepared_state,
        n_point)`` (one row per prepared state; one column per shot, or a single
        column for averaged data). Returns exactly :attr:`slice_keys`."""

    def derive(self, results: Dict[str, Any]) -> None:
        """Add the method's derived per-sweep curves to ``results`` in place
        (the stacked slice keys and the common ``separation`` are already
        there). Base: nothing to derive."""

    @abstractmethod
    def metric_ok(self, value: float, floor: float) -> bool:
        """Whether the selected point's metric value counts as success.
        ``floor`` is the estimator's ``fidelity_floor``; only a method whose
        metric is a fidelity consults it."""
