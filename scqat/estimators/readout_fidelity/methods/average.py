"""Average method: each prepared state's centre IS the average of its I/Q.

Nothing is fitted. The measurement is trusted to have prepared the state it
says it prepared, so the mean of prepared-state k's I/Q is taken as blob k's
centre — which is exactly what an FPGA-AVERAGED acquisition returns (one point
per prepared state, no shots at all). It works identically on per-shot data by
averaging over the shot axis first.

What that buys: it is fast, and it cannot fail the way a mixture fit fails on
merged or badly-seeded blobs. What it costs: with no fitted width there is no
std, no SNR, no outlier probability, no residue, no confusion matrix and hence
NO FIDELITY — so the best point is the one with the largest centre SEPARATION.
"""

from typing import Any, Dict

import numpy as np

from .base import ReadoutFidelityMethod


class AverageMethod(ReadoutFidelityMethod):
    """Centres by averaging; best point = max centre separation."""

    name = "average"
    requires_shots = False
    metric = "separation"
    slice_keys = ("mean",)
    knobs = frozenset()

    def reduce(self, I: np.ndarray, Q: np.ndarray, **knobs) -> Dict[str, Any]:
        # (n_prepared_state, n_point) -> one (I, Q) centre per prepared state.
        # n_point is the shot count on per-shot data and 1 on averaged data.
        return {"mean": np.column_stack([np.mean(I, axis=1), np.mean(Q, axis=1)])}

    def metric_ok(self, value: float, floor: float) -> bool:
        # ``floor`` is a fidelity floor and means nothing here: separation is in
        # raw IQ units, whose scale is the ADC's, not the physics'. The only
        # answerable question is whether the centres are apart at all.
        return bool(np.isfinite(value) and value > 0)
