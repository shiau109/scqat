"""GMM method: the trained 2-D Gaussian mixture per sweep point (the default).

Every sweep point is handed to the family-shared reduction
:func:`scqat.tools.discriminate.discriminate_states`, which trains one mixture
on that point's clouds and assigns every shot. This is the only method that
knows a blob WIDTH, so it is the only one that can report a std, an SNR, an
outlier probability, a normalised residue and a confusion matrix — and hence a
readout FIDELITY, which is what its best point maximises.
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

from scqat.tools.discriminate import DISCRIMINATE_KNOBS, discriminate_states

from .base import ReadoutFidelityMethod


class GmmMethod(ReadoutFidelityMethod):
    """Per-slice Gaussian-mixture discrimination; best point = max fidelity."""

    name = "gmm"
    requires_shots = True
    metric = "fidelity"
    slice_keys = ("std", "mean", "p_outlier", "norm_res", "gaussian_norms",
                  "direct_counts")
    #: ``outliers_threshold`` is the power subclass's selection constraint; it
    #: rides here because it is only answerable from this method's p_outlier.
    knobs = DISCRIMINATE_KNOBS | {"outliers_threshold"}

    def reduce(self, I: np.ndarray, Q: np.ndarray, **knobs) -> Dict[str, Any]:
        sd_knobs = {k: v for k, v in knobs.items() if k in DISCRIMINATE_KNOBS}
        res = discriminate_states(I, Q, **sd_knobs)
        trained = res["trained_paras"]
        return {
            "std": float(trained["std"]),
            "mean": np.asarray(trained["mean"], dtype=float),
            "p_outlier": np.asarray(res["outlier_probability"], dtype=float),
            "norm_res": np.asarray(res["norm_res"], dtype=float),
            "gaussian_norms": np.asarray(res["gaussian_norms"], dtype=float),
            "direct_counts": np.asarray(res["direct_counts"], dtype=float),
        }

    def derive(self, results: Dict[str, Any]) -> None:
        """The two curves the mixture makes possible: the correct-assignment
        fidelity and the separation in units of one blob's width."""
        results["fidelity"] = _fidelity_curve(results.get("direct_counts"))
        results["snr"] = _snr_curve(results.get("separation"), results.get("std"))

    def metric_ok(self, value: float, floor: float) -> bool:
        return bool(np.isfinite(value) and value >= floor)


def _fidelity_curve(direct_counts: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-sweep readout fidelity: mean of the confusion-matrix diagonal
    ``direct_counts[s, k, k]`` over the available states. Failed slices (all
    NaN) yield NaN."""
    if direct_counts is None:
        return None
    _, n_state, n_count = direct_counts.shape
    n = min(n_state, n_count)
    diag = direct_counts[:, np.arange(n), np.arange(n)]  # (S, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmean(diag, axis=1)


def _snr_curve(separation: Optional[np.ndarray],
               std: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Per-sweep readout SNR: the centre separation in units of one blob's GMM
    std. ``None`` when either curve is unavailable; failed slices yield NaN."""
    if separation is None or std is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(separation, dtype=float) / np.asarray(std, dtype=float)
