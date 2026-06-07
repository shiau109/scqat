"""Robust statistics helpers shared across protocols."""

import numpy as np


def mad_outliers(values: np.ndarray, valid: np.ndarray, n_sigma: float, rel_floor: float = 0.25):
    """Robust (median / MAD) outlier flagging among the ``valid`` points.

    A point is an outlier when it deviates from the median **both** by more than
    ``n_sigma * 1.4826 * MAD`` (1.4826 makes the scaled MAD a consistent estimator
    of the standard deviation for normal data) **and** by more than ``rel_floor``
    times the median. The relative floor keeps an almost-noise-free trace (MAD
    near zero) from flagging points that differ only by a negligible amount.
    With fewer than 4 valid points, or a zero MAD, nothing is flagged.

    Parameters
    ----------
    values : np.ndarray
        The quantity to test (e.g. per-flux fitted width or amplitude).
    valid : np.ndarray
        Boolean mask of the points eligible to be judged / flagged.
    n_sigma : float
        Robust-sigma threshold.
    rel_floor : float, optional
        Minimum relative deviation from the median required to flag (default 0.25).

    Returns
    -------
    (outlier_mask, median, mad)
        ``outlier_mask`` has the shape of ``values`` and is True only on flagged
        valid points.
    """
    values = np.asarray(values, dtype=float)
    finite_valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    outlier = np.zeros(values.shape, dtype=bool)
    v = values[finite_valid]
    if v.size < 4:
        return outlier, float("nan"), float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 0:
        return outlier, med, mad
    dev = np.abs(values - med)
    z = dev / robust_sigma
    rel = dev / max(abs(med), 1e-300)
    outlier = finite_valid & (z > n_sigma) & (rel > rel_floor)
    return outlier, med, mad
