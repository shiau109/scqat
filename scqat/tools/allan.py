"""Overlapping Allan deviation of a uniformly-sampled parameter trace.

The stability statistic for a tracked quantity (T1, a frequency estimate)
versus averaging time: sigma_y(tau) = sqrt( <(mean_next(tau) - mean_prev(tau))^2> / 2 )
over all overlapping windows of length tau = m * dt. Flat sigma_y(tau) means
white estimation noise averages down; a floor or rise marks the timescale where
real drift takes over — the answer to "how long may I average this quantity?".

Deliberately plain numpy (no ``allantools`` dependency — adding one is a
release-coupling decision). Discrete approximation of the overlapping
estimator; averaging times are log-spaced up to half the record.

Degraded behavior: fewer than 4 samples (or fewer than 2 usable averaging
times) returns two empty arrays, never raises. Non-finite samples propagate
into every window that touches them — drop them before calling.
"""

from typing import Tuple

import numpy as np


def overlapping_allan_deviation(series: np.ndarray, dt_s: float, *,
                                max_points: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of ``series`` sampled every ``dt_s`` seconds.

    Parameters
    ----------
    series : array-like
        The trace, uniformly spaced in time, same units throughout.
    dt_s : float
        Sample period in seconds (positive, finite).
    max_points : int
        Cap on the number of averaging times evaluated (log-spaced).

    Returns
    -------
    (tau_s, adev) : tuple of ndarray
        Averaging times (s) and the Allan deviation at each (series units).
        Both empty when the record is too short.
    """
    if dt_s is None or not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(
            f"dt_s must be a positive finite sample period in seconds, got {dt_s!r}"
        )
    series = np.asarray(series, dtype=float).ravel()
    n = series.size
    if n < 4:
        return np.array([]), np.array([])
    max_m = min(n // 2, int(max_points))
    if max_m < 2:
        return np.array([]), np.array([])

    ms = np.unique(np.geomspace(1, n // 2, num=max_m, dtype=int))
    taus, adevs = [], []
    for m in ms:
        means = np.convolve(series, np.ones(m) / m, mode="valid")
        if means.size <= m:
            continue
        deltas = means[m:] - means[:-m]
        taus.append(m * dt_s)
        adevs.append(float(np.sqrt(0.5 * np.mean(deltas ** 2))))
    return np.asarray(taus, dtype=float), np.asarray(adevs, dtype=float)
