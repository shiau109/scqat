"""One-sided Welch PSD of a real-valued time series.

The generic sibling of :func:`scqat.tools.telegraph_psd.telegraph_spectrum`,
which is documented (and framed) for 0/1 telegraph series. This one takes any
uniformly-sampled float series — a T1 trace, a frequency-estimate trace — and
returns the log-log-ready spectrum of its fluctuations. No fit is attached: a
parameter trace has no single reference model, so the fitting (if any) belongs
to the caller.

Result contract
---------------
``timeseries_psd(series, dt_s, ...)`` returns ``(freq_hz, psd)``:

* ``freq_hz`` — one-sided frequency axis (Hz), DC bin dropped.
* ``psd`` — power spectral density of the mean-subtracted series
  (units^2 / Hz), with any non-positive/non-finite bin dropped so the arrays
  are ready for a log-log plot or a log-space fit.

Degraded behavior: an internal ``scipy.signal.welch`` failure returns two
empty arrays, never raises — a spectrum is always optional to its caller.
"""

from typing import Dict, Optional

import numpy as np
from scipy.signal import welch

#: caller-selectable knobs — the single source of truth callers validate
#: against BEFORE any per-target loop.
TIMESERIES_PSD_KNOBS = frozenset({"nperseg", "window", "detrend"})


def validate_timeseries_psd_kwargs(knobs: Dict) -> None:
    """Raise ValueError for an unknown knob — call BEFORE per-target loops."""
    unknown = set(knobs) - TIMESERIES_PSD_KNOBS
    if unknown:
        raise ValueError(
            f"Unknown timeseries-PSD knob(s) {sorted(unknown)}; "
            f"valid: {sorted(TIMESERIES_PSD_KNOBS)}"
        )


def timeseries_psd(series: np.ndarray, dt_s: float, *,
                   nperseg: Optional[int] = None, window: str = "hann",
                   detrend: str = "constant") -> tuple:
    """The one-sided Welch PSD of a uniformly-sampled float series.

    Parameters
    ----------
    series : array-like
        The samples, uniformly spaced in time. Non-finite samples are the
        caller's problem — drop or fill them first (welch propagates NaN).
    dt_s : float
        Sample period in seconds (positive, finite).
    nperseg, window, detrend
        Passed to :func:`scipy.signal.welch`. ``nperseg`` defaults to
        ``min(n, max(256, n // 8))`` — several averaged segments on a long
        record, the whole record on a short one.

    Returns
    -------
    (freq_hz, psd) : tuple of ndarray
        See the module docstring. Empty arrays when welch fails internally.
    """
    if dt_s is None or not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(
            f"dt_s must be a positive finite sample period in seconds, got {dt_s!r}"
        )
    series = np.asarray(series, dtype=float).ravel()
    if series.size == 0:
        raise ValueError("series is empty — nothing to analyze")

    if nperseg is None:
        nperseg = min(series.size, max(256, series.size // 8))
    nperseg = int(min(int(nperseg), series.size))

    try:
        freq, psd = welch(series - float(np.mean(series)), fs=1.0 / float(dt_s),
                          window=window, nperseg=nperseg, detrend=detrend)
    except Exception:
        return np.array([]), np.array([])

    keep = (freq > 0) & (psd > 0) & np.isfinite(psd)
    return freq[keep], psd[keep]
