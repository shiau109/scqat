"""Generic 2-D peak-map tracker — one :func:`~scqat.tools.peak_fit.fit_peaks`
per row, pooled into a point-cloud with robust acceptance masks.

Shared by every "qubit line vs <axis>" experiment: the vs-flux map
(``flux_bias`` rows) and the parametric-drive map (``drive_amp`` rows) are the
same reduction with different axis names, so it lives here with GENERIC keys —
``x`` is the slow (row) axis, ``y`` the frequency axis — and each estimator
relabels the result into its own domain vocabulary. **All** peaks found in a
row are kept (several transitions may coexist per slice); assigning points to
individual transition branches belongs to the downstream model fit.

Acceptance: a peak is kept (``good``) when its centre lies strictly inside the
swept ``y`` window and its ``fwhm`` / ``|amplitude|`` are not robust
(median/MAD) outliers across the pooled set of detected peaks.

Result contract
---------------
``{x, y, full_freq?, peak_x, peak_x_index, peak_y, peak_full_freq?, peak_fwhm,
peak_amplitude, in_window, outlier, good, fwhm_median, fwhm_mad,
peak_amplitude_median, peak_amplitude_mad, n_x, n_peaks, n_in_window, n_good,
n_outlier}`` — the ``peak_*`` arrays are one flat point-cloud over all rows.
"""

from typing import Any, Dict, Optional

import numpy as np

from .peak_fit import fit_peaks, validate_peak_kwargs
from .robust import mad_outliers


def track_peaks(
    x: np.ndarray,
    y: np.ndarray,
    signal_map: np.ndarray,
    *,
    full_freq: Optional[np.ndarray] = None,
    n_sigma: float = 3.0,
    **peak_knobs,
) -> Dict[str, Any]:
    """Fit and pool every peak in every row of a 2-D spectrum map.

    Parameters
    ----------
    x : 1-D float array
        Slow (row) axis — e.g. flux bias or drive amplitude.
    y : 1-D float array
        Frequency axis of each row's spectrum (Hz).
    signal_map : 2-D array, shape ``(len(x), len(y))``
        Per-row signal — complex I + iQ or an already-real signal; each row is
        handed to :func:`~scqat.tools.peak_fit.fit_peaks` as-is.
    full_freq : 1-D float array, optional
        Absolute frequency (Hz) on the ``y`` axis; when present the cloud also
        carries ``peak_full_freq``.
    n_sigma : float, optional
        Robust-sigma threshold for the width / amplitude outlier test
        (default 3.0).
    **peak_knobs
        Knobs of :func:`~scqat.tools.peak_fit.fit_peaks` (``prominence``,
        ``max_peaks``, ...) — validated BEFORE the row loop; unknown names
        raise ValueError here, never silently vanish.
    """
    # Fail loudly BEFORE any per-row fit — a typo'd knob must never be
    # swallowed by the per-row fallback.
    try:
        validate_peak_kwargs(peak_knobs)
    except ValueError as err:
        raise ValueError(
            f"track_peaks: {err} (own keyword-only tunables: n_sigma)"
        ) from None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    signal_map = np.asarray(signal_map)
    has_full_freq = full_freq is not None

    # Flat point-cloud of all peaks found across all rows.
    pk_x_idx, pk_x = [], []
    pk_y, pk_full_freq, pk_fwhm, pk_amplitude = [], [], [], []
    for k in range(len(x)):
        try:
            r = fit_peaks(y, signal_map[k], full_freq=full_freq, **peak_knobs)
        except Exception:
            continue  # fit-domain failure only: kwargs were validated up front
        for pk in r["peaks"]:
            pk_x_idx.append(k)
            pk_x.append(float(x[k]))
            pk_y.append(float(pk["detuning"]))
            pk_fwhm.append(float(pk["fwhm"]))
            pk_amplitude.append(float(pk["amplitude"]))
            pk_full_freq.append(
                float(pk["full_freq"]) if (has_full_freq and "full_freq" in pk) else np.nan
            )

    peak_x_index = np.asarray(pk_x_idx, dtype=int)
    peak_x = np.asarray(pk_x, dtype=float)
    peak_y = np.asarray(pk_y, dtype=float)
    peak_fwhm = np.asarray(pk_fwhm, dtype=float)
    peak_amplitude = np.asarray(pk_amplitude, dtype=float)
    peak_full_freq = np.asarray(pk_full_freq, dtype=float)
    n_peaks = peak_y.size

    # (1) Strict window enforcement: the fitted peak centre must lie inside the
    # swept y window.
    y_lo, y_hi = float(y.min()), float(y.max())
    in_window = (
        np.isfinite(peak_y) & (peak_y > y_lo) & (peak_y < y_hi)
        if n_peaks else np.zeros(0, dtype=bool)
    )

    # (2) Robust outlier rejection on the pooled peak width and amplitude.
    outlier_fwhm, fwhm_med, fwhm_mad = mad_outliers(peak_fwhm, in_window, n_sigma)
    outlier_amp, amp_med, amp_mad = mad_outliers(np.abs(peak_amplitude), in_window, n_sigma)
    outlier = in_window & (outlier_fwhm | outlier_amp)
    good = in_window & ~outlier

    results: Dict[str, Any] = {
        "x": x,
        "y": y,
        "peak_x": peak_x,
        "peak_x_index": peak_x_index,
        "peak_y": peak_y,
        "peak_fwhm": peak_fwhm,
        "peak_amplitude": peak_amplitude,
        "in_window": in_window,
        "outlier": outlier,
        "good": good,
        "fwhm_median": fwhm_med,
        "fwhm_mad": fwhm_mad,
        "peak_amplitude_median": amp_med,
        "peak_amplitude_mad": amp_mad,
        "n_x": int(len(x)),
        "n_peaks": int(n_peaks),
        "n_in_window": int(in_window.sum()),
        "n_good": int(good.sum()),
        "n_outlier": int(outlier.sum()),
    }
    if has_full_freq:
        results["full_freq"] = np.asarray(full_freq, dtype=float).ravel()
        results["peak_full_freq"] = peak_full_freq

    return results
