"""Triangle-peak fit — the XY-Z (flux-line) delay reduction.

One difference trace (a relative-timing axis + the ``|e> - |g>`` readout
contrast) in, the peak position out. This is the pure-math reduction behind the
XY-Z delay experiment: the XY drive pulse and a same-length Z (flux) pulse are
slid past each other, and the overlap of two equal-duration rectangles is a
**triangle** in the relative shift, so the contrast peaks where the two lines
are aligned. The peak position IS the flux-line delay to apply.

It is a guarded per-trace reduction with a success flag and an argmax fallback
(the ``step_response_fit`` / ``fit_dip`` shape), not a single-model
``FunctionFitting`` subclass.

Ported from the QM reference implementation
(``qua-libs`` ``calibration_utils/xyz_delay/analysis.py``: ``triangle_peak`` +
``fit_delay_trace``) with two deliberate changes:

* **unit-agnostic time**: every threshold that is not already a pure ratio is
  expressed relative to ``xy_duration`` (the pulse width, passed in the same
  unit as ``t``), so the same code fits a ``t`` axis in seconds or in
  nanoseconds — ``t0`` comes out in the unit of the passed ``t``;
* **no printing**: the argmax-fallback notice goes to an optional ``log``
  callable (silent by default) so per-target estimator loops stay quiet.

Result contract (:func:`fit_triangle`)
---------------------------------------
    success     : bool — the guarded curve_fit converged AND passed every
                  acceptance gate (peak clear of the scan edges, a small enough
                  t0 uncertainty, a positive amplitude, and SNR above min_snr)
    t0          : the peak position (the delay) in the unit of ``t``; on a
                  failed fit this is the argmax of the baseline-subtracted trace
    t0_std      : 1-sigma uncertainty on t0 (NaN on the argmax fallback)
    amp, offset : triangle height above the floor, and the floor
    half_width  : fitted triangle half-width (~ xy_duration)
    snr         : amp / wing-noise std
    best_fit    : the triangle evaluated on ``t`` (all-NaN on the fallback)

On a degenerate input (too few samples, all-NaN) the dict comes back with
``success = False`` and a NaN ``t0`` — never a raise — so per-target estimator
loops degrade to an honest failed fit.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

#: Fewer samples than this cannot support the size-5 smoothing plus a meaningful
#: triangle fit — refuse honestly instead of fitting noise.
MIN_SAMPLES = 8


def triangle_peak(
    t: np.ndarray, amp: float, t0: float, half_width: float, offset: float
) -> np.ndarray:
    """Cross-correlation of two equal-duration rectangular pulses.

    Both the XY pulse and the Z (flux) pulse have the same length by
    construction, so their overlap integral versus relative shift is a triangle
    of half-width equal to the pulse duration, centred on the alignment point.
    """
    return amp * np.maximum(0.0, 1.0 - np.abs(t - t0) / half_width) + offset


def _failed(
    t: np.ndarray,
    y: np.ndarray,
    log: Optional[Callable[[str], None]],
    *,
    argmax: bool = True,
) -> Dict[str, Any]:
    """A failed fit (``success=False``). With ``argmax`` (a rejected curve_fit on
    usable data) t0 degrades to the peak of the baseline-subtracted trace; without
    it (degenerate input) t0 stays NaN because no estimate is meaningful."""
    result: Dict[str, Any] = {
        "success": False,
        "t0": float("nan"),
        "t0_std": float("nan"),
        "amp": float("nan"),
        "offset": float("nan"),
        "half_width": float("nan"),
        "snr": float("nan"),
        "best_fit": np.full_like(np.asarray(t, dtype=float), np.nan),
    }
    if argmax and t.size and np.any(np.isfinite(y)):
        baseline = float(np.nanpercentile(y, 10))
        result["t0"] = float(t[int(np.nanargmax(y - baseline))])
        result["offset"] = baseline
        if log:
            log(f"triangle fit failed — argmax fallback -> t0 = {result['t0']:.4g}")
    return result


def fit_triangle(
    t: np.ndarray,
    y: np.ndarray,
    xy_duration: float,
    *,
    min_snr: float = 3.0,
    max_t0_std: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Fit one XY-Z delay difference trace to a triangle peak.

    Parameters
    ----------
    t : 1-D float array
        The relative-timing axis (any unit — ``t0`` comes out in the same one).
    y : 1-D float array
        The ``|e> - |g>`` readout-contrast difference on that axis.
    xy_duration : float
        The XY/Z pulse duration, in the unit of ``t``. Seeds the triangle
        half-width and sets the wing/edge scale.
    min_snr : float, optional
        Minimum amp / wing-noise ratio for a successful fit (default 3.0).
    max_t0_std : float, optional
        Maximum 1-sigma uncertainty on t0 for a successful fit, in the unit of
        ``t``. Defaults to ``0.125 * xy_duration`` (5 ns for a 40 ns pi pulse,
        matching the QM reference's 5 ns gate).
    log : callable, optional
        Progress sink (silent when ``None``).
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if max_t0_std is None:
        max_t0_std = 0.125 * float(xy_duration)

    if t.size != y.size or t.size < MIN_SAMPLES or not np.any(np.isfinite(y)):
        return _failed(t, y, log, argmax=False)

    # Smooth before argmax so a single noise spike does not seed the fit.
    y_smooth = uniform_filter1d(y, size=5)
    t0_guess = float(t[int(np.argmax(y_smooth))])
    floor = float(np.percentile(y, 10))  # robust to the peak inflating the mean
    amp_guess = float(y_smooth.max()) - floor

    # Wing noise: samples far from the peak (the lines fully misaligned). Fall
    # back to the whole-trace std when too few wing points (peak near an edge).
    wing_mask = np.abs(t - t0_guess) > xy_duration
    noise_std = float(np.std(y[wing_mask])) if int(wing_mask.sum()) > 10 else float(np.std(y))

    try:
        p0 = [amp_guess, t0_guess, float(xy_duration), floor]
        bounds = (
            [0.0, float(t.min()), 0.5 * float(xy_duration), -np.inf],
            [np.inf, float(t.max()), 1.5 * float(xy_duration), np.inf],
        )
        popt, pcov = curve_fit(triangle_peak, t, y, p0=p0, bounds=bounds, maxfev=3000)
        amp, t0, half_width, offset = (float(v) for v in popt)
        t0_std = float(np.sqrt(pcov[1, 1]))
        snr = amp / noise_std if noise_std > 0 else float("inf")

        # Acceptance gates (any failure -> argmax fallback):
        #  - peak clear of the scan edges (avoid edge artefacts),
        #  - t0 well determined, positive amplitude, and a significant peak.
        if not (float(t.min()) + xy_duration < t0 < float(t.max()) - xy_duration):
            raise ValueError(f"peak at {t0:.4g} too close to a scan edge")
        if not (t0_std < max_t0_std):
            raise ValueError(f"t0 uncertainty {t0_std:.4g} exceeds {max_t0_std:.4g}")
        if not (amp > 0):
            raise ValueError("fitted amplitude is not positive")
        if not (snr > min_snr):
            raise ValueError(f"peak not significant: SNR {snr:.2f} <= {min_snr}")

        return {
            "success": True,
            "t0": t0,
            "t0_std": t0_std,
            "amp": amp,
            "offset": offset,
            "half_width": half_width,
            "snr": snr,
            "best_fit": triangle_peak(t, *popt),
        }
    except (RuntimeError, ValueError) as err:
        if log:
            log(f"triangle fit rejected ({err})")
        return _failed(t, y, log)
