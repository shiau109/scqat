"""Flux crosstalk from where a target's flux apex sits at each source amplitude.

A flux line that is not the target's own (the SOURCE line) still threads a little
flux through the target's SQUID. Seen from the target, a source move of ``b`` volts
acts like its own line moving by ``m * b`` volts, so the target's flux apex - located
on the target's OWN line - slides by ``-m * b``::

    apex(b) = apex(0) - m * b

One pure function: the apex positions at several source amplitudes in, the signed
coefficient ``m`` out. The ruler is the target's own line, so no arch model and no
stored fact enters, and anything that moves the apex HEIGHT without moving its
position (a coupler's dispersive shift, a drifting apex frequency) drops out.

The frequency probe behind the apex positions is the caller's business (Ramsey
fringes, a spectroscopy line); this reduction is shared by every such reading.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: source amplitudes needed for a slope AND an error on it (two points fit exactly)
MIN_POINTS = 3

_NAN = float("nan")


def fit_crosstalk_line(source: np.ndarray, apex: np.ndarray, apex_stderr: np.ndarray,
                       valid: Optional[np.ndarray] = None, *,
                       min_points: int = MIN_POINTS) -> Dict[str, Any]:
    """Weighted straight line through ``apex(source)``; ``m = -slope``.

    Parameters
    ----------
    source : array
        Source-line amplitudes (V), any order.
    apex : array
        The target's flux apex at each source amplitude, on the target's own line (V).
    apex_stderr : array
        Its standard error; used as relative weights only (a non-finite or
        non-positive entry takes the median of the others).
    valid : bool array, optional
        Which points to use (default: every finite one).
    min_points : int
        Fewer valid points than this -> no fit.

    Returns
    -------
    dict, always complete: ``success``, ``crosstalk`` (``m``, dimensionless),
    ``crosstalk_stderr``, ``apex_at_zero`` (the line's value at source = 0) and its
    ``apex_at_zero_stderr``, ``slope`` (``-m``), ``n_used``, ``chi2_red``,
    ``residual_rms`` and ``max_residual`` (V, over the used points), ``residuals``
    (per input point, NaN where unused) and ``used`` (0/1 per input point).

    The stderrs start from ``apex_stderr`` and are WIDENED by the scatter of the
    points about the line whenever that scatter is larger than the apex errors
    predict (``chi2_red`` > 1) - a per-point apex error from a quadratic through a
    few fringes is routinely optimistic. They are never narrowed: with one or two
    degrees of freedom a small scatter is luck, not precision.
    """
    source = np.asarray(source, dtype=float)
    apex = np.asarray(apex, dtype=float)
    err = np.asarray(apex_stderr, dtype=float)
    use = np.isfinite(source) & np.isfinite(apex)
    if valid is not None:
        use &= np.asarray(valid, dtype=bool)

    res: Dict[str, Any] = {
        "success": False,
        "crosstalk": _NAN, "crosstalk_stderr": _NAN,
        "apex_at_zero": _NAN, "apex_at_zero_stderr": _NAN,
        "slope": _NAN,
        "chi2_red": _NAN,
        "n_used": int(use.sum()),
        "residual_rms": _NAN, "max_residual": _NAN,
        "residuals": np.full(source.size, _NAN),
        "used": use.astype(int),
    }
    if use.sum() < max(int(min_points), MIN_POINTS):
        return res

    order = np.argsort(source[use], kind="stable")  # order-agnostic fit input
    b, a, e = source[use][order], apex[use][order], err[use][order]
    if float(b[-1] - b[0]) <= 0:
        return res
    good = np.isfinite(e) & (e > 0)
    floor = float(np.median(e[good])) if good.any() else 1.0
    w = 1.0 / np.where(good, e, floor)
    try:
        coef, cov = np.polyfit(b, a, 1, w=w, cov="unscaled")
    except (np.linalg.LinAlgError, ValueError):
        return res
    slope, intercept = float(coef[0]), float(coef[1])
    resid = a - (slope * b + intercept)
    chi2_red = float(np.sum((w * resid) ** 2) / (b.size - 2))
    # scatter larger than the apex errors predict widens the stderr; smaller never
    # narrows it (with one or two degrees of freedom a small scatter is luck)
    cov = cov * (max(1.0, chi2_red) if good.any() else chi2_red)
    out = np.full(source.size, _NAN)
    out[np.flatnonzero(use)[order]] = resid
    res.update(
        chi2_red=chi2_red,
        success=bool(np.isfinite(slope) and np.isfinite(cov[0, 0])),
        crosstalk=-slope,
        crosstalk_stderr=float(np.sqrt(max(cov[0, 0], 0.0))),
        apex_at_zero=intercept,
        apex_at_zero_stderr=float(np.sqrt(max(cov[1, 1], 0.0))),
        slope=slope,
        residual_rms=float(np.sqrt(np.mean(resid ** 2))),
        max_residual=float(np.max(np.abs(resid))),
        residuals=out,
    )
    return res
