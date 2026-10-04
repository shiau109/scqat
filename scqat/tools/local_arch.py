"""A qubit's LOCAL flux arch from Ramsey fringes - two pure reductions.

Near a sweet spot the transmon arch is a parabola (quadratic to 1e-4 over +-12 mV
on 5Q4C), so a handful of Ramsey fringes taken at different flux values pin its
vertex to a small fraction of a millivolt. Two steps, each usable on its own:

1. :func:`fringe_deltas` - one fringe per flux value in, ``delta_f = f_q - f_drive``
   per flux value out. The fringe frequency comes from
   :func:`scqat.tools.fringe_frequency.fringe_frequency`; with the SIGNED virtual
   detuning ``D`` the probe applied (the ``qubit_ramsey`` convention: the fringe
   sits at ``|D + (f_q - f_drive)|``) and a fringe that never crossed zero,
   ``delta_f = sign(D) * F - D``.
2. :func:`fit_local_arch` - a weighted quadratic through the valid ``delta_f(x)``
   points: its vertex (the flux apex and the apex height), the curvature, whether
   the vertex lies inside the sampled window, and whether the fitted fringe stays
   on one side of zero across it.

Both take plain arrays in whatever order the caller holds them and return per-point
arrays in THAT order; neither sorts for the caller (the quadratic sorts its own
input, so its result is order-free).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from scqat.tools.fringe_frequency import fringe_frequency

#: minimum valid points for the quadratic (three fix a parabola; two more give it an error)
MIN_VALID = 5

_NAN = float("nan")


def fringe_deltas(t: np.ndarray, rows: np.ndarray, ramp_detuning_hz: float, *,
                  min_snr: float = 30.0) -> Dict[str, Any]:
    """``delta_f = f_q - f_drive`` for each row of a fringe map.

    Parameters
    ----------
    t : array
        The idle times in SECONDS (one per column), any order.
    rows : array, shape (n, len(t))
        One real fringe per row.
    ramp_detuning_hz : float
        The SIGNED virtual detuning the probe applied; finite and nonzero.
    min_snr : float
        Periodogram peak / median power a row needs to count.

    Returns
    -------
    dict of per-row arrays ``fringe_hz``, ``fringe_stderr_hz``, ``snr``,
    ``amplitude``, ``valid`` (bool), ``delta_f_hz``, ``delta_f_stderr_hz``, plus the
    scalar ``span_s`` (the idle window the fringes were read over).
    """
    ramp = float(ramp_detuning_hz)
    if ramp == 0 or not np.isfinite(ramp):
        raise ValueError(f"ramp_detuning_hz must be finite and nonzero, got {ramp}")
    t = np.asarray(t, dtype=float)
    rows = np.asarray(rows, dtype=float)
    span = float(np.nanmax(t) - np.nanmin(t)) if t.size else _NAN

    n = rows.shape[0]
    fringe = np.full(n, _NAN)
    fringe_err = np.full(n, _NAN)
    snr = np.full(n, _NAN)
    amp = np.full(n, _NAN)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        r = fringe_frequency(t, rows[i])
        fringe[i], snr[i], amp[i] = r["frequency_hz"], r["snr"], r["amplitude"]
        err = r["frequency_stderr_hz"]
        if not np.isfinite(err) and np.isfinite(r["snr"]) and r["snr"] > 0 and span > 0:
            err = 1.0 / (span * np.sqrt(r["snr"]))  # periodogram-only fallback
        fringe_err[i] = err
        edge = 2.0 / span if span > 0 else _NAN
        valid[i] = bool(r["success"] and r["snr"] >= min_snr
                        and r["frequency_hz"] >= r["f_min_hz"] + edge
                        and r["frequency_hz"] <= r["f_max_hz"] - edge)

    sgn = 1.0 if ramp > 0 else -1.0
    return {
        "fringe_hz": fringe,
        "fringe_stderr_hz": fringe_err,
        "snr": snr,
        "amplitude": amp,
        "valid": valid,
        "delta_f_hz": sgn * fringe - ramp,
        "delta_f_stderr_hz": fringe_err.copy(),
        "span_s": span,
    }


def _vertex(coef: np.ndarray, cov: np.ndarray) -> tuple[float, float, float, float]:
    """Vertex (u0, stderr_u0, h0, stderr_h0) of a*u^2 + b*u + c."""
    a, b, c = coef
    u0 = -b / (2 * a)
    h0 = c - b * b / (4 * a)
    g_u = np.array([b / (2 * a * a), -1.0 / (2 * a), 0.0])
    g_h = np.array([b * b / (4 * a * a), -b / (2 * a), 1.0])
    su = float(np.sqrt(max(g_u @ cov @ g_u, 0.0)))
    sh = float(np.sqrt(max(g_h @ cov @ g_h, 0.0)))
    return float(u0), su, float(h0), sh


def fit_local_arch(x: np.ndarray, delta_f_hz: np.ndarray, delta_f_stderr_hz: np.ndarray,
                   valid: np.ndarray, *, ramp_detuning_hz: Optional[float] = None,
                   span_s: Optional[float] = None,
                   min_valid: int = MIN_VALID) -> Dict[str, Any]:
    """Weighted quadratic through the valid ``delta_f(x)`` points.

    ``ramp_detuning_hz`` and ``span_s`` (both from :func:`fringe_deltas`) switch on
    the fold check: the fitted fringe ``sign(D) * (D + delta_f)`` must stay above
    ``2 / span_s`` across the window. Without them ``fold_suspected`` stays 0.

    Returns a dict that is always complete (NaN / flags when there is no fit):
    ``fitted`` (bool), ``poly_center``, ``poly_coeffs`` (highest power first, about
    ``poly_center``), ``curvature_hz_per_v2``, ``curvature_stderr``, ``apex_flux``,
    ``apex_flux_stderr``, ``apex_delta_f_hz``, ``apex_delta_f_stderr_hz``,
    ``apex_not_bracketed`` (0/1), ``fold_suspected`` (0/1), ``window`` (the lowest
    and highest valid x), and the raw ``coef`` / ``cov`` arrays (None without a fit)
    for a caller that needs more than the vertex.
    """
    x = np.asarray(x, dtype=float)
    delta = np.asarray(delta_f_hz, dtype=float)
    delta_err = np.asarray(delta_f_stderr_hz, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    res: Dict[str, Any] = {
        "fitted": False,
        "poly_center": _NAN, "poly_coeffs": [_NAN, _NAN, _NAN],
        "curvature_hz_per_v2": _NAN, "curvature_stderr": _NAN,
        "apex_flux": _NAN, "apex_flux_stderr": _NAN,
        "apex_delta_f_hz": _NAN, "apex_delta_f_stderr_hz": _NAN,
        "apex_not_bracketed": 1,
        "fold_suspected": 0,
        "window": (_NAN, _NAN),
        "coef": None, "cov": None,
    }
    if valid.sum() < min_valid:
        return res

    xv, dv, ev = x[valid], delta[valid], delta_err[valid]
    order = np.argsort(xv, kind="stable")  # order-agnostic fit input
    xv, dv, ev = xv[order], dv[order], ev[order]
    x_c = float(np.mean(xv))
    u = xv - x_c
    floor = np.nanmedian(ev[np.isfinite(ev)]) if np.isfinite(ev).any() else 1.0
    w = 1.0 / np.where(np.isfinite(ev) & (ev > 0), ev, floor)
    try:
        coef, cov = np.polyfit(u, dv, 2, w=w, cov=True)
    except (np.linalg.LinAlgError, ValueError):
        return res
    res.update(fitted=True, coef=coef, cov=cov, poly_center=x_c,
               poly_coeffs=[float(v) for v in coef],
               curvature_hz_per_v2=float(coef[0]),
               curvature_stderr=float(np.sqrt(max(cov[0, 0], 0.0))))

    lo, hi = float(xv[0]), float(xv[-1])
    res["window"] = (lo, hi)
    u0, su0, h0, sh0 = _vertex(coef, cov)
    apex_ok = (coef[0] < 0 and coef[0] + 2 * res["curvature_stderr"] < 0
               and lo <= x_c + u0 <= hi)
    if coef[0] != 0:
        res.update(apex_flux=x_c + u0, apex_flux_stderr=su0,
                   apex_delta_f_hz=h0, apex_delta_f_stderr_hz=sh0)
    res["apex_not_bracketed"] = 0 if apex_ok else 1

    if ramp_detuning_hz is not None and span_s is not None:
        # the fitted fringe must stay on one side of zero across the window
        ramp = float(ramp_detuning_hz)
        sgn = 1.0 if ramp > 0 else -1.0
        grid = np.linspace(lo - x_c, hi - x_c, 201)
        fitted = sgn * (ramp + np.polyval(coef, grid))
        res["fold_suspected"] = int(np.min(fitted) < 2.0 / float(span_s))
    return res
