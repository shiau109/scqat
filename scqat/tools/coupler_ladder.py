"""Coupler lines and their multi-photon ladder - the reduction shared by the coupler
spectroscopy family (``pair_coupler_spectroscopy_swap``, ``pair_coupler_spectroscopy_zz``).

A tone on a pair member's drive line excites the coupler at its transitions; driven
hard, the coupler shows its multi-photon ladder BELOW its 0-1 line:
``f02/2 = f01 + alpha/2`` and ``f03/3 ~ f01 + alpha``. Through the more strongly
coupled member's line the f01 broadens until f02/2 is the strongest line, so height
never picks f01: it is the HIGHEST coupler line, and every other must sit on its
ladder for ONE alpha.

Two pure functions, one per stage:

* :func:`find_lines` - the positive lines of a real trace (a caller with dips passes
  the negated trace), each a line in its own right.
* :func:`read_ladder` - f01 = the highest of a set of lines, the rest placed on its
  ladder, alpha from f02/2.

Line dicts are ``tools.peak_fit.fit_peaks`` peaks (``full_freq``, ``detuning``,
``amplitude``, ``fwhm``, ``detuning_err``, ...), so a line needs the ``full_freq``
axis the tool is given here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .fit_lorentzian import lorentzian
from .peak_fit import fit_peaks, robust_noise

_NAN = float("nan")

#: the default anharmonicity window a ladder may take (Hz)
ALPHA_RANGE_HZ = (-400e6, -50e6)
#: the smallest rung tolerance (Hz); a rung is also as wide as the larger FWHM
LADDER_TOL_HZ = 15e6


def find_lines(freq: np.ndarray, signal: np.ndarray, *, min_snr: float = 6.0,
               prominence: float = 0.1, min_fwhm_steps: float = 2.0
               ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The positive lines of a real trace that are lines in their own right, in
    ascending frequency, and ``fit_peaks``' full result.

    ``fit_peaks``' own merge keeps the larger-AREA fit of two overlapping ones, so a
    broad sub-noise fit of a detected bump swallows a real narrow line beside it
    (0.015 x 178 MHz beat 0.26 x 2.6 MHz on hardware). So the merge is off here and
    every fit is gated on ITS height against the trace noise (``min_snr``) and on a
    width between ``min_fwhm_steps`` sweep steps (the tool's ``min_fwhm_factor``) and
    a quarter of the window, then duplicates are dropped strongest-first. A line one
    or two steps wide is under-sampled: re-measure it with a finer step.

    Only peaks, fixed so: the tool's automatic polarity takes two similar lines for a
    dip, and its dip path seeds each fit on the wrong side of the flipped trace - a
    caller looking for dips negates its trace instead. Each fit sees two estimated
    widths either side of its line, not the tool's five: a broadened f01 30 MHz wide
    otherwise reaches the narrow, stronger f02/2 74 MHz away and fits that."""
    freq = np.asarray(freq, dtype=float)
    mid = 0.5 * (float(np.min(freq)) + float(np.max(freq)))
    res = fit_peaks(freq - mid, signal, full_freq=freq, min_snr=min_snr,
                    prominence=prominence, merge_factor=0.0,
                    min_fwhm_factor=min_fwhm_steps, fit_window_factor=2.0,
                    polarity="peak")
    noise = robust_noise(res["signal_corrected"])
    span = float(np.max(freq) - np.min(freq))
    good = [p for p in res["peaks"]
            if np.isfinite(p["amplitude"]) and p["amplitude"] > 0
            and (noise <= 0 or p["amplitude"] >= min_snr * noise)
            and 0 < p["fwhm"] <= span / 4]
    kept: List[Dict[str, Any]] = []
    for p in sorted(good, key=lambda p: p["amplitude"], reverse=True):
        if all(abs(p["full_freq"] - q["full_freq"]) >= max(p["fwhm"], q["fwhm"]) for q in kept):
            kept.append(p)
    return sorted(kept, key=lambda p: p["full_freq"]), res


def lines_curve(freq: np.ndarray, lines: List[Dict[str, Any]],
                fit: Dict[str, Any]) -> np.ndarray:
    """The fitted lines on ``fit_peaks``' baseline, for a figure (NaN without lines)."""
    freq = np.asarray(freq, dtype=float)
    if not lines:
        return np.full(freq.size, _NAN)
    mid = 0.5 * (float(np.min(freq)) + float(np.max(freq)))
    return fit["baseline"] + sum(
        lorentzian(freq - mid, p["detuning"], p["amplitude"], p["fwhm"] / 2, 0.0)
        for p in lines)


def _place(f01: Dict[str, Any], others: List[Dict[str, Any]], *,
           alpha_range_hz: Tuple[float, float], tol_hz: float) -> Dict[str, list]:
    """Assign every other line to f01's multi-photon ladder.

    Each other line proposes an alpha twice - as f02/2 (``2 (f - f01)``) and as
    f03/3 (``f - f01``); every proposal inside ``alpha_range_hz`` is tried and the
    one placing the most lines wins. On a tie a proposal with an f02/2 line goes
    first (with only one other line the two readings are indistinguishable, and
    f02/2 is the stronger and so the likelier), then the one whose placed lines
    have the larger area (height x FWHM). A line sits on a rung when it is within
    ``max(tol_hz, larger FWHM)`` of it."""
    f0 = float(f01["full_freq"])
    lo, hi = sorted(alpha_range_hz)

    def assign(alpha: float) -> Dict[str, list]:
        rungs: Dict[str, list] = {"f02_half": [], "f03_third": [], "unexplained": []}
        for p in others:
            tol = max(tol_hz, float(f01["fwhm"]), float(p["fwhm"]))
            f = float(p["full_freq"])
            if abs(f - (f0 + alpha / 2)) <= tol:
                rungs["f02_half"].append(p)
            elif abs(f - (f0 + alpha)) <= tol:
                rungs["f03_third"].append(p)
            else:
                rungs["unexplained"].append(p)
        return rungs

    best = assign(_NAN)
    best_key = (0, False, 0.0)
    for p in others:
        for alpha in (2 * (float(p["full_freq"]) - f0), float(p["full_freq"]) - f0):
            if not lo <= alpha <= hi:
                continue
            rungs = assign(alpha)
            placed = rungs["f02_half"] + rungs["f03_third"]
            key = (len(placed), bool(rungs["f02_half"]),
                   sum(abs(float(q["amplitude"])) * float(q["fwhm"]) for q in placed))
            if key > best_key:
                best, best_key = rungs, key
    return best


def read_ladder(lines: List[Dict[str, Any]], *,
                alpha_range_hz: Tuple[float, float] = ALPHA_RANGE_HZ,
                tol_hz: float = LADDER_TOL_HZ) -> Optional[Dict[str, Any]]:
    """f01 = the HIGHEST of ``lines`` (all taken to be the coupler's), the rest placed
    on its ladder. ``None`` without lines.

    Returns ``{f01, f02_half, f03_third, unexplained, n_ladder_lines, alpha_hz,
    alpha_stderr_hz}``: the line dicts (the strongest on each rung, ``None`` when the
    rung is empty), the lines off the ladder, the count on it (f01 included), and
    ``alpha = 2 (f02/2 - f01)`` with its stderr - NaN without an f02/2 line."""
    if not lines:
        return None
    top = max(lines, key=lambda p: p["full_freq"])
    rungs = _place(top, [p for p in lines if p is not top],
                   alpha_range_hz=alpha_range_hz, tol_hz=tol_hz)
    strongest = lambda rung: (max(rung, key=lambda p: abs(p["amplitude"])) if rung else None)
    half, third = strongest(rungs["f02_half"]), strongest(rungs["f03_third"])
    out: Dict[str, Any] = {
        "f01": top, "f02_half": half, "f03_third": third,
        "unexplained": rungs["unexplained"],
        "n_ladder_lines": 1 + len(rungs["f02_half"]) + len(rungs["f03_third"]),
        "alpha_hz": _NAN, "alpha_stderr_hz": _NAN,
    }
    if half is not None:
        out["alpha_hz"] = 2 * (float(half["full_freq"]) - float(top["full_freq"]))
        out["alpha_stderr_hz"] = 2 * float(np.hypot(top["detuning_err"], half["detuning_err"]))
    return out
