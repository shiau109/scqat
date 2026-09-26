"""The frequency of one Ramsey-type fringe - a pure per-trace reduction.

One real trace ``y(t)`` in, its oscillation frequency out. The reading is the
frequency ONLY (no model selection, no T2* claim), so it stays trustworthy where a
full Ramsey fit is not: at a large virtual detuning ``tools.ramsey_fit`` can lock
onto the wrong frequency while the raw trace is clean (BACKLOG I24).

Two stages:

1. **Periodogram.** The DFT power of the mean-removed trace on a grid four times
   finer than the natural resolution ``1/T`` (``T`` = time span), from
   ``f_min`` (default ``1.5/T``: at least one and a half cycles) to ``f_max``
   (default: the Nyquist frequency of the median sample step), then refined on a
   fine local grid. The fringe must be the STRONGEST line in that band; that is
   what the caller's choice of detuning guarantees.
2. **Least-squares refinement.** ``a_c*cos + a_s*sin`` under ``exp(-k*t)`` plus an
   offset is LINEAR in ``(a_c, a_s, c)`` for fixed ``(f, k)``, so only ``(f, k)``
   are searched, each trial solving the linear part exactly. No phase seed, hence
   no anti-phase collapse (the trap ``tools.ramsey_fit`` needed a quadrature seed
   for). The refined frequency is accepted only when it stays within HALF a
   natural bin (``0.5/T``) of the periodogram peak; otherwise the periodogram
   value is reported and ``method`` says so.

The trace may arrive in any time order: it is sorted internally, so the result
does not depend on the order it was acquired in.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

#: grid points per natural bin (1/T) in the coarse periodogram
OVERSAMPLE = 4
#: fine-grid points across +-1 coarse step around the coarse peak
FINE_POINTS = 201


def _power(t: np.ndarray, y: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    phase = np.exp(-2j * np.pi * np.outer(freqs, t))
    return np.abs(phase @ y) ** 2


def _linear_part(t, y, f, k):
    """Exact (a_c, a_s, c) for fixed (f, k); returns (coef, model)."""
    env = np.exp(-k * t)
    basis = np.column_stack((env * np.cos(2 * np.pi * f * t),
                             env * np.sin(2 * np.pi * f * t),
                             np.ones_like(t)))
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return coef, basis @ coef


def _full_model(t, p):
    a_c, a_s, c, f, k = p
    env = np.exp(-k * t)
    return (env * (a_c * np.cos(2 * np.pi * f * t) + a_s * np.sin(2 * np.pi * f * t))
            + c)


def _refine(t, y, f0, span):
    """Least-squares (f, k) around f0; returns (params, stderr_f) or None."""
    from scipy.optimize import least_squares

    k0 = 1.0 / span

    def residual(v):
        return _linear_part(t, y, v[0], v[1])[1] - y

    try:
        sol = least_squares(residual, x0=[f0, k0],
                            bounds=([f0 - 1.0 / span, 0.0], [f0 + 1.0 / span, 50.0 / span]),
                            x_scale=[1.0 / span, 1.0 / span])
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not sol.success:
        return None
    f, k = (float(v) for v in sol.x)
    coef, model = _linear_part(t, y, f, k)
    params = np.array([coef[0], coef[1], coef[2], f, k])

    # stderr of f from the full 5-parameter Jacobian at the solution
    n = t.size
    dof = n - params.size
    if dof <= 0:
        return params, float("nan")
    rss = float(np.sum((model - y) ** 2))
    steps = np.array([1e-6, 1e-6, 1e-6, 1e-4 / span, 1e-4 / span])
    steps = np.where(np.abs(params) > 0, np.maximum(steps, 1e-6 * np.abs(params)), steps)
    jac = np.empty((n, params.size))
    for j in range(params.size):
        dp = np.zeros_like(params)
        dp[j] = steps[j]
        jac[:, j] = (_full_model(t, params + dp) - _full_model(t, params - dp)) / (2 * steps[j])
    try:
        cov = np.linalg.inv(jac.T @ jac) * rss / dof
        stderr_f = float(np.sqrt(cov[3, 3])) if cov[3, 3] >= 0 else float("nan")
    except np.linalg.LinAlgError:
        stderr_f = float("nan")
    return params, stderr_f


def fringe_frequency(
    t: np.ndarray,
    y: np.ndarray,
    *,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> Dict[str, Any]:
    """The fringe frequency of one trace.

    Parameters
    ----------
    t : array
        Sample times in SECONDS, any order (sorted internally).
    y : array
        The real signal (a population, or an IQ projection). Its sign and scale do
        not matter.
    f_min, f_max : float, optional
        Search band in Hz. Defaults: ``1.5/T`` and the Nyquist frequency of the
        median sample step.

    Returns
    -------
    dict with the REQUIRED keys
        ``frequency_hz`` (the reported frequency), ``frequency_stderr_hz`` (NaN when
        the refinement was not accepted), ``method`` (``"fit"`` | ``"periodogram"``),
        ``success`` (a finite frequency inside the band);
    and diagnostics
        ``periodogram_hz``, ``snr`` (peak power / median power over the band),
        ``amplitude`` (fitted oscillation amplitude at t = 0; NaN without a fit),
        ``decay_rate`` (1/s; NaN without a fit), ``f_min_hz``, ``f_max_hz``.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    order = np.argsort(t, kind="stable")
    t, y = t[order], y[order]

    nan = float("nan")
    out: Dict[str, Any] = {
        "frequency_hz": nan, "frequency_stderr_hz": nan, "method": "periodogram",
        "success": False, "periodogram_hz": nan, "snr": nan, "amplitude": nan,
        "decay_rate": nan, "f_min_hz": nan, "f_max_hz": nan,
    }
    if t.size < 8:
        return out
    span = float(t[-1] - t[0])
    dt = float(np.median(np.diff(t)))
    if not (span > 0 and dt > 0):
        return out
    lo = 1.5 / span if f_min is None else float(f_min)
    hi = 0.5 / dt if f_max is None else float(f_max)
    out["f_min_hz"], out["f_max_hz"] = lo, hi
    if not hi > lo:
        return out

    yc = y - y.mean()
    step = 1.0 / (OVERSAMPLE * span)
    coarse = np.arange(lo, hi + step, step)
    p_coarse = _power(t, yc, coarse)
    k = int(np.argmax(p_coarse))
    fine = np.linspace(coarse[k] - step, coarse[k] + step, FINE_POINTS)
    p_fine = _power(t, yc, fine)
    f_pg = float(fine[int(np.argmax(p_fine))])
    median = float(np.median(p_coarse))
    out["periodogram_hz"] = f_pg
    out["snr"] = float(np.max(p_fine) / median) if median > 0 else float("inf")
    out["frequency_hz"] = f_pg
    out["success"] = bool(lo <= f_pg <= hi)

    refined = _refine(t, y, f_pg, span)
    if refined is not None:
        params, stderr_f = refined
        f_fit = float(params[3])
        if abs(f_fit - f_pg) <= 0.5 / span and np.isfinite(f_fit):
            out.update(frequency_hz=f_fit, frequency_stderr_hz=stderr_f, method="fit",
                       amplitude=float(np.hypot(params[0], params[1])),
                       decay_rate=float(params[4]))
            out["success"] = bool(lo <= f_fit <= hi)
    return out
