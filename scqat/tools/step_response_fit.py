"""Multi-exponential step-response fit — the cryoscope tap extraction.

A settled step response ``y(t) ~ a_dc + sum_i amp_i * exp(-t / tau_i)`` in, the
``(amp, tau)`` components out. This is the pure-math reduction behind the
cryoscope experiment (flux-line transient characterization); it is a composite
sequential procedure, not a single-model fit, so it is a plain-function tool
(the ``telegraph_psd`` / ``dip_fit`` shape), not a ``FunctionFitting`` subclass.

Ported from the QM reference implementation
(``qua-libs`` ``calibration_utils/cryoscope/analysis.py``:
``sequential_exp_fit`` + ``optimize_start_fractions``) with two deliberate
changes:

* **unit-agnostic time**: every internal time scale (the tau lower bound, the
  initial tau guess) derives from the sample spacing ``t[1] - t[0]`` instead of
  a hardcoded "0.1 ns", so the same code fits a ``t`` axis in seconds or in
  nanoseconds — taus come out in the unit of the passed ``t``;
* **no printing**: progress goes to an optional ``log`` callable (silent by
  default) so per-slice estimator loops stay quiet.

How the sequential fit works
----------------------------
1. Estimate the constant term ``a_dc`` from the flattest tail of the data
   (rolling variance below 10 % of its mean), falling back to the last sample.
2. For each ``start_fraction`` (a 0-1 fraction of the record, DESCENDING —
   slowest component first), fit ``amp * exp(-t / tau)`` on the tail of the
   residual from that fraction onward, then subtract the fitted component from
   the WHOLE residual and continue with the next (faster) component.
3. :func:`fit_step_response` additionally Nelder-Mead-optimizes the start
   fractions (bounded ``+-bounds_scale`` around the initial values, refusing
   non-descending candidates) to minimize the residual RMS, and re-references
   the fitted amplitudes to ``t = 0`` (the fits run on ``t - t[0]``), so
   ``y(t) ~ a_dc + sum_i amp_i * exp(-t / tau_i)`` holds on the CALLER's axis.

Result contract (:func:`fit_step_response`)
-------------------------------------------
    success        : bool — optimizer converged AND every requested component
                     was fitted with a finite positive tau
    components     : list of (amp, tau) pairs, slowest first as requested;
                     amps referenced to t = 0, taus in the unit of ``t``
    a_dc           : the constant (settled) level — fitted, or the passed one
    rms            : root-mean-square of the final residual
    best_fractions : the optimized start fractions
    best_fit       : ``a_dc + sum_i amp_i * exp(-t / tau_i)`` on the input axis

On a degenerate input (too few samples, all-NaN) the dict comes back with
``success = False``, empty ``components`` and a NaN ``rms`` — never a raise —
so per-target estimator loops degrade to an honest failed fit.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import curve_fit, minimize

#: Fewer samples than this cannot support the rolling-variance a_dc estimate
#: (window >= 5) plus a meaningful tail fit — refuse honestly instead of
#: fitting noise.
MIN_SAMPLES = 16


def single_exp_decay(t: np.ndarray, amp: float, tau: float) -> np.ndarray:
    """Single exponential decay without offset: ``amp * exp(-t / tau)``."""
    return amp * np.exp(-t / tau)


def _estimate_a_dc(y: np.ndarray, log: Optional[Callable[[str], None]]) -> float:
    """The constant term from the flattest tail (rolling variance < 10 % of
    its mean), falling back to the last sample when no flat region exists."""
    window = max(5, len(y) // 20)
    rolling_var = np.array(
        [np.var(y[i:i + window]) for i in range(len(y) - window)]
    )
    try:
        var_threshold = float(np.mean(rolling_var)) * 0.1
        flat_start = np.where(rolling_var < var_threshold)[0][-1]
        return float(np.mean(y[flat_start:]))
    except (IndexError, ValueError):
        if log:
            log("no flat region found — using the last sample as a_dc")
        return float(y[-1])


def sequential_exp_fit(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: Sequence[float],
    fixed_taus: Optional[Sequence[float]] = None,
    a_dc: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[float, float]], float, np.ndarray]:
    """Fit a sum of exponentials sequentially, slowest component first.

    Parameters
    ----------
    t, y : 1-D arrays
        The time axis (any unit — taus come out in the same one) and the
        step-response values. ``t`` must be evenly spaced.
    start_fractions : sequence of float
        Fractions (0-1) of the record where each component's fit starts,
        DESCENDING (the slow component is fitted from the settled tail, the
        fast one from the front).
    fixed_taus : sequence of float, optional
        Pin the tau of each component (same length as ``start_fractions``);
        only amplitudes are fitted.
    a_dc : float, optional
        Pin the constant term instead of estimating it from the tail.
    log : callable, optional
        Progress sink (silent when ``None``).

    Returns
    -------
    (components, a_dc, residual) — ``components`` is a list of ``(amp, tau)``
    pairs, taus in the unit of ``t``, fitted on ``t - t[0]`` (see
    :func:`fit_step_response` for the t = 0 re-referencing); a component whose
    fit fails ends the loop early, so the list may be shorter than requested.

    The fit runs on a SAMPLE-normalized axis (``(t - t[0]) / dt``) so it is
    scale-invariant — the same data on a nanosecond or a second axis lands in
    the same minimum, with taus returned in the caller's unit.
    """
    components: List[Tuple[float, float]] = []
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    t_offset = t - t[0]
    dt = float(t_offset[1] - t_offset[0])
    # Work in sample units s = (t - t[0]) / dt = [0, 1, 2, ...]; convert taus
    # back to real units (tau = tau_s * dt) before returning.
    s = t_offset / dt

    if a_dc is None:
        a_dc = _estimate_a_dc(y, log)
    if log:
        log(f"constant term a_dc = {a_dc:.3e}")

    y_residual = y.copy() - a_dc

    for i, start_frac in enumerate(start_fractions):
        start_idx = int(len(t) * start_frac)
        s_fit = s[start_idx:]
        y_fit = y_residual[start_idx:]
        try:
            if fixed_taus is not None:
                tau_s_fixed = float(fixed_taus[i]) / dt
                popt, _ = curve_fit(
                    lambda ss, amp: single_exp_decay(ss, amp, tau_s_fixed),
                    s_fit, y_fit, p0=[y_fit[0]],
                )
                amp, tau_s = float(popt[0]), tau_s_fixed
            else:
                # tau floor = one-tenth of a sample; the guess must sit above
                # it or curve_fit refuses the fit as infeasible.
                p0 = [y_fit[0], max(s[start_idx] / 3.0, 1.0)]
                popt, _ = curve_fit(
                    single_exp_decay, s_fit, y_fit, p0=p0,
                    bounds=([-np.inf, 0.1], [np.inf, np.inf]),
                )
                amp, tau_s = float(popt[0]), float(popt[1])
            tau = tau_s * dt
            components.append((amp, tau))
            if log:
                log(f"component {i + 1}: amp = {amp:.3e}, tau = {tau:.4g}")
            y_residual -= amp * np.exp(-s / tau_s)
        except (RuntimeError, ValueError) as err:
            if log:
                log(f"component {i + 1} fit failed: {err}")
            break

    return components, float(a_dc), y_residual


def fit_step_response(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: Sequence[float],
    *,
    fixed_taus: Optional[Sequence[float]] = None,
    a_dc: Optional[float] = None,
    bounds_scale: float = 0.5,
    maxiter: int = 1000,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Extract multi-exponential taps from a settled step response.

    Optimizes the ``start_fractions`` (Nelder-Mead over the residual RMS,
    bounded ``+-bounds_scale`` around the initial values, non-descending
    candidates refused) around :func:`sequential_exp_fit`, then re-references
    the amplitudes to ``t = 0``. See the module docstring for the result
    contract; taus come out in the unit of ``t``.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    start_fractions = [float(f) for f in start_fractions]

    if not start_fractions or not np.all(np.diff(start_fractions) < 0):
        raise ValueError(
            "start_fractions must be non-empty and strictly DESCENDING "
            f"(slowest component first), got {start_fractions}"
        )
    if fixed_taus is not None:
        if len(fixed_taus) != len(start_fractions):
            raise ValueError(
                "fixed_taus must have the same length as start_fractions"
            )
        if any(tau <= 0 for tau in fixed_taus):
            raise ValueError("all fixed_taus values must be positive")

    failed: Dict[str, Any] = {
        "success": False, "components": [],
        "a_dc": float("nan") if a_dc is None else float(a_dc),
        "rms": float("nan"), "best_fractions": list(start_fractions),
        "best_fit": np.full_like(y, np.nan),
    }
    if t.size != y.size or t.size < MIN_SAMPLES or not np.all(np.isfinite(y)):
        return failed

    def objective(x: np.ndarray) -> float:
        if not np.all(np.diff(x) < 0):
            return 1e6
        components, _, residual = sequential_exp_fit(
            t, y, x, fixed_taus=fixed_taus, a_dc=a_dc,
        )
        if len(components) != len(start_fractions):
            return 1e6
        return float(np.sqrt(np.mean(residual ** 2)))

    bounds = [
        (start * (1.0 - bounds_scale), start * (1.0 + bounds_scale))
        for start in start_fractions
    ]
    result = minimize(
        objective, x0=start_fractions, bounds=bounds, method="Nelder-Mead",
        options={"disp": False, "maxiter": maxiter},
    )
    best_fractions = list(result.x) if result.success else list(start_fractions)

    components, fitted_dc, residual = sequential_exp_fit(
        t, y, best_fractions, fixed_taus=fixed_taus, a_dc=a_dc, log=log,
    )
    rms = float(np.sqrt(np.mean(residual ** 2)))
    # The fits ran on t - t[0]; move the amplitude reference to t = 0 so the
    # components reproduce y on the CALLER's axis.
    components = [
        (float(amp * np.exp(t[0] / tau)), float(tau)) for amp, tau in components
    ]

    best_fit = np.full_like(y, fitted_dc, dtype=float)
    for amp, tau in components:
        best_fit += amp * np.exp(-t / tau)

    success = bool(
        result.success
        and len(components) == len(start_fractions)
        and all(np.isfinite(tau) and tau > 0 for _, tau in components)
        and all(np.isfinite(amp) for amp, _ in components)
    )
    if log:
        log(
            f"fit {'succeeded' if success else 'FAILED'}: rms = {rms:.3e}, "
            f"components = {components}"
        )
    return {
        "success": success, "components": components, "a_dc": fitted_dc,
        "rms": rms, "best_fractions": [float(f) for f in best_fractions],
        "best_fit": best_fit,
    }
