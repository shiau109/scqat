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
    success        : bool — optimizer converged AND every kept component has a
                     finite positive tau and ``|amp| <= amp_max``
    components     : list of (amp, tau) pairs, slowest first; amps referenced
                     to t = 0, taus in the unit of ``t``. May be FEWER than
                     requested: a tau-degenerate pair (two components within
                     ``degen_tau_ratio`` in tau — one physical component split
                     into a cancelling pair) collapses the model by one
                     component and refits (see the degeneracy guard in
                     :func:`fit_step_response`).
    a_dc           : the constant (settled) level — fitted, or the passed one
    rms            : root-mean-square of the final residual
    best_fractions : the optimized start fractions (of the KEPT components)
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


def mpm_tau_seeds(
    t: np.ndarray,
    y: np.ndarray,
    *,
    a_dc: float = 1.0,
    max_modes: int = 4,
    mode_method: str = "mdl",
) -> Dict[str, Any]:
    """Model order + tau seeds for a UNIFORM-axis step response via the
    Hankel-SVD Matrix Pencil Method (:func:`scqat.tools.hankel.hankel_decompose`).

    MPM answers "how many exponentials does the data support, and roughly which
    taus" by linear algebra — no start-fraction heuristics, no sequential-
    subtraction bias — and is the standard SEEDING stage before a bounded
    nonlinear polish (pass the returned ``taus`` to :func:`fit_step_response`
    as ``tau_seeds``). Requires an evenly spaced ``t`` (the Hankel shift
    structure IS the uniform sampling); a log-spaced axis is refused by name —
    the spectroscopy cryoscope stays on the start-fractions path.

    Returns ``{"taus": [...descending, caller units...], "n_modes": the SVD
    model order, "oscillatory": bool — a significant complex-pole mode was
    found (ringing/reflection the exp-sum model cannot represent; consider an
    FIR correction), "modes": the kept raw mode dicts}``. Raises ``ValueError``
    on a non-uniform axis, too few samples, or non-finite data.
    """
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if t.size != y.size or t.size < MIN_SAMPLES:
        raise ValueError(f"mpm_tau_seeds needs >= {MIN_SAMPLES} samples")
    if not np.all(np.isfinite(y)):
        raise ValueError("mpm_tau_seeds: y contains non-finite values")
    steps = np.diff(t)
    if np.max(np.abs(steps - steps.mean())) > 1e-3 * abs(steps.mean()):
        raise ValueError(
            "mpm_tau_seeds needs a UNIFORM time axis (the Hankel shift "
            "structure is the uniform sampling) — got a non-uniform one; "
            "use the start-fractions path instead"
        )
    from scqat.tools.hankel import hankel_decompose  # tools -> tools is legal

    out = hankel_decompose(
        y - a_dc, t, mode_method=mode_method, recon_method="mpm"
    )
    step = float(steps.mean())
    span = float(t[-1] - t[0])
    nyquist = 0.5 / step
    max_amp = max((m["amplitude"] for m in out["modes"]), default=0.0)
    kept: List[Dict[str, Any]] = []
    oscillatory = False
    for m in out["modes"]:  # arrive amplitude-sorted (descending)
        significant = max_amp > 0 and m["amplitude"] >= 0.1 * max_amp
        if abs(m["freq_hz"]) >= 0.02 * nyquist:
            # a genuinely oscillating pole — not representable as a real
            # decaying exponential; flag it (the "consider FIR" signal).
            oscillatory = oscillatory or significant
            continue
        tau = m["time_constant"]
        if not (m["decay_rate"] < 0 and np.isfinite(tau)):
            continue
        if not (0.5 * step <= tau <= 10.0 * span):
            continue  # sub-sample or beyond-record: not identifiable
        # dedupe: a tau within 1.5x of an already-kept (stronger) mode would
        # hand the joint fit a near-degenerate p0 — keep the stronger one.
        if any(
            max(tau, k["time_constant"]) / min(tau, k["time_constant"]) < 1.5
            for k in kept
        ):
            continue
        kept.append(m)
    kept = kept[:max_modes]
    taus = sorted((float(m["time_constant"]) for m in kept), reverse=True)
    return {
        "taus": taus,
        "n_modes": int(out["n_modes"]),
        "oscillatory": bool(oscillatory),
        "modes": kept,
    }


def sequential_exp_fit(
    t: np.ndarray,
    y: np.ndarray,
    start_fractions: Sequence[float],
    fixed_taus: Optional[Sequence[float]] = None,
    a_dc: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
    amp_max: float = 2.0,
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
    amp_max : float, optional
        Bound on each component's |amplitude| (in units of the response, i.e.
        relative to ``a_dc ~ 1``). Physical flux-line taps are fractions of the
        settled level; unbounded amplitudes let a pair of near-identical taus
        cancel to a giant +-A degenerate solution (seen at +-22866 on real
        hardware).

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

    # Identifiability tau floor, in SAMPLE units (unit-agnostic — it scales
    # with the axis): a component faster than half the first sample time
    # t[0] is pure extrapolation (fully decayed before any data), and the
    # t = 0 re-referencing in fit_step_response would amplify its amplitude
    # by exp(t[0]/tau) — unbounded, this minted a +-22866 tap from a 40 ns
    # min-wait record. With the floor the factor is capped at e^2 ~ 7.4.
    tau_lo = max(0.1, 0.5 * t[0] / dt) if t[0] > 0 else 0.1

    for i, start_frac in enumerate(start_fractions):
        start_idx = int(len(t) * start_frac)
        s_fit = s[start_idx:]
        y_fit = y_residual[start_idx:]
        try:
            if fixed_taus is not None:
                tau_s_fixed = float(fixed_taus[i]) / dt
                popt, _ = curve_fit(
                    lambda ss, amp: single_exp_decay(ss, amp, tau_s_fixed),
                    s_fit, y_fit, p0=[np.clip(y_fit[0], -amp_max, amp_max)],
                    bounds=([-amp_max], [amp_max]),
                )
                amp, tau_s = float(popt[0]), tau_s_fixed
            else:
                # seeds must sit inside the bounds or curve_fit refuses the
                # fit as infeasible.
                p0 = [float(np.clip(y_fit[0], -amp_max, amp_max)),
                      max(s[start_idx] / 3.0, 1.0, tau_lo)]
                popt, _ = curve_fit(
                    single_exp_decay, s_fit, y_fit, p0=p0,
                    bounds=([-amp_max, tau_lo], [amp_max, np.inf]),
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


#: joint-path amplitude floor: a tap below this (0.2 % of the settled level) is
#: beneath measurement noise and hardware resolution — pruned, model refit.
AMP_PRUNE = 2e-3


def _joint_tau_fit(
    t: np.ndarray,
    y: np.ndarray,
    tau_seeds: List[float],
    *,
    a_dc: Optional[float],
    amp_max: float,
    degen_tau_ratio: float,
    log: Optional[Callable[[str], None]],
    failed: Dict[str, Any],
) -> Dict[str, Any]:
    """The tau-seeded JOINT path of :func:`fit_step_response`: one bounded
    ``curve_fit`` of ``a_dc + sum_i A_i*exp(-t/tau_i)`` with all amps + taus
    free, seeded by ``tau_seeds`` (e.g. from :func:`mpm_tau_seeds`) and their
    linear-LS amplitudes. Same floors/bounds as the sequential path; the
    degeneracy collapse drops the smallest-|amp| component and refits."""
    t_offset = t - t[0]
    dt = float(t_offset[1] - t_offset[0])
    s = t_offset / dt
    if a_dc is None:
        a_dc = _estimate_a_dc(y, log)
    tau_lo = max(0.1, 0.5 * t[0] / dt) if t[0] > 0 else 0.1
    # sample units, nudged above the floor so the seed is feasible
    seeds_s = [max(tau / dt, tau_lo * 1.01) for tau in tau_seeds]

    def _refit(seeds: List[float]):
        n = len(seeds)
        design = np.exp(-np.outer(s, 1.0 / np.asarray(seeds)))
        amps0, *_ = np.linalg.lstsq(design, y - a_dc, rcond=None)
        amps0 = np.clip(amps0, -0.999 * amp_max, 0.999 * amp_max)

        def model(ss, *p):
            out = np.zeros_like(ss)
            for a, tau_s in zip(p[:n], p[n:]):
                out = out + a * np.exp(-ss / tau_s)
            return out

        popt, _ = curve_fit(
            model, s, y - a_dc, p0=list(amps0) + list(seeds),
            bounds=([-amp_max] * n + [tau_lo] * n,
                    [amp_max] * n + [np.inf] * n),
        )
        comps = sorted(
            ((float(a), float(ts) * dt) for a, ts in zip(popt[:n], popt[n:])),
            key=lambda c: -c[1],  # slowest first, matching the contract
        )
        residual = (y - a_dc) - model(s, *popt)
        return comps, residual

    try:
        components, residual = _refit(seeds_s)
        while len(components) > 1:
            pair = _degenerate_pair(components, degen_tau_ratio)
            if pair is None:
                break
            # merge THE PAIR into one component (keep the stronger member's
            # tau) — dropping the globally smallest component instead can kill
            # a genuine component while the degenerate pair survives.
            i, j = pair
            keep_tau = components[i if abs(components[i][0])
                                  >= abs(components[j][0]) else j][1]
            seeds_s = [keep_tau / dt] + [
                tau / dt
                for k, (_, tau) in enumerate(components)
                if k not in (i, j)
            ]
            if log:
                log(
                    f"cancelling tau-degenerate pair — collapsing to "
                    f"{len(seeds_s)} component(s)"
                )
            components, residual = _refit(seeds_s)
        # prune negligible taps (an artifact-sized component the joint fit
        # dutifully models, e.g. a Savitzky-Golay edge bump) and refit once
        keep = [k for k, (a, _) in enumerate(components)
                if abs(a) >= AMP_PRUNE]
        if 0 < len(keep) < len(components):
            if log:
                log(f"pruning {len(components) - len(keep)} negligible tap(s)")
            seeds_s = [components[k][1] / dt for k in keep]
            components, residual = _refit(seeds_s)
    except (RuntimeError, ValueError) as err:
        if log:
            log(f"joint tau-seeded fit failed: {err}")
        return dict(failed)

    rms = float(np.sqrt(np.mean(residual ** 2)))
    # re-reference the amplitudes to t = 0 (the fit ran on t - t[0])
    components = [
        (float(amp * np.exp(t[0] / tau)), float(tau)) for amp, tau in components
    ]
    best_fit = np.full_like(y, a_dc, dtype=float)
    for amp, tau in components:
        best_fit += amp * np.exp(-t / tau)
    success = bool(
        all(np.isfinite(tau) and tau > 0 for _, tau in components)
        and all(np.isfinite(amp) for amp, _ in components)
        and all(abs(amp) <= amp_max for amp, _ in components)
    )
    if log:
        log(
            f"joint fit {'succeeded' if success else 'FAILED'}: "
            f"rms = {rms:.3e}, components = {components}"
        )
    return {
        "success": success, "components": components, "a_dc": float(a_dc),
        "rms": rms, "best_fractions": [],  # no fractions on the seeded path
        "best_fit": best_fit,
    }


def _degenerate_pair(
    components: Sequence[Tuple[float, float]], ratio: float
) -> Optional[Tuple[int, int]]:
    """Indices of a close-tau CANCELLING pair, or None.

    Two exponentials on (near-)identical taus whose amplitudes largely cancel
    are one physical component split into a degenerate pair — the parameters
    are non-identifiable regardless of how small the residual is (such a pair
    can even emulate a ``t*exp(-t/tau)`` shape, which is how +-22 amplitudes on
    taus equal to four digits scored a good rms on real hardware). A SAME-SIGN
    close pair merely splits one component's amplitude — harmless for any
    consumer that sums the taps — so it does not fire the collapse.
    """
    order = sorted(range(len(components)), key=lambda i: abs(components[i][1]))
    for i, j in zip(order, order[1:]):
        a_i, tau_i = components[i]
        a_j, tau_j = components[j]
        lo, hi = abs(tau_i), abs(tau_j)
        if lo <= 0 or hi / lo >= ratio:
            continue
        if a_i * a_j >= 0:
            continue  # same-sign split — harmless
        # (near-)IDENTICAL taus with opposite signs are degenerate regardless
        # of how much survives the cancellation — the pair IS one component.
        if hi / lo < 1.1 or abs(a_i + a_j) < 0.5 * max(abs(a_i), abs(a_j)):
            return (i, j)
    return None


def _has_degenerate_pair(
    components: Sequence[Tuple[float, float]], ratio: float
) -> bool:
    """Boolean face of :func:`_degenerate_pair` (the sequential path's check)."""
    return _degenerate_pair(components, ratio) is not None


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
    amp_max: float = 2.0,
    degen_tau_ratio: float = 1.5,
    tau_seeds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Extract multi-exponential taps from a settled step response.

    Optimizes the ``start_fractions`` (Nelder-Mead over the residual RMS,
    bounded ``+-bounds_scale`` around the initial values, non-descending
    candidates refused) around :func:`sequential_exp_fit`, then re-references
    the amplitudes to ``t = 0``. See the module docstring for the result
    contract; taus come out in the unit of ``t``.

    ``tau_seeds`` (e.g. from :func:`mpm_tau_seeds`) switches to the JOINT path:
    one bounded ``curve_fit`` of the full model with all amps + taus free,
    seeded by the given taus and their linear-LS amplitudes — no fractions, no
    sequential subtraction; ``best_fractions`` comes back empty. The model
    order is the seed count (collapse may still reduce it). Mutually exclusive
    with ``fixed_taus``.

    Degeneracy guard (both paths): amplitudes are bounded to ``+-amp_max`` and
    each tau to at least half the first sample time (see
    :func:`sequential_exp_fit`), and when a fit lands a CANCELLING pair within
    ``degen_tau_ratio`` in tau, the model is collapsed — the fastest start
    fraction (sequential) or the smallest-|amp| component (joint) is dropped
    and the fit rerun — until clean (or one component remains).
    ``components``/``best_fractions`` may therefore be SHORTER than requested;
    that is the honest model. ``fixed_taus`` disables the collapse (the caller
    pinned the model deliberately). A final component with ``|amp| > amp_max``
    after t = 0 re-referencing fails the fit instead of reporting a tap no
    hardware should see.
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
    if tau_seeds is not None:
        if fixed_taus is not None:
            raise ValueError("tau_seeds and fixed_taus are mutually exclusive")
        tau_seeds = [float(x) for x in tau_seeds]
        if not tau_seeds or any(
            not np.isfinite(tau) or tau <= 0 for tau in tau_seeds
        ):
            raise ValueError("tau_seeds must be non-empty positive finite taus")

    failed: Dict[str, Any] = {
        "success": False, "components": [],
        "a_dc": float("nan") if a_dc is None else float(a_dc),
        "rms": float("nan"), "best_fractions": list(start_fractions),
        "best_fit": np.full_like(y, np.nan),
    }
    if t.size != y.size or t.size < MIN_SAMPLES or not np.all(np.isfinite(y)):
        return failed

    if tau_seeds is not None:
        return _joint_tau_fit(
            t, y, tau_seeds, a_dc=a_dc, amp_max=amp_max,
            degen_tau_ratio=degen_tau_ratio, log=log, failed=failed,
        )

    def _fit_once(fracs: List[float]):
        """One full pass: Nelder-Mead over these fractions + the final fit."""

        def objective(x: np.ndarray) -> float:
            if not np.all(np.diff(x) < 0):
                return 1e6
            components, _, residual = sequential_exp_fit(
                t, y, x, fixed_taus=fixed_taus, a_dc=a_dc, amp_max=amp_max,
            )
            if len(components) != len(fracs):
                return 1e6
            return float(np.sqrt(np.mean(residual ** 2)))

        bounds = [
            (start * (1.0 - bounds_scale), start * (1.0 + bounds_scale))
            for start in fracs
        ]
        result = minimize(
            objective, x0=fracs, bounds=bounds, method="Nelder-Mead",
            options={"disp": False, "maxiter": maxiter},
        )
        best = list(result.x) if result.success else list(fracs)
        components, fitted_dc, residual = sequential_exp_fit(
            t, y, best, fixed_taus=fixed_taus, a_dc=a_dc, log=log,
            amp_max=amp_max,
        )
        return bool(result.success), best, components, fitted_dc, residual

    # Degeneracy collapse: two components landing on (near-)identical taus are
    # one physical component split into a cancelling pair — drop the fastest
    # start fraction and refit until the taus separate or one component is
    # left. Unconditional (no rms comparison): degenerate parameters are
    # meaningless no matter how well their sum scores.
    fracs = list(start_fractions)
    converged, best_fractions, components, fitted_dc, residual = _fit_once(fracs)
    while (
        fixed_taus is None
        and len(fracs) > 1
        and len(components) > 1
        and _has_degenerate_pair(components, degen_tau_ratio)
    ):
        fracs = fracs[:-1]
        if log:
            log(
                f"tau-degenerate pair detected — collapsing to "
                f"{len(fracs)} component(s)"
            )
        converged, best_fractions, components, fitted_dc, residual = _fit_once(
            fracs
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
        converged
        and len(components) == len(fracs)
        and all(np.isfinite(tau) and tau > 0 for _, tau in components)
        and all(np.isfinite(amp) for amp, _ in components)
        # the t = 0 re-reference can only blow past amp_max when tau crowds
        # its identifiability floor — an unphysical tap, refused honestly.
        and all(abs(amp) <= amp_max for amp, _ in components)
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
