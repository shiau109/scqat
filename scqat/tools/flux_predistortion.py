"""Flux predistortion — neutral (A_i, tau_i) sum <-> single-pole cascade converter.

Turns the recorded flux-distortion facts ``distortion_amp[]`` (relative amplitudes
``A_i = amp/a_dc``) + ``distortion_tau_s[]`` (time constants ``tau_i``) — a FORWARD
LTI step response ``s(t) = a_dc*(1 + sum_i A_i*exp(-t/tau_i))`` — into the
coefficient forms an instrument predistortion filter consumes. Pure math: it emits
GENERIC coefficient arrays (each DRIVER names them ``exponential_filter`` /
``exp_coeffs``), imports no vendor library, and stays instrument-agnostic.

The sum <-> cascade decomposition is ported faithfully from the QM (iqcc)
cryoscope toolkit (``cryoscope_tools.py``: ``decompose_exp_sum_to_cascade`` +
``add_rational_terms`` + ``get_rational_filter_single_exp_cont_time`` +
``get_scaling_of_v34_fpga_filter``). Firmware that takes the sum directly
(QM QOP >= 3.3, Qblox exp stages) needs no decomposition; the single-pole CASCADE
form is for QM QOP 3.4.1. The pre-QOP-3.3 ``feedback_filter`` FIR/IIR form (QM's
``single_exp``) is deliberately not ported — no current instrument consumes it.

Unit convention
---------------
Pass ``tau`` and the sample period ``ts`` in the SAME unit (SI seconds
recommended); ``tau_c`` comes back in that unit. Amplitudes ``A_i`` are
dimensionless (already ``amp/a_dc``). The decomposition arithmetic is
scale-invariant in the time unit, so the returned ``amps_c`` and ``scale`` are
identical whether you work in seconds or nanoseconds.
"""

import math
from functools import reduce
from typing import Any, Dict, Sequence, Tuple

import numpy as np
from numpy.polynomial import Polynomial as _P

#: below this settled level the response is high-pass (bias-tee droop), which the
#: single-pole cascade cannot represent — the same guard the QM toolkit uses.
MIN_A_DC = 0.2


def exp_sum_step_response(
    amps: Sequence[float],
    taus_s: Sequence[float],
    t_s: np.ndarray,
    *,
    a_dc: float = 1.0,
) -> np.ndarray:
    """The forward step response ``a_dc*(1 + sum_i A_i*exp(-t/tau_i))``.

    The verification workhorse: a set of ``(A_i, tau_i)`` taps and any
    representation derived from them must reproduce this curve. ``t_s`` and
    ``taus_s`` share a unit.
    """
    amps = np.asarray(amps, dtype=float)
    taus = np.asarray(taus_s, dtype=float)
    t = np.asarray(t_s, dtype=float)
    out = np.ones_like(t, dtype=float)
    for a, tau in zip(amps, taus):
        out = out + a * np.exp(-t / tau)
    return a_dc * out


def _single_exp_rational(a: float, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """One continuous-time single-exp term as ``(numerator, denominator)`` coeffs."""
    return np.array([a]), np.array([1.0, 1.0 / tau])


def _add_rational_terms(
    terms: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Sum rational terms over a common denominator; return ``(num, den)`` coeffs."""
    rational = [(_P(num), _P(den)) for num, den in terms]
    common_den = reduce(lambda acc, t: acc * t[1], rational, _P([1.0]))
    adjusted = []
    for num, den in rational:
        multiplier = common_den // den
        adjusted.append(num * multiplier)
    final_num = sum(adjusted, _P([0.0]))
    return final_num.coef, common_den.coef


def _fpga_cascade_scale(amps_c: np.ndarray, taus_c: np.ndarray, ts: float) -> float:
    """The QM-FPGA discretization compensation for the v3.4 cascade. Unit-invariant
    in ``ts``/``tau`` (numerator and denominator scale together)."""
    return float(np.prod((ts + 2 * taus_c) / (ts + 2 * taus_c * (1 + amps_c))))


def exp_sum_to_cascade(
    amps: Sequence[float],
    taus_s: Sequence[float],
    *,
    a_dc: float = 1.0,
    ts_s: float = 1e-9,
    compensate_fpga: bool = True,
) -> Dict[str, Any]:
    """Decompose the exponential SUM into a CASCADE of single-pole stages.

    ``s(t) = a_dc*(1 + sum_i A_i*exp(-t/tau_i))``  ->  a cascade of stages
    ``s_i(t) = 1 + A_c[i]*exp(-t/tau_c[i])`` times an overall ``scale``.

    Returns ``{"amps_c": [...], "taus_c_s": [...], "scale": float}`` (``tau_c`` in
    the unit of ``taus_s``/``ts_s``). Raises ``ValueError`` when the configuration
    is high-pass (``a_dc <= MIN_A_DC``) or cannot be written as a real-pole cascade
    (complex poles/zeros).
    """
    amps = [float(a) for a in amps]
    taus = [float(t) for t in taus_s]
    if not amps:
        raise ValueError("exp_sum_to_cascade needs at least one (A, tau) term")
    if a_dc <= MIN_A_DC:
        raise ValueError(
            f"high-pass/HPF mode (a_dc={a_dc:.4g} <= {MIN_A_DC}) is not supported "
            "by the single-pole cascade decomposition")

    ba_sum = [_single_exp_rational(a, t) for a, t in zip(amps, taus)]
    ba_sum.append((np.array([a_dc]), np.array([1.0])))
    b, a = _add_rational_terms(ba_sum)
    zeros = np.sort(np.roots(b))
    poles = np.sort(np.roots(a))
    # tolerant real-root check (the QM assert was strict; a tiny numeric imaginary
    # part on a genuinely real root must not spuriously refuse a valid decomposition)
    scale_ref = max(1.0, float(np.max(np.abs(np.real(np.concatenate([zeros, poles]))))))
    if not (np.all(np.abs(np.imag(zeros)) < 1e-9 * scale_ref)
            and np.all(np.abs(np.imag(poles)) < 1e-9 * scale_ref)):
        raise ValueError(
            "cannot decompose to a real-pole cascade — the exponential sum has "
            "complex poles/zeros (it is not representable as single-pole stages)")
    zeros = np.real(zeros)
    poles = np.real(poles)

    taus_c = -1.0 / poles
    amps_c = poles / zeros - 1.0
    scale = 1.0 / a_dc
    if compensate_fpga:
        scale *= _fpga_cascade_scale(amps_c, taus_c, ts_s)

    return {
        "amps_c": [float(x) for x in amps_c],
        "taus_c_s": [float(x) for x in taus_c],
        "scale": float(scale),
    }


def partition_exp_stages(
    amps: Sequence[float],
    taus_s: Sequence[float],
    *,
    max_stages: int,
    amp_range: Tuple[float, float] = (-1.0, 1.0),
    tau_min_s: float = 0.0,
    tau_max_s: float = math.inf,
) -> Dict[str, Any]:
    """Split ``(A_i, tau_i)`` taps into what a hardware exponential-overshoot stage
    bank can hold and what it cannot — generic (the caller passes its instrument's
    limits; this module never names a vendor).

    A stage is KEPT only if its amplitude is within ``amp_range`` and its tau within
    ``[tau_min_s, tau_max_s]``; the in-bounds stages are ranked by ``|A|`` (most
    significant first) and the first ``max_stages`` are kept. Everything else —
    out-of-bounds or beyond the limit — is returned as ``overflow`` (never silently
    dropped), for the caller to route to a wideband/FIR path or refuse.

    Returns ``{"kept": [(A, tau), ...], "overflow": [(A, tau), ...], "notes": [...]}``.
    """
    lo, hi = amp_range
    in_bounds: list[Tuple[float, float]] = []
    overflow: list[Tuple[float, float]] = []
    notes: list[str] = []
    for a_raw, t_raw in zip(amps, taus_s):
        a, tau = float(a_raw), float(t_raw)
        if not (lo <= a <= hi):
            overflow.append((a, tau))
            notes.append(f"amplitude {a:.4g} outside [{lo}, {hi}]")
        elif not (tau_min_s <= tau <= tau_max_s):
            overflow.append((a, tau))
            notes.append(f"tau {tau:.4g} outside [{tau_min_s}, {tau_max_s}]")
        else:
            in_bounds.append((a, tau))
    in_bounds.sort(key=lambda p: abs(p[0]), reverse=True)
    kept = in_bounds[:max_stages]
    beyond = in_bounds[max_stages:]
    if beyond:
        notes.append(
            f"{len(beyond)} in-bounds stage(s) beyond the {max_stages}-stage limit")
    return {"kept": kept, "overflow": overflow + beyond, "notes": notes}
