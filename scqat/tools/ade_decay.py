"""Analytical Decay Estimation (ADE) — closed-form T1 from three delays.

Reference: arXiv:2602.11912 (point estimate Sec. II, uncertainty App. F).

With an exponential ``P(t) = A * exp(-gamma * t) + B`` sampled at the three
delays ``t0``, ``t0 + dt``, ``t0 + 3*dt``, the offset and amplitude (and any
SPAM error absorbed into them) cancel in the ratio

    c = (P3 - P0) / (P1 - P0) = 1 + r + r^2,      r = exp(-gamma * dt)

so ``r = sqrt(c - 3/4) - 1/2`` and ``gamma = -ln(r) / dt`` — no confusion
matrix required. The 1:3 spacing is what makes the quadratic solvable in
closed form.

Validity domain: a genuine decay has ``P0 > P1 > P3``, i.e. ``1 < c < 3``
(``c <= 1`` means no decay resolved between the first two delays; ``c >= 3``
means gamma <= 0). Outside it — or on a zero denominator — every function
here reports NaN with ``valid=False``, never a clipped number: these are the
blocks an on-FPGA fixed-point implementation floors, and the host-side truth
is "no estimate", not the floor value.

Result contract (all vectorized over leading block axes, NaN on invalid):

* :func:`ade_gamma`             -> ``(gamma, valid)``; gamma in 1/s.
* :func:`ade_sigma_gamma`       -> analytic shot-noise sigma of gamma (1/s):
  binomial ``sigma_Pi = sqrt(P(1-P)/n)`` chained through
  ``dgamma/dc = -1 / (2 x (x+1/2) dt)`` and the quotient rule on ``c``.
* :func:`ade_bootstrap_sigma_t1` -> per-block bootstrap sigma of T1 (s) from
  the raw per-shot outcomes: resample each delay's shots independently with
  replacement, re-run the closed form, report ``(P84 - P16) / 2`` of the
  finite ``T1`` draws (NaN when fewer than ``min_finite`` draws survive).
"""

from typing import Tuple

import numpy as np

#: bootstrap draws that must survive (finite T1) for a percentile sigma to be
#: reported at all — below this the percentiles are noise about noise.
ADE_BOOTSTRAP_MIN_FINITE = 10


def ade_gamma(p0: np.ndarray, p1: np.ndarray, p3: np.ndarray,
              dt_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form decay rate from the three-delay populations.

    Parameters are broadcast together; ``dt_s`` is the delay spacing in
    seconds. Returns ``(gamma, valid)`` — gamma in 1/s, NaN where invalid.
    """
    p0, p1, p3, dt_s = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (p0, p1, p3, dt_s))
    )
    denom = p1 - p0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom != 0.0, (p3 - p0) / denom, np.nan)
        valid = np.isfinite(c) & (c > 1.0) & (c < 3.0) & (dt_s > 0)
        x = np.sqrt(np.where(valid, c - 0.75, np.nan)) - 0.5
        gamma = np.where(valid, -np.log(x) / dt_s, np.nan)
    return gamma, valid


def ade_sigma_gamma(p0: np.ndarray, p1: np.ndarray, p3: np.ndarray,
                    dt_s: np.ndarray, n_avg: int) -> np.ndarray:
    """Analytic shot-noise sigma of :func:`ade_gamma` (1/s; NaN where invalid).

    Binomial ``sigma_Pi`` at each delay (independent), chained through
    ``dgamma/dPi = dgamma/dc * dc/dPi``; ``dc/dP0`` uses the full quotient
    rule since P0 appears in numerator and denominator.
    """
    p0, p1, p3, dt_s = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (p0, p1, p3, dt_s))
    )
    _, valid = ade_gamma(p0, p1, p3, dt_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = p1 - p0
        denom_sq = denom * denom
        c = np.where(valid, (p3 - p0) / denom, np.nan)
        x = np.sqrt(c - 0.75) - 0.5
        dgamma_dc = -1.0 / (2.0 * x * (x + 0.5) * dt_s)
        dc_dp0 = (p3 - p1) / denom_sq
        dc_dp1 = -(p3 - p0) / denom_sq
        dc_dp3 = 1.0 / denom
        inv_n = 1.0 / float(n_avg)
        var = (
            (dgamma_dc * dc_dp0) ** 2 * p0 * (1.0 - p0) * inv_n
            + (dgamma_dc * dc_dp1) ** 2 * p1 * (1.0 - p1) * inv_n
            + (dgamma_dc * dc_dp3) ** 2 * p3 * (1.0 - p3) * inv_n
        )
        sigma = np.where(valid, np.sqrt(var), np.nan)
    return sigma


def ade_bootstrap_sigma_t1(shots0: np.ndarray, shots1: np.ndarray,
                           shots3: np.ndarray, dt_s: np.ndarray,
                           n_bootstrap: int, *, seed: int = 0,
                           min_finite: int = ADE_BOOTSTRAP_MIN_FINITE) -> np.ndarray:
    """Per-block bootstrap sigma of T1 (seconds) from raw per-shot outcomes.

    Parameters
    ----------
    shots0, shots1, shots3 : (n_blocks, n_avg) arrays of 0/1 outcomes
        The per-shot excited-state assignments at the three delays.
    dt_s : (n_blocks,) array
        Delay spacing per block, seconds.
    n_bootstrap : int
        Resamples per block; independent shot indices per delay.
    seed : int
        RNG seed — bootstrap output is deterministic for a given seed.
    min_finite : int
        Minimum finite T1 draws for a percentile sigma (else NaN).

    Returns
    -------
    (n_blocks,) array — ``(P84 - P16) / 2`` of the finite T1 draws, seconds.
    """
    shots0 = np.asarray(shots0, dtype=float)
    shots1 = np.asarray(shots1, dtype=float)
    shots3 = np.asarray(shots3, dtype=float)
    if shots0.ndim != 2 or shots0.shape != shots1.shape or shots0.shape != shots3.shape:
        raise ValueError(
            "shots0/shots1/shots3 must share one (n_blocks, n_avg) shape, got "
            f"{shots0.shape}, {shots1.shape}, {shots3.shape}"
        )
    n_blocks, n_avg = shots0.shape
    dt_s = np.broadcast_to(np.asarray(dt_s, dtype=float), (n_blocks,))
    if n_bootstrap <= 0:
        return np.full(n_blocks, np.nan)

    rng = np.random.default_rng(seed)
    sigma_t1 = np.full(n_blocks, np.nan)
    for b in range(n_blocks):
        idx = rng.integers(0, n_avg, size=(3, int(n_bootstrap), n_avg))
        p0 = shots0[b][idx[0]].mean(axis=1)
        p1 = shots1[b][idx[1]].mean(axis=1)
        p3 = shots3[b][idx[2]].mean(axis=1)
        gamma, _ = ade_gamma(p0, p1, p3, dt_s[b])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = 1.0 / gamma
        t1 = t1[np.isfinite(t1)]
        if t1.size >= min_finite:
            lo, hi = np.percentile(t1, [16.0, 84.0])
            sigma_t1[b] = 0.5 * (hi - lo)
    return sigma_t1
