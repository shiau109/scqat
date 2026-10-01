"""A repeated partial swap as a physical channel, and its fit to tomography.

THE MODEL. Two members, basis ``|00>, |01>, |10>, |11>`` (member 0 = ``high``
first). One step is

1. the exchange ``exp(-i theta (|10><01| + h.c.))``;
2. a relative Z phase ``phi``: ``|10>`` (and ``|11>``) gain ``exp(-i phi)``;
3. amplitude damping ``p_high`` on member 0 and ``p_low`` on member 1;
4. dephasing: a phase flip on member 0 with probability ``lam/2``, i.e. the
   single-excitation coherence shrinks by ``1 - lam`` per step.

The state after ``N`` steps is recorded through a fixed final Z rotation
``a_off`` (the share of the per-step phase that falls after the last exchange -
a frame offset, not a property of the swap). The prepared state is the excited
member's ``|1>`` with probability ``1 - eps`` and ``|00>`` otherwise.

In the single-excitation subspace (up = ``|10>``) one step is
``Rz(phi) Rx(2 theta)``, a rotation by ``omega`` about ``n`` with

    cos(omega/2) = cos(theta) cos(phi/2),
    n ~ (sin theta cos(phi/2), sin theta sin(phi/2), cos theta sin(phi/2)),

so a 3D trajectory fixes ``theta`` and ``phi`` separately - population data
alone only see ``omega``. A detuned exchange is EXACTLY an equatorial one of a
slightly smaller angle between two equal Z rotations, so a detuning shows up
here as part of ``phi`` (first order) and of ``theta`` (second order); the model
cannot and does not separate it.

WHY A CHANNEL AND NOT A SHRINK FACTOR. Dephasing shrinks only the transverse
components, T1 moves population out of the subspace, and unequal T1 tilts z. A
single isotropic shrink per step pulls ``theta`` low whenever ``phi != 0``
(-6 % at theta 0.4, -14..-31 % at 0.126 in a Monte Carlo at 5Q4C's numbers);
this channel is unbiased on the same data.

The fit works on five unnormalized features per step, from the reconstructed
density matrix: ``2 Re rho_{10,01}``, ``-2 Im rho_{10,01}``, ``rho_{10,10}``,
``rho_{01,01}``, ``rho_{00,00}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

#: names of the fitted parameters, in vector order.
PARAM_NAMES: tuple[str, ...] = ("theta", "phi", "a_off", "p_high", "p_low", "lam", "eps")
_LOWER = np.array([0.0, -np.pi, -np.pi, 0.0, 0.0, 0.0, 0.0])
_UPPER = np.array([np.pi / 2, np.pi, np.pi, 0.5, 0.5, 0.9, 0.5])

_SM = np.array([[0.0, 1.0], [0.0, 0.0]])
_I2 = np.eye(2)


def _exchange(theta: float) -> np.ndarray:
    u = np.eye(4, dtype=complex)
    u[1, 1] = u[2, 2] = np.cos(theta)
    u[1, 2] = u[2, 1] = -1j * np.sin(theta)
    return u


def _phase_high(phi: float) -> np.ndarray:
    return np.diag([1.0, 1.0, np.exp(-1j * phi), np.exp(-1j * phi)])


def _damping(p: float, first: bool) -> list[np.ndarray]:
    e1 = np.diag([1.0, np.sqrt(max(1.0 - p, 0.0))])
    e2 = np.sqrt(max(p, 0.0)) * _SM
    return [np.kron(k, _I2) if first else np.kron(_I2, k) for k in (e1, e2)]


def _dephase_high(lam: float) -> list[np.ndarray]:
    z = np.kron(np.diag([1.0, -1.0]), _I2)
    return [np.sqrt(max(1.0 - lam / 2, 0.0)) * np.eye(4), np.sqrt(max(lam / 2, 0.0)) * z]


def _superop(kraus: list[np.ndarray]) -> np.ndarray:
    """The 16x16 map on row-major vec(rho): vec(K rho K^dag) = (K (x) K*) vec(rho)."""
    return sum(np.kron(k, k.conj()) for k in kraus)


def features_of(rho: np.ndarray) -> np.ndarray:
    """The five fit features of one (or many, leading axes) density matrices."""
    r = np.asarray(rho)
    c = r[..., 2, 1]
    return np.stack([2 * np.real(c), -2 * np.imag(c), np.real(r[..., 2, 2]),
                     np.real(r[..., 1, 1]), np.real(r[..., 0, 0])], axis=-1)


def channel_states(params, n_max: int, excite_high: bool = True) -> np.ndarray:
    """Density matrices after 0..n_max steps, shape ``(n_max + 1, 4, 4)``."""
    theta, phi, a_off, p_high, p_low, lam, eps = (float(v) for v in params)
    rho = np.zeros((4, 4), dtype=complex)
    rho[2 if excite_high else 1, 2 if excite_high else 1] = 1.0 - eps
    rho[0, 0] = eps
    # one step = unitary, then each member's damping, then the dephasing
    step = (_superop(_dephase_high(lam)) @ _superop(_damping(p_low, False))
            @ _superop(_damping(p_high, True))
            @ _superop([_phase_high(phi) @ _exchange(theta)]))
    record = _superop([_phase_high(a_off)])
    vec = rho.reshape(16)
    out = np.empty((n_max + 1, 16), dtype=complex)
    for n in range(n_max + 1):
        out[n] = vec
        vec = step @ vec
    return (out @ record.T).reshape(n_max + 1, 4, 4)


def channel_features(params, counts: np.ndarray, excite_high: bool = True) -> np.ndarray:
    """The model's five features at the given (non-negative integer) counts."""
    counts = np.asarray(counts, dtype=int)
    states = channel_states(params, int(counts.max()), excite_high)
    return features_of(states[counts])


# ---------------------------------------------------------------- the seed
def _rotation_seed(features: np.ndarray, counts: np.ndarray) -> tuple[float, float, float] | None:
    """(theta, phi, a_off) from the best rotation between consecutive steps.

    Kabsch on the normalized subspace Bloch vectors: the per-step map seen in
    the recorded frame is ``Rz(a_off) C Rz(-a_off)``, whose angle and axis tilt
    give theta and phi and whose axis azimuth (phi/2 + a_off) gives a_off.
    None when fewer than two usable consecutive pairs exist.
    """
    p_sub = features[:, 2] + features[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.stack([features[:, 0], features[:, 1],
                      features[:, 2] - features[:, 3]], axis=1) / p_sub[:, None]
    index = {int(c): i for i, c in enumerate(counts)}
    pairs = [(index[c], index[c + 1]) for c in index if c + 1 in index]
    pairs = [(i, j) for i, j in pairs
             if np.all(np.isfinite(v[i])) and np.all(np.isfinite(v[j]))
             and p_sub[i] > 0.2 and p_sub[j] > 0.2]
    if len(pairs) < 2:
        return None
    u = np.array([v[i] for i, _ in pairs])
    w = np.array([v[j] for _, j in pairs])
    h = u.T @ w
    left, _, right_t = np.linalg.svd(h)
    d = np.sign(np.linalg.det(right_t.T @ left.T)) or 1.0
    rot = right_t.T @ np.diag([1.0, 1.0, d]) @ left.T
    cos_omega = np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)
    omega = float(np.arccos(cos_omega))
    if omega < 1e-6:
        return None
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return None
    axis /= norm
    s, c = np.sin(omega / 2), np.cos(omega / 2)
    theta = float(np.arcsin(np.clip(s * np.hypot(axis[0], axis[1]), 0.0, 1.0)))
    phi = float(2 * np.arctan2(axis[2] * s, c))
    a_off = float(np.angle(np.exp(1j * (np.arctan2(axis[1], axis[0]) - phi / 2))))
    return theta, phi, a_off


def _wrap(angle: float) -> float:
    return float(np.angle(np.exp(1j * angle)))


def fit_swap_channel(counts, features, excite_high: bool = True) -> dict[str, Any]:
    """Fit the channel to measured features ``(len(counts), 5)``.

    Starts from the rotation seed plus a small grid in phi, keeps the best
    least-squares solution, and returns every parameter with a standard error
    (from the Jacobian, scaled by the residual variance), ``rms`` of the
    residuals, the model features at the counts, and ``success``.
    Never raises on bad data: a failed fit returns NaN values and
    ``success = False``.
    """
    counts = np.asarray(counts, dtype=int)
    data = np.asarray(features, dtype=float)
    nan = {name: float("nan") for name in PARAM_NAMES}
    failed = {**nan, **{f"{k}_err": float("nan") for k in PARAM_NAMES},
              "rms": float("nan"), "success": False,
              "model": np.full(data.shape, np.nan)}
    good = np.all(np.isfinite(data), axis=1)
    if counts.size < 4 or good.sum() < 4 or counts.min() < 0:
        return failed
    counts, data = counts[good], data[good]

    def residual(p):
        return (channel_features(p, counts, excite_high) - data).ravel()

    # seed: transfer after the first step for theta, the rotation seed for the rest
    order = np.argsort(counts)
    p0_sub = data[order[0], 2] + data[order[0], 3]
    eps0 = float(np.clip(1.0 - p0_sub, 0.0, 0.4))
    theta0 = 0.3
    one = np.flatnonzero(counts == 1)
    if one.size:
        partner = data[one[0], 3] if excite_high else data[one[0], 2]
        sub = data[one[0], 2] + data[one[0], 3]
        if sub > 0.2:
            theta0 = float(np.arcsin(np.sqrt(np.clip(partner / sub, 0.0, 1.0))))
    def start(theta_s, phi_s, a_s):
        return np.clip([max(theta_s, 1e-3), _wrap(phi_s), _wrap(a_s), 0.02, 0.02, 0.05, eps0],
                       _LOWER + 1e-9, _UPPER - 1e-9)

    # The rotation seed is usually right; a coarse phi x a_off grid, screened by
    # its starting cost, guards the rest. Only the seed and the two best grid
    # points are optimized.
    grid = [start(theta0, phi0, a0)
            for phi0 in np.linspace(-np.pi, np.pi, 8, endpoint=False)
            for a0 in (0.0, np.pi / 2, np.pi, -np.pi / 2)]
    grid.sort(key=lambda x0: float(np.sum(residual(x0) ** 2)))
    seed = _rotation_seed(data, counts)
    starts = ([start(*seed)] if seed is not None else []) + grid[:2]
    best = None
    for x0 in starts:
        try:
            sol = least_squares(residual, x0, bounds=(_LOWER, _UPPER), x_scale="jac")
        except (ValueError, np.linalg.LinAlgError):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None or not best.success:
        return failed

    x = best.x.copy()
    x[1], x[2] = _wrap(x[1]), _wrap(x[2])
    dof = max(best.fun.size - x.size, 1)
    sigma2 = 2.0 * best.cost / dof
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * sigma2
        err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        err = np.full(x.size, np.nan)
    out: dict[str, Any] = {name: float(v) for name, v in zip(PARAM_NAMES, x)}
    out.update({f"{name}_err": float(e) for name, e in zip(PARAM_NAMES, err)})
    out["rms"] = float(np.sqrt(np.mean(best.fun ** 2)))
    model = np.full((good.size, 5), np.nan)
    model[good] = channel_features(x, counts, excite_high)
    out["model"] = model
    out["success"] = bool(np.isfinite(x).all())
    return out


def phase_root(amps, phases) -> tuple[float, bool]:
    """The stark amplitude where the per-step phase crosses zero (mod 2 pi).

    ``phases`` are wrapped to (-pi, pi]; they are unwrapped along ascending
    ``amps`` and fitted with a polynomial of degree ``min(2, n - 1)``. Among the
    roots of ``poly(a) = 2 pi k`` the one inside the swept range wins (closest to
    the amplitude whose phase is smallest); with none inside, the nearest one
    outside is returned and flagged ``extrapolated``. ``(nan, False)`` when fewer
    than two finite points exist.
    """
    a = np.asarray(amps, dtype=float)
    ph = np.asarray(phases, dtype=float)
    ok = np.isfinite(a) & np.isfinite(ph)
    if ok.sum() < 2:
        return float("nan"), False
    a, ph = a[ok], ph[ok]
    order = np.argsort(a)
    a, ph = a[order], np.unwrap(ph[order])
    deg = min(2, a.size - 1)
    coeffs = np.polyfit(a, ph, deg)
    lo, hi = float(a.min()), float(a.max())
    span = hi - lo
    values = np.polyval(coeffs, np.linspace(lo - span, hi + span, 41))
    ks = range(int(np.floor(values.min() / (2 * np.pi))) - 1,
               int(np.ceil(values.max() / (2 * np.pi))) + 2)
    roots = []
    for k in ks:
        shifted = coeffs.copy()
        shifted[-1] -= 2 * np.pi * k
        for r in np.roots(shifted):
            if abs(r.imag) < 1e-9:
                roots.append(float(r.real))
    if not roots:
        return float("nan"), False
    anchor = float(a[np.argmin(np.abs(np.angle(np.exp(1j * ph))))])
    inside = [r for r in roots if lo <= r <= hi]
    if inside:
        return min(inside, key=lambda r: abs(r - anchor)), False
    return min(roots, key=lambda r: min(abs(r - lo), abs(r - hi))), True
