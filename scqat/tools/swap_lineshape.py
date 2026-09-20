"""Per-trace central-peak fit of a FIXED-TIME swap lineshape.

One column of a fixed-duration swap map (the transfer versus the detuning knob,
at one coupler setting) in, the exchange ANGLE out. This is the per-trace
REDUCTION shared by every fixed-time swap reading, and it lives in ``tools/``
for the same reason :func:`scqat.tools.fit_cosine.fit_swap_oscillation` does:
sharing a numerical routine is how two estimators share math without a second
model binding.

THE MODEL AND ITS CONVENTION
----------------------------
With detuning ``delta`` and coupling ``J`` (both Hz), the transfer after a pulse
of length ``t`` is::

    P(delta) = (2J)^2 / omega^2 * sin^2(pi * omega * t)
    omega    = sqrt(delta^2 + (2J)^2)

so ON RESONANCE ``omega = 2J`` and ``P_peak = sin^2(theta)`` with::

    theta = 2 * pi * J * t        (a full swap is theta = pi/2, i.e. t = 1/(4J))

That is the Hz convention the ``j_hz`` catalog field and the swap simulators use
(``t_full_swap = 1/(4 J)``). Note SCQO's ``TUTORIAL.md`` states the same physics
as ``J = pi/(2 t_pi)``, which is the ANGULAR (rad/s) convention — a factor 2*pi
away. Everything here is Hz.

WHAT THIS FIT USES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
Only the PEAK HEIGHT carries J, and only if the trace is normalized so the
amplitude is pinned (the single-excitation normalization
``p_transfer / (p01 + p10)``, whose denominator is detuning-independent under a
common T1). The fitted WIDTH is NOT a second measurement of J: solving the
half-maximum condition of the model above gives ``hwhm_delta * t`` of about
0.44 / 0.43 / 0.40 / 0.32 for ``theta`` = pi/8 / pi/4 / pi/2 / 3pi/4, i.e. the
width is dominated by the pulse's own Fourier limit ``~1/(2t)`` and it NARROWS
as the coupling grows. ``hwhm`` is reported as a DIAGNOSTIC only.

THE BRANCH. ``arcsin`` returns the principal branch ``theta`` in [0, pi/2]. A
map taken at a duration near a full swap will push the strong-coupling end past
pi/2, where ``sin^2`` folds back and J is UNDER-reported. The peak does not
split when that happens — for ``theta`` in (0, pi) the maximum stays exactly on
resonance (the derivative of ``sin^2(pi omega t)/omega^2`` is negative while
``omega t < 1``), it merely comes back down — so the fold is invisible in a
single trace. The caller detects it across a curve (peak and width falling
TOGETHER) and the real fix is a shorter pulse, not an unfolding.

The central peak is not exactly Lorentzian (the ``sin^2`` factor varies across
it), so the fit is restricted to a window around the maximum: the top, which is
the physics, is not dragged by the wings. The residual shape mismatch biases
``theta`` HIGH by a few percent at intermediate angles (measured on synthetic
traces: exact at ``theta`` = pi/10 and pi/2, about +7% near 1.1 rad). That is a
bring-up-grade number, not a spectroscopy-grade one — quote it as such.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .fit_lorentzian import FitLorentzian

__all__ = ["fit_swap_peak", "theta_from_peak", "j_hz_from_theta",
           "theta_from_j_poly", "coupler_flux_for_theta", "parabolic_vertex"]

#: points in the smooth curve returned for plotting.
_DENSE_POINTS = 501

#: a fitted peak above this is not a population — the fit ran away.
_MAX_PEAK = 1.05


def parabolic_vertex(x: np.ndarray, y: np.ndarray, i: int) -> tuple:
    """The peak's SUB-GRID position, as ``(value, refined)``.

    The sweep lands the true optimum between two swept points as often as on
    one, and a peak's three top points fix a parabola whose vertex recovers
    where it actually sat. Degrades to the grid point itself (``refined = 0``)
    when the maximum is at an end of the swept range or the three points do not
    curve downwards -- there is nothing to interpolate through then.

    Lives here rather than in an estimator because both swap-family estimators
    that refine a swept optimum need it (``qc_n_stark_amp`` along the stark
    axis, ``qc_swap_flux_stark`` along both), and the repo rule is that anything
    two estimators use is a ``tools/`` routine.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if i <= 0 or i >= y.size - 1:
        return float(x[i]), 0
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    curvature = y0 - 2.0 * y1 + y2
    if not np.isfinite(curvature) or curvature >= 0:
        return float(x[i]), 0
    # in units of the grid step, and bounded to the bin the peak belongs to
    delta = float(np.clip(0.5 * (y0 - y2) / curvature, -0.5, 0.5))
    step = float(x[i + 1] - x[i]) if delta > 0 else float(x[i] - x[i - 1])
    return float(x[i] + delta * step), 1


def theta_from_peak(peak: float) -> float:
    """Principal-branch exchange angle (rad) from a normalized peak transfer.

    ``P_peak = sin^2(theta)``, so ``theta = arcsin(sqrt(P_peak))`` in [0, pi/2];
    a full swap is ``pi/2``. The input is clipped to [0, 1] so shot noise just
    above 1 saturates instead of returning NaN. See the module docstring on why
    this is the PRINCIPAL branch only.
    """
    if not np.isfinite(peak):
        return float("nan")
    return float(np.arcsin(np.sqrt(np.clip(float(peak), 0.0, 1.0))))


def j_hz_from_theta(theta_rad, swap_time_ns):
    """Exchange coupling J (Hz) from the angle and the duration ACTUALLY played.

    ``theta = 2 pi J t`` so ``J = theta / (2 pi t)``. ``swap_time_ns`` None (a
    shaped pulse plays its own native length, which SCQO does not know) gives
    NaN — the angle stays valid, the Hz value is simply not available.
    """
    theta = np.asarray(theta_rad, dtype=float)
    usable = (swap_time_ns is not None
              and np.isfinite(float(swap_time_ns)) and float(swap_time_ns) > 0)
    if not usable:
        out = np.full(theta.shape, np.nan)
    else:
        out = theta / (2.0 * np.pi * float(swap_time_ns) * 1e-9)
    return out if out.ndim else float(out)


def theta_from_j_poly(coupler_flux_v, j2_coeffs, swap_time_ns):
    """Evaluate the empirical curve FORWARD: coupler flux -> angle.

    ``j2_coeffs`` are the polynomial in ``J^2`` (numpy order, highest power
    first) that ``pair_swap_flux_map`` fits and publishes. Fitting in ``J^2``
    rather than ``|J|`` is what keeps it smooth through the decouple point, so
    the coupling is ``sqrt(max(poly, 0))`` and the angle ``2*pi*J*t``.

    The polynomial is EMPIRICAL and interpolating: outside the coupler window it
    was fitted over it is an extrapolation with no data behind it, and near the
    decouple point one arm of a parabola is often held up by two or three points.
    Check the window before believing a value.
    """
    j_hz = np.sqrt(np.clip(np.polyval(np.asarray(j2_coeffs, dtype=float),
                                      np.asarray(coupler_flux_v, dtype=float)),
                           0.0, None))
    usable = (swap_time_ns is not None
              and np.isfinite(float(swap_time_ns)) and float(swap_time_ns) > 0)
    if not usable:
        return np.full(np.shape(j_hz), np.nan)
    return 2.0 * np.pi * j_hz * float(swap_time_ns) * 1e-9


def coupler_flux_for_theta(
    theta_rad: float, j2_coeffs, swap_time_ns: float, *,
    window=None, prefer: float | None = None,
) -> Dict[str, Any]:
    """Invert the empirical curve: which coupler flux delivers ``theta_rad``?

    Solves ``poly(V) = J_target^2`` with ``J_target = theta / (2 pi t)``. This is
    the conversion a gate calibration actually needs — the coupler flux is the
    angle knob and ``theta(Phi_c)`` is nowhere near linear, so a target angle has
    to be solved for, not scaled to.

    TWO ROOTS, ALWAYS, and that is physics not numerics: ``|J|`` has a MINIMUM at
    the decouple point and rises on both sides, so every reachable angle is
    delivered by one coupler flux below the decouple point and one above. Pass
    ``prefer`` — a coupler flux near where the sweep actually took data, e.g. the
    median of the successful columns — and the root nearest it is returned as
    ``coupler_flux_v``; every in-window root is returned in ``roots`` regardless,
    because picking for the caller without showing the alternative is how the
    wrong branch gets used silently.

    Parameters
    ----------
    theta_rad : float
        The wanted exchange angle (a full swap is ``pi/2``).
    j2_coeffs : array_like
        The published ``J^2`` polynomial, numpy order (highest power first).
    swap_time_ns : float
        The pulse length the angle is wanted AT. The polynomial is in J and is
        pulse-independent; the ANGLE is not.
    window : (float, float), optional
        The coupler range the polynomial was fitted over. Roots outside it are
        reported in ``roots_all`` but never chosen, and ``in_window`` is False.
    prefer : float, optional
        Reference coupler flux; the nearest root wins. Without it the smallest
        in-window root is returned, which is a coin toss on a two-branch curve.

    Returns
    -------
    dict
        ``coupler_flux_v`` (NaN when unreachable), ``j_hz`` (the target
        coupling), ``roots`` (real roots inside the window, ascending),
        ``roots_all`` (every real root), ``in_window`` and ``reason``.
    """
    out: Dict[str, Any] = {
        "coupler_flux_v": float("nan"), "j_hz": float("nan"),
        "roots": [], "roots_all": [], "in_window": False, "reason": "",
    }
    coeffs = np.asarray(j2_coeffs, dtype=float)
    if coeffs.size < 2 or not np.isfinite(coeffs).all():
        out["reason"] = "no usable J^2 polynomial"
        return out
    if not swap_time_ns or not np.isfinite(float(swap_time_ns)) or float(swap_time_ns) <= 0:
        out["reason"] = "the angle needs the duration actually played"
        return out
    if not np.isfinite(theta_rad) or theta_rad < 0:
        out["reason"] = "theta must be finite and non-negative"
        return out

    j_target = float(theta_rad) / (2.0 * np.pi * float(swap_time_ns) * 1e-9)
    out["j_hz"] = j_target

    shifted = coeffs.copy()
    shifted[-1] -= j_target ** 2
    try:
        roots = np.roots(shifted)
    except Exception:  # noqa: BLE001 - a degenerate polynomial is not an error
        out["reason"] = "the polynomial could not be solved"
        return out
    # A complex-conjugate pair means the curve never reaches this angle at all.
    real = sorted(float(r.real) for r in np.atleast_1d(roots)
                  if abs(np.imag(r)) <= 1e-9 * max(1.0, abs(float(np.real(r)))))
    out["roots_all"] = real
    if not real:
        out["reason"] = ("the fitted curve never reaches this angle "
                         "(below the decouple minimum, or past its maximum)")
        return out

    inside = real
    if window is not None:
        lo, hi = float(min(window)), float(max(window))
        inside = [r for r in real if lo <= r <= hi]
    out["roots"] = inside
    if not inside:
        out["reason"] = ("every solution lies outside the measured coupler "
                         "window — widen the sweep instead of extrapolating")
        return out

    chosen = (min(inside, key=lambda r: abs(r - float(prefer)))
              if prefer is not None and np.isfinite(prefer) else inside[0])
    out["coupler_flux_v"] = float(chosen)
    out["in_window"] = True
    if len(inside) > 1:
        out["reason"] = (f"{len(inside)} coupler fluxes deliver this angle "
                         f"(both sides of the decouple point); returned the one "
                         f"nearest {prefer}")
    return out


def _degenerate(x: np.ndarray, reason: str) -> Dict[str, Any]:
    """The NaN result a bad trace returns; never raises, never omits a key."""
    dense = (np.linspace(float(np.min(x)), float(np.max(x)), _DENSE_POINTS)
             if x.size > 1 else np.full(_DENSE_POINTS, np.nan))
    return {
        "peak": float("nan"), "x0": float("nan"), "hwhm": float("nan"),
        "offset": float("nan"), "theta_rad": float("nan"),
        "r_squared": float("nan"), "success": False,
        "best_fit": np.full(x.shape, np.nan),
        "x_dense": dense,
        "best_fit_dense": np.full(_DENSE_POINTS, np.nan),
        "fit_report": reason,
    }


def _width_seed(x: np.ndarray, y: np.ndarray, i_peak: int, step: float) -> float:
    """Half-width seed: the span where the trace stays above half its height."""
    base = float(np.nanmedian(y))
    height = float(y[i_peak]) - base
    if not np.isfinite(height) or height <= 0:
        return 5.0 * step
    above = np.isfinite(y) & (y >= base + 0.5 * height)
    if above.sum() >= 2:
        seed = 0.5 * float(np.ptp(x[above]))
        if seed > 0:
            return seed
    return 5.0 * step


def fit_swap_peak(
    x, y, *, window_factor: float = 3.0, min_contrast: float = 0.05,
    min_r_squared: float = 0.5,
) -> Dict[str, Any]:
    """Fit the central peak of ONE fixed-time swap trace and report its angle.

    Parameters
    ----------
    x : array_like
        The detuning knob axis — for ``pair_swap_flux_map`` the member's flux
        amplitude in volts. Any monotonic axis works; the returned ``x0`` and
        ``hwhm`` carry its units.
    y : array_like
        The NORMALIZED transfer in 0..1 (see the module docstring: the peak
        height is the physics only once the amplitude is pinned). NaN entries
        are ignored.
    window_factor : float, optional
        The fit runs within ``window_factor`` seeded half-widths of the maximum,
        so the wings cannot drag the top. Default 3.0.
    min_contrast : float, optional
        Reject a flat fit: the fitted amplitude must exceed
        ``min_contrast * ptp(y)`` in the window. Default 0.05.
    min_r_squared : float, optional
        Reject a noise fit: the Lorentzian must beat a constant by this much.
        Default 0.5 (the same scale-free gate ``fit_swap_oscillation`` uses).

    Returns
    -------
    dict
        ``peak`` (fitted height at ``x0``, i.e. ``amplitude + offset``), ``x0``
        (the resonance point), ``hwhm`` (DIAGNOSTIC, not a J measurement),
        ``offset``, ``theta_rad`` (principal branch), ``r_squared``, ``success``,
        plus the arrays ``best_fit`` (sampled at the FULL input ``x``),
        ``x_dense`` / ``best_fit_dense`` and the lmfit ``fit_report``.

        **Never raises.** A trace that is too short, all-NaN or unfittable
        degrades to NaN fields with ``success=False``, so a caller looping over
        map columns keeps its raw figure — the "raw data must always be
        plottable" rule.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Four free parameters (x0, amplitude, gamma, offset) need four points.
    finite = np.isfinite(y) & np.isfinite(x)
    if x.size < 4 or finite.sum() < 4:
        return _degenerate(x, "too few finite points to fit a peak")

    step = float(np.min(np.abs(np.diff(x)))) if x.size > 1 else 0.0
    if not np.isfinite(step) or step <= 0:
        return _degenerate(x, "degenerate x axis")

    idx = np.flatnonzero(finite)
    local = int(np.argmax(y[idx]))
    i_peak = int(idx[local])
    half = _width_seed(x[idx], y[idx], local, step)

    # The window around the maximum. Widen it until it holds enough points — a
    # peak sitting on the edge of the sweep still has to be fittable.
    keep = finite & (np.abs(x - x[i_peak]) <= window_factor * half)
    grow = window_factor
    while keep.sum() < 5 and grow < 64.0:
        grow *= 2.0
        keep = finite & (np.abs(x - x[i_peak]) <= grow * half)
    if keep.sum() < 4:
        keep = finite
    xw, yw = x[keep], y[keep]

    lo, hi = float(np.min(xw)), float(np.max(xw))
    span = hi - lo if hi > lo else step
    try:
        # The gamma floor is a QUARTER of the grid step, deliberately below the
        # acceptance gate: an unresolved peak must be free to converge under the
        # gate and be rejected, not get pinned ON it and pass by a rounding hair.
        fitter = FitLorentzian(yw, x=xw,
                               bounds={"x0": (lo, hi),
                                       "gamma": (0.25 * step, 2.0 * span)})
        fitter.guess()
        fitter.params["x0"].set(value=float(x[i_peak]))
        fitter.params["amplitude"].set(
            value=max(float(y[i_peak]) - float(np.nanmedian(yw)), 1e-3),
            min=0.0, max=_MAX_PEAK)
        fitter.params["offset"].set(
            value=float(np.clip(np.nanmin(yw), 0.0, 1.0)), min=0.0, max=_MAX_PEAK)
        fitter.params["gamma"].set(
            value=float(np.clip(half, 0.25 * step, 2.0 * span)))
        result = fitter.fit()
    except Exception as err:  # noqa: BLE001 - a bad column must not sink the map
        return _degenerate(x, f"{type(err).__name__}: {err}")

    p = {k: v.value for k, v in result.params.items()}
    peak = float(p["amplitude"] + p["offset"])

    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    r_squared = 1.0 - float(result.chisqr) / ss_tot if ss_tot > 0 else float("nan")

    ptp = float(np.ptp(yw))
    # Every gate, in one place. A fit pinned on the x0 bound walked out of the
    # window rather than converging inside it; a "peak" above 1 is not a
    # population; and a half-width under one grid step means the sweep never
    # RESOLVED the peak, so its height is one sample and not a measurement
    # (the gamma FLOOR sits at a quarter step so an unresolved fit can fall
    # below this gate instead of resting on the bound and passing).
    success = bool(
        bool(result.success)
        and np.isfinite(r_squared) and r_squared > min_r_squared
        and ptp > 0 and p["amplitude"] > min_contrast * ptp
        and 0.0 < peak <= _MAX_PEAK
        and lo < p["x0"] < hi
        and step < p["gamma"] < 2.0 * span
    )

    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), _DENSE_POINTS)
    return {
        "peak": peak,
        "x0": float(p["x0"]),
        # NOT a J measurement — see the module docstring.
        "hwhm": float(p["gamma"]),
        "offset": float(p["offset"]),
        "theta_rad": theta_from_peak(peak),
        "r_squared": float(r_squared),
        "success": success,
        # evaluated over the FULL axis so a caller can store it as one column of
        # the map, not just over the window the fit ran in
        "best_fit": np.asarray(result.eval(x=x), dtype=float),
        "x_dense": x_dense,
        "best_fit_dense": np.asarray(result.eval(x=x_dense), dtype=float),
        "fit_report": result.fit_report(),
    }
