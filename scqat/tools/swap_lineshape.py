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

__all__ = ["fit_swap_peak", "theta_from_peak", "j_hz_from_theta"]

#: points in the smooth curve returned for plotting.
_DENSE_POINTS = 501

#: a fitted peak above this is not a population — the fit ran away.
_MAX_PEAK = 1.05


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
