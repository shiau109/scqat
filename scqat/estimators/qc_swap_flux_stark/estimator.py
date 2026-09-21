"""The fixed-N flux x AC-Stark map, and the stark amplitude that compensates.

Draws the four joint two-qubit populations over the flux-amplitude x
stark-amplitude grid at a FIXED number of swaps, and then reads the COMPENSATING
stark amplitude off that map: the control flux brings the members onto resonance
while the Stark tone nulls the phase they accumulate between swaps, and the two
are coupled, so the optimum is a RIDGE in the plane rather than a point.

THE CLOSED FORM, and why a fixed N is enough. In the single-excitation subspace
each round is an exchange by ``theta`` followed by a relative phase ``phi``, so
after N rounds the transfer is

    T(N) = sin^2(theta) * [ sin(N*theta_eff) / sin(theta_eff) ]^2
    cos(theta_eff) = cos(phi/2) * cos(theta)

i.e. ``T(N)/sin^2(theta)`` is the squared Chebyshev polynomial
``U_{N-1}(cos theta_eff)^2``. Three consequences drive everything below:

* ``T(1) = sin^2(theta)`` EXACTLY, with no ``phi`` in it. A single swap cannot
  see the between-round phase, which is why N=1 leaves this map's stark axis
  inert by construction -- and why the prior angle comes from a single-swap
  experiment (``pair_swap_flux_map``) in the first place.
* ``U_{N-1}`` is monotone on ``[cos(pi/N), 1]``, so as long as
  ``theta <= pi/N`` the largest transfer along a FIXED flux row is at
  ``theta_eff = theta``, i.e. exactly at ``phi = 0``. That makes the per-row
  argmax over the stark axis a compensation measurement, not a heuristic --
  and it makes the row's own PEAK VALUE the transfer on the ridge, with no
  model in between.
* On the ridge ``theta_eff = theta``, so that peak value is ``sin^2(N*theta)``
  -- an N-fold amplified angle reading, whose ``arcsin`` is unambiguous only
  while ``N*theta <= pi/2``.

TWO GATES, DELIBERATELY SEPARATE. They are not the same condition and collapsing
them into one throws away a result the map really does carry:

  ``ridge_ok``   ``N*theta <= pi``    the per-row argmax is still ``phi = 0``
                                      AND the arch along the ridge folds at
                                      most ONCE, so the resonance and
                                      ``compensating_stark_amp`` are valid
  ``branch_ok``  ``N*theta <= pi/2``  the arch does not fold at all, so the
                                      fitted angle is on the principal branch
                                      and ``swap_angle_rad_refined`` is reported

Both come from the PRIOR angle (``swap_angle_rad``), because ``theta`` is what
the map cannot measure on its own at a fixed N. The prior is a BRANCH SELECTOR,
not a fitted parameter: it picks the band the arch fit's angle may live in, and
it only has to be good to about ``pi/(2N)`` to pick the right one, so a
30%-accurate prior still yields a percent-level refined angle.

THE RESONANCE IS FITTED, NOT PICKED. Along the ridge the transfer is an ARCH,
``T_N = sin^2(N * arcsin(sin(theta*s)/s))`` with ``s = sqrt(1 + (beta*(V -
V0))^2)`` (derived in :mod:`scqat.tools.swap_lineshape`), and its top is flat:
on 5Q4C's N=5 map of 2026-09-21 (``123240-776``, 200 averages) the largest row
moved by +-0.79 mV under a shot-noise bootstrap, and a 2-D interpolation or
smoothing of the map did no better (+-0.72 to +-0.76), because they still pick
a point on the top. Fitting the whole arch lets every row on both flanks vote,
and gave +-0.10 mV on the same data -- with a reduced chi-square of 0.8-1.2
against pure shot noise on all four runs of that bring-up, so the error it
reports is one to act on. The raw largest row is still reported
(``ridge_peak_flux_amp_v``), because it needs no prior.

A fitted centre can be WRONG WITHOUT BEING NOISY, so it is refused, not
reported, in two cases, with its error left in place to show why:

  ``resonance_at_edge``     the centre lies outside the live rows (less half a
                            step) -- the arch was extrapolated, not measured
  ``resonance_unresolved``  its error exceeds a tenth of the live span, or
                            moves the compensation by more than a stark step,
                            or the fit failed

(a 3 mV window at the same flux, ``105115-861``, read ``-151.01 +- 1.98`` mV
against a window starting at -151.00; the resolved runs read +-0.02 to +-0.10.)

THE RIDGE IS READ LOCALLY, NOT FITTED GLOBALLY. The compensating amplitude at
the resonance is interpolated from the rows AROUND the resonance and nowhere
else, because a global model would have to assume a shape the amplitude axis
does not have. The per-round phase is linear in the FLUX (detuning x time) but
the stark tone's own phase is not linear in its amplitude -- measured on 5Q4C q1
(``qubit_stark_phase_echo``, 2026-09-20): quadratic near zero, essentially
linear past ~0.6, and 2*pi at amplitude 0.993 where a pure ``a^2`` law would put
it at 1.100, an 11% error. So ``phi = 0`` is a straight line in PHASE and a
curved one in amplitude, and fitting a line across the whole axis costs a
factor of ~3 in residual (0.042 against 0.016 stark units on that same map).
Locally none of that matters, and no phase curve has to be carried in.

THE RIDGE IS ALSO PERIODIC. ``phi = 0`` means ``phi = 0 mod 2*pi``, so the
locus is a FAMILY of parallel curves and the per-row optimum jumps a whole
period whenever the flux window drives ``phi`` past ``2*pi``. The jump is
undone before anything is interpolated (:func:`_unwrap_optima`). Calibrating the
stark window to exactly one period -- ``max_stark_amp`` = the echo's
``amp_2pi``, passed here as ``stark_amp_2pi`` -- makes that wrap exact without a
phase curve, guarantees every row's compensation is reachable exactly once
(a window SHORTER than a period pins those rows' argmax against the edge and
invents ridge points), and gives a free consistency check against the wrap the
map measures for itself.

WHY THE CONTRAST FILTER IS MODEL-FREE. A row's available signal is
``U_{N-1}`` swept over ``[0, cos theta]``, which collapses where the swap is
near-full (for N=2 it is ``cos^2(phi/2) * sin^2(2*theta)``, and ``sin^2(2*theta)``
vanishes at ``theta = pi/2``). Such a row's argmax is noise, so rows count only
when their observed swing clears ``min_row_contrast``. Measured, never fitted --
the lesson ``qc_n_stark_amp``'s ``trace_contrast`` records. If the resonance
falls INSIDE a run of such rows -- which is where a folded arch puts it -- the
arch fit still reaches it from the two flanks (5Q4C's N=2 map: ``-149.164 +-
0.024`` mV across a seven-row dead band, within 0.1 mV of three independent
reads), but the COMPENSATION there would have to be interpolated across the
band, and the stark axis is not a linear picture of phase. So only that number
is withheld (``compensation_in_gap``: the live rows bracketing the resonance are
more than two flux steps apart).

The transfer used throughout is the NORMALIZED one
(:func:`~scqat.estimators._pair_swap_maps.pair_swap_normalized_transfer`), which
divides out the prep fidelity and the common decay so a peak height IS
``sin^2(theta)``.

Record-only for the DEVICE: nothing is proposed and nothing is written back; the
SUCCESS / ``min_transfer`` verdict stays in SCQO.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` — dims ``(joint_state, flux_amp_v, stark_amp)``
           in any order; ``joint_state`` labels ``"00"/"01"/"10"/"11"``
           (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``flux_amp_v`` (V) / ``stark_amp`` (dimensionless
           factor of the stark operation's baked amplitude)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) — selects the transfer partner
           ``flux_side`` / ``high_name`` / ``low_name`` — role labels for the figure
           ``swap_count`` — the run's FIXED N; without it no angle is reported
           ``swap_angle_rad`` — the prior exchange angle at the coupler flux the
           swap macro bakes, from ``pair_swap_flux_map``'s ``theta_rad`` column
           ``stark_amp_2pi`` — the stark amplitude worth one full 2*pi, from
           ``qubit_stark_phase_echo``'s ``amp_2pi_factor``
           ``min_row_contrast`` — a row's swing must clear this to count
"""

import math
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import (
    pair_swap_normalized_transfer,
    pair_swap_plot_data,
    summarize_pair_swap,
)
from scqat.estimators.qc_swap_flux_stark.visualization import (
    plot_qc_swap_flux_stark,
    plot_swap_flux_stark_ridge,
)
from scqat.tools.swap_lineshape import (
    fit_compensated_swap_arch,
    parabolic_vertex,
    theta_from_peak,
)

AXIS0 = "flux_amp_v"
AXIS1 = "stark_amp"

#: the dense axis the fitted arch is drawn on, as a dim of its own.
DENSE_AXIS = "flux_amp_v_dense"

#: points per measured row on the drawn arch.
_FIT_CURVE_DENSITY = 10

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``qc_swap_flux_stark.png`` and
#: ``qc_swap_flux_stark_ridge.png``.
FIG_MAP = "qc_swap_flux_stark"
FIG_RIDGE = "ridge"

#: the per-flux-row columns, as ``{results key: dtype}``. Carried both in the
#: metadata (as lists) and in plot_data (as columns over ``flux_amp_v``).
#: ``ridge_stark_amp`` is the UNWRAPPED optimum (the drawn ridge), which may sit
#: outside the swept window by whole periods; ``row_stark_amp`` is where the
#: peak actually was.
_RIDGE_COLUMNS = {
    "row_stark_amp": float,
    "row_contrast": float,
    "ridge_stark_amp": float,
    "ridge_transfer": float,
    "ridge_theta_rad": float,
    "row_ok": int,
}

#: live rows needed on each side before the resonance can be interpolated.
_LOCAL_HALF_WINDOW = 2

#: live rows needed at all: the arch has three parameters, and its error is
#: estimated from the residual, which needs a couple of degrees of freedom.
_MIN_RIDGE_ROWS = 5

#: a resonance whose error exceeds this fraction of the live-row span is
#: refused. 5Q4C's resolved runs sit at <= 0.014 and its 3 mV window at 0.66, a
#: factor of ~45 apart; 0.1 is close to their geometric middle.
_MAX_RESONANCE_ERR_FRACTION = 0.1

#: the compensation is withheld when the live rows bracketing the resonance are
#: further apart than this many flux steps; one dead row in between is fine.
_MAX_BRACKET_STEPS = 2

#: the gates, in units of ``N * theta``.
_RIDGE_LIMIT = math.pi
_BRANCH_LIMIT = math.pi / 2
#: float slack so a prior sitting exactly on a limit is not rejected by rounding.
_GATE_TOL = 1e-9

#: the fold's flanking maxima ARE ``N*theta = pi/2``, so the latch level is this
#: fraction of the ridge transfer's own maximum — never an absolute number,
#: which a real map's imperfect contrast walks straight past.
_FOLD_FRACTION = 0.9

#: a per-row optimum that jumps by more than this fraction of the swept stark
#: span is the next ``2*pi`` branch entering the window, not a measurement.
_WRAP_JUMP_FRACTION = 0.4

#: how far the measured wrap may sit from a supplied ``stark_amp_2pi``.
_WRAP_TOLERANCE = 0.1


def _empty_ridge() -> Dict[str, Any]:
    """Every scalar this estimator can report, at its "nothing was read" value.

    Never absent, per "fit-derived fields degrade to NaN": SCQO lifts this key
    set into a flat scalar surface and a missing key would read as a schema
    change rather than as a failed fit.
    """
    return {
        "compensating_stark_amp": float("nan"),
        "compensating_stark_err": float("nan"),
        "compensation_in_gap": 0,
        "resonance_flux_amp_v": float("nan"),
        "resonance_flux_err_v": float("nan"),
        "resonance_at_edge": 0,
        "resonance_unresolved": 0,
        "arch_r_squared": float("nan"),
        "ridge_peak_flux_amp_v": float("nan"),
        "ridge_peak_transfer": float("nan"),
        "swap_angle_rad_refined": float("nan"),
        "swap_angle_err_rad": float("nan"),
        "swap_angle_rad_prior": float("nan"),
        "swap_angle_consistent": 0,
        "ridge_slope_per_v": float("nan"),
        "ridge_local_rms": float("nan"),
        "ridge_wrap_amp": float("nan"),
        "stark_amp_2pi_prior": float("nan"),
        "wrap_consistent": 0,
        "n_ridge_rows": 0,
        "n_fold_rows": 0,
        "max_row_contrast": float("nan"),
        "ridge_ok": 0,
        "branch_ok": 0,
    }


def dense_flux(flux: np.ndarray) -> np.ndarray:
    """The flux axis at :data:`_FIT_CURVE_DENSITY` times the sampling, ascending."""
    flux = np.asarray(flux, dtype=float)
    if flux.size < 2:
        return flux
    return np.linspace(float(np.min(flux)), float(np.max(flux)),
                       _FIT_CURVE_DENSITY * (flux.size - 1) + 1)


def _grid_step(axis: np.ndarray) -> float:
    """The sweep's pitch, from either end of either direction."""
    axis = np.asarray(axis, dtype=float)
    if axis.size < 2:
        return float("nan")
    return float(np.median(np.abs(np.diff(axis))))


def _column(values, size: int, dtype):
    """A per-row column of the right length, NaN/0-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the fit, so a missing column degrades rather than
    raising -- rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan if dtype is float else 0, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _validate(swap_count, swap_angle_rad, stark_amp_2pi, min_row_contrast) -> None:
    """Check the knobs ONCE, before any per-row loop.

    A per-row loop that swallows its own failures would turn a typo'd knob into
    an all-NaN map that looks like a bad measurement.
    """
    if swap_count is not None and int(swap_count) < 1:
        raise ValueError(f"swap_count must be >= 1, got {swap_count!r}")
    if swap_angle_rad is not None:
        angle = float(swap_angle_rad)
        if not np.isfinite(angle) or not 0.0 < angle <= math.pi / 2:
            raise ValueError(
                "swap_angle_rad is an exchange angle in (0, pi/2] rad "
                f"(a full swap is pi/2), got {swap_angle_rad!r}"
            )
    if stark_amp_2pi is not None:
        period = float(stark_amp_2pi)
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError(
                "stark_amp_2pi is the stark amplitude worth one full 2*pi and "
                f"must be positive, got {stark_amp_2pi!r}"
            )
    contrast = float(min_row_contrast)
    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError(
            f"min_row_contrast must be in [0, 1], got {min_row_contrast!r}")


def _row_optima(stark: np.ndarray, transfer: np.ndarray, min_row_contrast: float):
    """Per flux row: its observed swing, where it peaks, and how high.

    Returns ``(row_stark_amp, row_peak, row_contrast, row_ok)``. The peak
    position is refined to sub-grid by
    :func:`~scqat.tools.swap_lineshape.parabolic_vertex`, because nothing makes
    the compensation land on a swept amplitude. The peak VALUE needs no model:
    under ``ridge_ok`` the row's maximum IS its transfer at ``phi = 0``.
    """
    n_rows = transfer.shape[0]
    star = np.full(n_rows, np.nan)
    peak = np.full(n_rows, np.nan)
    contrast = np.full(n_rows, np.nan)
    ok = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        row = transfer[i]
        if np.isfinite(row).sum() < 3:
            continue
        contrast[i] = float(np.nanmax(row) - np.nanmin(row))
        peak[i] = float(np.nanmax(row))
        star[i] = parabolic_vertex(stark, row, int(np.nanargmax(row)))[0]
        ok[i] = int(contrast[i] >= min_row_contrast)
    return star, peak, contrast, ok


def _unwrap_optima(flux: np.ndarray, star: np.ndarray, ok: np.ndarray,
                   span: float, stark_amp_2pi: Optional[float]):
    """Undo the ``2*pi`` jumps in the per-row optimum. ``(unwrapped, period)``.

    ``phi = 0`` is ``phi = 0 mod 2*pi``, so the optimum re-enters the swept
    window from the far end whenever the flux drives the phase past a turn. The
    jump is the whole period, which is much larger than the row-to-row step, so
    it is identified by SIZE and not by a model.

    The period is ``stark_amp_2pi`` when the operator calibrated it (the echo's
    ``amp_2pi``) and the measured jump otherwise. Calibrating it matters: the
    period is constant in PHASE, never in amplitude, so a window that is not one
    turn long has a jump whose size depends on where in the curve it happened.
    """
    unwrapped = np.array(star, dtype=float)
    live = np.flatnonzero((np.asarray(ok) > 0) & np.isfinite(star))
    measured = float("nan")
    if live.size < 2:
        return unwrapped, measured
    shift = 0.0
    for n, (a, b) in enumerate(zip(live[:-1], live[1:])):
        if abs(star[b] - star[a]) > _WRAP_JUMP_FRACTION * span:
            # The RAW jump understates the period by however far the ridge
            # would have moved between the two rows anyway, so the local slope
            # is taken out first (13% of the period on 5Q4C's N=5 map).
            past = live[max(0, n - 4): n + 1]
            slope = (np.polyfit(flux[past], unwrapped[past], 1)[0]
                     if past.size >= 2 else 0.0)
            # both sides in the RAW frame: the accumulated shift is constant
            # between wraps, so the slope carries over but the offset must not
            expected = star[a] + slope * (flux[b] - flux[a])
            jump = star[b] - expected
            if not np.isfinite(measured):
                measured = abs(jump)
            period = float(stark_amp_2pi) if stark_amp_2pi is not None else abs(jump)
            shift -= math.copysign(period, jump)
        unwrapped[b] = star[b] + shift
    return unwrapped, measured


def _unfold_along_ridge(ridge_transfer: np.ndarray, expect_fold: bool):
    """``N*theta`` per row from the ridge transfer, with ONE fold undone.

    ``arcsin`` returns the principal branch, so a row past ``N*theta = pi/2``
    reads back as ``pi - N*theta`` and the resonance -- the LARGEST angle --
    would look like a local MINIMUM flanked by two maxima.

    WHETHER to unfold is the PRIOR's call, not a threshold's: a fold exists iff
    ``N*theta > pi/2``, which is exactly the condition that separates the two
    gates. WHERE it starts is then read off the curve -- the flanking maxima ARE
    the ``pi/2`` crossings, so the latch level is a fraction of the observed
    maximum rather than a fixed number. (A fixed 0.95 fails on real data: 5Q4C
    q1_q2's N=2 ridge peaked at 0.936, and the fold went undetected.) The flux
    window brackets the resonance, so both ends of the axis are unfolded: walk
    inward from each and take the rows BOTH walks latched.

    Returns ``(n_theta, folded)``.
    """
    principal = np.array([theta_from_peak(v) for v in ridge_transfer])
    folded = np.zeros(principal.size, dtype=bool)
    if not expect_fold or not np.isfinite(ridge_transfer).any():
        return principal, folded
    level = _FOLD_FRACTION * float(np.nanmax(ridge_transfer))
    inward = []
    n = principal.size
    for start, stop, step in ((0, n, 1), (n - 1, -1, -1)):
        latched = np.zeros(n, dtype=bool)
        on = False
        for i in range(start, stop, step):
            if np.isfinite(ridge_transfer[i]) and ridge_transfer[i] >= level:
                on = True
            latched[i] = on
        inward.append(latched)
    folded = inward[0] & inward[1] & np.isfinite(principal)
    return np.where(folded, math.pi - principal, principal), folded


def _bracketing_rows(flux: np.ndarray, live: np.ndarray, at: float):
    """The nearest live row at or below ``at`` and at or above it, or None.

    Direction-agnostic: the flux axis may be swept either way.
    """
    below = live[flux[live] <= at]
    above = live[flux[live] >= at]
    if below.size == 0 or above.size == 0:
        return None
    return (int(below[np.argmin(at - flux[below])]),
            int(above[np.argmin(flux[above] - at)]))


def _theta_band(n_swaps: int, branch_ok: int):
    """The band the arch fit's single-round angle may live in.

    Picked by the prior through the gates: an unfolded arch keeps
    ``N*theta <= pi/2``, a folded one sits in ``[pi/2, pi]``. The two give
    mirror-image heights on resonance, which is exactly the ambiguity the prior
    is there to settle.
    """
    edge = math.pi / (2 * n_swaps)
    return (0.0, edge) if branch_ok else (edge, 2.0 * edge)


def _local_compensation(flux: np.ndarray, unwrapped: np.ndarray,
                        ok: np.ndarray, at: float, period: float):
    """The compensating amplitude at one flux, from the rows AROUND it.

    The value is a straight interpolation between the two live rows that
    BRACKET the resonance -- as local as the grid allows, because the ridge is
    straight in PHASE and the amplitude axis is not a linear picture of phase.
    The error of a linear-in-amplitude step is set by how fast ``dphi/da``
    varies across it, which is worst near zero amplitude (on 5Q4C's echo,
    ``dphi/da`` runs 3.3 rad at a = 0.2 against 8.9 at 0.6), so widening the
    window buys noise averaging at the price of a bias that does not average
    out. Two rows it is.

    ``slope`` and ``rms`` come from a wider window (``_LOCAL_HALF_WINDOW`` rows
    each side) and are diagnostics only: the slope says how tightly the flux has
    to be held, and the rms says whether the stark axis resolved the ridge at
    all. A run whose rms approaches the stark step is under-sampled, and its
    compensation is worth no more than that step.

    Returns ``(value, slope, rms)``; the value is folded back into
    ``[0, period)`` when a period is known, since the compensation is only
    defined modulo one turn.
    """
    live = np.flatnonzero((np.asarray(ok) > 0) & np.isfinite(unwrapped))
    if live.size < 2:
        return float("nan"), float("nan"), float("nan")
    bracket = _bracketing_rows(flux, live, at)
    if bracket is None:
        return float("nan"), float("nan"), float("nan")
    lo, hi = bracket
    if lo == hi:
        value = float(unwrapped[lo])
    else:
        w = (at - flux[lo]) / (flux[hi] - flux[lo])
        value = float(unwrapped[lo] + w * (unwrapped[hi] - unwrapped[lo]))
    if np.isfinite(period) and period > 0:
        value = value % period

    slope = rms = float("nan")
    order = live[np.argsort(np.abs(flux[live] - at))]
    window = np.sort(order[: 2 * _LOCAL_HALF_WINDOW + 1])
    if window.size >= 3:
        s, intercept = np.polyfit(flux[window], unwrapped[window], 1)
        residual = unwrapped[window] - (s * flux[window] + intercept)
        slope, rms = float(s), float(np.sqrt(np.mean(residual ** 2)))
    return value, slope, rms


def _ridge_pick(flux: np.ndarray, stark: np.ndarray, transfer: np.ndarray,
                swap_count: Optional[int], swap_angle_rad: Optional[float],
                stark_amp_2pi: Optional[float], min_row_contrast: float):
    """The whole reading: per-row optima, the unwrap, the arch and the local read.

    Returns ``(scalars, columns, arch)``, where ``arch`` is the arch fitter's
    result -- kept whole so the caller can draw it -- or None when no fit ran.
    """
    out = _empty_ridge()
    star, peak, contrast, ok = _row_optima(stark, transfer, min_row_contrast)
    span = float(np.max(stark) - np.min(stark)) or 1.0
    unwrapped, measured_wrap = _unwrap_optima(flux, star, ok, span,
                                              stark_amp_2pi)
    columns = {
        "row_stark_amp": star,
        "row_contrast": contrast,
        "ridge_stark_amp": unwrapped,
        "ridge_transfer": np.where(np.asarray(ok) > 0, peak, np.nan),
        "ridge_theta_rad": np.full(flux.size, np.nan),
        "row_ok": np.asarray(ok, dtype=int),
    }
    if np.isfinite(contrast).any():
        out["max_row_contrast"] = float(np.nanmax(contrast))
    out["n_ridge_rows"] = int(np.sum(np.asarray(ok) > 0))
    out["ridge_wrap_amp"] = measured_wrap

    # Where the phase-compensated transfer is largest. This needs NO prior --
    # it is a measurement, not a claim -- so it is the flux an operator can act
    # on from any run. It equals `resonance_flux_amp_v` whenever the angle has
    # not folded; past `pi/2` it is one of the two flanks instead, which is
    # exactly what the prior is there to tell you.
    on_ridge = columns["ridge_transfer"]
    if np.isfinite(on_ridge).any():
        kp = int(np.nanargmax(on_ridge))
        out["ridge_peak_flux_amp_v"] = parabolic_vertex(flux, on_ridge, kp)[0]
        out["ridge_peak_transfer"] = float(on_ridge[kp])
    if swap_angle_rad is not None:
        out["swap_angle_rad_prior"] = float(swap_angle_rad)
    if stark_amp_2pi is not None:
        out["stark_amp_2pi_prior"] = float(stark_amp_2pi)
        if np.isfinite(measured_wrap):
            out["wrap_consistent"] = int(
                abs(measured_wrap - float(stark_amp_2pi))
                <= _WRAP_TOLERANCE * float(stark_amp_2pi))

    # The gates are the PRIOR's judgement of what this N could resolve, so they
    # are set BEFORE anything is read -- an open gate over an empty ridge says
    # "the run was answerable and the data did not answer", which is a different
    # diagnosis from "this N cannot answer it".
    n_swaps = None if swap_count is None else int(swap_count)
    if n_swaps is not None and swap_angle_rad is not None:
        n_theta = n_swaps * float(swap_angle_rad)
        out["ridge_ok"] = int(n_theta <= _RIDGE_LIMIT + _GATE_TOL)
        out["branch_ok"] = int(out["ridge_ok"]
                               and n_theta <= _BRANCH_LIMIT + _GATE_TOL)

    if n_swaps is not None:
        # a per-row angle for the FIGURE; the resonance no longer reads it --
        # the arch fit handles the fold through its angle band instead
        unfolded, folded = _unfold_along_ridge(
            columns["ridge_transfer"],
            expect_fold=bool(out["ridge_ok"] and not out["branch_ok"]))
        columns["ridge_theta_rad"] = unfolded / n_swaps
        out["n_fold_rows"] = int(folded.sum())

    if not out["ridge_ok"]:
        return out, columns, None
    # From here on the gate is open: the run WAS answerable, so a missing
    # answer is the data's, and it is flagged rather than left blank.
    live = np.flatnonzero((np.asarray(ok) > 0)
                          & np.isfinite(columns["ridge_transfer"]))
    if live.size < _MIN_RIDGE_ROWS:
        out["resonance_unresolved"] = 1
        return out, columns, None

    arch = fit_compensated_swap_arch(
        flux[live], columns["ridge_transfer"][live], n_swaps,
        theta_bounds=_theta_band(n_swaps, out["branch_ok"]),
        theta_guess=float(swap_angle_rad))
    centre, centre_err = arch["x0"], arch["x0_err"]
    out["arch_r_squared"] = arch["r_squared"]
    out["resonance_flux_err_v"] = centre_err
    if out["branch_ok"]:
        out["swap_angle_err_rad"] = arch["theta_err"]
    if not np.isfinite(centre):
        out["resonance_unresolved"] = 1
        return out, columns, arch

    period = (float(stark_amp_2pi) if stark_amp_2pi is not None
              else measured_wrap)
    value, slope, rms = _local_compensation(flux, unwrapped, ok, centre, period)
    out["ridge_slope_per_v"] = slope
    out["ridge_local_rms"] = rms
    # what the flux uncertainty does to the number the operator sets
    out["compensating_stark_err"] = abs(slope) * centre_err

    step = _grid_step(flux)
    lo_live, hi_live = float(np.min(flux[live])), float(np.max(flux[live]))
    out["resonance_at_edge"] = int(
        not lo_live + 0.5 * step <= centre <= hi_live - 0.5 * step)
    moves_comp = (np.isfinite(out["compensating_stark_err"])
                  and out["compensating_stark_err"] > _grid_step(stark))
    out["resonance_unresolved"] = int(
        not arch["success"] or not np.isfinite(centre_err)
        or centre_err > _MAX_RESONANCE_ERR_FRACTION * (hi_live - lo_live)
        or bool(moves_comp))
    bracket = _bracketing_rows(flux, live, centre)
    out["compensation_in_gap"] = int(
        bracket is not None
        and abs(flux[bracket[1]] - flux[bracket[0]])
        > _MAX_BRACKET_STEPS * step * (1.0 + 1e-9))

    if out["resonance_at_edge"] or out["resonance_unresolved"]:
        return out, columns, arch
    out["resonance_flux_amp_v"] = centre
    if not out["compensation_in_gap"]:
        out["compensating_stark_amp"] = value
    if out["branch_ok"]:
        out["swap_angle_rad_refined"] = arch["theta_rad"]
        out["swap_angle_consistent"] = int(
            abs(arch["theta_rad"] - float(swap_angle_rad))
            <= math.pi / (2 * n_swaps))
    return out, columns, arch


class QcSwapFluxStarkEstimator(BaseEstimator):
    """Read the AC-Stark compensation off the fixed-N flux x stark map."""

    estimator_name = "qc_swap_flux_stark"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "qc_swap_flux_stark estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_swap_flux_stark estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        swap_count: Optional[int] = None,
        swap_angle_rad: Optional[float] = None,
        stark_amp_2pi: Optional[float] = None,
        min_row_contrast: float = 0.3, **kwargs
    ) -> Dict[str, Any]:
        """Summarise the map, then read the compensation off it.

        ``swap_count`` is the run's FIXED number of swaps (not an axis); without
        it no angle can be read and the gated scalars stay NaN.
        ``swap_angle_rad`` is the PRIOR exchange angle in (0, pi/2] at the
        coupler flux the swap macro bakes -- read it from
        ``pair_swap_flux_map``'s ``theta_rad`` column, and do not use a column
        that carries ``branch_warn``. ``stark_amp_2pi`` is the stark amplitude
        worth one full turn of phase, from ``qubit_stark_phase_echo``; supplying
        it makes the ridge's ``2*pi`` unwrap exact instead of measured.
        ``min_row_contrast`` is the swing a flux row must show along the stark
        axis before its optimum is believed.
        """
        _validate(swap_count, swap_angle_rad, stark_amp_2pi, min_row_contrast)
        results = summarize_pair_swap(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        projected = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        transfer = pair_swap_normalized_transfer(projected, drive_side)
        flux = np.asarray(projected[AXIS0].values, dtype=float)
        stark = np.asarray(projected[AXIS1].values, dtype=float)

        pick, columns, arch = _ridge_pick(flux, stark, transfer, swap_count,
                                          swap_angle_rad, stark_amp_2pi,
                                          min_row_contrast)
        results.update(pick)
        for key, dtype in _RIDGE_COLUMNS.items():
            results[key] = [dtype(v) for v in columns[key]]
        # provenance (Layered-analyses rule 5): the knobs that shaped the read
        results["swap_count"] = (float("nan") if swap_count is None
                                 else int(swap_count))
        results["min_row_contrast"] = float(min_row_contrast)
        # bulky intermediates — the `_` prefix keeps them out of the metadata
        # JSON. The arch is resampled from the fitter's own dense curve rather
        # than re-evaluated here, so the drawn line cannot drift from the fit;
        # it is left NaN outside the live rows it was fitted on. It is kept
        # when the resonance is REFUSED too: the arch is how the operator sees
        # why (its centre off the window, or its top flat across it).
        dense = dense_flux(flux)
        curve = np.full(dense.size, np.nan)
        if arch is not None and np.isfinite(arch["best_fit_dense"]).any():
            curve = np.interp(dense, arch["x_dense"], arch["best_fit_dense"],
                              left=np.nan, right=np.nan)
        results["_transfer"] = transfer
        results["_arch_fit"] = curve
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Everything but the bulky 2-D intermediates."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        swap_count: Optional[int] = None,
        swap_angle_rad: Optional[float] = None,
        stark_amp_2pi: Optional[float] = None,
        min_row_contrast: float = 0.3, **kwargs
    ) -> Optional[xr.Dataset]:
        out = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        transfer = results.get("_transfer")
        if transfer is None:
            transfer = pair_swap_normalized_transfer(out, drive_side)
        out["transfer"] = ((AXIS0, AXIS1), np.asarray(transfer, dtype=float))
        n_rows = out.sizes[AXIS0]
        for key, dtype in _RIDGE_COLUMNS.items():
            out[key] = ((AXIS0,), _column(results.get(key), n_rows, dtype))
        # The fitted arch, on a dense flux axis of its own; NaN, never absent,
        # so the replot path and a failed fit still draw the raw data.
        dense = dense_flux(np.asarray(out[AXIS0].values, dtype=float))
        arch = results.get("_arch_fit")
        if arch is None or np.shape(arch) != dense.shape:
            arch = np.full(dense.size, np.nan)
        out.coords[DENSE_AXIS] = dense
        out["arch_fit"] = ((DENSE_AXIS,), np.asarray(arch, dtype=float))
        defaults = {**_empty_ridge(), "swap_count": float("nan"),
                    "min_row_contrast": float(min_row_contrast)}
        for key, default in defaults.items():
            value = results.get(key, default)
            # netCDF attrs take numbers and strings only
            out.attrs[key] = type(default)(value) if value is not None else default
        return out

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(
                dataset, results,
                drive_side=kwargs.get("drive_side", "low"),
                flux_side=kwargs.get("flux_side"),
                high_name=kwargs.get("high_name"),
                low_name=kwargs.get("low_name"),
                swap_count=kwargs.get("swap_count"),
                swap_angle_rad=kwargs.get("swap_angle_rad"),
                stark_amp_2pi=kwargs.get("stark_amp_2pi"),
                min_row_contrast=kwargs.get("min_row_contrast", 0.3),
            )
        return render_figures(
            {
                FIG_MAP: lambda: plot_qc_swap_flux_stark(plot_data),
                FIG_RIDGE: lambda: plot_swap_flux_stark_ridge(plot_data),
            },
            label=self.estimator_name,
        )
