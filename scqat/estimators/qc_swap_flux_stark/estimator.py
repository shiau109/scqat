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
                                      AND the angle along the ridge folds at
                                      most ONCE, so the resonance and
                                      ``compensating_stark_amp`` are valid
  ``branch_ok``  ``N*theta <= pi/2``  the principal ``arcsin`` does not fold at
                                      all, so ``swap_angle_rad_refined`` is a
                                      number rather than an unfolding

Both come from the PRIOR angle (``swap_angle_rad``), because ``theta`` is what
the map cannot measure on its own at a fixed N. The prior is a BRANCH SELECTOR,
not a fitted parameter: it only has to be good to about ``pi/(2N)``, so a
30%-accurate prior still yields a percent-level refined angle.

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
falls INSIDE a run of such rows there is nothing local to interpolate, and the
numbers are refused (``resonance_in_gap``) rather than reached by a model.

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
from scqat.tools.swap_lineshape import parabolic_vertex, theta_from_peak

AXIS0 = "flux_amp_v"
AXIS1 = "stark_amp"

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

#: live rows needed at all.
_MIN_RIDGE_ROWS = 4

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
        "compensating_is_refined": 0,
        "resonance_flux_amp_v": float("nan"),
        "resonance_in_gap": 0,
        "swap_angle_rad_refined": float("nan"),
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
    below = live[flux[live] <= at]
    above = live[flux[live] >= at]
    if below.size == 0 or above.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo = int(below[np.argmin(at - flux[below])])
    hi = int(above[np.argmin(flux[above] - at)])
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
    """The whole reading: per-row optima, the unwrap, and the local read."""
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
        unfolded, folded = _unfold_along_ridge(
            columns["ridge_transfer"],
            expect_fold=bool(out["ridge_ok"] and not out["branch_ok"]))
        columns["ridge_theta_rad"] = unfolded / n_swaps
        out["n_fold_rows"] = int(folded.sum())

    if not out["ridge_ok"] or out["n_ridge_rows"] < _MIN_RIDGE_ROWS:
        return out, columns

    theta = columns["ridge_theta_rad"]
    if not np.isfinite(theta).any():
        return out, columns
    k = int(np.nanargmax(theta))
    # The resonance is the largest angle; if the rows next to it were rejected,
    # the true maximum may sit inside that gap and nothing local can reach it.
    neighbours = [j for j in (k - 1, k + 1) if 0 <= j < flux.size]
    out["resonance_in_gap"] = int(any(ok[j] == 0 for j in neighbours))
    if out["resonance_in_gap"]:
        return out, columns

    resonance, refined = parabolic_vertex(flux, theta, k)
    period = (float(stark_amp_2pi) if stark_amp_2pi is not None
              else measured_wrap)
    value, slope, rms = _local_compensation(flux, unwrapped, ok, resonance,
                                            period)
    out["resonance_flux_amp_v"] = resonance
    out["compensating_is_refined"] = refined
    out["compensating_stark_amp"] = value
    out["ridge_slope_per_v"] = slope
    out["ridge_local_rms"] = rms
    if out["branch_ok"]:
        out["swap_angle_rad_refined"] = float(theta[k])
        out["swap_angle_consistent"] = int(
            abs(theta[k] - float(swap_angle_rad)) <= math.pi / (2 * n_swaps))
    return out, columns


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

        pick, columns = _ridge_pick(flux, stark, transfer, swap_count,
                                    swap_angle_rad, stark_amp_2pi,
                                    min_row_contrast)
        results.update(pick)
        for key, dtype in _RIDGE_COLUMNS.items():
            results[key] = [dtype(v) for v in columns[key]]
        # provenance (Layered-analyses rule 5): the knobs that shaped the read
        results["swap_count"] = (float("nan") if swap_count is None
                                 else int(swap_count))
        results["min_row_contrast"] = float(min_row_contrast)
        # bulky intermediate — the `_` prefix keeps it out of the metadata JSON
        results["_transfer"] = transfer
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
