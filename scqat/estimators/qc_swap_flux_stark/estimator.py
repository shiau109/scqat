"""The fixed-N flux x AC-Stark map, and the stark amplitude that compensates.

Draws the four joint two-qubit populations over the flux-amplitude x
stark-amplitude grid at a FIXED number of swaps, and then reads the COMPENSATING
stark amplitude off that map: the control flux brings the members onto resonance
while the Stark tone nulls the phase they accumulate between swaps, and the two
are coupled, so the optimum is a RIDGE in the plane rather than a point. A
one-knob-at-a-time scan lands somewhere on that ridge; this estimator fits it.

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
  argmax over the stark axis a compensation measurement, not a heuristic.
  Past it the argmax jumps to a side lobe and the criterion INVERTS (N=3 flips
  once ``theta > pi/3``) -- the same trap ``qc_n_stark_amp`` records for its
  period criterion.
* On the ridge ``theta_eff = theta``, so the transfer sampled along the fitted
  ridge is ``sin^2(N*theta)`` -- an N-fold amplified angle reading, whose
  ``arcsin`` is unambiguous only while ``N*theta <= pi/2``.

TWO GATES, DELIBERATELY SEPARATE. They are not the same condition and collapsing
them into one throws away a result the map really does carry:

  ``ridge_ok``   ``N*theta <= pi``    the per-row argmax is still ``phi = 0``
                                      AND the angle along the ridge folds at
                                      most ONCE, so the ridge, the resonance
                                      and ``compensating_stark_amp`` are valid
  ``branch_ok``  ``N*theta <= pi/2``  the principal ``arcsin`` does not fold at
                                      all, so ``swap_angle_rad_refined`` is a
                                      number rather than an unfolding

The single fold ``ridge_ok`` admits IS undone (:func:`_unfold_along_ridge`) --
it has to be, or the resonance would read as a local minimum and the ridge would
be evaluated at the wrong flux. The reported ANGLE stays behind the tighter gate
because an unfolded angle inherits the latch's assumptions, while a POSITION
found through it is off by at most a row.

Both are decided from the PRIOR angle (``swap_angle_rad``), because ``theta`` is
what the map cannot measure on its own at a fixed N. The prior is a BRANCH
SELECTOR, not a fitted parameter: it only has to be good to about ``pi/(2N)``,
so a 30%-accurate prior still yields a percent-level refined angle. Without a
prior both gates are 0 and only the raw ridge coefficients are reported.

WHY THE CONTRAST FILTER IS MODEL-FREE. A row's available signal is
``U_{N-1}`` swept over ``[0, cos theta]``, which collapses where the swap is
near-full (for N=2 it is ``cos^2(phi/2) * sin^2(2*theta)``, and ``sin^2(2*theta)``
vanishes at ``theta = pi/2``). Such a row's argmax is noise, so rows enter the
ridge fit only when their observed swing clears ``min_row_contrast``. Measured,
never fitted -- the lesson ``qc_n_stark_amp``'s ``trace_contrast`` records.

The transfer used throughout is the NORMALIZED one
(:func:`~scqat.estimators._pair_swap_maps.pair_swap_normalized_transfer`), which
divides out the prep fidelity and the common decay so a peak height IS
``sin^2(theta)``. The raw marginal would make the angle read low by whatever the
pi-pulse and T1 cost.

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
           ``min_row_contrast`` — a row's swing must clear this to enter the fit
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
from scqat.tools.robust import mad_outliers
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
_RIDGE_COLUMNS = {
    "row_stark_amp": float,
    "row_contrast": float,
    "ridge_stark_amp": float,
    "ridge_transfer": float,
    "ridge_theta_rad": float,
    "row_ok": int,
}

#: rows needed before a straight line through the per-row optima means anything.
_MIN_RIDGE_ROWS = 4

#: MAD width for dropping rows whose optimum sits off the ridge (a row that
#: cleared the contrast gate on a side lobe rather than the main one).
_RIDGE_OUTLIER_SIGMA = 3.0

#: the gates, in units of ``N * theta``.
_RIDGE_LIMIT = math.pi
_BRANCH_LIMIT = math.pi / 2
#: float slack so a prior sitting exactly on a limit is not rejected by rounding.
_GATE_TOL = 1e-9

#: the fold's flanking maxima ARE ``N*theta = pi/2``, so the latch level is this
#: fraction of the ridge transfer's own maximum — never an absolute number,
#: which a real map's imperfect contrast walks straight past.
_FOLD_FRACTION = 0.9


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
        "swap_angle_rad_refined": float("nan"),
        "swap_angle_rad_prior": float("nan"),
        "swap_angle_consistent": 0,
        "ridge_slope_per_v": float("nan"),
        "ridge_intercept": float("nan"),
        "ridge_rms": float("nan"),
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


def _validate(swap_count, swap_angle_rad, min_row_contrast) -> None:
    """Check the three knobs ONCE, before any per-row loop.

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
    contrast = float(min_row_contrast)
    if not np.isfinite(contrast) or not 0.0 <= contrast <= 1.0:
        raise ValueError(
            f"min_row_contrast must be in [0, 1], got {min_row_contrast!r}")


def _row_optima(stark: np.ndarray, transfer: np.ndarray, min_row_contrast: float):
    """Per flux row: its observed swing, and where along stark it peaks.

    Returns ``(row_stark_amp, row_contrast, row_ok)``. The peak is refined to
    sub-grid by :func:`~scqat.tools.swap_lineshape.parabolic_vertex`, because
    nothing makes the compensation land on a swept amplitude.
    """
    n_rows = transfer.shape[0]
    star = np.full(n_rows, np.nan)
    contrast = np.full(n_rows, np.nan)
    ok = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        row = transfer[i]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        contrast[i] = float(np.nanmax(row) - np.nanmin(row))
        star[i] = parabolic_vertex(stark, row, int(np.nanargmax(row)))[0]
        ok[i] = int(contrast[i] >= min_row_contrast)
    return star, contrast, ok


def _fit_ridge(flux: np.ndarray, star: np.ndarray, ok: np.ndarray):
    """Straight line through the per-row optima, as ``(slope, intercept, rms, keep)``.

    The compensation is one phase budget split between the flux pulse's own
    detuning and the stark tone, so where the members are brought is linear in
    what the tone has to undo over the swept window -- a line, not a curve. Rows
    whose optimum sits far off it (a side lobe that cleared the contrast gate)
    are dropped by MAD and the line refit; ``keep`` is the mask that survived.
    """
    keep = (ok > 0) & np.isfinite(star) & np.isfinite(flux)
    if keep.sum() < _MIN_RIDGE_ROWS:
        return float("nan"), float("nan"), float("nan"), np.zeros_like(keep)
    slope, intercept = np.polyfit(flux[keep], star[keep], 1)
    residual = star - (slope * flux + intercept)
    # the residuals' median is ~0, so mad_outliers' relative floor is inert here
    # and the n_sigma test decides on its own
    outlier, _, _ = mad_outliers(residual, keep, _RIDGE_OUTLIER_SIGMA)
    trimmed = keep & ~np.asarray(outlier, dtype=bool)
    if trimmed.sum() >= _MIN_RIDGE_ROWS:
        keep = trimmed
        slope, intercept = np.polyfit(flux[keep], star[keep], 1)
        residual = star - (slope * flux + intercept)
    rms = float(np.sqrt(np.mean(residual[keep] ** 2)))
    return float(slope), float(intercept), rms, keep


def _sample_along(stark: np.ndarray, transfer: np.ndarray,
                  at: np.ndarray) -> np.ndarray:
    """The transfer of each row read at that row's ridge position.

    This is the phase-COMPENSATED transfer curve: it is defined on every row,
    including the ones whose own stark axis carried no signal, which is what
    lets the resonance be located across a dead band.
    """
    out = np.full(transfer.shape[0], np.nan)
    if not np.isfinite(at).any():
        return out
    lo, hi = float(np.min(stark)), float(np.max(stark))
    for i in range(transfer.shape[0]):
        row = transfer[i]
        finite = np.isfinite(row)
        if finite.sum() < 2 or not np.isfinite(at[i]):
            continue
        out[i] = float(np.interp(np.clip(at[i], lo, hi),
                                 stark[finite], row[finite]))
    return out


def _unfold_along_ridge(ridge_transfer: np.ndarray, expect_fold: bool):
    """``N*theta`` per row from the ridge transfer, with ONE fold undone.

    ``arcsin`` returns the principal branch, so a row past ``N*theta = pi/2``
    reads back as ``pi - N*theta`` and the resonance -- the LARGEST angle --
    would look like a local MINIMUM flanked by two maxima. Left alone, the
    resonance search would then pick one of the flanks and the ridge would be
    evaluated some millivolts away.

    WHETHER to unfold is the PRIOR's call, not a threshold's: a fold exists iff
    ``N*theta > pi/2``, which is exactly the condition that separates the two
    gates. WHERE it starts is then read off the curve -- the flanking maxima ARE
    the ``pi/2`` crossings, so the latch level is a fraction of the observed
    maximum rather than a fixed number. (A fixed 0.95 fails on real data: 5Q4C
    q1_q2's N=2 ridge peaked at 0.936, and the fold went undetected.) The flux
    window brackets the resonance, so both ends of the axis are unfolded: walk
    inward from each and take the rows BOTH walks latched.

    Same latch ``pair_swap_flux_map._branch_warnings`` uses to WARN, used here
    to correct -- legitimate only because the gate has bounded the folds at one.
    A row near a crossing may be unfolded a little early, which moves its own
    angle and not the position of the deep interior maximum.

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


def _ridge_pick(flux: np.ndarray, stark: np.ndarray, transfer: np.ndarray,
                swap_count: Optional[int], swap_angle_rad: Optional[float],
                min_row_contrast: float) -> Dict[str, Any]:
    """The whole reading: per-row optima, the ridge, and the two gated scalars."""
    out = _empty_ridge()
    star, contrast, ok = _row_optima(stark, transfer, min_row_contrast)
    slope, intercept, rms, keep = _fit_ridge(flux, star, ok)
    columns = {
        "row_stark_amp": star,
        "row_contrast": contrast,
        "ridge_stark_amp": np.full(flux.size, np.nan),
        "ridge_transfer": np.full(flux.size, np.nan),
        "ridge_theta_rad": np.full(flux.size, np.nan),
        "row_ok": np.asarray(keep, dtype=int),
    }
    if np.isfinite(contrast).any():
        out["max_row_contrast"] = float(np.nanmax(contrast))
    out["n_ridge_rows"] = int(keep.sum())
    if swap_angle_rad is not None:
        out["swap_angle_rad_prior"] = float(swap_angle_rad)

    # The gates are the PRIOR's judgement of what this N could resolve, so they
    # are set BEFORE anything is fitted -- an open gate over an empty ridge says
    # "the run was answerable and the data did not answer", which is a different
    # diagnosis from "this N cannot answer it".
    n_swaps = None if swap_count is None else int(swap_count)
    if n_swaps is not None and swap_angle_rad is not None:
        n_theta = n_swaps * float(swap_angle_rad)
        out["ridge_ok"] = int(n_theta <= _RIDGE_LIMIT + _GATE_TOL)
        out["branch_ok"] = int(out["ridge_ok"]
                               and n_theta <= _BRANCH_LIMIT + _GATE_TOL)
    if not np.isfinite(slope):
        return out, columns

    out["ridge_slope_per_v"] = slope
    out["ridge_intercept"] = intercept
    out["ridge_rms"] = rms
    ridge = slope * flux + intercept
    columns["ridge_stark_amp"] = ridge
    columns["ridge_transfer"] = _sample_along(stark, transfer, ridge)

    if n_swaps is not None:
        unfolded, folded = _unfold_along_ridge(
            columns["ridge_transfer"],
            expect_fold=bool(out["ridge_ok"] and not out["branch_ok"]))
        columns["ridge_theta_rad"] = unfolded / n_swaps
        out["n_fold_rows"] = int(folded.sum())

    if not out["ridge_ok"]:
        return out, columns

    theta = columns["ridge_theta_rad"]
    if not np.isfinite(theta).any():
        return out, columns
    peak = int(np.nanargmax(theta))
    resonance, is_refined = parabolic_vertex(flux, theta, peak)
    out["resonance_flux_amp_v"] = resonance
    out["compensating_is_refined"] = is_refined
    out["compensating_stark_amp"] = float(slope * resonance + intercept)
    if out["branch_ok"]:
        out["swap_angle_rad_refined"] = float(theta[peak])
        out["swap_angle_consistent"] = int(
            abs(theta[peak] - float(swap_angle_rad)) <= math.pi / (2 * n_swaps))
    return out, columns


class QcSwapFluxStarkEstimator(BaseEstimator):
    """Fit the AC-Stark compensation ridge of the fixed-N flux x stark map."""

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
        min_row_contrast: float = 0.3, **kwargs
    ) -> Dict[str, Any]:
        """Summarise the map, then fit the compensation ridge across it.

        ``swap_count`` is the run's FIXED number of swaps (not an axis); without
        it no angle can be read and the gated scalars stay NaN.
        ``swap_angle_rad`` is the PRIOR exchange angle in (0, pi/2] at the
        coupler flux the swap macro bakes -- read it from
        ``pair_swap_flux_map``'s ``theta_rad`` column, and do not use a column
        that carries ``branch_warn``. ``min_row_contrast`` is the swing a flux
        row must show along the stark axis before its optimum is believed.
        """
        _validate(swap_count, swap_angle_rad, min_row_contrast)
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
                                    swap_angle_rad, min_row_contrast)
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
                min_row_contrast=kwargs.get("min_row_contrast", 0.3),
            )
        return render_figures(
            {
                FIG_MAP: lambda: plot_qc_swap_flux_stark(plot_data),
                FIG_RIDGE: lambda: plot_swap_flux_stark_ridge(plot_data),
            },
            label=self.estimator_name,
        )
