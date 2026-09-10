"""The N-swap x AC-Stark-amplitude map, and the stark amplitude that compensates.

Draws the four joint two-qubit populations over the stark-amplitude x swap-count
grid so an operator can see how the AC-Stark drive detunes the swap and how a
small miscalibration amplifies with the number of swaps -- and then reads the
COMPENSATING stark amplitude off the map by measuring, at every stark amplitude,
the transfer oscillation along the swap count.

The error-amplification sibling of ``qc_n_swap_amp``: identical
``joint_population`` form and shared drawing, but the outer axis is the AC-Stark
drive amplitude (``stark_amp``) instead of the swap flux amplitude.

WHY BOTH CRITERIA POINT AT ONE AMPLITUDE. Between two swaps the members
accumulate a relative phase ``phi``, and an exchange followed by a Z rotation
does not commute, so the repeated unit is a composite rotation with

    cos(theta_eff) = cos(phi/2) * cos(theta_exchange)

Hence ``theta_eff >= theta_exchange`` ALWAYS -- an uncompensated phase can only
INFLATE the per-swap angle -- while the oscillation's CONTRAST falls as ``phi``
grows. The stark tone is the knob that nulls ``phi``, so along the transfer's
oscillation in N the compensating amplitude is where

  * the oscillation CONTRAST is largest (it is maximal on compensation), and
  * the PERIOD is longest (``period = pi / theta_eff``, and ``theta_eff`` bottoms
    out at ``theta_exchange``).

Both are reported on their own, plus a combined pick (the geometric mean of the
two normalised criteria) and an ``osc_criteria_agree`` flag -- when the two
disagree the map is telling you something, and hiding that behind one number
would be a lie. The pick is also interpolated between grid points
(``compensating_stark_amp_refined``), since nothing makes the optimum land on a
swept amplitude.

THE STANDING ASSUMPTION: NO ROW EXCEEDS A FULL SWAP. Every ``theta_eff`` in the
map is taken to be at most ``pi/2`` -- equivalently every period is at least TWO
counts, the Nyquist period of an integer-N axis. That is what makes a fitted
period a measurement at all: anything faster ALIASES, and the faster it truly is
the SLOWER it reads, so an unguarded map would hand the slowest-period criterion
to its own far wing. The assumption cannot be checked from the fitted periods --
an aliased row is indistinguishable from a slow one -- so it is the OPERATOR's to
keep: run this map on a PARTIAL swap, and keep the stark window narrow enough
that the fastest row stays clear of the limit.

``min_osc_period`` is the dashboard for it. It is the fastest oscillation in the
map, in counts per cycle, and ``pi / min_osc_period`` is the largest per-swap
angle: at 2.0 the sweep is AT the limit and the reading past it is aliased, not
slow. The 5Q4C q1_q2 map of 2026-09-08 sat at 2.02 -- inside by one percent --
and resolved 4.5 counts per cycle at its compensation point.

One thing the assumption does NOT buy is the fitted amplitude. Near the limit the
cosine's ``a`` goes degenerate -- at ``f = 1/2`` the samples are
``a*cos(phi)*(-1)**N + c``, so only the PRODUCT is determined -- and it degrades
continuously, not at the endpoint alone: on 5Q4C the row at ``f = 0.4946`` fitted
``a = 0.355`` while swinging 0.178, and that phantom won the criterion. So the
amplitude criterion is measured off the RAW trace, as half its observed swing
(:func:`trace_contrast`): coarse sampling can only make a trace MISS crests, so a
contrast may under-report but can never invent a winner.

Record-only for the DEVICE: nothing is proposed and nothing is written back; the
SUCCESS / ``min_transfer`` verdict stays in SCQO.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` -- dims ``(joint_state, stark_amp, swap_count)``
           in any order; ``joint_state`` labels ``"00"/"01"/"10"/"11"``
           (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``stark_amp`` (dimensionless amplitude prefactor) /
           ``swap_count`` (dimensionless N)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) -- selects the transfer partner
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import (
    pair_swap_plot_data,
    pair_swap_transfer_marginal,
    summarize_pair_swap,
)
from scqat.estimators.qc_n_stark_amp.visualization import (
    plot_qc_n_stark_amp,
    plot_stark_compensation,
)
from scqat.tools.fit_cosine import fit_swap_oscillation

AXIS0 = "stark_amp"
AXIS1 = "swap_count"

#: the dense axis the fitted curve is drawn on, as a dim of its own.
DENSE_AXIS = "swap_count_dense"

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``qc_n_stark_amp.png`` and
#: ``qc_n_stark_amp_compensation.png``.
FIG_MAP = "qc_n_stark_amp"
FIG_COMPENSATION = "compensation"

#: the per-stark-amplitude columns, as ``{results key: dtype}``. Carried both in
#: the metadata (as lists) and in plot_data (as columns over ``stark_amp``).
#: ``osc_contrast`` is the AMPLITUDE criterion and is measured off the raw trace;
#: ``osc_amplitude`` is the cosine's own ``a``, kept as the diagnostic that shows
#: why the criterion is not read off it (see :func:`_compensation_pick`).
_OSC_COLUMNS = {
    "osc_contrast": float,
    "osc_amplitude": float,
    "osc_period": float,
    "osc_r_squared": float,
    "osc_success": int,
}

#: points per measured count on the drawn fit curve. The sweep takes a handful of
#: integer counts and the oscillation runs a few cycles across them, so the fit
#: sampled at the data points alone draws as a zigzag polyline.
_FIT_CURVE_DENSITY = 10


def dense_counts(counts: np.ndarray) -> np.ndarray:
    """The swap-count axis at :data:`_FIT_CURVE_DENSITY` times the sampling."""
    counts = np.asarray(counts, dtype=float)
    if counts.size < 2:
        return counts
    return np.linspace(float(np.min(counts)), float(np.max(counts)),
                       _FIT_CURVE_DENSITY * (counts.size - 1) + 1)


def _column(values, size: int, dtype):
    """A per-amplitude column of the right length, NaN/0-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the fit, so a missing column degrades rather than
    raising -- rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan if dtype is float else 0, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def trace_contrast(row: np.ndarray) -> float:
    """The oscillation amplitude MEASURED off one trace: half its observed swing.

    Deliberately the crudest possible statistic, because what matters is the
    direction of its error. Coarse sampling makes a trace MISS crests, never
    invent them, so this under-reports a badly sampled oscillation and reports a
    well sampled one exactly -- it cannot manufacture a winner. (An RMS measure
    would not do: at the Nyquist period every sample sits ON a crest, so
    ``sqrt(2) * std`` hands that row a factor of sqrt(2) over a well sampled one
    -- a bias pointing the wrong way.) Noise adds a small floor, but the same one
    to every row of the same count axis, so it does not move the comparison.

    The cosine's fitted ``a`` carries no such guarantee -- see the module
    docstring -- which is why the amplitude criterion is read from here.
    """
    finite = np.isfinite(row)
    if finite.sum() < 2:
        return float("nan")
    seen = np.asarray(row, dtype=float)[finite]
    return 0.5 * float(np.max(seen) - np.min(seen))


def _empty_pick() -> Dict[str, Any]:
    """The pick fields when nothing could be picked -- NaN, never absent."""
    return {
        "compensating_stark_amp": float("nan"),
        "compensating_stark_amp_refined": float("nan"),
        "compensating_is_refined": 0,
        "compensation_score": float("nan"),
        "compensating_osc_contrast": float("nan"),
        "compensating_osc_period": float("nan"),
        "compensating_theta_rad": float("nan"),
        "max_osc_contrast_stark_amp": float("nan"),
        "max_osc_contrast": float("nan"),
        "max_osc_period_stark_amp": float("nan"),
        "max_osc_period": float("nan"),
        "min_osc_period": float("nan"),
        "osc_criteria_agree": 0,
    }


def _parabolic_vertex(x: np.ndarray, y: np.ndarray, i: int) -> tuple:
    """The peak's SUB-GRID position, as ``(value, refined)``.

    The sweep lands the true optimum between two swept amplitudes as often as on
    one, and a peak's three top points fix a parabola whose vertex recovers where
    it actually sat. Degrades to the grid point itself (``refined = 0``) when the
    maximum is at an end of the swept range or the three points do not curve
    downwards -- there is nothing to interpolate through then.
    """
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


def _compensation_pick(
    amps: np.ndarray, contrast: np.ndarray, period: np.ndarray, ok: np.ndarray,
) -> Dict[str, Any]:
    """The stark amplitude whose N-oscillation is strongest AND slowest.

    Only rows whose cosine fit CONVERGED are considered: a rejected row carries a
    NaN period, and letting one win either criterion would invent a compensation
    point out of a flat trace.

    Both criteria are reported on their own (``max_osc_contrast_stark_amp`` /
    ``max_osc_period_stark_amp``) before being combined into one pick by the
    GEOMETRIC mean of the ratios ``c / c_max`` and ``T / T_max`` -- each
    dimensionless and in ``0..1``, and the geometric mean (unlike the arithmetic
    one) refuses to let a row that is excellent on one criterion and poor on the
    other win. ``osc_criteria_agree`` is 1 when the two land on the same swept
    amplitude or on neighbouring ones, which is what a true optimum sitting
    between two grid points looks like.

    Reading the period as a measurement rests on the standing assumption that no
    row swaps by more than ``pi/2`` (module docstring); ``min_osc_period`` is
    reported so a map that ran into that limit says so.
    """
    out = _empty_pick()
    good = np.flatnonzero(
        ok & np.isfinite(contrast) & np.isfinite(period) & (period > 0)
    )
    if good.size == 0:
        return out
    x, c, t = amps[good], contrast[good], period[good]

    i_contrast = int(np.argmax(c))
    i_period = int(np.argmax(t))
    out["max_osc_contrast_stark_amp"] = float(x[i_contrast])
    out["max_osc_contrast"] = float(c[i_contrast])
    out["max_osc_period_stark_amp"] = float(x[i_period])
    out["max_osc_period"] = float(t[i_period])
    # the fastest row in the map: the assumption's dashboard, since pi over it is
    # the largest per-swap angle and 2.0 counts per cycle is where reading a
    # period stops meaning anything
    out["min_osc_period"] = float(np.min(t))
    if float(c[i_contrast]) <= 0:
        return out

    score = np.sqrt((c / float(np.max(c))) * (t / float(np.max(t))))
    i_best = int(np.argmax(score))
    refined, is_refined = _parabolic_vertex(x, score, i_best)
    out.update({
        "compensating_stark_amp": float(x[i_best]),
        "compensating_stark_amp_refined": refined,
        "compensating_is_refined": is_refined,
        "compensation_score": float(score[i_best]),
        "compensating_osc_contrast": float(c[i_best]),
        "compensating_osc_period": float(t[i_best]),
        # theta_eff = pi / period: the angle ONE swap applies at the pick, which
        # is the exchange angle only once phi is nulled (module docstring).
        "compensating_theta_rad": float(np.pi / t[i_best]),
        "osc_criteria_agree": int(abs(i_period - i_contrast) <= 1),
    })
    return out


class QcNStarkAmpEstimator(BaseEstimator):
    """Draw the N-swap AC-Stark map and locate the compensating stark amplitude."""

    estimator_name = "qc_n_stark_amp"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "qc_n_stark_amp estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_n_stark_amp estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        # The map summary (where transfer peaks, the marginal ranges) is the same
        # reduction every pair-swap map reports; the oscillation fit below is
        # this reading's own.
        results = summarize_pair_swap(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        projected = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        transfer = pair_swap_transfer_marginal(projected, drive_side)  # (amp, N)
        amps = np.asarray(dataset[AXIS0].values, dtype=float)
        counts = np.asarray(dataset[AXIS1].values, dtype=float)

        # One cosine fit per stark amplitude, plus the model-free contrast the
        # amplitude criterion is actually read from. A row that cannot be fitted
        # degrades to NaN with success False -- the maps still draw.
        dense = dense_counts(counts)
        contrast = np.full(amps.size, np.nan)
        amplitude = np.full(amps.size, np.nan)
        period = np.full(amps.size, np.nan)
        r_squared = np.full(amps.size, np.nan)
        ok = np.zeros(amps.size, dtype=bool)
        best_fit = np.full((amps.size, dense.size), np.nan)
        for i in range(amps.size):
            contrast[i] = trace_contrast(transfer[i])
            fit = fit_swap_oscillation(counts, transfer[i])
            amplitude[i] = fit["a"]
            period[i] = fit["swap_period"]
            r_squared[i] = fit["r_squared"]
            ok[i] = bool(fit["success"])
            # resampled from the fitter's own dense curve rather than re-evaluated
            # here, so the drawn line cannot drift from the fitted model
            if np.isfinite(fit["x_dense"]).all():
                best_fit[i] = np.interp(dense, fit["x_dense"], fit["best_fit_dense"])

        results["n_osc_ok"] = int(ok.sum())
        results.update(_compensation_pick(amps, contrast, period, ok))
        # The per-amplitude curves themselves, as plain lists so the metadata
        # JSON stays portable.
        results["osc_contrast"] = [float(v) for v in contrast]
        results["osc_amplitude"] = [float(v) for v in amplitude]
        results["osc_period"] = [float(v) for v in period]
        results["osc_r_squared"] = [float(v) for v in r_squared]
        results["osc_success"] = [int(v) for v in ok]
        # underscore-prefixed: bulky arrays kept for build_plot_data only
        results["_transfer"] = transfer
        results["_transfer_fit"] = best_fit
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the bulky per-row maps; the per-amplitude curves are small and stay."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None, **kwargs
    ) -> Optional[xr.Dataset]:
        out = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        dims = (AXIS0, AXIS1)
        transfer = results.get("_transfer")
        if transfer is None:
            transfer = pair_swap_transfer_marginal(out, drive_side)
        out["transfer"] = (dims, np.asarray(transfer, dtype=float))
        # The fitted curve is carried at DENSITY times the sampling, on a dim of
        # its own: a swap oscillation runs several cycles across a handful of
        # integer counts, so the fit sampled at the data points draws as a zigzag.
        dense = dense_counts(np.asarray(out[AXIS1].values, dtype=float))
        fit = results.get("_transfer_fit")
        if fit is None:
            fit = np.full((np.shape(transfer)[0], dense.size), np.nan)
        # The fit degrades to NaN, never to absent, so the raw transfer map still
        # draws when every row failed.
        out.coords[DENSE_AXIS] = dense
        out["transfer_fit"] = ((AXIS0, DENSE_AXIS), np.asarray(fit, dtype=float))
        n_amp = int(out.sizes[AXIS0])
        for name, dtype in _OSC_COLUMNS.items():
            out[name] = ((AXIS0,), _column(results.get(name), n_amp, dtype))
        # netCDF-safe attrs: floats and ints only, NaN where the fit failed.
        defaults = {**_empty_pick(), "n_osc_ok": 0}
        out.attrs.update({key: type(default)(results.get(key, default))
                          for key, default in defaults.items()})
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
            )
        return render_figures(
            {
                FIG_MAP: lambda: plot_qc_n_stark_amp(plot_data),
                FIG_COMPENSATION: lambda: plot_stark_compensation(plot_data),
            },
            label=self.estimator_name,
        )
