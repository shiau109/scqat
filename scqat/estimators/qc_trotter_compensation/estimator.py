"""Which AC-Stark compensation amplitude the Trotter chain wants -- record-only.

The chain repeats a Trotter step (partial swap source->relay, partial swap
relay->sink, parametric reset of the relay, per-qubit AC-Stark phase
compensation) ``N`` times. The sink does not simply collect the source: it
accumulates amplitude from EVERY round, so the round-to-round phase decides
whether those contributions add or cancel. Propagating the single-excitation
amplitudes through one round gives

    |K_M| = sin(t1) sin(t2) |S0| * | SUM_j cos^(M-1-j)(t2) cos^j(t1) e^{i j dPhi} |

with ``dPhi = phi_source - phi_sink``. Three things follow, and this estimator is
shaped by all three:

* only the DIFFERENTIAL phase is observable -- a common-mode phase cancels, so
  ONE compensation amplitude is swept, not one per qubit;
* the sum peaks at ``dPhi = 0``, so the sweep has a single optimum;
* when the rounds cancel, only the LAST round survives and the sink peaks at
  ``N = 1``; when they add, the peak moves out to several rounds. So
  ``n_at_max`` reads the phase condition more sharply than the peak height
  does, and it is reported beside it rather than buried.

TWO READINGS OF THE OPTIMUM. ``best_compensation_amp`` is the raw one: the
amplitude of the single brightest (amplitude, round) pixel. One noisy pixel
moves it, and on 5Q4C it landed 0.02-0.04 from the ridge centre on four scans.
``best_compensation_amp_refined`` reads the ridge instead: the sink averaged over
every round from ``REFINE_FIRST_ROUND`` on (earlier rounds hold at most one path,
so they carry no phase), a parabola within ``REFINE_HALF_WIDTH`` of that curve's
grid best, and its vertex. ``best_compensation_amp_err`` is the vertex's spread
when the rounds are resampled. A curve that is not peaked inside the window --
no interior maximum, a vertex beyond it, too few points -- gets NaN and
``compensation_unresolved = 1`` instead of a number, with the reason in
``refine_reason``.

Record-only: this estimator proposes nothing and fits no physics model -- the
parabola is a local shape fit to the round-averaged sink. It locates the
optimum, reports both discriminators, and draws the map -- the SUCCESS verdict
stays in SCQO.

Dataset contract:
  vars   : ``population`` -- dims ``(qubit, compensation_amp, round_count)`` in
           any order: the averaged marginal P(level >= 1) of each chain qubit.
  coords : ``qubit`` (chain qubit names, chain order) / ``compensation_amp``
           (dimensionless factor of the stark operation's baked amplitude) /
           ``round_count`` (dimensionless Trotter-step count N)
  kwargs : ``source`` / ``relay`` / ``sink`` -- chain role names; ``sink`` picks
           the trace the optimum is read from and is what makes the answer
           meaningful. ``compensation_target`` names the qubit whose tone was
           swept, for the figure only.
"""

import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qc_trotter_compensation.visualization import (
    plot_compensation_scan,
    plot_sink_map,
)

QUBIT_DIM = "qubit"
AMP_AXIS = "compensation_amp"
ROUND_AXIS = "round_count"

#: chain role kwargs -- the labels the figure reads back.
ROLES = ("source", "relay", "sink")

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``qc_trotter_compensation.png`` and
#: ``qc_trotter_compensation_map.png``.
FIG_SCAN = "qc_trotter_compensation"
FIG_MAP = "map"

#: the first round the refined optimum averages. Round 0 holds an empty sink and
#: round 1 a single path, so neither can interfere: they carry no phase.
REFINE_FIRST_ROUND = 2

#: half-width, in amplitude factor, of the parabola window around the grid best
#: of the round-averaged sink. On 5Q4C (2026-09-22) that curve's peak had a
#: Gaussian sigma of about 0.067, so +-0.08 spans its top on both the coarse
#: (0.033) and the fine (0.01) grid. A chip whose peak is several times narrower
#: or wider needs this revisited.
REFINE_HALF_WIDTH = 0.08

#: round resamples behind ``best_compensation_amp_err``, with a fixed seed:
#: re-analysing a run must reproduce its numbers.
REFINE_BOOTSTRAP = 200
_REFINE_SEED = 0

#: the axis the fitted parabola is drawn on (plot_data).
DENSE_AXIS = "compensation_amp_dense"
N_DENSE = 101


def _roles(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """The chain role names supplied by the caller, blank when unknown."""
    return {role: str(kwargs.get(role) or "") for role in ROLES}


def _qubit_names(dataset: xr.Dataset) -> List[str]:
    return [str(v) for v in np.atleast_1d(dataset[QUBIT_DIM].values)]


def _peak_per_amp(rounds: np.ndarray, sink: np.ndarray) -> Dict[str, np.ndarray]:
    """Per compensation amplitude: the sink peak and the N at which it occurs.

    ``sink`` is ``(amp, round)``. An all-NaN row (a failed acquisition) degrades
    to NaN rather than raising, so the map still draws.
    """
    n_amp = sink.shape[0]
    p_max = np.full(n_amp, np.nan)
    n_at_max = np.full(n_amp, np.nan)
    p_final = np.full(n_amp, np.nan)
    for i in range(n_amp):
        row = sink[i]
        if not np.isfinite(row).any():
            continue
        j = int(np.nanargmax(row))
        p_max[i] = float(row[j])
        n_at_max[i] = float(rounds[j])
        p_final[i] = float(row[-1])
    return {"sink_p_max": p_max, "sink_n_at_max": n_at_max, "sink_p_final": p_final}


def _nanmean_rows(block: np.ndarray) -> np.ndarray:
    """Row means ignoring NaN; an all-NaN row is NaN, without the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(block, axis=1)


def _parabola(x: np.ndarray, y: np.ndarray):
    """``(vertex, coefficients)`` of a least-squares parabola through the finite
    points. The coefficients are None below three points; the vertex is NaN
    unless the parabola opens downward."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan"), None
    coeffs = np.polyfit(x[ok], y[ok], 2)
    if not coeffs[0] < 0:
        return float("nan"), coeffs
    return float(-coeffs[1] / (2.0 * coeffs[0])), coeffs


def _refine_optimum(amps: np.ndarray, rounds: np.ndarray,
                    sink: np.ndarray) -> Dict[str, Any]:
    """The ridge reading of the optimum -- see the module docstring.

    ``sink`` is ``(amp, round)``. Never raises: every way the curve can fail to
    show an interior peak degrades to NaN, ``compensation_unresolved = 1`` and a
    ``refine_reason``, while the averaged curve and whatever parabola was fitted
    stay drawable.
    """
    out: Dict[str, Any] = {
        "best_compensation_amp_refined": float("nan"),
        "best_compensation_amp_err": float("nan"),
        "compensation_unresolved": 1.0,
        "refine_reason": "",
        "sink_round_avg": np.full(amps.size, np.nan),
        "_fit_x": None,
        "_fit_y": None,
    }
    use = rounds >= REFINE_FIRST_ROUND
    if not use.any():
        out["refine_reason"] = f"no round >= {REFINE_FIRST_ROUND}: nothing interferes yet"
        return out
    block = sink[:, use]                                      # (amp, rounds used)
    avg = _nanmean_rows(block)
    out["sink_round_avg"] = avg
    if not np.isfinite(avg).any():
        out["refine_reason"] = "no finite sink population"
        return out

    best = int(np.nanargmax(avg))
    window = np.abs(amps - amps[best]) <= REFINE_HALF_WIDTH + 1e-9
    x = amps[window]
    vertex, coeffs = _parabola(x, avg[window])
    if coeffs is None:
        out["refine_reason"] = (f"fewer than 3 points within +-{REFINE_HALF_WIDTH} of "
                                f"the averaged curve's best")
        return out
    lo, hi = float(np.min(x)), float(np.max(x))
    out["_fit_x"] = np.linspace(lo, hi, N_DENSE)
    out["_fit_y"] = np.polyval(coeffs, out["_fit_x"])
    if not np.isfinite(vertex):
        out["refine_reason"] = "the averaged curve has no interior maximum (parabola opens upward)"
        return out
    if not lo <= vertex <= hi:
        out["refine_reason"] = (f"vertex {vertex:.4g} outside the fit window "
                                f"[{lo:.4g}, {hi:.4g}]: the optimum is at or past the scan edge")
        return out

    # Spread: the same window, the rounds resampled. Fewer than three rounds
    # cannot resample into anything but themselves, so they give no spread.
    err = float("nan")
    cols = block[window]
    if cols.shape[1] >= 3:
        rng = np.random.default_rng(_REFINE_SEED)
        draws = []
        for _ in range(REFINE_BOOTSTRAP):
            pick = rng.integers(0, cols.shape[1], cols.shape[1])
            v, _c = _parabola(x, _nanmean_rows(cols[:, pick]))
            if np.isfinite(v):
                draws.append(v)
        if len(draws) >= REFINE_BOOTSTRAP // 2:
            err = float(np.std(draws))

    out.update({
        "best_compensation_amp_refined": vertex,
        "best_compensation_amp_err": err,
        "compensation_unresolved": 0.0,
    })
    return out


class QcTrotterCompensationEstimator(BaseEstimator):
    """Locate the compensation amplitude that maximises chain transport."""

    estimator_name = "qc_trotter_compensation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "population" not in dataset.data_vars:
            raise ValueError(
                "qc_trotter_compensation estimator requires the population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in (QUBIT_DIM, AMP_AXIS, ROUND_AXIS):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_trotter_compensation estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        amps = np.asarray(dataset[AMP_AXIS].values, dtype=float)
        rounds = np.asarray(dataset[ROUND_AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"]
            .transpose(QUBIT_DIM, AMP_AXIS, ROUND_AXIS).values,
            dtype=float,
        )

        results: Dict[str, Any] = {
            "qubits": qubits,
            "n_compensation_amp": int(amps.size),
            "n_round_count": int(rounds.size),
            "compensation_target": str(kwargs.get("compensation_target") or ""),
            **roles,
        }

        sink = roles["sink"]
        if sink not in qubits:
            # Without the sink there is no transport to optimise. Say so in the
            # results rather than guessing a trace -- SCQO turns this into FAILED.
            results.update({
                "success": False,
                "best_compensation_amp": float("nan"),
                "best_sink_p_max": float("nan"),
                "best_n_at_max": float("nan"),
                "worst_compensation_amp": float("nan"),
                "worst_sink_p_max": float("nan"),
                "contrast": float("nan"),
            })
            results["_sink"] = np.full((amps.size, rounds.size), np.nan)
            results["_pop"] = pop
            results["_curves"] = {k: v for k, v in _peak_per_amp(
                rounds, np.full((amps.size, rounds.size), np.nan)).items()}
            refine = _refine_optimum(amps, rounds, results["_sink"])
            refine["refine_reason"] = f"the sink {sink!r} is not among the chain qubits"
            _merge_refine(results, refine)
            return results

        sink_map = pop[qubits.index(sink)]                      # (amp, round)
        curves = _peak_per_amp(rounds, sink_map)
        p_max = curves["sink_p_max"]
        _merge_refine(results, _refine_optimum(amps, rounds, sink_map))

        if np.isfinite(p_max).any():
            best = int(np.nanargmax(p_max))
            worst = int(np.nanargmin(p_max))
            results.update({
                "success": True,
                "best_compensation_amp": float(amps[best]),
                "best_sink_p_max": float(p_max[best]),
                "best_n_at_max": float(curves["sink_n_at_max"][best]),
                "best_sink_p_final": float(curves["sink_p_final"][best]),
                "worst_compensation_amp": float(amps[worst]),
                "worst_sink_p_max": float(p_max[worst]),
                # How much the phase knob is worth on this chain: 1.0 means the
                # sweep found no dependence at all, so compensation buys nothing.
                "contrast": float(p_max[best] / p_max[worst])
                if p_max[worst] > 0 else float("nan"),
            })
            # Every qubit's trace AT the optimum -- the population-vs-N curve the
            # run exists to produce, so it lands in the metadata, not only a figure.
            results["per_qubit_at_best"] = {
                q: [float(v) for v in pop[k, best]] for k, q in enumerate(qubits)
            }
        else:
            results.update({
                "success": False,
                "best_compensation_amp": float("nan"),
                "best_sink_p_max": float("nan"),
                "best_n_at_max": float("nan"),
                "worst_compensation_amp": float("nan"),
                "worst_sink_p_max": float("nan"),
                "contrast": float("nan"),
            })

        results["compensation_amp"] = [float(v) for v in amps]
        results["round_count"] = [float(v) for v in rounds]
        for key, values in curves.items():
            results[key] = [float(v) for v in values]
        # underscore-prefixed: bulky arrays kept for build_plot_data only
        results["_sink"] = sink_map
        results["_pop"] = pop
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the bulky maps; the per-amp curves and the optimum stay."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """The measured arrays, carried unconditionally (there is no fit to fail)."""
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        amps = np.asarray(dataset[AMP_AXIS].values, dtype=float)
        rounds = np.asarray(dataset[ROUND_AXIS].values, dtype=float)
        pop = results.get("_pop")
        if pop is None:
            pop = np.asarray(
                dataset["population"]
                .transpose(QUBIT_DIM, AMP_AXIS, ROUND_AXIS).values,
                dtype=float,
            )
        sink_map = results.get("_sink")
        if sink_map is None:
            sink = roles["sink"]
            sink_map = (pop[qubits.index(sink)] if sink in qubits
                        else np.full((amps.size, rounds.size), np.nan))

        n_amp = amps.size
        fit_x, fit_y = results.get("_fit_x"), results.get("_fit_y")
        if fit_x is None or fit_y is None:
            # no parabola: a NaN curve on the scanned range keeps the layout fixed
            lo, hi = (float(np.min(amps)), float(np.max(amps))) if n_amp else (0.0, 1.0)
            fit_x, fit_y = np.linspace(lo, hi, N_DENSE), np.full(N_DENSE, np.nan)
        out = xr.Dataset(
            {
                "population": ((QUBIT_DIM, AMP_AXIS, ROUND_AXIS),
                               np.asarray(pop, dtype=float)),
                "sink": ((AMP_AXIS, ROUND_AXIS), np.asarray(sink_map, dtype=float)),
                "sink_p_max": ((AMP_AXIS,),
                               _as_row(results.get("sink_p_max"), n_amp)),
                "sink_n_at_max": ((AMP_AXIS,),
                                  _as_row(results.get("sink_n_at_max"), n_amp)),
                "sink_p_final": ((AMP_AXIS,),
                                 _as_row(results.get("sink_p_final"), n_amp)),
                "sink_round_avg": ((AMP_AXIS,),
                                   _as_row(results.get("sink_round_avg"), n_amp)),
                "round_avg_fit": ((DENSE_AXIS,), np.asarray(fit_y, dtype=float)),
            },
            coords={QUBIT_DIM: qubits, AMP_AXIS: amps, ROUND_AXIS: rounds,
                    DENSE_AXIS: np.asarray(fit_x, dtype=float)},
        )
        # netCDF-safe attrs only: bools as int, absent roles as empty strings.
        out.attrs.update({
            "success": int(bool(results.get("success", False))),
            "best_compensation_amp": float(
                results.get("best_compensation_amp", float("nan"))),
            "best_sink_p_max": float(results.get("best_sink_p_max", float("nan"))),
            "best_n_at_max": float(results.get("best_n_at_max", float("nan"))),
            "worst_compensation_amp": float(
                results.get("worst_compensation_amp", float("nan"))),
            "contrast": float(results.get("contrast", float("nan"))),
            "best_compensation_amp_refined": float(
                results.get("best_compensation_amp_refined", float("nan"))),
            "best_compensation_amp_err": float(
                results.get("best_compensation_amp_err", float("nan"))),
            # absent (a results dict from before the refinement) reads as unresolved
            "compensation_unresolved": int(bool(results.get("compensation_unresolved", 1.0))),
            "refine_reason": str(results.get("refine_reason", "")),
            "refine_first_round": int(REFINE_FIRST_ROUND),
            "refine_half_width": float(REFINE_HALF_WIDTH),
            "compensation_target": str(results.get("compensation_target", "")),
            **roles,
        })
        return out

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return render_figures(
            {
                FIG_SCAN: lambda: plot_compensation_scan(plot_data),
                FIG_MAP: lambda: plot_sink_map(plot_data),
            },
            label=self.estimator_name,
        )


def _merge_refine(results: Dict[str, Any], refine: Dict[str, Any]) -> None:
    """Fold ``_refine_optimum``'s output into ``results`` (JSON-safe scalars and
    lists; the parabola samples stay underscore-private for build_plot_data)."""
    for key in ("best_compensation_amp_refined", "best_compensation_amp_err",
                "compensation_unresolved"):
        results[key] = float(refine[key])
    results["refine_reason"] = str(refine["refine_reason"])
    results["refine_first_round"] = int(REFINE_FIRST_ROUND)
    results["refine_half_width"] = float(REFINE_HALF_WIDTH)
    results["sink_round_avg"] = [float(v) for v in refine["sink_round_avg"]]
    results["_fit_x"] = refine["_fit_x"]
    results["_fit_y"] = refine["_fit_y"]


def _as_row(values, size: int) -> np.ndarray:
    """A per-amplitude column of the right length, NaN-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the curves, so a missing one degrades rather than
    raising -- rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan)
    return np.asarray(values, dtype=float)
