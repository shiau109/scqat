"""Excitation transport along a Trotterized unidirectional chain — record-only.

The sequence excites ONE qubit of a three-qubit chain, then repeats a Trotter
step (partial swap source->relay, partial swap relay->sink, parametric reset of
the relay, per-qubit AC-Stark phase compensation) ``N`` times before reading
every qubit out in the same shot. This estimator does NOT fit the transport: it
draws the measured populations against ``N`` and extracts a small
self-describing summary of where each qubit's excitation peaks. The SUCCESS /
``min_transfer`` verdict stays in SCQO.

The relay reset is what makes the coupling one-way, so the picture to read off
the figure is: the source decays, the sink grows, and the relay stays small.

IDEAL CURVES, computed and never fitted. Given the two swaps' per-application
angles (``theta_first_rad`` source->relay, ``theta_second_rad`` relay->sink) the
estimator overlays the closed form of a perfect round — exact exchanges, a
perfect relay reset, the source-sink phase compensated to zero, no decay and
perfect readout:

    P_source(N) = cos(theta1)^(2N)
    P_sink(N)   = [sin(theta1) sin(theta2) sum_{j<N} cos(theta2)^(N-1-j) cos(theta1)^j]^2

The source curve needs only ``theta1``; the sink needs both. A missing angle
(``None``/NaN) leaves that curve NaN — the measured traces are drawn either way.
The gap between a measured trace and its ideal is the data: decay and readout
contrast pull both down, a wrong compensation pulls the sink alone.

Unlike the pair estimators (``qc_n_swap_amp``, ``pair_swap_chevron``) this one
consumes the WHOLE multi-qubit dataset rather than a per-target slice: the joint
panel is a cross-qubit quantity, and a per-target split cannot draw it.

Dataset contract:
  vars   : ``population`` — dims ``(qubit, round_count)`` in any order: the
           averaged marginal P(level >= 1) of each chain qubit.
           ``joint_population`` — OPTIONAL, dims ``(joint_state, round_count)``:
           the joint distribution over the chain's basis states, present only
           when the run kept every shot (SCQO's ``readout_mode="shot"``).
  coords : ``qubit`` (chain qubit names, chain order) / ``round_count``
           (dimensionless Trotter-step count N) / ``joint_state`` (per-qubit
           level digits, leftmost digit = the first ``qubit``) when joint.
  kwargs : ``source`` / ``relay`` / ``sink`` — chain role names used to label
           the figure and to pick the transport summary; all optional.
           ``theta_first_rad`` / ``theta_second_rad`` — the two swaps'
           per-application angles (rad) for the ideal curves; optional, 0 for
           a step that plays no swap.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.qc_unidirectional_trotter.visualization import (
    plot_chain_joint,
    plot_chain_populations,
)

QUBIT_DIM = "qubit"
AXIS = "round_count"
JOINT_DIM = "joint_state"

#: chain role kwargs, in chain order — the labels the figure reads back.
ROLES = ("source", "relay", "sink")

#: figure keys — stable across runs and readout modes, so a saved plotdata.nc
#: always replots under the same PNG names. ``save_figures`` prefixes the
#: estimator name unless the key IS it, so these land as
#: ``qc_unidirectional_trotter.png`` and ``qc_unidirectional_trotter_joint.png``.
FIG_POPULATIONS = "qc_unidirectional_trotter"
FIG_JOINT = "joint"


#: the swap-angle kwargs, in chain order (source->relay, relay->sink).
THETA_KEYS = ("theta_first_rad", "theta_second_rad")


def _roles(kwargs: Dict[str, Any]) -> Dict[str, str]:
    """The chain role names supplied by the caller, blank when unknown."""
    return {role: str(kwargs.get(role) or "") for role in ROLES}


def _thetas(kwargs: Dict[str, Any]) -> Dict[str, float]:
    """The two swap angles supplied by the caller, NaN when unknown."""
    out = {}
    for key in THETA_KEYS:
        value = kwargs.get(key)
        try:
            out[key] = float(value) if value is not None else float("nan")
        except (TypeError, ValueError):
            out[key] = float("nan")
    return out


def ideal_transport(
    rounds: np.ndarray, theta_first: float, theta_second: float
) -> Dict[str, np.ndarray]:
    """``{"source": ..., "sink": ...}`` of a perfect round, per round count.

    The closed form in the module docstring. The sink amplitude is summed
    term by term rather than through the geometric closed form, so equal
    angles (whose closed form divides by zero) need no special case. A NaN
    angle gives a NaN curve: the source needs ``theta_first``, the sink both.
    """
    n = np.rint(np.asarray(rounds, dtype=float))
    c1, s1 = np.cos(theta_first), np.sin(theta_first)
    c2, s2 = np.cos(theta_second), np.sin(theta_second)
    # explicit, because nan ** 0 is 1: an unknown angle must not claim N=0
    source = c1 ** (2.0 * n) if np.isfinite(theta_first) else np.full(n.shape, np.nan)
    sink = np.full(n.shape, np.nan)
    if np.isfinite(theta_first) and np.isfinite(theta_second):
        for i, m in enumerate(n.astype(int)):
            j = np.arange(max(m, 0))
            amp = s1 * s2 * np.sum(c2 ** (m - 1 - j) * c1 ** j)
            sink[i] = amp ** 2
    return {"source": source, "sink": sink}


def _qubit_names(dataset: xr.Dataset) -> List[str]:
    return [str(v) for v in np.atleast_1d(dataset[QUBIT_DIM].values)]


def _trace_summary(rounds: np.ndarray, trace: np.ndarray) -> Dict[str, float]:
    """One qubit's population-vs-N summary. All-NaN degrades to NaN, never raises."""
    trace = np.asarray(trace, dtype=float)
    out = {
        "p_initial": float(trace[0]) if trace.size else float("nan"),
        "p_final": float(trace[-1]) if trace.size else float("nan"),
    }
    if not np.isfinite(trace).any():
        out.update(p_max=float("nan"), p_min=float("nan"), n_at_max=float("nan"))
        return out
    i = int(np.nanargmax(trace))
    out.update(
        p_max=float(trace[i]),
        p_min=float(np.nanmin(trace)),
        n_at_max=float(rounds[i]),
    )
    return out


class QcUnidirectionalTrotterEstimator(BaseEstimator):
    """Draw the chain's transport curves (and joint distribution) vs N."""

    estimator_name = "qc_unidirectional_trotter"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "population" not in dataset.data_vars:
            raise ValueError(
                "qc_unidirectional_trotter estimator requires the population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in (QUBIT_DIM, AXIS):
            if axis not in dataset.coords:
                raise ValueError(
                    f"qc_unidirectional_trotter estimator requires a {axis!r} coordinate"
                )
        if "joint_population" in dataset.data_vars and JOINT_DIM not in dataset.coords:
            raise ValueError(
                "qc_unidirectional_trotter estimator requires a "
                f"{JOINT_DIM!r} coordinate alongside joint_population"
            )

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        roles = _roles(kwargs)
        qubits = _qubit_names(dataset)
        rounds = np.asarray(dataset[AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"].transpose(QUBIT_DIM, AXIS).values, dtype=float
        )

        per_qubit = {q: _trace_summary(rounds, pop[k]) for k, q in enumerate(qubits)}
        results: Dict[str, Any] = {
            "qubits": qubits,
            "n_round_count": int(rounds.size),
            "max_round_count": float(rounds[-1]) if rounds.size else float("nan"),
            "per_qubit": per_qubit,
            **roles,
        }
        # The transport headline, named rather than left to a reader of
        # per_qubit: SCQO's min_transfer verdict is made against exactly this.
        sink = roles["sink"]
        if sink in per_qubit:
            results["sink_p_max"] = per_qubit[sink]["p_max"]
            results["sink_n_at_max"] = per_qubit[sink]["n_at_max"]
            results["sink_p_final"] = per_qubit[sink]["p_final"]

        # The ideal curves' inputs and the ideal sink's own peak — what the
        # measured sink_p_max is to be read against.
        thetas = _thetas(kwargs)
        results.update(thetas)
        ideal = ideal_transport(rounds, *thetas.values())
        peak = _trace_summary(rounds, ideal["sink"])
        results["ideal_sink_p_max"] = peak["p_max"]
        results["ideal_sink_n_at_max"] = peak["n_at_max"]

        has_joint = "joint_population" in dataset.data_vars
        results["has_joint"] = bool(has_joint)
        if has_joint:
            joint = dataset["joint_population"].transpose(JOINT_DIM, AXIS)
            labels = [str(v) for v in np.atleast_1d(joint[JOINT_DIM].values)]
            values = np.asarray(joint.values, dtype=float)
            results["joint_p_max"] = {
                label: (float(np.nanmax(row)) if np.isfinite(row).any() else float("nan"))
                for label, row in zip(labels, values)
            }
        return results

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """The measured arrays, carried unconditionally (there is no fit to
        fail), plus the ideal curves on the source and sink rows — NaN where an
        angle or a role is unknown, and always NaN on the relay (a perfect
        reset leaves it empty, which the figure does not need a line for)."""
        roles = _roles(kwargs)
        thetas = _thetas(kwargs)
        qubits = _qubit_names(dataset)
        rounds = np.asarray(dataset[AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"].transpose(QUBIT_DIM, AXIS).values, dtype=float
        )
        ideal = np.full(pop.shape, np.nan)
        curves = ideal_transport(rounds, *thetas.values())
        for role in ("source", "sink"):
            if roles[role] in qubits:
                ideal[qubits.index(roles[role])] = curves[role]
        out = xr.Dataset(
            {"population": ((QUBIT_DIM, AXIS), pop),
             "ideal_population": ((QUBIT_DIM, AXIS), ideal)},
            coords={QUBIT_DIM: qubits, AXIS: rounds},
        )
        has_joint = "joint_population" in dataset.data_vars
        if has_joint:
            joint = dataset["joint_population"].transpose(JOINT_DIM, AXIS)
            out["joint_population"] = (
                (JOINT_DIM, AXIS),
                np.asarray(joint.values, dtype=float),
            )
            out = out.assign_coords(
                {JOINT_DIM: [str(v) for v in np.atleast_1d(joint[JOINT_DIM].values)]}
            )
        # netCDF-safe attrs only: the bool as int, absent roles as empty
        # strings, absent angles as NaN floats.
        out.attrs.update({"has_joint": int(has_joint), **roles, **thetas})
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
        builders = {FIG_POPULATIONS: lambda: plot_chain_populations(plot_data)}
        if "joint_population" in plot_data.data_vars:
            builders[FIG_JOINT] = lambda: plot_chain_joint(plot_data)
        return render_figures(builders, label=self.estimator_name)
