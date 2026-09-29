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

TWO THEORIES, and decoherence when it is known. The state stays in the chain's
single-excitation manifold (the relay is emptied every round), so three numbers
carry it: the source and sink populations and their coherence ``r = <T|rho|S>``.

* TROTTER (discrete, the sequence itself): per round, the swap-swap-reset Kraus
  map ``P_S <- c1^2 P_S``, ``r <- c1 c2 r - s1 s2 c1 P_S``,
  ``P_T <- c2^2 P_T + (s1 s2)^2 P_S - 2 s1 s2 c2 r``, then the round's decay.
* MASTER EQUATION (continuous, the collisional limit ``gamma_k = theta_k^2`` per
  round): the cascaded Lindblad equation in the same variables,
  ``P_S' = -(g1 + G1s) P_S``, ``r' = -((g1+g2)/2 + (G1s+G1t)/2 + Gps + Gpt) r - k P_S``,
  ``P_T' = -(g2 + G1t) P_T - 2 k r`` with ``k = sqrt(g1 g2)``.

Decay enters only when ``round_duration_ns`` and BOTH ends' ``T1`` and ``T2*``
are given: per round ``G1 = t_round / T1`` (amplitude damping) and
``Gp = t_round (1/T2* - 1/(2 T1))`` (pure dephasing, clipped at 0). The relay's
own decay is left out: it holds the excitation only between the two swaps of a
round. Without decay both models are the ideal ones, and the discrete one equals
the closed form above.

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
           ``round_duration_ns``, ``source_t1_s``, ``source_t2_star_s``,
           ``sink_t1_s``, ``sink_t2_star_s`` — the decay the two models carry;
           optional, all five needed for any decay.
"""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.linalg import expm

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

#: the decay kwargs: the round period that turns times into per-round decay, and
#: each END's T1 / T2* (the relay's own decay is left out, module docstring).
DECOHERENCE_KEYS = ("round_duration_ns", "source_t1_s", "source_t2_star_s",
                    "sink_t1_s", "sink_t2_star_s")

#: points on the continuous (master-equation) curve's round axis.
MODEL_POINTS = 301
MODEL_DIM = "model_round"


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


def _as_float(value: Any) -> float:
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _decoherence(kwargs: Dict[str, Any]) -> Dict[str, float]:
    """The five decay inputs supplied by the caller, NaN when unknown."""
    return {key: _as_float(kwargs.get(key)) for key in DECOHERENCE_KEYS}


def per_round_decay(inputs: Dict[str, float]) -> Optional[Dict[str, float]]:
    """``{g1_source, gphi_source, g1_sink, gphi_sink}`` per round, or ``None``
    when any of the five inputs is missing (then both models stay ideal).

    ``g1 = t_round / T1`` is the amplitude-damping exponent and
    ``gphi = t_round (1/T2* - 1/(2 T1))`` the pure-dephasing one, clipped at 0
    because a measured T2* can exceed 2 T1 within its error bar."""
    values = [inputs.get(key, float("nan")) for key in DECOHERENCE_KEYS]
    if not all(np.isfinite(v) and v > 0 for v in values):
        return None
    t_round = inputs["round_duration_ns"] * 1e-9
    out = {}
    for end in ("source", "sink"):
        t1, t2 = inputs[f"{end}_t1_s"], inputs[f"{end}_t2_star_s"]
        out[f"g1_{end}"] = t_round / t1
        out[f"gphi_{end}"] = max(t_round * (1.0 / t2 - 0.5 / t1), 0.0)
    return out


def _no_decay() -> Dict[str, float]:
    return {"g1_source": 0.0, "gphi_source": 0.0, "g1_sink": 0.0, "gphi_sink": 0.0}


def trotter_transport(
    rounds: np.ndarray, theta_first: float, theta_second: float,
    decay: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """The DISCRETE model: ``{"source": P_S, "sink": P_T}`` per round count.

    Iterates the swap-swap-reset Kraus map on (P_S, r, P_T) and then the round's
    decay (``per_round_decay``; ``None`` = ideal, which reproduces
    ``ideal_transport``). NaN angles give NaN curves."""
    n = np.rint(np.asarray(rounds, dtype=float)).astype(int)
    nan = np.full(n.shape, np.nan)
    if not (np.isfinite(theta_first) and np.isfinite(theta_second)) or n.size == 0:
        return {"source": nan.copy(), "sink": nan.copy()}
    d = decay or _no_decay()
    c1, s1 = np.cos(theta_first), np.sin(theta_first)
    c2, s2 = np.cos(theta_second), np.sin(theta_second)
    keep_s, keep_t = np.exp(-d["g1_source"]), np.exp(-d["g1_sink"])
    keep_r = np.exp(-0.5 * (d["g1_source"] + d["g1_sink"])
                    - d["gphi_source"] - d["gphi_sink"])
    ps, r, pt = 1.0, 0.0, 0.0
    source, sink = [ps], [pt]
    for _ in range(max(int(n.max()), 0)):
        ps, r, pt = (c1 ** 2 * ps,
                     c1 * c2 * r - s1 * s2 * c1 * ps,
                     c2 ** 2 * pt + (s1 * s2) ** 2 * ps - 2.0 * s1 * s2 * c2 * r)
        ps, r, pt = ps * keep_s, r * keep_r, pt * keep_t
        source.append(ps)
        sink.append(pt)
    idx = np.clip(n, 0, None)
    return {"source": np.asarray(source)[idx], "sink": np.asarray(sink)[idx]}


def master_transport(
    times: np.ndarray, theta_first: float, theta_second: float,
    decay: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """The CONTINUOUS model: the cascaded master equation at ``times`` (in
    rounds), with ``gamma_k = theta_k^2`` per round and the same decay as
    ``trotter_transport``. Solved exactly as a 3x3 linear system."""
    t = np.asarray(times, dtype=float)
    nan = np.full(t.shape, np.nan)
    if not (np.isfinite(theta_first) and np.isfinite(theta_second)):
        return {"source": nan.copy(), "sink": nan.copy()}
    d = decay or _no_decay()
    g1, g2 = theta_first ** 2, theta_second ** 2
    k = np.sqrt(g1 * g2)
    gen = np.array([
        [-(g1 + d["g1_source"]), 0.0, 0.0],
        [-k, -(0.5 * (g1 + g2) + 0.5 * (d["g1_source"] + d["g1_sink"])
               + d["gphi_source"] + d["gphi_sink"]), 0.0],
        [0.0, -2.0 * k, -(g2 + d["g1_sink"])],
    ])
    states = np.array([expm(gen * ti) @ np.array([1.0, 0.0, 0.0]) for ti in t])
    if states.size == 0:
        return {"source": nan.copy(), "sink": nan.copy()}
    return {"source": states[:, 0], "sink": states[:, 2]}


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

        # The discrete model WITH decay, when all five inputs are known — the
        # curve the measured sink is expected to follow up to readout contrast.
        inputs = _decoherence(kwargs)
        decay = per_round_decay(inputs)
        results.update(inputs)
        results["decoherence_applied"] = decay is not None
        model = (trotter_transport(rounds, *thetas.values(), decay)
                 if decay is not None else {"sink": np.full(rounds.shape, np.nan)})
        peak = _trace_summary(rounds, model["sink"])
        results["model_sink_p_max"] = peak["p_max"]
        results["model_sink_n_at_max"] = peak["n_at_max"]

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
        fail), plus the theory on the source and sink rows — NaN where an angle
        or a role is unknown, and always NaN on the relay (a perfect reset
        leaves it empty, which the figure does not need a line for):

        * ``ideal_population`` — the discrete closed form, no decay;
        * ``trotter_population`` — the discrete model, with decay when known;
        * ``master_population`` / ``master_ideal_population`` — the continuous
          model with and without decay, on a fine ``model_round`` axis.
        """
        roles = _roles(kwargs)
        thetas = _thetas(kwargs)
        inputs = _decoherence(kwargs)
        decay = per_round_decay(inputs)
        qubits = _qubit_names(dataset)
        rounds = np.asarray(dataset[AXIS].values, dtype=float)
        pop = np.asarray(
            dataset["population"].transpose(QUBIT_DIM, AXIS).values, dtype=float
        )
        finite = rounds[np.isfinite(rounds)]
        model_round = np.linspace(0.0, float(finite.max()) if finite.size else 0.0,
                                  MODEL_POINTS)

        def rows(curves, length):
            out_rows = np.full((len(qubits), length), np.nan)
            for role in ("source", "sink"):
                if roles[role] in qubits:
                    out_rows[qubits.index(roles[role])] = curves[role]
            return out_rows

        n = rounds.size
        ideal = rows(ideal_transport(rounds, *thetas.values()), n)
        trotter = rows(trotter_transport(rounds, *thetas.values(), decay), n)
        master = rows(master_transport(model_round, *thetas.values(), decay),
                      MODEL_POINTS)
        master_ideal = rows(master_transport(model_round, *thetas.values()),
                            MODEL_POINTS)
        out = xr.Dataset(
            {"population": ((QUBIT_DIM, AXIS), pop),
             "ideal_population": ((QUBIT_DIM, AXIS), ideal),
             "trotter_population": ((QUBIT_DIM, AXIS), trotter),
             "master_population": ((QUBIT_DIM, MODEL_DIM), master),
             "master_ideal_population": ((QUBIT_DIM, MODEL_DIM), master_ideal)},
            coords={QUBIT_DIM: qubits, AXIS: rounds, MODEL_DIM: model_round},
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
        # netCDF-safe attrs only: bools as int, absent roles as empty strings,
        # absent angles and decay inputs as NaN floats.
        out.attrs.update({"has_joint": int(has_joint), **roles, **thetas,
                          **inputs, "decoherence_applied": int(decay is not None)})
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
