"""Figures for the unidirectional-coupling Trotter chain.

Both plotters draw from ``plot_data`` ONLY (the estimator-output-contract rule)
and neither fits anything. The population figure overlays COMPUTED theory: the
discrete Trotter model (hollow points) and the continuous master equation
(line), with T1/T2* decay when it was known, and the ideal master equation as a
faint ceiling beside them. Each overlay is drawn only where finite, so a run
without angles (or a plot_data.nc written before an overlay existed) shows the
measured points alone. Neither plotter may assume a finite value anywhere: the
population axis is pinned to 0..1 rather than taken from the data, and an
all-NaN acquisition renders an annotated empty panel instead of raising (which
would cost the run every other figure).
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_chain_populations", "plot_chain_joint"]

#: chain roles, in chain order — the words appended to a qubit's legend entry.
_ROLE_TAGS = ("source", "relay", "sink")


def _role_of(plot_data: xr.Dataset) -> Dict[str, str]:
    """``{qubit name: role word}`` from the plot-data attrs (may be empty)."""
    out: Dict[str, str] = {}
    for role in _ROLE_TAGS:
        name = str(plot_data.attrs.get(role, "") or "")
        if name:
            out[name] = role
    return out


def _names(plot_data: xr.Dataset, dim: str) -> List[str]:
    return [str(v) for v in np.atleast_1d(plot_data[dim].values)]


def _annotate_empty(ax: plt.Axes, message: str) -> None:
    """A panel with nothing finite to draw still has to render."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _rows(plot_data: xr.Dataset, name: str, dim: str, shape) -> np.ndarray:
    """A (qubit, dim) theory array, or all-NaN when this plot data predates it."""
    if name not in plot_data.data_vars:
        return np.full(shape, np.nan)
    return np.asarray(plot_data[name].transpose("qubit", dim).values, dtype=float)


def plot_chain_populations(plot_data: xr.Dataset) -> plt.Figure:
    """P(|1>) against the Trotter-step count N, one colour per chain qubit.

    MEASURED data are filled points. The source and sink carry two theories in
    their own colour: the DISCRETE Trotter model as hollow points on the round
    axis, and the continuous MASTER EQUATION as a line. Both include T1/T2*
    decay when the estimator had it; the ideal master equation is then drawn
    as a faint dotted line, the no-decay ceiling. A theory absent from the plot
    data (older runs) or NaN is simply not drawn."""
    rounds = np.asarray(plot_data["round_count"].values, dtype=float)
    qubits = _names(plot_data, "qubit")
    pop = np.asarray(
        plot_data["population"].transpose("qubit", "round_count").values, dtype=float
    )
    trotter = _rows(plot_data, "trotter_population", "round_count", pop.shape)
    if not np.isfinite(trotter).any():
        # plot data from before the decay models carried only the discrete ideal
        trotter = _rows(plot_data, "ideal_population", "round_count", pop.shape)
    has_model = "model_round" in plot_data.coords
    model_round = (np.asarray(plot_data["model_round"].values, dtype=float)
                   if has_model else np.zeros(1))
    master = _rows(plot_data, "master_population", "model_round",
                   (len(qubits), model_round.size))
    master_ideal = _rows(plot_data, "master_ideal_population", "model_round",
                         (len(qubits), model_round.size))
    decayed = bool(int(plot_data.attrs.get("decoherence_applied", 0)))
    role_of = _role_of(plot_data)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    drawn = False
    for k, qubit in enumerate(qubits):
        role = role_of.get(qubit, "")
        label = f"{qubit} ({role})" if role else qubit
        # fixed per chain position, so q1/q2/q3 keep their colour across runs
        color = f"C{k % 10}"
        ax.plot(rounds, pop[k], linestyle="none", marker="o", markersize=5,
                color=color, label=label, zorder=3)
        if np.isfinite(master[k]).any():
            ax.plot(model_round, master[k], color=color, linewidth=1.6, zorder=2)
        if decayed and np.isfinite(master_ideal[k]).any():
            ax.plot(model_round, master_ideal[k], color=color, linewidth=1.0,
                    linestyle=":", alpha=0.55, zorder=1)
        if np.isfinite(trotter[k]).any():
            ax.plot(rounds, trotter[k], linestyle="none", marker="o",
                    markersize=8.5, markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.1, zorder=2)
        drawn = drawn or bool(np.isfinite(pop[k]).any())
    ax.set_xlabel("Trotter steps N")
    ax.set_ylabel("P(|1>)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    if drawn:
        from matplotlib.lines import Line2D

        grey = "0.45"
        suffix = " (T1, T2*)" if decayed else " (ideal)"
        styles = [Line2D([], [], color=grey, linestyle="none", marker="o",
                         markersize=5, label="data")]
        if np.isfinite(trotter).any():
            styles.append(Line2D([], [], color=grey, linestyle="none", marker="o",
                                 markersize=8.5, markerfacecolor="none",
                                 label="Trotter theory" + suffix))
        if np.isfinite(master).any():
            styles.append(Line2D([], [], color=grey, linewidth=1.6,
                                 label="master equation" + suffix))
        if decayed and np.isfinite(master_ideal).any():
            styles.append(Line2D([], [], color=grey, linewidth=1.0, linestyle=":",
                                 label="master equation (ideal)"))
        handles, _labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + styles, fontsize=8.5, ncol=2)
    else:
        _annotate_empty(ax, "no finite population data")
    ax.set_title(_title(plot_data, decayed))
    return fig


def _title(plot_data: xr.Dataset, decayed: bool) -> str:
    """The run's theory inputs, so a saved figure says what it was drawn with."""
    title = "unidirectional Trotter chain — excitation transport vs N"
    angles = [float(plot_data.attrs.get(key, np.nan))
              for key in ("theta_first_rad", "theta_second_rad")]
    if any(np.isfinite(angles)):
        shown = ", ".join("?" if not np.isfinite(a) else f"{a:.3f}" for a in angles)
        title += "\n" + f"swap angles ({shown}) rad"
    if decayed:
        a = plot_data.attrs
        title += (f"; round {float(a['round_duration_ns']):.0f} ns;"
                  f" T1/T2* source {float(a['source_t1_s']) * 1e6:.1f}/"
                  f"{float(a['source_t2_star_s']) * 1e6:.1f} us,"
                  f" sink {float(a['sink_t1_s']) * 1e6:.1f}/"
                  f"{float(a['sink_t2_star_s']) * 1e6:.1f} us")
    return title


def plot_chain_joint(plot_data: xr.Dataset) -> plt.Figure:
    """The joint chain distribution against N — one row per basis state.

    Present only when the run kept every shot; each label's digit order is the
    chain-qubit order (leftmost digit = the first ``qubit``).

    ``imshow`` rather than ``pcolormesh``: the round axis is an integer grid, so
    the extent is exact, and a single-column map (``max_rounds=0``) still draws
    instead of failing on cell-boundary inference."""
    rounds = np.asarray(plot_data["round_count"].values, dtype=float)
    labels = _names(plot_data, "joint_state")
    joint = np.asarray(
        plot_data["joint_population"].transpose("joint_state", "round_count").values,
        dtype=float,
    )
    qubits = _names(plot_data, "qubit")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if rounds.size == 0 or not labels:
        _annotate_empty(ax, "no joint-population data")
        return fig
    im = ax.imshow(
        joint,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(rounds[0] - 0.5, rounds[-1] + 0.5, -0.5, len(labels) - 0.5),
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Trotter steps N")
    ket = "".join(qubits)
    ax.set_ylabel(f"joint state  |{ket}>" if ket else "joint state")
    fig.colorbar(im, ax=ax, label="joint population")
    if not np.isfinite(joint).any():
        _annotate_empty(ax, "no finite joint-population data")
    ax.set_title("unidirectional Trotter chain — joint populations vs N")
    return fig
