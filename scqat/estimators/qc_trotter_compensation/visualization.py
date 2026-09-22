"""Compensation-scan plotting for ``qc_trotter_compensation``.

Consumes the **plot_data** Dataset built by
``QcTrotterCompensationEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``qubit``, ``compensation_amp``, ``round_count``,
         ``compensation_amp_dense`` (the parabola's own axis)
vars   : ``population`` (qubit, compensation_amp, round_count),
         ``sink`` (compensation_amp, round_count),
         ``sink_p_max`` / ``sink_n_at_max`` / ``sink_p_final`` /
         ``sink_round_avg`` (compensation_amp),
         ``round_avg_fit`` (compensation_amp_dense)
attrs  : ``success``, ``best_compensation_amp``, ``best_sink_p_max``,
         ``best_n_at_max``, ``worst_compensation_amp``, ``contrast``,
         ``best_compensation_amp_refined``, ``best_compensation_amp_err``,
         ``compensation_unresolved``, ``refine_reason``, ``refine_first_round``,
         ``refine_half_width``, ``compensation_target``,
         ``source`` / ``relay`` / ``sink``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

QUBIT_DIM = "qubit"
AMP_AXIS = "compensation_amp"
ROUND_AXIS = "round_count"
DENSE_AXIS = "compensation_amp_dense"

REFINED_COLOR = "tab:orange"


def _refined(plot_data: xr.Dataset):
    """``(value, err)`` of the refined optimum; NaN when it was refused."""
    value = float(plot_data.attrs.get("best_compensation_amp_refined", float("nan")))
    err = float(plot_data.attrs.get("best_compensation_amp_err", float("nan")))
    return value, err


def _mark_refined(ax, value: float, err: float, color: str, label: bool = True) -> None:
    """The refined optimum as a solid line, and its spread as a band."""
    if not np.isfinite(value):
        return
    text = f"refined {value:.4g}" + (f" +- {err:.2g}" if np.isfinite(err) else "")
    ax.axvline(value, color=color, lw=1.6, ls="-", label=text if label else None)
    if np.isfinite(err) and err > 0:
        ax.axvspan(value - err, value + err, color=color, alpha=0.2, lw=0)


def plot_compensation_scan(plot_data: xr.Dataset) -> plt.Figure:
    """The optimum, and the chain transport at it.

    Top: the sink peak population against the swept compensation amplitude, with
    ``n_at_max`` on a twin axis. Those are the two independent readings of the
    same phase condition -- when the rounds cancel only the last one survives and
    the peak sits at N=1, so ``n_at_max`` dropping to 1-2 marks the destructive
    amplitudes as sharply as the height does. Over them, the sink averaged over
    the rounds that carry phase, its parabola, and the refined optimum (solid,
    with its spread as a band); the brightest pixel stays as a dotted line.

    Bottom: every chain qubit against N at the best amplitude, with the sink at
    the WORST amplitude overlaid, so the size of the effect is visible rather
    than asserted.
    """
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8.5), constrained_layout=True
    )

    amps = np.asarray(plot_data[AMP_AXIS].values, dtype=float)
    rounds = np.asarray(plot_data[ROUND_AXIS].values, dtype=float)
    p_max = np.asarray(plot_data["sink_p_max"].values, dtype=float)
    n_at_max = np.asarray(plot_data["sink_n_at_max"].values, dtype=float)

    sink_name = plot_data.attrs.get("sink", "") or "sink"
    target = plot_data.attrs.get("compensation_target", "") or "?"

    # --- top: the scan (raw, always drawn) -------------------------------
    ax_top.plot(amps, p_max, "o-", color="tab:blue", markersize=5,
                label=f"{sink_name} peak population")
    # The ridge reading: the same sink averaged over the rounds that carry
    # phase, and the parabola whose vertex is the refined optimum.
    if "sink_round_avg" in plot_data:
        first = int(plot_data.attrs.get("refine_first_round", 2))
        ax_top.plot(amps, np.asarray(plot_data["sink_round_avg"].values, dtype=float),
                    "s-", color=REFINED_COLOR, markersize=4,
                    label=f"{sink_name} averaged over N >= {first}")
    if "round_avg_fit" in plot_data:
        fit = np.asarray(plot_data["round_avg_fit"].values, dtype=float)
        if np.isfinite(fit).any():
            ax_top.plot(np.asarray(plot_data[DENSE_AXIS].values, dtype=float), fit,
                        "-", color="k", lw=1.0, alpha=0.7, label="parabola")
    ax_top.set_xlabel(f"compensation amplitude on {target}  (factor of baked stark amp)")
    ax_top.set_ylabel("sink population", color="tab:blue")
    ax_top.tick_params(axis="y", labelcolor="tab:blue")

    ax_n = ax_top.twinx()
    ax_n.plot(amps, n_at_max, "s--", color="tab:grey", markersize=4, alpha=0.8,
              label="N at peak")
    ax_n.set_ylabel("N at peak", color="tab:grey")
    ax_n.tick_params(axis="y", labelcolor="tab:grey")

    best = float(plot_data.attrs.get("best_compensation_amp", float("nan")))
    worst = float(plot_data.attrs.get("worst_compensation_amp", float("nan")))
    if np.isfinite(best):
        ax_top.axvline(best, color="tab:green", lw=1.4, ls=":",
                       label=f"brightest pixel {best:.4g}")
    if np.isfinite(worst):
        ax_top.axvline(worst, color="tab:red", lw=1.2, ls=":")
    refined, err = _refined(plot_data)
    _mark_refined(ax_top, refined, err, REFINED_COLOR)

    contrast = float(plot_data.attrs.get("contrast", float("nan")))
    title = "compensation scan"
    if np.isfinite(refined):
        title += f"     refined optimum {refined:.4g}"
        if np.isfinite(err):
            title += f" +- {err:.2g}"
    else:
        reason = str(plot_data.attrs.get("refine_reason", "") or "")
        title += "     refined optimum unresolved" + (f":\n{reason}" if reason else "")
    if np.isfinite(contrast):
        title += f"     best/worst = {contrast:.2f}x"
    ax_top.set_title(title)

    handles, labels = ax_top.get_legend_handles_labels()
    h2, l2 = ax_n.get_legend_handles_labels()
    ax_top.legend(handles + h2, labels + l2, loc="best", fontsize=9)

    # --- bottom: the transport curves at the optimum ---------------------
    population = np.asarray(plot_data["population"].values, dtype=float)
    qubits = [str(v) for v in np.atleast_1d(plot_data[QUBIT_DIM].values)]
    sink_map = np.asarray(plot_data["sink"].values, dtype=float)

    if np.isfinite(best) and amps.size:
        i_best = int(np.argmin(np.abs(amps - best)))
        for k, name in enumerate(qubits):
            ax_bot.plot(rounds, population[k, i_best], "o-", markersize=4,
                        label=f"{name}")
        if np.isfinite(worst):
            i_worst = int(np.argmin(np.abs(amps - worst)))
            ax_bot.plot(rounds, sink_map[i_worst], "x--", color="tab:red",
                        markersize=5, alpha=0.8,
                        label=f"{sink_name} at worst amp ({worst:.4g})")
        ax_bot.set_title(f"chain transport at the brightest-pixel amplitude ({best:.4g})")
    else:
        # No optimum: still show the raw traces so the run is not figure-less.
        for k, name in enumerate(qubits):
            ax_bot.plot(rounds, population[k].T, "-", alpha=0.35,
                        label=name)
        ax_bot.set_title("no optimum found - every compensation amplitude shown")

    ax_bot.set_xlabel("Trotter steps N")
    ax_bot.set_ylabel("population")
    ax_bot.set_ylim(0.0, 1.0)
    if ax_bot.get_legend_handles_labels()[0]:
        ax_bot.legend(loc="best", fontsize=9)

    return fig


def plot_sink_map(plot_data: xr.Dataset) -> plt.Figure:
    """The sink population over (compensation amplitude, N).

    The scan curve is a projection of this map; the map itself shows HOW the
    transport fails off the optimum -- the peak walking back to N=1 as the
    round-to-round contributions stop adding.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)

    amps = np.asarray(plot_data[AMP_AXIS].values, dtype=float)
    rounds = np.asarray(plot_data[ROUND_AXIS].values, dtype=float)
    sink_map = np.asarray(plot_data["sink"].values, dtype=float)   # (amp, round)

    mesh = ax.pcolormesh(*np.meshgrid(amps, rounds), sink_map.T,
                         shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="sink population")

    n_at_max = np.asarray(plot_data["sink_n_at_max"].values, dtype=float)
    if np.isfinite(n_at_max).any():
        ax.plot(amps, n_at_max, "w.--", lw=1.2, markersize=5, alpha=0.9,
                label="N at peak")

    best = float(plot_data.attrs.get("best_compensation_amp", float("nan")))
    if np.isfinite(best):
        ax.axvline(best, color="w", lw=1.4, ls=":", label=f"brightest pixel {best:.4g}")
    refined, err = _refined(plot_data)
    _mark_refined(ax, refined, err, REFINED_COLOR)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize=9)

    sink_name = plot_data.attrs.get("sink", "") or "sink"
    target = plot_data.attrs.get("compensation_target", "") or "?"
    ax.set_xlabel(f"compensation amplitude on {target}")
    ax.set_ylabel("Trotter steps N")
    ax.set_title(f"{sink_name} population vs (compensation amplitude, N)")
    return fig
