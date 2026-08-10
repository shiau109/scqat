"""Shared helpers for the pair-swap raw-population map estimators.

``pair_swap_chevron``, ``pair_swap_flux_map`` and ``qc_n_swap_amp`` are the same
analysis over different 2-D sweep grids: excite one member of a pair, read BOTH
members out jointly, and draw the four joint state populations as maps. Only the
two axis NAMES differ, so the summary, the plot-data projection and the figure
live here once and each estimator supplies its own coordinate names.

Like ``_iq_plane.py`` this is a plain shared FUNCTION module (function-level
sharing is what the estimator-layering rule permits); it lives outside ``tools/``
because it is presentation, not math, and it imports no estimator.

Dataset contract (the unified readout schema's digital + average + joint form):
  vars   : ``joint_population`` — probabilities over the target's joint states,
           dims ``(joint_state, <axis0>, <axis1>)`` in any order (transposed by
           name here)
  coords : ``joint_state`` — per-member level-digit labels, leftmost digit = the
           HIGH member (``"00"``/``"01"``/``"10"``/``"11"``; an f-resolved
           dataset may carry more labels — this estimator draws the four
           computational-basis panels and ignores the rest), plus the two sweep
           axes
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) — selects the transfer partner

plot_data contract (consumed by :func:`plot_pair_swap_map`):
  vars   : ``p00`` / ``p01`` / ``p10`` / ``p11`` — each ``(axis0, axis1)``, the
           joint basis populations in 0..1 (digit order ``high, low``)
  coords : ``<axis0>`` / ``<axis1>``
  attrs  : ``axis0`` / ``axis1`` (the coord names), ``drive_side``,
           ``prepared_state`` / ``transfer_state`` (the two single-excitation
           joint-state names the figure tags)
"""

from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["summarize_pair_swap", "pair_swap_plot_data", "plot_pair_swap_map"]

#: the four computational-basis joint states this analysis draws, in figure
#: (row-major) order. Digit order (high, low).
_BASIS_LABELS = ("00", "01", "10", "11")

#: joint_state label -> plot_data variable name.
_POP_VAR = {label: f"p{label}" for label in _BASIS_LABELS}

#: the four joint basis populations, in figure (row-major) order (digits: high, low).
_POPULATIONS = tuple(_POP_VAR[label] for label in _BASIS_LABELS)

#: human axis labels for the coordinate names the estimators use.
_AXIS_LABELS = {
    "flux_amp_v": "flux amplitude (V)",
    "swap_time_ns": "swap time (ns)",
    "swap_count": "swap count N",
    "qubit_flux_v": "qubit flux (V)",
    "coupler_flux_v": "coupler flux (V)",
    "stark_amp": "AC-Stark amplitude",
}

#: panel titles: (P-label, ket) per joint basis state, digit order (high, low).
_POP_TITLES = {
    "p00": (r"$P_{00}$", "|gg>"),
    "p01": (r"$P_{01}$", "|ge>"),
    "p10": (r"$P_{10}$", "|eg>"),
    "p11": (r"$P_{11}$", "|ee>"),
}


def _prepared_transfer_states(drive_side: str) -> tuple[str, str]:
    """The two single-excitation joint-state LABELS, as ``(prepared, transfer)``.

    Driving the LOW member prepares ``"01"`` (high=g, low=e) and the excitation
    transfers to ``"10"``; driving HIGH is the mirror."""
    return ("01", "10") if drive_side == "low" else ("10", "01")


def _axis_label(name: str) -> str:
    return _AXIS_LABELS.get(name, name)


def _basis_populations(dataset: xr.Dataset, axis0: str, axis1: str) -> Dict[str, np.ndarray]:
    """The four computational-basis joint populations as ``{label: (axis0, axis1)}``.

    Transposes by NAME (order-invariant) and selects the four binary labels off
    the ``joint_state`` coordinate — an f-resolved dataset's extra labels are
    simply not drawn."""
    jp = dataset["joint_population"].transpose("joint_state", axis0, axis1)
    labels = [str(v) for v in jp["joint_state"].values]
    missing = [label for label in _BASIS_LABELS if label not in labels]
    if missing:
        raise ValueError(
            f"joint_population lacks computational-basis labels {missing} "
            f"(joint_state={labels})"
        )
    return {
        label: np.asarray(jp.sel(joint_state=label).values, dtype=float)
        for label in _BASIS_LABELS
    }


def _member_marginal(basis: Dict[str, np.ndarray], side: str) -> np.ndarray:
    """P(member excited) by partial trace over the four basis populations.

    ``side`` names the member by role: ``"high"`` is the leftmost label digit.
    """
    digit = 0 if side == "high" else 1
    return sum(pop for label, pop in basis.items() if label[digit] == "1")


def summarize_pair_swap(
    dataset: xr.Dataset, axis0: str, axis1: str, drive_side: str,
    flux_side: str | None = None,
    high_name: str | None = None, low_name: str | None = None,
) -> Dict[str, Any]:
    """Minimal, JSON-scalar metadata for one pair's transfer map.

    Records WHERE the transfer peaks (the undriven member's marginal, traced out
    of ``joint_population``) and each marginal's range, which role is excited
    (``drive_side``) / carries the swept flux pulse (``flux_side``), and the
    actual member qubit names behind the high/low roles (``high_name`` /
    ``low_name``). This is the estimator's self-describing metadata only — the
    SUCCESS / ``min_transfer`` verdict is SCQO's and is not made here.
    ``success`` means merely that the map held a finite peak (a failed
    acquisition is all-NaN)."""
    basis = _basis_populations(dataset, axis0, axis1)
    partner_side = "high" if drive_side == "low" else "low"
    p_high = _member_marginal(basis, "high")
    p_low = _member_marginal(basis, "low")
    transfer = p_high if partner_side == "high" else p_low
    out: Dict[str, Any] = {
        "drive_side": str(drive_side),
        "flux_side": str(flux_side) if flux_side is not None else "",
        "high_name": str(high_name) if high_name is not None else "",
        "low_name": str(low_name) if low_name is not None else "",
        "partner": f"p_{partner_side}",
        "p_high_min": float(np.nanmin(p_high)),
        "p_high_max": float(np.nanmax(p_high)),
        "p_low_min": float(np.nanmin(p_low)),
        "p_low_max": float(np.nanmax(p_low)),
        "p_ee_max": float(np.nanmax(basis["11"])),
        f"n_{axis0}": int(dataset.sizes[axis0]),
        f"n_{axis1}": int(dataset.sizes[axis1]),
    }
    if np.isfinite(transfer).any():
        i, j = np.unravel_index(int(np.nanargmax(transfer)), transfer.shape)
        out["best_transfer"] = float(transfer[i, j])
        out[f"best_{axis0}"] = float(dataset[axis0].values[i])
        out[f"best_{axis1}"] = float(dataset[axis1].values[j])
        out["success"] = True
    else:
        # an all-NaN map is a failed acquisition, not a zero-transfer chip
        out["best_transfer"] = float("nan")
        out["success"] = False
    return out


def pair_swap_plot_data(
    dataset: xr.Dataset, axis0: str, axis1: str, drive_side: str,
    flux_side: str | None = None,
    high_name: str | None = None, low_name: str | None = None,
) -> xr.Dataset:
    """Project the four JOINT basis-population maps + coords + attrs to redraw from."""
    basis = _basis_populations(dataset, axis0, axis1)
    dims = (axis0, axis1)
    prepared, transfer = _prepared_transfer_states(drive_side)
    out = xr.Dataset(
        {_POP_VAR[label]: (dims, basis[label]) for label in _BASIS_LABELS},
        coords={
            axis0: np.asarray(dataset[axis0].values, dtype=float),
            axis1: np.asarray(dataset[axis1].values, dtype=float),
        },
    )
    out.attrs.update(
        {
            "axis0": axis0,
            "axis1": axis1,
            "drive_side": str(drive_side),
            "flux_side": str(flux_side) if flux_side is not None else "",
            "high_name": str(high_name) if high_name is not None else "",
            "low_name": str(low_name) if low_name is not None else "",
            "prepared_state": _POP_VAR[prepared],
            "transfer_state": _POP_VAR[transfer],
        }
    )
    return out


def plot_pair_swap_map(plot_data: xr.Dataset) -> plt.Figure:
    """2x2 pcolormesh of the four JOINT basis populations over the sweep grid.

    Draws ONLY from ``plot_data`` (the estimator-output-contract rule): the two
    axis names and the prepared/transfer joint states come from
    ``plot_data.attrs``. Every panel is fixed to 0..1 so the four populations are
    directly comparable, and the two single-excitation panels are tagged so the
    swap (prepared -> transfer) reads at a glance."""
    axis0 = plot_data.attrs["axis0"]
    axis1 = plot_data.attrs["axis1"]
    prepared = plot_data.attrs.get("prepared_state", "")
    transfer = plot_data.attrs.get("transfer_state", "")
    # Prefer actual qubit names (q0/q1); fall back to the high/low role words when
    # the roster mapping was not supplied (e.g. a standalone/unit-test call).
    high_name = plot_data.attrs.get("high_name", "") or "high"
    low_name = plot_data.attrs.get("low_name", "") or "low"
    role_name = {"high": high_name, "low": low_name}
    excited_q = role_name.get(plot_data.attrs.get("drive_side", ""), "?")
    flux_q = role_name.get(plot_data.attrs.get("flux_side", ""), "?")

    x = np.asarray(plot_data[axis0].values, dtype=float)
    y = np.asarray(plot_data[axis1].values, dtype=float)
    X, Y = np.meshgrid(x, y)  # -> (len(y), len(x))

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True
    )
    im = None
    for idx, (ax, name) in enumerate(zip(axes.ravel(), _POPULATIONS)):
        pop = np.asarray(plot_data[name].values, dtype=float)  # (axis0, axis1)
        im = ax.pcolormesh(
            X, Y, pop.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0
        )
        plabel, ket = _POP_TITLES[name]
        if name == transfer:
            tag = "  — transfer"
        elif name == prepared:
            tag = "  — prepared"
        else:
            tag = ""
        ax.set_title(f"{plabel}  {ket}{tag}")
        if idx // 2 == 1:  # bottom row
            ax.set_xlabel(_axis_label(axis0))
        if idx % 2 == 0:  # left column
            ax.set_ylabel(_axis_label(axis1))
    fig.colorbar(im, ax=axes.ravel().tolist(), label="population", shrink=0.9)
    # Spell out, by ACTUAL qubit name, which member is excited (the pi pulse) and
    # which carries the swept flux pulse, plus the ket digit order (high, low).
    fig.suptitle(
        "pair swap — joint 2-qubit populations\n"
        f"excited qubit (π pulse): {excited_q}     ·     "
        f"flux-pulse qubit: {flux_q}     ·     |eg> = |{high_name}, {low_name}>",
        fontsize=12,
    )
    return fig
