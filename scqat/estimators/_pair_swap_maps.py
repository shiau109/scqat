"""Shared helpers for the pair-swap raw-population map estimators.

``pair_swap_chevron`` and ``pair_swap_flux_map`` are the same analysis over two
different 2-D sweep grids: excite one member of a pair, read BOTH members out
jointly, and draw the four joint state populations (``p_high`` / ``p_low`` /
``p_ee`` / ``p_gg``) as maps. Only the two axis NAMES differ, so the summary, the
plot-data projection and the figure live here once and each estimator supplies
its own coordinate names.

Like ``_iq_plane.py`` this is a plain shared FUNCTION module (function-level
sharing is what the estimator-layering rule permits); it lives outside ``tools/``
because it is presentation, not math, and it imports no estimator.

The four panels are the JOINT two-qubit computational-basis populations — what a
joint state discrimination on a multiplexed readout measures directly: P00 / P01
/ P10 / P11 (bit order ``high, low``). The probe stores them reduced to role
marginals (``p_high`` = P(high=e) = P10+P11, ``p_low`` = P(low=e) = P01+P11) plus
``p_ee`` (P11) and ``p_gg`` (P00); that is a LOSSLESS encoding, so the
single-excitation joints come back by removing the double-excitation overlap
(``P10 = p_high - p_ee``, ``P01 = p_low - p_ee``).

plot_data contract (consumed by :func:`plot_pair_swap_map`):
  vars   : ``p00`` / ``p01`` / ``p10`` / ``p11`` — each ``(axis0, axis1)``, the
           joint basis populations in 0..1 (bit order ``high, low``)
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

#: the four joint basis populations, in figure (row-major) order (bits: high, low).
_POPULATIONS = ("p00", "p01", "p10", "p11")

#: human axis labels for the coordinate names the two estimators use.
_AXIS_LABELS = {
    "flux_amp_v": "flux amplitude (V)",
    "swap_time_ns": "swap time (ns)",
    "qubit_flux_v": "qubit flux (V)",
    "coupler_flux_v": "coupler flux (V)",
}

#: panel titles: (P-label, ket) per joint basis state, bit order (high, low).
_POP_TITLES = {
    "p00": (r"$P_{00}$", "|gg>"),
    "p01": (r"$P_{01}$", "|ge>"),
    "p10": (r"$P_{10}$", "|eg>"),
    "p11": (r"$P_{11}$", "|ee>"),
}


def _partner(drive_side: str) -> str:
    """The role marginal the excitation transfers ONTO (the undriven member)."""
    return "p_high" if drive_side == "low" else "p_low"


def _prepared_transfer_states(drive_side: str) -> tuple[str, str]:
    """The two single-excitation joint states, as ``(prepared, transfer)``.

    Driving the LOW member prepares ``|ge>`` (high=g, low=e = ``p01``) and the
    excitation transfers to ``|eg>`` (``p10``); driving HIGH is the mirror."""
    return ("p01", "p10") if drive_side == "low" else ("p10", "p01")


def _axis_label(name: str) -> str:
    return _AXIS_LABELS.get(name, name)


def summarize_pair_swap(
    dataset: xr.Dataset, axis0: str, axis1: str, drive_side: str,
    flux_side: str | None = None,
    high_name: str | None = None, low_name: str | None = None,
) -> Dict[str, Any]:
    """Minimal, JSON-scalar metadata for one pair's transfer map.

    Records WHERE the transfer peaks (the undriven member's population) and each
    population's range, which role is excited (``drive_side``) / carries the swept
    flux pulse (``flux_side``), and the actual member qubit names behind the
    high/low roles (``high_name`` / ``low_name``). This is the estimator's
    self-describing metadata only — the SUCCESS / ``min_transfer`` verdict is
    SCQO's and is not made here. ``success`` means merely that the map held a
    finite peak (a failed acquisition is all-NaN)."""
    ds = dataset.transpose(axis0, axis1)
    partner = _partner(drive_side)
    transfer = np.asarray(ds[partner].values, dtype=float)
    out: Dict[str, Any] = {
        "drive_side": str(drive_side),
        "flux_side": str(flux_side) if flux_side is not None else "",
        "high_name": str(high_name) if high_name is not None else "",
        "low_name": str(low_name) if low_name is not None else "",
        "partner": partner,
        "p_high_min": float(np.nanmin(ds["p_high"].values)),
        "p_high_max": float(np.nanmax(ds["p_high"].values)),
        "p_low_min": float(np.nanmin(ds["p_low"].values)),
        "p_low_max": float(np.nanmax(ds["p_low"].values)),
        "p_ee_max": float(np.nanmax(ds["p_ee"].values)),
        f"n_{axis0}": int(ds.sizes[axis0]),
        f"n_{axis1}": int(ds.sizes[axis1]),
    }
    if np.isfinite(transfer).any():
        i, j = np.unravel_index(int(np.nanargmax(transfer)), transfer.shape)
        out["best_transfer"] = float(transfer[i, j])
        out[f"best_{axis0}"] = float(ds[axis0].values[i])
        out[f"best_{axis1}"] = float(ds[axis1].values[j])
        out["success"] = True
    else:
        # an all-NaN map is a failed acquisition, not a zero-transfer chip
        out["best_transfer"] = float("nan")
        out["success"] = False
    return out


def _p_gg(ds: xr.Dataset) -> np.ndarray:
    """``p_gg`` (P00) verbatim when present, else the four-population closure."""
    if "p_gg" in ds.data_vars:
        return np.asarray(ds["p_gg"].values, dtype=float)
    p_high = np.asarray(ds["p_high"].values, dtype=float)
    p_low = np.asarray(ds["p_low"].values, dtype=float)
    p_ee = np.asarray(ds["p_ee"].values, dtype=float)
    # same formula as SCQO's simulate()/reduce_raw (_role_populations)
    return np.clip(1.0 - (p_high + p_low) + p_ee, 0.0, 1.0)


def joint_populations(ds: xr.Dataset) -> Dict[str, np.ndarray]:
    """The four joint 2-qubit basis populations (bit order high, low).

    Recovered from the role marginals the probe stores: ``P10 = p_high - p_ee``
    (only the high member excited), ``P01 = p_low - p_ee`` (only the low member),
    with ``P00 = p_gg`` and ``P11 = p_ee``. Small negatives from readout noise are
    clipped to 0."""
    p_high = np.asarray(ds["p_high"].values, dtype=float)
    p_low = np.asarray(ds["p_low"].values, dtype=float)
    p_ee = np.asarray(ds["p_ee"].values, dtype=float)
    return {
        "p00": _p_gg(ds),
        "p01": np.clip(p_low - p_ee, 0.0, 1.0),
        "p10": np.clip(p_high - p_ee, 0.0, 1.0),
        "p11": np.clip(p_ee, 0.0, 1.0),
    }


def pair_swap_plot_data(
    dataset: xr.Dataset, axis0: str, axis1: str, drive_side: str,
    flux_side: str | None = None,
    high_name: str | None = None, low_name: str | None = None,
) -> xr.Dataset:
    """Project the four JOINT basis-population maps + coords + attrs to redraw from."""
    ds = dataset.transpose(axis0, axis1)
    dims = (axis0, axis1)
    joints = joint_populations(ds)
    prepared, transfer = _prepared_transfer_states(drive_side)
    out = xr.Dataset(
        {name: (dims, joints[name]) for name in _POPULATIONS},
        coords={
            axis0: np.asarray(ds[axis0].values, dtype=float),
            axis1: np.asarray(ds[axis1].values, dtype=float),
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
            "prepared_state": prepared,
            "transfer_state": transfer,
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
    # which carries the swept flux pulse, plus the ket bit order (high, low).
    fig.suptitle(
        "pair swap — joint 2-qubit populations\n"
        f"excited qubit (π pulse): {excited_q}     ·     "
        f"flux-pulse qubit: {flux_q}     ·     |eg> = |{high_name}, {low_name}>",
        fontsize=12,
    )
    return fig
