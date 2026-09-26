"""Figures for :mod:`scqat.estimators.readout_time_of_flight`.

Draws from ``plot_data`` ONLY (the estimator-output contract). The raw trace is
drawn unconditionally and every fit overlay is guarded, so a run whose edge was
never found still produces its trace figure — that is the whole point of the
figure on a failed fit: you look at it to see WHY.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

TIME_COORD = "readout_time_ns"


def _finite(plot_data: xr.Dataset, key: str) -> float | None:
    value = float(plot_data.attrs.get(key, float("nan")))
    return value if np.isfinite(value) else None


def plot_trace(plot_data: xr.Dataset) -> plt.Figure:
    """The averaged raw ADC trace with the arrival edge marked.

    Two panels: |IQ| with the threshold and the located edge, and the two ADC
    quadratures underneath — the quadratures are what show a phase that was not
    reset per shot, which averages the step away and is the most likely reason
    an edge goes missing.
    """
    times = np.asarray(plot_data[TIME_COORD].values, dtype=float)
    magnitude = np.asarray(plot_data["magnitude"].values, dtype=float)

    fig, (ax, ax_iq) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]})

    ax.plot(times, magnitude, lw=1.0, color="0.25", label="|IQ| (averaged)")

    baseline = _finite(plot_data, "baseline")
    plateau = _finite(plot_data, "plateau")
    threshold = _finite(plot_data, "threshold")
    arrival = _finite(plot_data, "arrival_ns")
    if threshold is not None:
        ax.axhline(threshold, color="tab:orange", ls=":", lw=1.0,
                   label="threshold")
    for level, name in ((baseline, "baseline"), (plateau, "plateau")):
        if level is not None:
            ax.axhline(level, color="0.7", ls="--", lw=0.8)
            ax.annotate(name, (times[0], level), textcoords="offset points",
                        xytext=(2, 2), fontsize=7, color="0.45")
    if arrival is not None:
        ax.axvline(arrival, color="tab:red", lw=1.4, label="arrival")

    ax.set_ylabel("|IQ| (V)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(_title(plot_data))

    ax_iq.plot(times, np.asarray(plot_data["adc_i"].values, dtype=float),
               lw=0.8, label="ADC I")
    ax_iq.plot(times, np.asarray(plot_data["adc_q"].values, dtype=float),
               lw=0.8, label="ADC Q")
    if arrival is not None:
        ax_iq.axvline(arrival, color="tab:red", lw=1.0)
    ax_iq.set_xlabel("time from window origin (ns)")
    ax_iq.set_ylabel("ADC (V)")
    ax_iq.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    return fig


def _title(plot_data: xr.Dataset) -> str:
    """One line that says the answer, or names what stopped it."""
    delay = _finite(plot_data, "time_of_flight_ns")
    old = _finite(plot_data, "old_delay_ns")
    flags = [name for name, key in (
        ("edge not found", "arrival_unresolved"),
        ("edge at the window rim", "arrival_at_edge"),
        ("ADC saturated", "adc_saturated"),
    ) if float(plot_data.attrs.get(key, float("nan"))) == 1.0]

    if delay is None:
        head = "time of flight: NOT MEASURED"
    else:
        head = f"time of flight: {delay:.0f} ns"
        if old is not None:
            head += f"  (was {old:.0f} ns)"
    return head + (f"  [{', '.join(flags)}]" if flags else "")
