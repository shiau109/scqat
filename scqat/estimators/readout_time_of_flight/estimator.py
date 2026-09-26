"""
Readout Time-of-Flight Estimator
================================
Where the readout pulse arrives in the digitizer's own raw trace (SCQO's
``readout_time_of_flight``).

Reading
-------
The instrument emits a readout pulse and opens its acquisition window at a
DECLARED early origin — as early as the hardware allows — then digitizes the
raw ADC samples and averages them over shots. The trace reads noise while the
pulse is still in the cables, then a step, then a plateau. The step edge is the
round trip through the output chain, the fridge and the input chain.

The reduction is :func:`scqat.tools.pulse_arrival.find_pulse_arrival`: a
Savitzky-Golay smooth, a threshold midway between a robust baseline and a
robust plateau, and an interpolated first crossing. What this estimator adds is
the frame: the crossing is relative to the window origin, so the ABSOLUTE delay
the vendor field wants is ``window_start_ns + arrival_ns``, rounded onto the
instrument's timing grid.

Why the origin is declared and not the current setting
------------------------------------------------------
If the window opened at the value already configured and that value were
correct, the pulse would be present at sample 0: no pre-arrival baseline, no
threshold, no measurement. Opening early is what makes the edge visible, and it
also makes the answer independent of how wrong the current setting is.

Averaging and phase
-------------------
The trace is averaged over shots, so the caller must reset the digital
oscillator's phase each shot or the cosine averages toward zero and the step
vanishes. That is the probe's job; this estimator sees only the result, and a
washed-out step shows up as ``arrival_unresolved``.

Grid
----
``grid_ns`` is the instrument's timing granularity (4 ns on both shipped
backends). The absolute delay is rounded to it because that is what the vendor
field accepts; ``arrival_ns`` keeps the unrounded measurement beside it.

Expected ``xarray.Dataset`` contract
------------------------------------
The ``target``/``qubit`` dimension is already removed (``repetition_data``).

Coordinates:
    - readout_time_ns : 1-D float — sample time from the window origin, ns.
Data variables:
    - IQdata (complex) or I and Q : (readout_time_ns,) — the AVERAGED raw
      digitizer trace, in volts at the ADC.
Attributes or kwargs:
    - window_start_ns : the declared origin the probe realized (REQUIRED; a
      relative arrival is meaningless without it).
    - grid_ns         : timing granularity for the rounded answer (default 4).
    - full_scale_v    : digitizer full scale, enabling the saturation flag.
    - old_delay_ns    : the value the channel carried at acquisition time,
      reported back so the writeback hint can say "was X, write Y".
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.core.figures import render_figures
from scqat.estimators.readout_time_of_flight.visualization import plot_trace
from scqat.tools.pulse_arrival import find_pulse_arrival

#: the coordinate this estimator reads.
TIME_COORD = "readout_time_ns"

#: default instrument timing granularity for the rounded answer (ns).
DEFAULT_GRID_NS = 4.0


class ReadoutTimeOfFlightEstimator(BaseEstimator):
    """Locate the readout pulse's arrival edge in the averaged raw ADC trace."""

    estimator_name = "readout_time_of_flight"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if TIME_COORD not in dataset.coords and TIME_COORD not in dataset.dims:
            raise ValueError(
                f"readout_time_of_flight needs the '{TIME_COORD}' coordinate "
                f"(sample time from the window origin, ns); got "
                f"{list(dataset.coords)}")
        if "IQdata" not in dataset.data_vars and not (
                "I" in dataset.data_vars and "Q" in dataset.data_vars):
            raise ValueError(
                "readout_time_of_flight needs the averaged raw trace as "
                "'IQdata' or as 'I' and 'Q'; got "
                f"{list(dataset.data_vars)}")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        nan = float("nan")
        ds = with_iqdata(dataset)
        times = np.asarray(ds[TIME_COORD].values, dtype=float)
        trace = np.asarray(ds["IQdata"].transpose(TIME_COORD).values)
        magnitude = np.abs(trace)

        window_start = _scalar(kwargs.get("window_start_ns",
                                          ds.attrs.get("window_start_ns")))
        grid = _scalar(kwargs.get("grid_ns", ds.attrs.get("grid_ns")))
        grid = DEFAULT_GRID_NS if not np.isfinite(grid) or grid <= 0 else grid
        full_scale = _scalar(kwargs.get("full_scale_v",
                                        ds.attrs.get("full_scale_v")))
        old_delay = _scalar(kwargs.get("old_delay_ns",
                                       ds.attrs.get("old_delay_ns")))

        found = find_pulse_arrival(
            times, magnitude,
            full_scale=None if not np.isfinite(full_scale) else full_scale)

        # The window origin is not optional: without it the arrival is a number
        # with no frame. Report the arrival, refuse the absolute.
        delay_ns = nan
        if np.isfinite(found["arrival_ns"]) and np.isfinite(window_start):
            delay_ns = float(np.round(
                (window_start + found["arrival_ns"]) / grid) * grid)

        results: Dict[str, Any] = {
            **found,
            "time_of_flight_ns": delay_ns,
            "window_start_ns": window_start,
            "old_delay_ns": old_delay,
            "grid_ns": float(grid),
            "delta_ns": (delay_ns - old_delay
                         if np.isfinite(delay_ns) and np.isfinite(old_delay)
                         else nan),
            "origin_missing": float(not np.isfinite(window_start)),
        }
        results["success"] = bool(
            np.isfinite(delay_ns)
            and results["arrival_unresolved"] == 0.0
            and results["arrival_at_edge"] == 0.0)
        return results

    def build_plot_data(self, dataset: xr.Dataset, results: Dict[str, Any],
                        **kwargs) -> xr.Dataset:
        ds = with_iqdata(dataset)
        times = np.asarray(ds[TIME_COORD].values, dtype=float)
        trace = np.asarray(ds["IQdata"].transpose(TIME_COORD).values)
        plot = xr.Dataset(
            {
                "magnitude": ((TIME_COORD,), np.abs(trace)),
                "adc_i": ((TIME_COORD,), np.real(trace)),
                "adc_q": ((TIME_COORD,), np.imag(trace)),
            },
            coords={TIME_COORD: times},
        )
        plot[TIME_COORD].attrs = {"long_name": "time from window origin",
                                  "units": "ns"}
        plot["magnitude"].attrs = {"long_name": "|IQ|", "units": "V"}
        # fit-derived fields degrade to NaN, never absent: the raw figure must
        # render on a failed fit.
        for key in ("arrival_ns", "threshold", "baseline", "plateau",
                    "rise_time_ns", "plateau_snr", "time_of_flight_ns",
                    "window_start_ns", "old_delay_ns", "delta_ns",
                    "arrival_unresolved", "arrival_at_edge", "adc_saturated"):
            plot.attrs[key] = float(results.get(key, float("nan")))
        return plot

    def generate_figures(self, dataset: xr.Dataset, results: Dict[str, Any],
                         plot_data: Optional[xr.Dataset] = None,
                         **kwargs) -> Dict[str, plt.Figure]:
        if plot_data is None:
            return {}
        return render_figures(
            {"readout_time_of_flight": lambda: plot_trace(plot_data)},
            label=self.estimator_name)


def _scalar(value: Any) -> float:
    """A float, or NaN — an absent or unparseable input is not a zero."""
    try:
        out = float(np.asarray(value).reshape(()).item())
    except Exception:
        return float("nan")
    return out
