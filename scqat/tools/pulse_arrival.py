"""When a pulse arrives in a raw digitizer trace — the time-of-flight reduction.

The instrument opens its acquisition window at a DECLARED early origin and
digitizes the readout pulse it just emitted. The trace therefore reads: noise
while the pulse is still in the cables, then a step, then a plateau for the rest
of the pulse. The arrival edge is the round trip through the output chain, the
fridge and the input chain — what QM calls ``resonator.time_of_flight`` and
Qblox calls ``measure.acq_delay``.

Why the window must open EARLY, not at the current setting: if the current value
is already right the pulse is present at sample 0, there is no pre-arrival
baseline, and the threshold has nothing to sit between. The caller opens the
window as early as the instrument allows and this function reports the arrival
RELATIVE to that origin; adding the two gives the absolute delay.

The threshold is the midpoint between a robust baseline and a robust plateau,
both medians over a fraction of the trace, and the crossing is interpolated
between the two bracketing samples — the digitizer grid is 1 ns but the vendor
field lands on a 4 ns grid, so rounding is the caller's last step, not a
resolution limit here.

Pure and array-only: no xarray, no instrument, no estimator import.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

#: Fraction of the trace taken as the pre-arrival baseline and as the plateau.
#: Deliberately generous - a step anywhere in the middle 80 % is measurable, and
#: a step inside either end zone is what ``arrival_at_edge`` is for.
EDGE_FRACTION = 0.10

#: Savitzky-Golay smoothing of the magnitude before thresholding. The window is
#: in SAMPLES; at 1 GSa/s that is nanoseconds.
SMOOTH_WINDOW = 11
SMOOTH_ORDER = 3

#: |trace| at or above this fraction of full scale is clipping. An ADC that
#: saturates flattens the very edge this function measures, so the arrival reads
#: EARLY and nothing else notices.
SATURATION_FRACTION = 0.98


def _smooth(y: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky-Golay, degrading to the input when the trace is too short."""
    from scipy.signal import savgol_filter

    window = int(window)
    if window % 2 == 0:
        window += 1
    if y.size < window or window <= order:
        return np.asarray(y, dtype=float)
    return savgol_filter(np.asarray(y, dtype=float), window, order)


def _first_crossing(times: np.ndarray, y: np.ndarray, level: float) -> float:
    """Interpolated time of the first upward crossing of ``level``, or NaN."""
    above = y >= level
    if not above.any() or above[0]:
        # never crosses, or is already above at the first sample - the caller
        # distinguishes these through arrival_at_edge / arrival_unresolved
        return float(times[0]) if above.any() and above[0] else float("nan")
    idx = int(np.argmax(above))
    y0, y1 = float(y[idx - 1]), float(y[idx])
    if y1 == y0:
        return float(times[idx])
    frac = (level - y0) / (y1 - y0)
    return float(times[idx - 1] + frac * (times[idx] - times[idx - 1]))


def find_pulse_arrival(
    times_ns: np.ndarray,
    magnitude: np.ndarray,
    *,
    full_scale: float | None = None,
    edge_fraction: float = EDGE_FRACTION,
    smooth_window: int = SMOOTH_WINDOW,
    smooth_order: int = SMOOTH_ORDER,
) -> Dict[str, Any]:
    """Locate the pulse edge in one averaged magnitude trace.

    Parameters
    ----------
    times_ns
        Sample times from the window origin, ascending, in ns.
    magnitude
        ``|I + jQ|`` of the averaged raw trace, same length.
    full_scale
        Digitizer full scale in the same units as ``magnitude``; enables the
        saturation flag. ``None`` leaves ``adc_saturated`` NaN rather than 0 -
        an unchecked flag must not read as a passed check.

    Returns a dict of plain floats: ``arrival_ns`` (relative to the origin),
    ``rise_time_ns`` (10-90 %), ``baseline`` / ``plateau`` / ``threshold``,
    ``plateau_snr`` (step over baseline noise), and the three flags
    ``arrival_unresolved`` / ``arrival_at_edge`` / ``adc_saturated`` as 1.0,
    0.0 or NaN. Every key is always present; a failure is NaN, never absent.
    """
    nan = float("nan")
    times_ns = np.asarray(times_ns, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)

    out: Dict[str, Any] = {
        "arrival_ns": nan, "rise_time_ns": nan, "baseline": nan, "plateau": nan,
        "threshold": nan, "plateau_snr": nan,
        "arrival_unresolved": 1.0, "arrival_at_edge": nan, "adc_saturated": nan,
    }

    finite = np.isfinite(magnitude)
    if times_ns.size < 4 or times_ns.size != magnitude.size or finite.sum() < 4:
        return out

    if full_scale is not None and np.isfinite(full_scale) and full_scale > 0:
        out["adc_saturated"] = float(
            np.nanmax(np.abs(magnitude)) >= SATURATION_FRACTION * float(full_scale))

    smooth = _smooth(np.where(finite, magnitude, np.nan), smooth_window, smooth_order)
    n_edge = max(2, int(round(edge_fraction * smooth.size)))
    baseline = float(np.nanmedian(smooth[:n_edge]))
    plateau = float(np.nanmedian(smooth[-n_edge:]))
    noise = float(np.nanstd(smooth[:n_edge]))
    out["baseline"], out["plateau"] = baseline, plateau

    step = plateau - baseline
    out["plateau_snr"] = float(step / noise) if noise > 0 else nan
    if not np.isfinite(step) or step <= 0:
        # no step: the window missed the pulse entirely, or the pulse was
        # already on at sample 0 and the "baseline" is plateau too. Either way
        # there is no edge to locate and the caller must not round a number.
        return out

    threshold = baseline + 0.5 * step
    out["threshold"] = threshold
    arrival = _first_crossing(times_ns, smooth, threshold)
    if not np.isfinite(arrival):
        return out

    out["arrival_ns"] = arrival
    out["arrival_unresolved"] = 0.0
    # An edge inside either end zone means the window is mis-placed by more than
    # this trace can see: the number would be a bound, not a measurement.
    out["arrival_at_edge"] = float(
        arrival <= times_ns[n_edge - 1] or arrival >= times_ns[-n_edge])

    low = _first_crossing(times_ns, smooth, baseline + 0.1 * step)
    high = _first_crossing(times_ns, smooth, baseline + 0.9 * step)
    if np.isfinite(low) and np.isfinite(high) and high >= low:
        out["rise_time_ns"] = float(high - low)
    return out
