"""Per-trace qubit peak fit — the reduction shared by an experiment FAMILY.

One spectrum trace (frequency axis + signal) in, a list of fitted peaks out.
This is the pure-math reduction behind every qubit-line experiment: 1-D qubit
spectroscopy uses it once, the vs-flux / parametric-drive maps call it once per
slice, and the AC-Stark / readout-photon traces call it once per slice with
``max_peaks=1``. Per the repo rule ("anything used by more than one estimator
lives in tools/"), it lives here — estimators compose it, they never call each
other.

Pipeline (single method): distance signal -> polynomial baseline through the
quietest quantile -> ``scipy.signal.find_peaks`` on both polarities (keep the
stronger; a MAD-based ``min_snr`` gate rejects noise-only traces) -> windowed
Lorentzian fit per peak (:class:`scqat.tools.fit_lorentzian.FitLorentzian`) ->
merge duplicate fits of one line -> ``max_peaks`` area cap.

Signal convention
-----------------
``signal`` may be **complex** (demodulated I + iQ; the fitted quantity is
``|signal - ref|`` with ``ref`` auto-estimated as the complex median when not
given) or **real** (e.g. a state population; used as-is, ``ref`` does not
apply and ``ref_iq`` is reported ``None``).

Result contract
---------------
``{signal, baseline, signal_corrected, ref_iq, inverted, peaks}`` where
``peaks`` is a list (ascending ``detuning``) of::

    {detuning, amplitude, fwhm, offset,
     detuning_err, amplitude_err, fwhm_err,
     fit_x, fit_y, full_freq?}

``full_freq`` is present iff the ``full_freq`` axis was supplied. Callers may
rely on ``detuning``/``amplitude``/``fwhm`` of each peak; the ``fit_x``/``fit_y``
arrays are figure fodder.

Callers that loop over slices and collect tunables in a dict should call
:func:`validate_peak_kwargs` ONCE before the loop, so a typo'd knob dies loudly
instead of being swallowed by a per-slice ``try/except``.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks
import xarray as xr

from .fit_lorentzian import FitLorentzian, lorentzian
from .iq_reduce import ground_ref, radial

#: caller-selectable knobs of :func:`fit_peaks` (everything after ``full_freq``)
#: — the single source of truth dict-collecting callers validate against.
PEAK_KNOBS = frozenset({
    "ref", "prominence", "min_snr", "max_peaks",
    "merge_factor", "min_fwhm_factor", "fit_window_factor",
})


def validate_peak_kwargs(knobs: Dict) -> None:
    """Raise ValueError for an unknown knob — call BEFORE slice loops."""
    unknown = set(knobs) - PEAK_KNOBS
    if unknown:
        raise ValueError(
            f"Unknown knob(s) {sorted(unknown)} for the peak fit; "
            f"valid: {sorted(PEAK_KNOBS)}"
        )


def _estimate_baseline(x: np.ndarray, y: np.ndarray, order: int = 1,
                       quantile: float = 0.25):
    """
    Fit a polynomial baseline through the *quietest* portion of the
    spectrum (bottom ``quantile`` fraction sorted by absolute deviation
    from the median).
    """
    med = np.median(y)
    dev = np.abs(y - med)
    threshold = np.quantile(dev, quantile)
    mask = dev <= threshold
    coeffs = np.polyfit(x[mask], y[mask], deg=order)
    return np.polyval(coeffs, x)


def _merge_overlapping_peaks(peaks: List[Dict[str, Any]],
                             merge_factor: float) -> List[Dict[str, Any]]:
    """Collapse peaks whose Lorentzians overlap within their summed half-widths.

    Two peaks ``i, j`` are treated as the same physical line when their centres
    are closer than a fraction of their combined linewidth::

        |x0_i - x0_j| < merge_factor * (fwhm_i + fwhm_j) / 2

    Within each overlapping group only the peak of largest Lorentzian **area**
    (``|amplitude| * fwhm``) is kept; the rest are discarded.  This removes the
    duplicate fits that arise when one transition is detected as two adjacent
    ``find_peaks`` maxima whose overlapping fit windows both converge onto it,
    while leaving genuinely separated transitions untouched.

    A falsy ``merge_factor`` (``0`` / ``None``) disables merging and returns the
    input unchanged.  Operates on the peak dicts, which already carry
    ``detuning``, ``amplitude`` and ``fwhm`` — no extra fields required.
    """
    if not merge_factor or len(peaks) < 2:
        return peaks

    def _area(p: Dict[str, Any]) -> float:
        return abs(p["amplitude"]) * p["fwhm"]

    kept = list(peaks)
    while True:
        # Find the closest still-overlapping pair (smallest centre gap relative
        # to its overlap threshold), drop its smaller-area member, and repeat.
        drop_idx = None
        best_gap = np.inf
        for a in range(len(kept)):
            for b in range(a + 1, len(kept)):
                gap = abs(kept[a]["detuning"] - kept[b]["detuning"])
                threshold = merge_factor * (kept[a]["fwhm"] + kept[b]["fwhm"]) / 2.0
                if gap < threshold and gap < best_gap:
                    best_gap = gap
                    drop_idx = a if _area(kept[a]) < _area(kept[b]) else b
        if drop_idx is None:
            return kept
        kept.pop(drop_idx)


def fit_peaks(
    detuning: np.ndarray,
    signal: np.ndarray,
    full_freq: Optional[np.ndarray] = None,
    *,
    ref: Optional[complex] = None,
    prominence: float = 0.1,
    min_snr: float = 6.0,
    max_peaks: Optional[int] = None,
    merge_factor: float = 1.0,
    min_fwhm_factor: float = 0.5,
    fit_window_factor: float = 5.0,
) -> Dict[str, Any]:
    """Detect and fit every peak in one spectrum trace (see module docstring).

    Parameters
    ----------
    detuning : 1-D float array
        Drive-frequency detuning axis (Hz, relative to the LO).
    signal : 1-D array
        Complex demodulated I + iQ (fitted as ``|signal - ref|``) or an
        already-real signal (used as-is; ``ref`` does not apply).
    full_freq : 1-D float array, optional
        Absolute drive frequency (Hz) on the same axis; when present each peak
        also carries its absolute ``full_freq`` centre.
    ref : complex, optional
        Reference point in the IQ plane for complex input. When omitted, the
        reference is auto-estimated as the median of the complex IQ data
        (works well when most sweep points are off-resonance).
    prominence : float, optional
        Minimum prominence for ``find_peaks`` relative to the baseline-
        subtracted signal span.  Default 0.1 (10 %).
    min_snr : float, optional
        Significance gate: a peak's prominence must also exceed
        ``min_snr * robust_sigma`` (robust_sigma = 1.4826 * MAD of the
        baseline-corrected signal). Rejects noise-only sweeps (returns no
        peaks) and keeps all genuine lines regardless of count. Default 6.0.
    max_peaks : int or None, optional
        Maximum number of peaks to return.  Applied *after* merging, so a
        duplicate fit can never consume a slot ahead of a genuine line.
        When set, the *max_peaks* largest-area peaks are kept (sorted by
        Lorentzian area ``|amplitude|*fwhm``) and the rest discarded.  The
        returned ``peaks`` list is then sorted by ascending ``detuning``.
        Default ``None`` (keep all).
    merge_factor : float, optional
        De-duplication strength.  Two fitted peaks are merged into one
        (keeping the larger-area fit) when their centres are closer than
        ``merge_factor * (fwhm_i + fwhm_j) / 2`` — i.e. they overlap within
        their summed half-widths.  Default ``1.0``; set ``0`` to disable.
    min_fwhm_factor : float, optional
        Sub-resolution spike guard.  A fitted peak is dropped when its
        ``fwhm`` is below ``min_fwhm_factor * median(diff(detuning))``,
        removing single-sample noise spikes fit as delta-like Lorentzians.
        Default ``0.5``; set ``0`` to disable.
    fit_window_factor : float, optional
        Each peak is fitted inside a window of
        ``fit_window_factor * estimated_width`` around the peak centre.
        Default 5.
    """
    detuning = np.asarray(detuning, dtype=float)

    # --- Resolve the 1-D fitted signal ---
    signal = np.asarray(signal).ravel()
    if np.iscomplexobj(signal):
        # Radial reduction: distance from the off-resonance/ground cluster (the
        # complex median when no ref given). See scqat.tools.iq_reduce.
        ref_iq = complex(ref) if ref is not None else ground_ref(signal.real, signal.imag)
        signal = radial(signal.real, signal.imag, ref=ref_iq)
    else:
        signal = signal.astype(float)
        ref_iq = None

    # --- Baseline subtraction ---
    baseline = _estimate_baseline(detuning, signal)
    signal_corrected = signal - baseline

    # --- Peak detection ---
    # Try both polarities; keep the one whose most prominent
    # peak is larger (handles both absorption dips and emission peaks).
    span = signal_corrected.max() - signal_corrected.min()
    # Significance gate: a peak must rise above the noise, not merely be the most
    # prominent bump within the span. robust sigma = 1.4826 * MAD (median-based, so a
    # few strong peaks don't inflate it). This rejects noise-only sweeps (no peak found)
    # while keeping every genuine line, e.g. both peaks of a two-transition sweep.
    robust_sigma = 1.4826 * np.median(np.abs(signal_corrected - np.median(signal_corrected)))
    abs_prom = max(prominence * span, min_snr * robust_sigma)

    idx_pos, props_pos = find_peaks(
        signal_corrected, prominence=abs_prom, width=1,
    )
    idx_neg, props_neg = find_peaks(
        -signal_corrected, prominence=abs_prom, width=1,
    )

    best_pos = props_pos["prominences"].max() if len(idx_pos) else 0
    best_neg = props_neg["prominences"].max() if len(idx_neg) else 0

    if best_neg > best_pos:
        peak_indices, properties = idx_neg, props_neg
        signal_corrected = -signal_corrected
        inverted = True
    else:
        peak_indices, properties = idx_pos, props_pos
        inverted = False

    # --- Fit Lorentzians to each peak ---
    # All detected peaks are fitted; de-duplication (merge) and the
    # ``max_peaks`` cap are applied afterwards on the fitted results, so a
    # duplicate of one line can't crowd out a genuine separate transition.
    min_fwhm = min_fwhm_factor * abs(float(np.median(np.diff(detuning)))) if len(detuning) > 1 else 0.0
    peaks_info: List[Dict[str, Any]] = []
    for i, idx in enumerate(peak_indices):
        est_width_pts = properties["widths"][i] if "widths" in properties else 10
        hw = int(fit_window_factor * max(est_width_pts, 3))
        lo = max(idx - hw, 0)
        hi = min(idx + hw + 1, len(detuning))

        x_win = detuning[lo:hi]
        y_win = signal_corrected[lo:hi]

        da_win = xr.DataArray(y_win, coords={'x': x_win}, dims='x')
        x_lo_b = float(detuning[lo])
        x_hi_b = float(detuning[hi - 1])
        gamma_max = float(detuning[-1] - detuning[0])
        fitter = FitLorentzian(
            da_win,
            inverted=inverted,
            bounds={'x0': (x_lo_b, x_hi_b), 'gamma': (0.0, gamma_max)},
        )
        try:
            result = fitter.fit()
            p = result.params
            popt = np.array([p['x0'].value, p['amplitude'].value,
                             p['gamma'].value, p['offset'].value])
            perr = np.array([
                p['x0'].stderr if p['x0'].stderr is not None else np.nan,
                p['amplitude'].stderr if p['amplitude'].stderr is not None else np.nan,
                p['gamma'].stderr if p['gamma'].stderr is not None else np.nan,
                p['offset'].stderr if p['offset'].stderr is not None else np.nan,
            ])
        except Exception:
            # Fall back to initial guess
            center_guess = detuning[idx]
            amp_guess = signal_corrected[idx] if not inverted else -signal_corrected[idx]
            gamma_guess = abs(detuning[min(idx + max(int(est_width_pts // 2), 1), len(detuning) - 1)]
                              - detuning[idx])
            if gamma_guess == 0:
                gamma_guess = abs(detuning[1] - detuning[0]) * 5
            popt = np.array([center_guess, amp_guess, gamma_guess, 0.0])
            perr = np.full(4, np.nan)

        det_fit = popt[0]
        fwhm = 2 * abs(popt[2])

        # Drop sub-resolution spikes: a fit whose linewidth collapsed below
        # the sampling step is a single noise sample, not a real line.
        if min_fwhm > 0 and fwhm < min_fwhm:
            continue

        peak_entry: Dict[str, Any] = {
            "detuning": float(det_fit),
            "amplitude": float(popt[1]),
            "fwhm": float(fwhm),
            "offset": float(popt[3]),
            "detuning_err": float(perr[0]),
            "amplitude_err": float(perr[1]),
            "fwhm_err": float(2 * perr[2]),
            "fit_x": x_win,
            "fit_y": lorentzian(x_win, *popt),
        }

        # Report absolute frequency if available
        if full_freq is not None:
            freq_vals = np.asarray(full_freq, dtype=float).ravel()
            order = np.argsort(detuning)
            peak_entry["full_freq"] = float(
                np.interp(det_fit, detuning[order], freq_vals[order])
            )

        peaks_info.append(peak_entry)

    # Merge duplicate fits of the same line (keep the larger-area one), then
    # cap to the strongest ``max_peaks`` by area so a duplicate can't crowd
    # out a genuine transition.
    peaks_info = _merge_overlapping_peaks(peaks_info, merge_factor)
    if max_peaks is not None and len(peaks_info) > max_peaks:
        peaks_info = sorted(
            peaks_info, key=lambda p: abs(p["amplitude"]) * p["fwhm"], reverse=True
        )[:max_peaks]

    # Sort peaks by detuning
    peaks_info.sort(key=lambda p: p["detuning"])

    return {
        "signal": signal,
        "baseline": baseline,
        "signal_corrected": signal_corrected,
        "ref_iq": ref_iq,
        "inverted": inverted,
        "peaks": peaks_info,
    }
