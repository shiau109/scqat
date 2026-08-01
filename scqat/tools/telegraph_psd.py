"""Telegraph-signal PSD fit — the parity-switching-rate reduction.

One discriminated 0/1 shot trace (uniform cadence ``dt_s``) in, the switching
rate out. This is the pure-math reduction behind the parity-switch experiment;
per the repo rule ("anything used by more than one estimator lives in tools/",
and simulation callers reuse tools without estimator wrapping) it lives here.

Rate convention (pinned)
------------------------
A symmetric random telegraph signal with per-direction switching rate Gamma
(up == down) has autocovariance ``R(tau) = Delta^2 * exp(-2*Gamma*|tau|)``,
hence a Lorentzian one-sided PSD

    S(f) = A / (1 + (f / f_c)^2) + B,        2*pi*f_c = 2*Gamma

so the reported rate is

    parity_rate_hz = Gamma = pi * f_c        (per direction)

For an asymmetric telegraph (Gamma_up != Gamma_down) the Lorentzian corner is
``Gamma_up + Gamma_down``, so ``parity_rate_hz`` is the MEAN per-direction
rate ``(Gamma_up + Gamma_down) / 2``.

Why the PSD knee and not transition counting: a readout error flips ONE shot
and fakes TWO transitions, so a counted rate is inflated by ``~2*p_err/dt``.
Uncorrelated readout errors are spectrally white — they raise the fitted floor
``B`` and leave the corner ``f_c`` unbiased. ``n_transitions`` is therefore
reported as a diagnostic only, never folded into the rate.

The rate scales linearly with ``dt_s``: a shot-period bookkeeping error in the
caller shifts the rate proportionally (which is also how to detect one — vary
the between-shot wait on hardware and check the rate is invariant).

Result contract (two tiers)
---------------------------
REQUIRED keys — identical meaning/unit always; the only keys a caller may rely
on:

    parity_rate_hz  : per-direction switching rate Gamma = pi * f_c, Hz
    psd_corner_hz   : fitted Lorentzian corner f_c, Hz
    psd_amplitude   : fitted low-frequency plateau A, 1/Hz
    psd_white_floor : fitted white floor B, 1/Hz
    n_transitions   : raw nearest-neighbour flip count (diagnostic;
                      inflated by readout errors)
    p_excited       : mean occupancy of the trace
    success         : bool — fit trustworthy (corner resolved inside the
                      spectral window)
    method          : "welch_lorentzian"

On a failed fit the rate is NaN, ``success`` is False, and the tier-2 arrays
are still returned (possibly empty for a degenerate trace). Tier-2 plot-data
arrays: ``psd_freq_hz``, ``psd``, ``psd_fit``.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

#: caller-selectable knobs — the single source of truth callers validate
#: against BEFORE any per-target loop.
TELEGRAPH_PSD_KNOBS = frozenset({"nperseg", "window", "detrend"})


def validate_telegraph_psd_kwargs(knobs: Dict) -> None:
    """Raise ValueError for an unknown knob — call BEFORE per-target loops."""
    unknown = set(knobs) - TELEGRAPH_PSD_KNOBS
    if unknown:
        raise ValueError(
            f"Unknown telegraph-PSD knob(s) {sorted(unknown)}; "
            f"valid: {sorted(TELEGRAPH_PSD_KNOBS)}"
        )


def lorentzian_knee(f: np.ndarray, amplitude: float, corner_hz: float,
                    floor: float) -> np.ndarray:
    """The one-sided RTS spectrum model: ``A / (1 + (f/f_c)^2) + B``."""
    return amplitude / (1.0 + (f / corner_hz) ** 2) + floor


def _fit_knee(freq: np.ndarray, psd: np.ndarray) -> tuple:
    """Fit the Lorentzian knee in log space; returns (A, f_c, B).

    Log-parameterization keeps every parameter positive, and fitting
    ``log(S)`` weights the decades evenly — a linear-space fit is dominated
    by the handful of plateau bins and barely constrains the corner.
    """
    high = psd[freq >= freq[-1] / 3.0]
    floor0 = float(np.median(high)) if high.size else float(np.min(psd))
    floor0 = max(floor0, 1e-12 * float(np.max(psd)))
    plateau0 = max(
        float(np.mean(psd[: max(3, psd.size // 50)])) - floor0, floor0
    )
    below = np.nonzero(psd < 0.5 * plateau0 + floor0)[0]
    corner0 = float(freq[below[0]]) if below.size else float(
        np.sqrt(freq[0] * freq[-1])
    )
    corner0 = float(np.clip(corner0, freq[0], freq[-1]))

    p0 = np.log([plateau0, corner0, floor0])
    lower = np.log([1e-12 * plateau0, 0.5 * freq[0], 1e-6 * floor0])
    upper = np.log([1e12 * plateau0, freq[-1], 1e6 * max(floor0, plateau0)])

    def log_model(f, log_a, log_fc, log_b):
        return np.log(
            lorentzian_knee(f, np.exp(log_a), np.exp(log_fc), np.exp(log_b))
        )

    popt, _ = curve_fit(log_model, freq, np.log(psd), p0=p0,
                        bounds=(lower, upper), maxfev=20000)
    a, fc, b = np.exp(popt)
    return float(a), float(fc), float(b)


def fit_telegraph_psd(states: np.ndarray, dt_s: float, *,
                      nperseg: Optional[int] = None, window: str = "hann",
                      detrend: str = "constant") -> Dict[str, Any]:
    """Extract the switching rate of a 0/1 telegraph trace from its PSD knee.

    Parameters
    ----------
    states : 1-D array of 0/1 values
        The discriminated shot trace, uniformly sampled.
    dt_s : float
        Shot-to-shot period in seconds (the sampling interval).
    nperseg : int, optional
        Welch segment length. Default ``min(n, max(256, n // 8))`` — about 8
        averaged segments, keeping low-frequency reach on long traces.
    window, detrend
        Passed to :func:`scipy.signal.welch`.

    See the module docstring for the rate convention and the result contract.
    """
    if dt_s is None or not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(
            f"dt_s must be a positive finite shot period in seconds, got {dt_s!r}"
        )
    states = np.asarray(states, dtype=float).ravel()
    if states.size == 0:
        raise ValueError("states is empty — nothing to analyze")

    rounded = np.rint(states)
    n_transitions = int(np.count_nonzero(np.diff(rounded)))
    p_excited = float(np.mean(states))

    out: Dict[str, Any] = {
        "parity_rate_hz": float("nan"),
        "psd_corner_hz": float("nan"),
        "psd_amplitude": float("nan"),
        "psd_white_floor": float("nan"),
        "n_transitions": n_transitions,
        "p_excited": p_excited,
        "success": False,
        "method": "welch_lorentzian",
        "psd_freq_hz": np.array([]),
        "psd": np.array([]),
        "psd_fit": np.array([]),
    }

    if nperseg is None:
        nperseg = min(states.size, max(256, states.size // 8))
    nperseg = int(min(int(nperseg), states.size))

    try:
        freq, psd = welch(states - p_excited, fs=1.0 / float(dt_s),
                          window=window, nperseg=nperseg, detrend=detrend)
    except Exception:
        return out

    keep = (freq > 0) & (psd > 0) & np.isfinite(psd)
    freq, psd = freq[keep], psd[keep]
    out.update(psd_freq_hz=freq, psd=psd,
               psd_fit=np.full_like(psd, np.nan))
    if freq.size < 8:
        return out

    try:
        amplitude, corner, floor = _fit_knee(freq, psd)
    except Exception:
        return out

    out.update(psd_corner_hz=corner, psd_amplitude=amplitude,
               psd_white_floor=floor,
               psd_fit=lorentzian_knee(freq, amplitude, corner, floor))
    # A corner pinned at (or outside) the spectral window is unresolved: too
    # slow to see in this trace length, or faster than the shot cadence.
    resolved = bool(freq[0] < corner < freq[-1] and np.isfinite(corner))
    if resolved:
        out["parity_rate_hz"] = float(np.pi * corner)
        out["success"] = True
    return out
