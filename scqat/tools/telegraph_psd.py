"""Telegraph-signal PSD fit — the parity-switching-rate reduction.

One 0/1 series at uniform cadence ``dt_s`` in, the switching rate out. This is
the pure-math reduction behind the parity-switch experiment; per the repo rule
("anything used by more than one estimator lives in tools/", and simulation
callers reuse tools without estimator wrapping) it lives here.

THE INPUT MUST BE THE SERIES THAT *IS* THE TELEGRAPH
----------------------------------------------------
Everything below assumes the passed array samples the two-valued process whose
switching rate you want. For the parity-switch experiment that is the PARITY
series, NOT the raw readout — and the difference is not cosmetic.

That experiment runs y90 - idle - x90 with NO qubit reset, and the sequence is
a unitary, so it maps antipodal Bloch vectors to antipodal ones: |0> and |1>
can never give the same outcome. The measured outcome therefore INVERTS with
the pole the previous shot left the qubit in, i.e.

    s[i] = s[i-1] XOR parity[i]

The readout trace is the running XOR of the parity, and the pair series
``q[i] = s[i] XOR s[i+1]`` IS the parity telegraph. Feed q. Fed s instead, the
fit is of an integrated telegraph and the returned rate is meaningless —
measured: 50.53 Hz recovered from q against a 50 Hz truth, while s fails
outright. (scqat/tests/test_parity_switch_estimator.py pins this.)

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

Why the PSD knee and not transition counting: on a DIRECTLY sampled telegraph a
readout error flips one sample and fakes two transitions, so a counted rate is
inflated by ``~2*p_err/dt``, whereas uncorrelated errors are spectrally white —
they raise the fitted floor ``B`` and leave the corner ``f_c`` unbiased.
``n_transitions`` is therefore a diagnostic only, never folded into the rate.

READOUT ERROR IS *NOT* BENIGN ON AN XOR-DERIVED INPUT
-----------------------------------------------------
That white-floor argument holds only when the input is sampled directly. When
the input is the parity recovered from a no-reset readout (the case above), an
error on ONE readout sample flips TWO ADJACENT parity samples, so the
corruption enters as ``e[i] XOR e[i+1]`` — lag-1 correlated, not white — and it
biases the corner itself. Measured, planting Gamma = 50 Hz at dt = 100 us
(Gamma*dt = 0.005):

    readout error   eps/(Gamma*dt)   reported rate
        0                0.00           47.6 Hz   (0.95x)
        0.05 %           0.10           58.9 Hz   (1.18x)
        0.1  %           0.20           69.8 Hz   (1.40x)
        0.6  %           1.20            331 Hz   (6.63x)
        1.0  %           2.00           1133 Hz   (22.7x)

i.e. the rate inflates as ``1 + 2*eps/(Gamma*dt)``. Two consequences for
running the experiment:

* the readout must satisfy ``eps << Gamma*dt`` — about ``eps < 0.1*Gamma*dt``
  buys a ~20 % accurate rate. This is a real fidelity requirement, not a
  nicety;
* sampling SLOWER raises ``Gamma*dt`` and therefore suppresses the bias, so the
  best cadence is the slowest one that still leaves the corner comfortably
  inside the spectral window — the opposite of the "measure faster" instinct.

Correcting this bias (fitting the lag-1 term, or subtracting the known
``2*eps(1-eps)`` contribution) is not implemented; the rate is reported as
measured.

The reference parameterization (mapping fidelity F)
---------------------------------------------------
The literature writes the same spectrum as

    S_P(f) = 4*F^2*Gamma / ((2*Gamma)^2 + (2*pi*f)^2) + (1 - F^2)*dt

for a +-1-valued parity, two-sided, where F is the SEQUENCE MAPPING FIDELITY —
the correlation between the true parity and the measured one, ``F = 1 - 2*eps``
for a per-sample mapping-error probability ``eps``. That is the same function as
ours in different variables. Our series is 0/1 (a quarter of the variance) and
welch returns the one-sided density (twice), so ``S_ours = S_P / 2`` and

    f_c = Gamma / pi           <=>   Gamma = pi * f_c
    A   = F^2 / (2*Gamma)      <=>   F^2   = 2 * pi * f_c * A
    B   = (1 - F^2) * dt / 2   <=>   F^2   = 1 - 2 * B / dt

No refit is needed to get F: the existing three-parameter fit ALREADY IS the
reference model, with ``A`` and ``B`` exactly its ``4*F^2`` and ``(1-F^2)*dt``
terms carried as free parameters. The normalization is self-consistent by
construction — the Lorentzian integrates to ``F^2/4`` and the floor to
``(1-F^2)/4``, summing to the 1/4 variance of a balanced 0/1 series (checked on
the real chipA run: 0.2329 reconstructed vs 0.2490 measured, 6 %).

TWO estimates of F fall out, and they are NOT redundant:

    mapping_fidelity        sqrt(2*pi*f_c*A)    — from the plateau
    mapping_fidelity_floor  sqrt(1 - 2*B/dt)    — from the floor
    mapping_fidelity_ratio  their ratio         — the diagnostic

On a DIRECTLY sampled telegraph with genuinely white mapping error the FLOOR
estimate is the accurate one — measured at 1.000x of truth across Gamma =
2-200 Hz and eps = 0-5 %, while the plateau estimate runs ~2 % low from Welch
window leakage. But on the XOR-derived parity of a no-reset run the induced
noise is lag-1 correlated rather than white (previous section), and then the
floor estimate SATURATES AT 1 — it reports a flawless sequence — while the
plateau estimate collapses. Measured, planting Gamma = 20 Hz at dt = 50 us:

    readout error   corner bias   F_plateau   F_floor
        0              0.97x        0.978      1.000
        0.05 %         2.2x         0.913      1.000
        0.2  %          71x         0.285      1.000
        1.0  %         226x         0.358      1.000

So ``mapping_fidelity`` (the plateau) is the headline, and the RATIO is what to
read: near 1 the Lorentzian-plus-white-floor model describes the data; far below
1 it does not, which is the readable symptom of the correlated readout error
that also inflates the rate. Note the contrast gate does NOT catch this — those
corrupted fits keep a large ``A/B``.

It is REPORTED, never gated. The failure it exposes is real, but there is no
calibrated threshold yet, and a wrong one would reject good runs — which is
exactly how the old ``p_switch`` ceiling nearly refused a textbook chipA fit.
That run scores F = 0.578 (plateau) / 0.634 (floor), ratio 0.91.

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
    n_transitions   : raw nearest-neighbour flip count of the INPUT
                      (diagnostic; inflated by readout errors)
    p_switch        : n_transitions / (n - 1) — the fraction of consecutive
                      samples that DIFFER. See "Is it even resolved?" below.
    p_high          : mean level of the input telegraph. Generic on purpose:
                      fed the parity series this is the odd-parity fraction
                      (~0.5 is HEALTHY), not an excited-state population.
    psd_freq_min_hz : lowest fitted frequency bin = 1/(nperseg*dt). THE limit on
                      how slow a rate this trace can see (see below).
    psd_freq_max_hz : highest bin (Nyquist, 1/(2*dt)).
    psd_contrast    : A/B — the plateau over its own white floor. The "is there
                      a knee at all" number; gated by MIN_PSD_CONTRAST.
    mapping_fidelity : sequence mapping fidelity F from the PLATEAU,
                      sqrt(2*pi*f_c*A). F = 1 - 2*eps for a per-sample
                      mapping-error probability eps. The headline F.
    mapping_fidelity_floor : the same F from the FLOOR, sqrt(1 - 2*B/dt).
                      Independent of the plateau — NaN if the fitted floor
                      exceeds the model's whole noise budget (B > dt/2).
    mapping_fidelity_ratio : plateau/floor. ~1 means the model fits; well below
                      1 means it does not. See "The reference parameterization".
    corner_margin_low : corner / psd_freq_min_hz — how much low-frequency
                      headroom the fit had. REPORTED, not gated: below ~5 the
                      plateau is thinly sampled and a longer record would help.
    success         : bool — fit trustworthy (corner inside the spectral window
                      AND a real plateau above the floor)
    method          : "welch_lorentzian"

On a failed fit the rate is NaN, ``success`` is False, and the tier-2 arrays
are still returned (possibly empty for a degenerate trace). Tier-2 plot-data
arrays: ``psd_freq_hz``, ``psd``, ``psd_fit``.

Is it even resolved?
--------------------
Two independent things can go wrong, and only one of them is the fit's fault.

**No knee at all.** On uncorrelated data the spectrum is white, yet ``curve_fit``
still returns a finite corner — the Lorentzian cannot self-diagnose this. What
DOES separate the cases is the plateau-to-floor contrast ``A/B``
(:data:`MIN_PSD_CONTRAST`), which lands ~1e-9 on white data and 1e3–1e6 on a
real telegraph. Failing it sets ``success = False`` with a NaN rate while
KEEPING the corner and the arrays, so the failure stays diagnosable.

**Not enough low-frequency reach.** The lowest bin is
``psd_freq_min_hz = 1/(nperseg*dt)``, and with the default ``nperseg = n/8``
that is ``8 / T_record``. A corner near that bin is fitted from only a handful
of plateau points. ``corner_margin_low`` reports the headroom and is
deliberately NOT gated — the remedy is a longer record, which is a decision for
the caller, and refusing the run would throw away a usable measurement. Rule of
thumb: aim for a margin above ~5, i.e. ``T_record > 40 / f_c``.

The switches-per-sample follows from ``p_switch`` when wanted:
``Gamma * dt = -ln(1 - 2*p_switch)/2``. It is reported but does NOT gate — any
shot-to-shot noise drives it toward 0.5 regardless of whether a knee exists.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

#: caller-selectable knobs — the single source of truth callers validate
#: against BEFORE any per-target loop.
TELEGRAPH_PSD_KNOBS = frozenset({"nperseg", "window", "detrend"})

#: Refuse a fit whose Lorentzian plateau is not meaningfully above its own white
#: floor: ``A/B`` below this means there is no knee, only noise with a curve
#: drawn through it. Measured separation is enormous — real telegraphs score
#: 7.4e2 to 7.8e6 (including one buried under 30 % spurious flips), while
#: uncorrelated data scores ~1e-9 because the fitter drives ``A`` to zero. 3.0
#: therefore sits eight orders of magnitude clear of both populations.
#:
#: This REPLACED a threshold on ``p_switch``. That was the wrong quantity: any
#: shot-to-shot noise pushes it toward 0.5 whether or not a knee exists, and a
#: real chipA run with a clean fit (A/B ~ 7800) reported p_switch = 0.294
#: against a 0.40 ceiling — within 1.4x of a false refusal. ``p_switch`` is
#: still reported, it just no longer decides.
MIN_PSD_CONTRAST = 3.0


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


def telegraph_spectrum(series: np.ndarray, dt_s: float, *,
                       nperseg: Optional[int] = None, window: str = "hann",
                       detrend: str = "constant") -> tuple:
    """The one-sided Welch PSD of a 0/1 series — the Welch half alone.

    Split out from :func:`fit_telegraph_psd` so a caller can spectrum a series
    it does NOT want fitted: the parity-switch estimator draws the raw readout's
    spectrum as a diagnostic beside the fitted parity one. Returns
    ``(freq, psd)`` with the DC bin and any non-positive/non-finite bin dropped,
    so the arrays are ready for a log-log plot or a log-space fit.
    """
    if dt_s is None or not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(
            f"dt_s must be a positive finite sample period in seconds, got {dt_s!r}"
        )
    series = np.asarray(series, dtype=float).ravel()
    if series.size == 0:
        raise ValueError("series is empty — nothing to analyze")

    if nperseg is None:
        nperseg = min(series.size, max(256, series.size // 8))
    nperseg = int(min(int(nperseg), series.size))

    try:
        freq, psd = welch(series - float(np.mean(series)), fs=1.0 / float(dt_s),
                          window=window, nperseg=nperseg, detrend=detrend)
    except Exception:
        return np.array([]), np.array([])

    keep = (freq > 0) & (psd > 0) & np.isfinite(psd)
    return freq[keep], psd[keep]


def fit_telegraph_psd(states: np.ndarray, dt_s: float, *,
                      nperseg: Optional[int] = None, window: str = "hann",
                      detrend: str = "constant") -> Dict[str, Any]:
    """Extract the switching rate of a 0/1 telegraph from its PSD knee.

    Parameters
    ----------
    states : 1-D array of 0/1 values
        The series that IS the telegraph, uniformly sampled. For the
        parity-switch experiment that is the PARITY series, never the raw
        readout — see the module docstring, this is the whole ballgame.
    dt_s : float
        Sample-to-sample period in seconds.
    nperseg : int, optional
        Welch segment length. Default ``min(n, max(256, n // 8))`` — about 8
        averaged segments, keeping low-frequency reach on long traces.
    window, detrend
        Passed to :func:`scipy.signal.welch`.

    See the module docstring for the rate convention and the result contract.
    """
    if dt_s is None or not (np.isfinite(dt_s) and dt_s > 0):
        raise ValueError(
            f"dt_s must be a positive finite sample period in seconds, got {dt_s!r}"
        )
    states = np.asarray(states, dtype=float).ravel()
    if states.size == 0:
        raise ValueError("states is empty — nothing to analyze")

    rounded = np.rint(states)
    n_transitions = int(np.count_nonzero(np.diff(rounded)))
    p_high = float(np.mean(states))
    p_switch = (float(n_transitions) / (states.size - 1)
                if states.size >= 2 else float("nan"))

    out: Dict[str, Any] = {
        "parity_rate_hz": float("nan"),
        "psd_corner_hz": float("nan"),
        "psd_amplitude": float("nan"),
        "psd_white_floor": float("nan"),
        "n_transitions": n_transitions,
        "p_switch": p_switch,
        "p_high": p_high,
        "psd_freq_min_hz": float("nan"),
        "psd_freq_max_hz": float("nan"),
        "psd_contrast": float("nan"),
        "corner_margin_low": float("nan"),
        "mapping_fidelity": float("nan"),
        "mapping_fidelity_floor": float("nan"),
        "mapping_fidelity_ratio": float("nan"),
        "success": False,
        "method": "welch_lorentzian",
        "psd_freq_hz": np.array([]),
        "psd": np.array([]),
        "psd_fit": np.array([]),
    }

    freq, psd = telegraph_spectrum(states, dt_s, nperseg=nperseg,
                                   window=window, detrend=detrend)

    out.update(psd_freq_hz=freq, psd=psd,
               psd_fit=np.full_like(psd, np.nan))
    if freq.size < 8:
        return out
    out.update(psd_freq_min_hz=float(freq[0]), psd_freq_max_hz=float(freq[-1]))

    try:
        amplitude, corner, floor = _fit_knee(freq, psd)
    except Exception:
        return out

    contrast = float(amplitude / floor) if floor > 0 else float("inf")
    # the same fit read in the reference variables: A and B ARE its 4F^2 and
    # (1-F^2)dt terms, so F falls out twice over with no refit (module
    # docstring). Two independent estimates on purpose — their ratio is the
    # only thing here that notices a correlated-noise model failure.
    f_plateau = float(np.sqrt(2.0 * np.pi * corner * amplitude))
    floor_budget = 1.0 - 2.0 * floor / dt_s
    # B > dt/2 means the fitted floor carries more power than the model allows
    # for ANY fidelity — not a low F, a broken model. NaN, not a clamp to 0.
    f_floor = float(np.sqrt(floor_budget)) if floor_budget > 0 else float("nan")
    out.update(psd_corner_hz=corner, psd_amplitude=amplitude,
               psd_white_floor=floor, psd_contrast=contrast,
               corner_margin_low=float(corner / freq[0]),
               mapping_fidelity=f_plateau,
               mapping_fidelity_floor=f_floor,
               mapping_fidelity_ratio=(float(f_plateau / f_floor)
                                       if f_floor > 0 else float("nan")),
               psd_fit=lorentzian_knee(freq, amplitude, corner, floor))
    # A corner pinned at (or outside) the spectral window is unresolved: too
    # slow to see in this trace length, or faster than the sample cadence.
    resolved = bool(freq[0] < corner < freq[-1] and np.isfinite(corner))
    # ... and there must be an actual KNEE, not a curve drawn through noise.
    # curve_fit returns a finite corner on a white spectrum, so the fit cannot
    # self-diagnose that; the plateau-to-floor contrast can (module docstring).
    has_knee = bool(np.isfinite(contrast) and contrast >= MIN_PSD_CONTRAST)
    if resolved and has_knee:
        out["parity_rate_hz"] = float(np.pi * corner)
        out["success"] = True
    return out
