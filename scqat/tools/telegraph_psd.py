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

TWO FIT MODELS, selected by ``model=`` (:data:`TELEGRAPH_MODELS`). Both fit the
one-sided form above; they differ in whether F is shared or free.

**"constrained"** (default) — the reference model verbatim: ONE F, shared across
the amplitude and the floor, with ``dt`` FIXED to the known sample period. A
2-parameter fit ``{F, Gamma}`` of

    S(f) = [F^2/(2*Gamma)] / (1 + (pi*f/Gamma)^2) + (1 - F^2)*dt/2

so ``mapping_fidelity`` is the fitted F directly, and ``A``, ``f_c``, ``B`` are
DERIVED from ``{F, Gamma, dt}``. There is no independent floor estimate — the
model couples the two — so ``mapping_fidelity_floor`` and
``mapping_fidelity_ratio`` are NaN under this model. Its own fit-quality number
is ``psd_fit_residual`` (log-space RMS): a shared F that cannot fit the plateau
and the floor at once leaves a large residual. Note this model assumes a
SYMMETRIC (balanced) telegraph — its normalization bakes in the 1/4 variance —
which the parity is by physics; on a deliberately asymmetric telegraph its rate
is biased and ``independent`` is the one to use.

**"independent"** — the unconstrained fit: free ``{A, f_c, B}``, from which F is
read back TWICE. ``A`` and ``B`` are exactly the reference model's ``4*F^2`` and
``(1-F^2)*dt`` terms carried as free parameters, so no refit is needed — the
normalization is self-consistent by construction (the Lorentzian integrates to
``F^2/4`` and the floor to ``(1-F^2)/4``, summing to the 1/4 variance of a
balanced 0/1 series; checked on the real chipA run, 0.2329 vs 0.2490, 6 %). The
two estimates

    mapping_fidelity        sqrt(2*pi*f_c*A)    — from the plateau
    mapping_fidelity_floor  sqrt(1 - 2*B/dt)    — from the floor
    mapping_fidelity_ratio  their ratio         — the diagnostic

are NOT redundant. On a DIRECTLY sampled telegraph with genuinely white mapping
error the FLOOR estimate is the accurate one — measured at 1.000x of truth across
Gamma = 2-200 Hz and eps = 0-5 %, while the plateau runs ~2 % low from Welch
window leakage. But on the XOR-derived parity of a no-reset run the induced noise
is lag-1 correlated rather than white (previous section), and then the floor
estimate SATURATES AT 1 — it reports a flawless sequence — while the plateau
collapses. Measured, planting Gamma = 20 Hz at dt = 50 us:

    readout error   corner bias   F_plateau   F_floor
        0              0.97x        0.978      1.000
        0.05 %         2.2x         0.913      1.000
        0.2  %          71x         0.285      1.000
        1.0  %         226x         0.358      1.000

So the RATIO is what to read: near 1 the Lorentzian-plus-white-floor model
describes the data; far below 1 it does not, which is the readable symptom of the
correlated readout error that also inflates the rate. The contrast gate does NOT
catch this — those corrupted fits keep a large ``A/B``.

Which to use: ``constrained`` is the default because it returns F directly the
way the reference reports it and is more robust (2 params, coupled), and its
residual flags a bad fit. Reach for ``independent`` when you specifically want the
plateau/floor cross-check — the ratio is the sharper correlated-noise diagnostic.
The diagnostics are REPORTED, never gated: the failures are real, but there is no
calibrated threshold yet and a wrong one would reject good runs, exactly how the
old ``p_switch`` ceiling nearly refused a textbook chipA fit (which scores F =
0.578 plateau / 0.634 floor, ratio 0.91 under ``independent``).

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
    mapping_fidelity : sequence mapping fidelity F. F = 1 - 2*eps for a
                      per-sample mapping-error probability eps. Under
                      model="constrained" it is the fitted single F; under
                      model="independent" it is the PLATEAU estimate
                      sqrt(2*pi*f_c*A). The headline F either way.
    mapping_fidelity_floor : the floor estimate sqrt(1 - 2*B/dt) — ONLY under
                      model="independent" (NaN under "constrained", which has no
                      independent floor). NaN too if B > dt/2 (broken model).
    mapping_fidelity_ratio : plateau/floor — ONLY under model="independent"
                      (NaN under "constrained"). ~1 means the model fits, well
                      below 1 means it does not.
    psd_model       : which model was fitted ("constrained" | "independent").
    psd_fit_residual : log-space RMS residual of the fit over the band. The
                      constrained model's fit-quality number.
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

#: The two spectral fit models, selected by ``fit_telegraph_psd(model=...)``.
#: Both fit the SAME one-sided RTS spectrum; they differ only in whether the
#: sequence mapping fidelity F is a shared parameter or free per-term (see the
#: module docstring's "reference parameterization" section):
#:
#:   "constrained" — the reference model, ONE shared F in both the Lorentzian
#:                   amplitude and the floor, with dt FIXED to the sample period:
#:                   a 2-parameter fit {F, Gamma}. F comes out directly. DEFAULT.
#:   "independent" — free {A, f_c, B}: F is read back TWICE (plateau + floor),
#:                   and their disagreement is the model-consistency diagnostic.
#:
#: SCQO mirrors this as the ``psd_model`` Literal (its test_estimator_method_sync
#: keeps the two equal).
TELEGRAPH_MODELS = ("constrained", "independent")


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


def _fit_independent(freq: np.ndarray, psd: np.ndarray, dt_s: float) -> Dict:
    """The unconstrained fit: free {A, f_c, B}, then F read back TWICE.

    A and B are the reference model's ``4F^2`` and ``(1-F^2)dt`` terms carried as
    free parameters, so F falls out of the plateau AND the floor with no refit —
    two independent estimates whose ratio notices a correlated-noise model
    failure the contrast gate is blind to (module docstring).
    """
    amplitude, corner, floor = _fit_knee(freq, psd)
    f_plateau = float(np.sqrt(2.0 * np.pi * corner * amplitude))
    floor_budget = 1.0 - 2.0 * floor / dt_s
    # B > dt/2 means the fitted floor carries more power than the model allows
    # for ANY fidelity — a broken model, not a low F. NaN, not a clamp to 0.
    f_floor = float(np.sqrt(floor_budget)) if floor_budget > 0 else float("nan")
    ratio = float(f_plateau / f_floor) if f_floor > 0 else float("nan")
    return {
        "psd_amplitude": float(amplitude),
        "psd_corner_hz": float(corner),
        "psd_white_floor": float(floor),
        "mapping_fidelity": f_plateau,
        "mapping_fidelity_floor": f_floor,
        "mapping_fidelity_ratio": ratio,
    }


def _fit_constrained(freq: np.ndarray, psd: np.ndarray, dt_s: float) -> Dict:
    """The reference fit: ONE shared F, dt FIXED, a 2-parameter {F, Gamma}.

    Fits ``S(f) = [F^2/(2G)] / (1 + (pi*f/G)^2) + (1-F^2)*dt/2`` in log space —
    the 0/1 one-sided form of ``F^2*4G/((2G)^2+(2*pi*f)^2) + (1-F^2)*dt``. F is
    the fitted parameter, so it is returned directly; ``A``, ``f_c`` and ``B``
    are DERIVED from ``{F, Gamma, dt}``. There is no independent floor estimate
    here — the model couples the two by construction — so ``mapping_fidelity_*``
    floor/ratio are NaN (use ``model="independent"`` for that cross-check).

    Seeded from the cheap unconstrained knee fit so the corner and fidelity start
    near the data even on a short trace.
    """
    a0, fc0, b0 = _fit_knee(freq, psd)  # seed only
    gamma0 = np.pi * fc0
    f0 = float(np.clip(np.sqrt(max(2.0 * np.pi * fc0 * a0, 1e-12)), 1e-3, 1.0))

    # fit {F, log(Gamma)} — F stays in (0, 1], Gamma positive by the log
    def log_model(f, fidelity, log_gamma):
        gamma = np.exp(log_gamma)
        amp = fidelity ** 2 / (2.0 * gamma)
        floor = (1.0 - fidelity ** 2) * dt_s / 2.0
        return np.log(amp / (1.0 + (np.pi * f / gamma) ** 2) + floor)

    p0 = [f0, np.log(gamma0)]
    lower = [1e-3, np.log(0.5 * np.pi * freq[0])]
    upper = [1.0, np.log(np.pi * freq[-1])]
    popt, _ = curve_fit(log_model, freq, np.log(psd), p0=p0,
                        bounds=(lower, upper), maxfev=20000)
    fidelity = float(popt[0])
    gamma = float(np.exp(popt[1]))

    corner = gamma / np.pi
    amplitude = fidelity ** 2 / (2.0 * gamma)
    floor = (1.0 - fidelity ** 2) * dt_s / 2.0
    return {
        "psd_amplitude": float(amplitude),
        "psd_corner_hz": float(corner),
        "psd_white_floor": float(floor),
        "mapping_fidelity": fidelity,
        "mapping_fidelity_floor": float("nan"),
        "mapping_fidelity_ratio": float("nan"),
    }


#: model name -> fitter. Both return the same key set; the caller shares the
#: contrast/residual/gate logic downstream.
_TELEGRAPH_FITTERS = {
    "constrained": _fit_constrained,
    "independent": _fit_independent,
}


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
                      model: str = "constrained",
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
    model : {"constrained", "independent"}
        Which spectral model to fit (see :data:`TELEGRAPH_MODELS`).
        ``"constrained"`` (default) is the reference RTS model with a single
        shared fidelity F and dt fixed; ``"independent"`` frees the floor and
        reads F twice for the plateau/floor cross-check.
    nperseg : int, optional
        Welch segment length. Default ``min(n, max(256, n // 8))`` — about 8
        averaged segments, keeping low-frequency reach on long traces.
    window, detrend
        Passed to :func:`scipy.signal.welch`.

    See the module docstring for the rate convention and the result contract.
    """
    if model not in _TELEGRAPH_FITTERS:
        raise ValueError(
            f"Unknown telegraph model {model!r}; valid: {list(TELEGRAPH_MODELS)}"
        )
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
        "psd_model": model,
        "psd_fit_residual": float("nan"),
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
        fid = _TELEGRAPH_FITTERS[model](freq, psd, dt_s)
    except Exception:
        return out

    amplitude = fid["psd_amplitude"]
    corner = fid["psd_corner_hz"]
    floor = fid["psd_white_floor"]
    contrast = float(amplitude / floor) if floor > 0 else float("inf")
    psd_fit = lorentzian_knee(freq, amplitude, corner, floor)
    # log-space RMS residual over the fitted band: the constrained model's own
    # fit-quality number (it has no plateau/floor ratio to fall back on), and a
    # useful goodness check for the independent model too.
    with np.errstate(divide="ignore", invalid="ignore"):
        resid = np.log(psd) - np.log(psd_fit)
    residual = (float(np.sqrt(np.mean(resid ** 2)))
                if np.all(np.isfinite(resid)) else float("nan"))
    out.update(psd_corner_hz=corner, psd_amplitude=amplitude,
               psd_white_floor=floor, psd_contrast=contrast,
               corner_margin_low=float(corner / freq[0]),
               mapping_fidelity=fid["mapping_fidelity"],
               mapping_fidelity_floor=fid["mapping_fidelity_floor"],
               mapping_fidelity_ratio=fid["mapping_fidelity_ratio"],
               psd_fit_residual=residual,
               psd_fit=psd_fit)
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
