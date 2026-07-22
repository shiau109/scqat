"""Per-trace Ramsey fit — the reduction shared by an experiment FAMILY.

One time trace (idle-time axis + real signal) in, the fitted fringe model out.
This is the pure-math reduction behind every Ramsey-type experiment: the 1-D
Ramsey experiment uses it once, the charge-gate map calls it once per gate
voltage. Per the repo rule ("anything used by more than one estimator lives in
tools/"), it lives here — estimators compose it, they never call each other.

The lab sequence is ``x90 -> idle -> y90``, so the fringe is a **sine** whose
phase is seeded at 0. Model selection is principled rather than heuristic:

1. **frequency gate** — if the dominant fringe spans fewer than
   :data:`MIN_CYCLES` oscillations over the idle window (i.e. the frequency
   is too close to 0 to resolve), fit a pure **exponential decay**
   (relaxation / T1-like) and report the frequency as 0;
2. otherwise fit a single damped sine and a two-frequency **beat** and keep
   the **beat** only when it improves the Bayesian information criterion by
   at least :data:`DELTA_BIC` (so a single-frequency signal is not upgraded
   to a beat by fitting noise). The beat case (charge dispersion) calibrates
   the qubit with the **mean** of the two frequencies.

Result contract (model-polymorphic on ``model_type``)
-----------------------------------------------------
Always present: ``model_type`` (``'single'``/``'beat'``/``'relaxation'``),
``a_1, kappa_1, tau_1, f_1, phi_1, c, success`` and the diagnostics
``best_fit, fft_freq, fft_amp, fit_report``. The ``'beat'`` model additionally
carries ``a_2, kappa_2, tau_2, f_2, phi_2`` — consumers must branch on
``model_type`` (or ``.get``) before touching any ``_2`` key.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
from lmfit.model import ModelResult

from .fit_damped_oscillation import FitDampedOscillation
from .fit_damping_beat import FitDampingBeat
from .fit_exp_decay import FitExponentialDecay
from .function_fitting import robust_dt

#: Minimum number of fringe oscillations across the idle window for a real
#: frequency to be resolvable. A pure decay concentrates its spectrum in the
#: first non-DC bin, which corresponds to just under one cycle across the
#: window ((N-1)/N), so a threshold of 1.0 catches it while a genuine
#: multi-cycle fringe (cycles > 1) passes through to the BIC comparison.
MIN_CYCLES = 1.0
#: Bayesian-information-criterion margin by which the beat model must beat the
#: single model to be accepted (Kass-Raftery "strong evidence").
DELTA_BIC = 6.0

#: Valid ``force_model`` values (``None`` = automatic selection).
RAMSEY_MODELS = ("single", "beat", "relaxation")


def _ramsey_fft(idle_time: np.ndarray, signal: np.ndarray):
    """One-sided FFT amplitude spectrum (DC removed)."""
    n = len(idle_time)
    dt = robust_dt(idle_time) if n > 1 else 1.0

    amp = np.fft.fft(signal)[: n // 2]
    freq = np.fft.fftfreq(n, dt)[: len(amp)]
    amp[0] = 0  # remove DC
    return freq, np.abs(amp)


def _fit_single(fit_data, f_seed: float) -> Tuple[Dict[str, Any], ModelResult]:
    """Single damped sine: ``a*exp(-kappa*x)*sin(2*pi*f*x + phi) + c``."""
    fitter = FitDampedOscillation(fit_data, basis="sin")
    fitter.guess()
    if f_seed and f_seed > 0:
        fitter.params['f'].set(value=abs(f_seed))
    fit_result = fitter.fit()
    p = {k: v.value for k, v in fit_result.params.items()}
    results = {
        'model_type': 'single',
        'a_1': p['a'],
        'kappa_1': p['kappa'],
        'tau_1': 1.0 / p['kappa'] if p['kappa'] != 0 else float('nan'),
        'f_1': p['f'],
        'phi_1': p['phi'],
        'c': p['c'],
    }
    return results, fit_result


def _fit_beat(fit_data) -> Tuple[Dict[str, Any], ModelResult]:
    """Two damped sines (charge dispersion); both components kept free."""
    fitter = FitDampingBeat(fit_data, basis="sin")
    fitter.guess(force_two_components=True)
    fit_result = fitter.fit()
    p = {k: v.value for k, v in fit_result.params.items()}
    results = {
        'model_type': 'beat',
        'a_1': p['a_1'],
        'kappa_1': p['kappa_1'],
        'tau_1': 1.0 / p['kappa_1'] if p['kappa_1'] != 0 else float('nan'),
        'f_1': p['f_1'],
        'phi_1': p['phi_1'],
        'a_2': p['a_2'],
        'kappa_2': p['kappa_2'],
        'tau_2': 1.0 / p['kappa_2'] if p['kappa_2'] != 0 else float('nan'),
        'f_2': p['f_2'],
        'phi_2': p['phi_2'],
        'c': p['c'],
    }
    return results, fit_result


def _try_fit_beat(fit_data) -> Optional[Tuple[Dict[str, Any], ModelResult]]:
    """Beat fit guarded for the model comparison; ``None`` if it fails."""
    try:
        return _fit_beat(fit_data)
    except Exception:
        return None


def _fit_relaxation(fit_data) -> Tuple[Dict[str, Any], ModelResult]:
    """Pure exponential decay: ``a*exp(-x/tau) + c``; frequency reported as 0."""
    fitter = FitExponentialDecay(fit_data)
    fitter.guess()
    fit_result = fitter.fit()
    p = {k: v.value for k, v in fit_result.params.items()}  # a, tau, c
    tau = p['tau']
    results = {
        'model_type': 'relaxation',
        'a_1': p['a'],
        'kappa_1': 1.0 / tau if tau != 0 else float('nan'),
        'tau_1': tau,
        'f_1': 0.0,
        'phi_1': 0.0,
        'c': p['c'],
    }
    return results, fit_result


def fit_ramsey(
    idle_time: np.ndarray,
    signal: np.ndarray,
    *,
    force_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Fit one Ramsey trace and extract oscillation/decay parameters.

    Parameters
    ----------
    idle_time : 1-D float array
        Idle-time axis (s or ns — the fitted ``f``/``tau`` are in its inverse
        / its own units).
    signal : 1-D float array
        Measured Ramsey signal on that axis.
    force_model : str, optional
        Force ``'single'``, ``'beat'`` or ``'relaxation'`` instead of the
        automatic frequency-gate + BIC selection.
    """
    if force_model not in (None, *RAMSEY_MODELS):
        raise ValueError(
            f"force_model must be None, 'single', 'beat' or 'relaxation', got {force_model!r}."
        )

    idle_time = np.asarray(idle_time, dtype=float)
    signal = np.asarray(signal, dtype=float).ravel()
    fit_data = xr.DataArray(signal, coords={'x': idle_time}, dims='x')

    # FFT once: feeds both the diagnostic spectrum and the frequency gate.
    fft_freq, fft_amp = _ramsey_fft(idle_time, signal)
    f_dom = float(fft_freq[int(np.argmax(fft_amp))]) if fft_amp.size else 0.0

    span = float(idle_time[-1] - idle_time[0]) if idle_time.size > 1 else 0.0
    cycles = abs(f_dom) * span

    # Select and fit the model.
    if force_model == 'relaxation' or (force_model is None and cycles < MIN_CYCLES):
        results, fit_result = _fit_relaxation(fit_data)
    elif force_model == 'single':
        results, fit_result = _fit_single(fit_data, f_dom)
    elif force_model == 'beat':
        results, fit_result = _fit_beat(fit_data)
    else:
        # Auto: compare a single damped sine against a genuine two-frequency
        # beat and keep the beat only on a decisive BIC improvement.
        res_single, fr_single = _fit_single(fit_data, f_dom)
        beat = _try_fit_beat(fit_data)
        if beat is not None and beat[1].bic < fr_single.bic - DELTA_BIC:
            results, fit_result = beat
        else:
            results, fit_result = res_single, fr_single

    results['success'] = bool(fit_result.success)
    results['best_fit'] = fit_result.best_fit
    results['fft_freq'] = fft_freq
    results['fft_amp'] = fft_amp
    results['fit_report'] = fit_result.fit_report()

    return results
