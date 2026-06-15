from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from lmfit.model import ModelResult

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.fit_damped_oscillation import FitDampedOscillation
from scqat.tools.fit_damping_beat import FitDampingBeat
from scqat.tools.fit_exp_decay import FitExponentialDecay
from scqat.tools.function_fitting import robust_dt
from scqat.estimators.ramsey.visualization import plot_time_domain, plot_fft


class RamseyEstimator(BaseEstimator):
    """Estimator for a Ramsey experiment: fits a *model* to the probed *Dataset*
    and returns the extracted model parameters (the fringe frequency, which
    calibrates the qubit frequency, and the decay time T2*).

    Expects an ``xarray.Dataset`` with:
        - Variable: ``'signal'``
        - Coordinate: ``'idle_time'``

    The lab sequence is ``x90 -> idle -> y90``, so the fringe is a **sine** whose
    phase is seeded at 0. Model selection is principled rather than heuristic:

    1. **frequency gate** — if the dominant fringe spans fewer than
       :attr:`MIN_CYCLES` oscillations over the idle window (i.e. the frequency
       is too close to 0 to resolve), fit a pure **exponential decay**
       (relaxation / T1-like) and report the frequency as 0;
    2. otherwise fit a single damped sine and a two-frequency **beat** and keep
       the **beat** only when it improves the Bayesian information criterion by
       at least :attr:`DELTA_BIC` (so a single-frequency signal is not upgraded
       to a beat by fitting noise). The beat case (charge dispersion) calibrates
       the qubit with the **mean** of the two frequencies.

    The QM node's ``estimate`` step calls :meth:`analyze`; ``update`` then writes
    ``f_01`` / ``charge_dispersion`` from the returned ``model_type`` + ``f_1``/``f_2``.
    """

    estimator_name = "ramsey"

    #: Minimum number of fringe oscillations across the idle window for a real
    #: frequency to be resolvable. A pure decay concentrates its spectrum in the
    #: first non-DC bin, which corresponds to just under one cycle across the
    #: window ((N-1)/N), so a threshold of 1.0 catches it while a genuine
    #: multi-cycle fringe (cycles > 1) passes through to the BIC comparison.
    MIN_CYCLES = 1.0
    #: Bayesian-information-criterion margin by which the beat model must beat the
    #: single model to be accepted (Kass-Raftery "strong evidence").
    DELTA_BIC = 6.0

    def _check_data(self, dataset: xr.Dataset) -> None:
        if 'signal' not in dataset:
            raise ValueError("Ramsey analysis requires a 'signal' variable in the dataset.")
        if 'idle_time' not in dataset.coords:
            raise ValueError("Ramsey analysis requires an 'idle_time' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Fit the Ramsey signal and extract oscillation/decay parameters.

        Kwargs:
            force_model (str): Force ``'single'``, ``'beat'`` or ``'relaxation'``
                instead of the automatic frequency-gate + BIC selection.

        Returns a dict with:
            model_type, a_1, kappa_1, tau_1, f_1, phi_1, c, success,
            best_fit, fft_freq, fft_amp, fit_report,
            and (for the beat model) a_2, kappa_2, tau_2, f_2, phi_2.
        """
        force_model = kwargs.get('force_model', None)
        if force_model not in (None, 'single', 'beat', 'relaxation'):
            raise ValueError(
                f"force_model must be None, 'single', 'beat' or 'relaxation', got {force_model!r}."
            )

        # Prepare a DataArray with an 'x' coordinate for the fitters.
        fit_data = dataset['signal'].rename({'idle_time': 'x'}).squeeze()

        # FFT once: feeds both the diagnostic spectrum and the frequency gate.
        fft_freq, fft_amp = self._compute_fft(dataset)
        f_dom = float(fft_freq[int(np.argmax(fft_amp))]) if fft_amp.size else 0.0

        x = np.asarray(fit_data.coords['x'].values, dtype=float)
        span = float(x[-1] - x[0]) if x.size > 1 else 0.0
        cycles = abs(f_dom) * span

        # Select and fit the model.
        if force_model == 'relaxation' or (force_model is None and cycles < self.MIN_CYCLES):
            results, fit_result = self._fit_relaxation(fit_data)
        elif force_model == 'single':
            results, fit_result = self._fit_single(fit_data, f_dom)
        elif force_model == 'beat':
            results, fit_result = self._fit_beat(fit_data)
        else:
            # Auto: compare a single damped sine against a genuine two-frequency
            # beat and keep the beat only on a decisive BIC improvement.
            res_single, fr_single = self._fit_single(fit_data, f_dom)
            beat = self._try_fit_beat(fit_data)
            if beat is not None and beat[1].bic < fr_single.bic - self.DELTA_BIC:
                results, fit_result = beat
            else:
                results, fit_result = res_single, fr_single

        results['success'] = bool(fit_result.success)
        results['best_fit'] = fit_result.best_fit
        results['fft_freq'] = fft_freq
        results['fft_amp'] = fft_amp
        results['fit_report'] = fit_result.fit_report()

        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the model type and fit parameters; drop the diagnostic arrays."""
        drop = {'best_fit', 'fft_freq', 'fft_amp', 'fit_report'}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the raw signal + best-fit curve (over ``idle_time``) and the FFT
        amplitude spectrum (over ``fft_freq``); fit parameters live in ``.attrs``
        so the time-domain and FFT figures need no recomputation.
        """
        idle_time = dataset.coords['idle_time'].values
        signal = np.asarray(dataset['signal'].squeeze().values, dtype=float)
        best_fit = np.asarray(results['best_fit'], dtype=float)
        fft_freq = np.asarray(results['fft_freq'], dtype=float)
        fft_amp = np.asarray(results['fft_amp'], dtype=float)

        attr_keys = ('model_type', 'a_1', 'kappa_1', 'tau_1', 'f_1', 'phi_1', 'c',
                     'a_2', 'kappa_2', 'tau_2', 'f_2', 'phi_2', 'success')
        attrs = {k: results[k] for k in attr_keys if k in results}
        if 'success' in attrs:
            attrs['success'] = int(bool(attrs['success']))

        return xr.Dataset(
            {
                'signal': ('idle_time', signal),
                'best_fit': ('idle_time', best_fit),
                'fft_amp': ('fft_freq', fft_amp),
            },
            coords={'idle_time': idle_time, 'fft_freq': fft_freq},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the time-domain fit plot and FFT spectrum plot, drawing
        strictly from ``plot_data`` so the figures stay reconstructable
        downstream; rebuild it only when called outside ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {
            'time_domain': plot_time_domain(plot_data),
            'fft_spectrum': plot_fft(plot_data),
        }

    # ------------------------------------------------------------------
    # Model fits (each returns the uniform results dict + the lmfit result)
    # ------------------------------------------------------------------

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _try_fit_beat(cls, fit_data) -> Optional[Tuple[Dict[str, Any], ModelResult]]:
        """Beat fit guarded for the model comparison; ``None`` if it fails."""
        try:
            return cls._fit_beat(fit_data)
        except Exception:
            return None

    @staticmethod
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fft(dataset: xr.Dataset):
        """Compute one-sided FFT amplitude spectrum (DC removed)."""
        idle_times = dataset.coords['idle_time'].values
        y = dataset['signal'].values
        n = len(idle_times)
        dt = robust_dt(idle_times) if n > 1 else 1.0

        amp = np.fft.fft(y)[: n // 2]
        freq = np.fft.fftfreq(n, dt)[: len(amp)]
        amp[0] = 0  # remove DC
        return freq, np.abs(amp)
