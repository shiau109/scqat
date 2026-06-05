from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_damped_oscillation import FitDampedOscillation
from scqat.math_tools.fit_damping_beat import FitDampingBeat
from scqat.protocols.ramsey.visualization import plot_time_domain, plot_fft


class RamseyAnalyzer(BaseAnalyzer):
    """
    Analyzes Ramsey experiment data to extract decay rate and oscillation frequency.

    Expects an xarray.Dataset with:
        - Variable: 'signal'
        - Coordinate: 'idle_time'

    Automatically detects whether the data contains a single damped oscillation
    or a damped beat (two frequencies) using FFT peak analysis,
    then fits the appropriate model.
    """

    protocol_name = "ramsey"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if 'signal' not in dataset:
            raise ValueError("Ramsey analysis requires a 'signal' variable in the dataset.")
        if 'idle_time' not in dataset.coords:
            raise ValueError("Ramsey analysis requires an 'idle_time' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the Ramsey signal and extract oscillation/decay parameters.

        Kwargs:
            force_model (str): Force 'single' or 'beat' model instead of auto-detect.

        Returns a dict with:
            model_type, a_1, kappa_1, tau_1, f_1, phi_1, c, best_fit,
            fft_freq, fft_amp, fit_report,
            and (for beat model) a_2, kappa_2, tau_2, f_2, phi_2.
        """
        force_model = kwargs.get('force_model', None)

        # Prepare DataArray with 'x' coordinate for the fitters
        fit_data = dataset['signal'].rename({'idle_time': 'x'}).squeeze()

        # Compute FFT (always returned for diagnostics)
        fft_freq, fft_amp = self._compute_fft(dataset)

        # Auto-detect or use forced model
        detected_type, peak_freqs = self._detect_model_type(fit_data)
        if force_model == 'single':
            model_type = 'single'
        elif force_model == 'beat':
            model_type = 'beat'
        else:
            model_type = detected_type

        # Fit
        if model_type == 'beat':
            fitter = FitDampingBeat(fit_data)
            fitter.guess()
            if len(peak_freqs) >= 2:
                fitter.params['f_1'].set(value=peak_freqs[0])
                fitter.params['f_2'].set(value=peak_freqs[1])
            elif len(peak_freqs) == 1:
                fitter.params['f_1'].set(value=peak_freqs[0])
            fit_result = fitter.fit()
            params = {k: v.value for k, v in fit_result.params.items()}

            results = {
                'model_type': 'beat',
                'a_1': params['a_1'],
                'kappa_1': params['kappa_1'],
                'tau_1': 1.0 / params['kappa_1'] if params['kappa_1'] != 0 else float('nan'),
                'f_1': params['f_1'],
                'phi_1': params['phi_1'],
                'a_2': params['a_2'],
                'kappa_2': params['kappa_2'],
                'tau_2': 1.0 / params['kappa_2'] if params['kappa_2'] != 0 else float('nan'),
                'f_2': params['f_2'],
                'phi_2': params['phi_2'],
                'c': params['c'],
            }
        else:
            fitter = FitDampedOscillation(fit_data)
            fitter.guess()
            if len(peak_freqs) >= 1:
                fitter.params['f'].set(value=peak_freqs[0])
            fit_result = fitter.fit()
            params = {k: v.value for k, v in fit_result.params.items()}

            results = {
                'model_type': 'single',
                'a_1': params['a'],
                'kappa_1': params['kappa'],
                'tau_1': 1.0 / params['kappa'] if params['kappa'] != 0 else float('nan'),
                'f_1': params['f'],
                'phi_1': params['phi'],
                'c': params['c'],
            }

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
                     'a_2', 'kappa_2', 'tau_2', 'f_2', 'phi_2')
        attrs = {k: results[k] for k in attr_keys if k in results}

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
        """Generate time-domain fit plot and FFT spectrum plot from plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {
            'time_domain': plot_time_domain(plot_data),
            'fft_spectrum': plot_fft(plot_data),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fft(dataset: xr.Dataset):
        """Compute one-sided FFT amplitude spectrum."""
        idle_times = dataset.coords['idle_time'].values
        y = dataset['signal'].values
        n = len(idle_times)
        dt = idle_times[1] - idle_times[0] if n > 1 else 1.0

        amp = np.fft.fft(y)[: n // 2]
        freq = np.fft.fftfreq(n, dt)[: len(amp)]
        amp[0] = 0  # remove DC
        return freq, np.abs(amp)

    @staticmethod
    def _detect_model_type(fit_data):
        """Auto-detect single vs beat by checking FFT for a second dominant peak.

        Returns:
            (model_type, peak_freqs): model_type is 'single' or 'beat';
                peak_freqs is a list of detected frequencies sorted by power
                (descending), used as initial guesses for the fitter.
        """
        y = fit_data.values
        x = fit_data.coords['x'].values
        dt = float(x[1] - x[0])

        amp = np.fft.fft(y)[: len(y) // 2]
        freq = np.fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0
        power = np.abs(amp)

        # Find local maxima in the power spectrum
        local_peak_indices, _ = find_peaks(power, height=float(power.max()) * 0.1)
        if len(local_peak_indices) == 0:
            # Fallback: use global max
            peak_freqs = [float(np.abs(freq[np.argmax(power)]))]
            return 'single', peak_freqs

        # Sort local peaks by power (descending)
        sorted_local = local_peak_indices[power[local_peak_indices].argsort()[::-1]]
        peak_freqs = [float(np.abs(freq[idx])) for idx in sorted_local]

        if len(sorted_local) < 2:
            return 'single', peak_freqs

        # Estimate noise floor as median of the power spectrum
        noise_floor = float(np.median(power[power > 0]))

        for idx in sorted_local[1:]:
            is_strong_relative = power[idx] / power[sorted_local[0]] > 0.5
            is_above_noise = power[idx] > 2 * noise_floor
            is_separated = abs(int(idx) - int(sorted_local[0])) > 2
            if is_strong_relative and is_above_noise and is_separated:
                return 'beat', peak_freqs
        return 'single', peak_freqs
