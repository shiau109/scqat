from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.ramsey_fit import fit_ramsey
from scqat.estimators.ramsey.visualization import plot_time_domain, plot_fft


class RamseyEstimator(BaseEstimator):
    """Estimator for a Ramsey experiment: fits a *model* to the probed *Dataset*
    and returns the extracted model parameters (the fringe frequency, which
    calibrates the qubit frequency, and the decay time T2*).

    Expects an ``xarray.Dataset`` with:
        - Variable: ``'signal'``
        - Coordinate: ``'idle_time'``

    The heavy lifting — the frequency-gate + BIC model selection over the
    single / beat / relaxation fits — is the family-shared per-trace reduction
    :func:`scqat.tools.ramsey_fit.fit_ramsey`; this estimator only resolves the
    dataset into arrays, forwards its flat kwarg surface, and owns the
    artifacts (metadata / plot data / figures).

    The QM node's ``estimate`` step calls :meth:`analyze`; ``update`` then writes
    ``f_01`` / ``charge_dispersion`` from the returned ``model_type`` + ``f_1``/``f_2``.
    """

    estimator_name = "ramsey"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if 'signal' not in dataset:
            raise ValueError("Ramsey analysis requires a 'signal' variable in the dataset.")
        if 'idle_time' not in dataset.coords:
            raise ValueError("Ramsey analysis requires an 'idle_time' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Fit the Ramsey signal and extract oscillation/decay parameters.

        Kwargs — flat and fully owned; unknown names raise:
            force_model (str): Force ``'single'``, ``'beat'`` or ``'relaxation'``
                instead of the automatic frequency-gate + BIC selection.

        Returns a dict with:
            model_type, a_1, kappa_1, tau_1, f_1, phi_1, c, success,
            best_fit, fft_freq, fft_amp, fit_report,
            and (for the beat model) a_2, kappa_2, tau_2, f_2, phi_2.
        """
        force_model = kwargs.pop('force_model', None)
        if kwargs:
            raise ValueError(
                f"Unknown keyword argument(s) {sorted(kwargs)} for "
                f"RamseyEstimator; valid: ['force_model']"
            )

        signal = dataset['signal'].squeeze()
        idle_time = np.asarray(signal.coords['idle_time'].values, dtype=float)
        return fit_ramsey(idle_time, signal.values, force_model=force_model)

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

