from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator, reduced_signal, with_iqdata
from scqat.tools.ramsey_fit import fit_ramsey
from scqat.tools.iq_reduce import AXIAL_KNOBS, validate_iq_reduce_kwargs
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
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
        has_iq = "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars)
        if 'signal' not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Ramsey analysis requires a 'signal' variable, or complex 'IQdata', "
                "or both 'I' and 'Q'."
            )
        if 'idle_time' not in dataset.coords:
            raise ValueError("Ramsey analysis requires an 'idle_time' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Fit the Ramsey signal and extract oscillation/decay parameters.

        Kwargs — flat and fully owned; unknown names raise:
            force_model (str): Force ``'single'``, ``'beat'`` or ``'relaxation'``
                instead of the automatic frequency-gate + BIC selection.
            angle, positions, pca_sign
                IQ->1-D axial-reduction knobs (see :func:`scqat.tools.iq_reduce.axial`);
                ignored when the dataset already carries a real ``signal``.

        Returns a dict with:
            model_type, a_1, kappa_1, tau_1, f_1, phi_1, c, success,
            signal, reduction_method, reduction_angle,
            best_fit, fft_freq, fft_amp, fit_report,
            and (for the beat model) a_2, kappa_2, tau_2, f_2, phi_2.
        """
        force_model = kwargs.pop('force_model', None)
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)

        sig = reduced_signal(dataset, **kwargs)
        idle_time = np.asarray(sig.coords['idle_time'].values, dtype=float)
        results = fit_ramsey(idle_time, sig.values, force_model=force_model)
        results['signal'] = np.asarray(sig.values, dtype=float)
        results['reduction_method'] = sig.attrs.get('reduction_method')
        results['reduction_angle'] = sig.attrs.get('reduction_angle')
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the model type and fit parameters; drop the diagnostic arrays."""
        drop = {'best_fit', 'fft_freq', 'fft_amp', 'fit_report', 'signal'}
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
        signal = np.asarray(results['signal'], dtype=float)
        best_fit = np.asarray(results['best_fit'], dtype=float)
        fft_freq = np.asarray(results['fft_freq'], dtype=float)
        fft_amp = np.asarray(results['fft_amp'], dtype=float)

        attr_keys = ('model_type', 'a_1', 'kappa_1', 'tau_1', 'f_1', 'phi_1', 'c',
                     'a_2', 'kappa_2', 'tau_2', 'f_2', 'phi_2', 'success')
        attrs = {k: results[k] for k in attr_keys if k in results}
        if 'success' in attrs:
            attrs['success'] = int(bool(attrs['success']))
        attrs['reduction_method'] = str(results.get('reduction_method', 'signal'))
        # 0.0 is a legitimate angle (axis on I) — only None becomes NaN
        attrs['reduction_angle'] = (float(results['reduction_angle'])
                                    if results.get('reduction_angle') is not None else float('nan'))

        data_vars = {
            'signal': ('idle_time', signal),
            'best_fit': ('idle_time', best_fit),
            'fft_amp': ('fft_freq', fft_amp),
        }
        # the raw IQ cloud for the shared IQ-plane panel (absent on pre-reduced input)
        if "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars):
            iq = with_iqdata(dataset)["IQdata"].squeeze().values
            data_vars['iq_i'] = ('idle_time', np.real(iq).astype(float))
            data_vars['iq_q'] = ('idle_time', np.imag(iq).astype(float))

        return xr.Dataset(
            data_vars,
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
        figs = {
            'time_domain': plot_time_domain(plot_data),
            'fft_spectrum': plot_fft(plot_data),
        }
        if has_iq_plane(plot_data):
            figs['iq_plane'] = plot_iq_plane(plot_data)
        return figs

