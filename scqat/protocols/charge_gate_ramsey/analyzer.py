from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.ramsey.analyzer import RamseyAnalyzer
from scqat.math_tools.fit_abscos import FitAbsCos
from scqat.protocols.charge_gate_ramsey.visualization import (
    plot_raw_2d_colormap,
    plot_2d_spectrum,
    plot_2d_spectrum_with_fit,
    plot_1d_frequencies,
)


class ChargeGateRamseyAnalyzer(BaseAnalyzer):
    """
    Analyzes 2D Ramsey data swept over charge_gate voltage.

    Expects an xarray.Dataset with:
        - Variable: 'signal' with dims (charge_gate, idle_time)
        - Coordinate: 'idle_time'
        - Coordinate: 'charge_gate'

    For each charge_gate slice, a RamseyAnalyzer extracts f_1 (and f_2 for
    beat-mode fits).  The centre frequency f_c is computed as the mean of
    (f_1 + f_2) / 2 across all charge gates where a beat was detected, unless
    the user supplies ``f_c_fixed``.

    Finally, |f_1 − f_c| merged with |f_2 − f_c| is fitted to an absolute
    cosine model as a function of charge_gate.
    """

    protocol_name = "charge_gate_ramsey"

    # ------------------------------------------------------------------
    # BaseAnalyzer interface
    # ------------------------------------------------------------------

    def _check_data(self, dataset: xr.Dataset) -> None:
        if 'signal' not in dataset:
            raise ValueError("Charge-gate Ramsey requires a 'signal' variable.")
        if 'idle_time' not in dataset.coords:
            raise ValueError("Charge-gate Ramsey requires an 'idle_time' coordinate.")
        if 'charge_gate' not in dataset.coords:
            raise ValueError("Charge-gate Ramsey requires a 'charge_gate' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run Ramsey fits per charge-gate and fit the |cos| dispersion.

        Kwargs:
            f_c_fixed (float | None): User-supplied centre frequency.
                If None, f_c is computed from the data.
            force_model (str | None): Forwarded to RamseyAnalyzer
                ('single', 'beat', or None for auto-detect).
            abscos_frequency_fixed (float | None): If given, fix the
                FitAbsCos ``frequency`` parameter to this value (not varied
                during the fit).  Analogous to ``f_c_fixed``.

        Returns a dict with:
            charge_gates, f_1, f_2, model_types,
            per_gate_results (list of per-gate result dicts),
            fft_freqs, fft_spectra,
            f_c,
            abscos_fit_result (lmfit ModelResult or None),
            abscos_params (dict or None).
        """
        force_model = kwargs.get('force_model', None)
        f_c_fixed = kwargs.get('f_c_fixed', None)

        charge_gates = dataset.coords['charge_gate'].values
        ramsey = RamseyAnalyzer()

        f_1_list = []
        f_2_list = []
        model_types = []
        per_gate_results = []
        fft_spectra = []
        fft_freqs = None

        for cg in charge_gates:
            slice_ds = dataset.sel(charge_gate=cg)
            try:
                res = ramsey.extract_parameters(slice_ds, force_model=force_model)
            except Exception:
                res = None

            if res is None:
                f_1_list.append(np.nan)
                f_2_list.append(np.nan)
                model_types.append(None)
                per_gate_results.append(None)
                fft_spectra.append(None)
                continue

            per_gate_results.append(res)
            model_types.append(res['model_type'])
            f_1_list.append(res['f_1'])
            f_2_list.append(res.get('f_2', np.nan))

            fft_spectra.append(res['fft_amp'])
            if fft_freqs is None:
                fft_freqs = res['fft_freq']

        f_1 = np.array(f_1_list)
        f_2 = np.array(f_2_list)

        # ---- compute f_c ------------------------------------------------
        if f_c_fixed is not None:
            f_c = float(f_c_fixed)
        else:
            # Only use points where both f_1 and f_2 are valid (beat model)
            beat_mask = np.array([m == 'beat' for m in model_types])
            if np.any(beat_mask):
                f_c = float(np.nanmean((f_1[beat_mask] + f_2[beat_mask]) / 2.0))
            else:
                # Fallback: use all valid f_1 values
                f_c = float(np.nanmean(f_1))

        # ---- build spectrum dataset for plotting -------------------------
        spectrum_dataset = self._build_spectrum_dataset(
            fft_spectra, fft_freqs, charge_gates
        )

        # ---- fit |cos| dispersion ----------------------------------------
        abscos_result, abscos_params = self._fit_abscos_dispersion(
            charge_gates, f_1, f_2, f_c,
            frequency_hint=kwargs.get('abscos_frequency_hint', None),
            frequency_fixed=kwargs.get('abscos_frequency_fixed', None),
            phase_bounds=kwargs.get('abscos_phase_bounds', None),
        )

        return {
            'charge_gates': charge_gates,
            'f_1': f_1,
            'f_2': f_2,
            'model_types': model_types,
            'per_gate_results': per_gate_results,
            'spectrum_dataset': spectrum_dataset,
            'f_c': f_c,
            'abscos_fit_result': abscos_result,
            'abscos_params': abscos_params,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist only the key parameters. The bulky intermediates kept in
        ``results`` for plotting/inspection — ``spectrum_dataset`` (an
        ``xr.Dataset``), ``per_gate_results``, and the lmfit ``abscos_fit_result``
        — are deliberately excluded from the metadata file.
        """
        return {
            'charge_gates': results['charge_gates'],
            'f_1': results['f_1'],
            'f_2': results['f_2'],
            'model_types': results['model_types'],
            'f_c': results['f_c'],
            'abscos_params': results['abscos_params'],
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """
        Assemble every array the four figures need into a single self-sufficient
        Dataset, so the plots can be reconstructed downstream with no
        recalculation.

        Variables: ``raw_signal`` (charge_gate, idle_time), ``spectrum``
        (charge_gate, frequency), ``f_1``/``f_2`` (charge_gate), and the
        pre-evaluated |cos| fit curves ``fit_curve_even``/``fit_curve_odd``/
        ``fit_abscos`` (cg_fine).  The centre frequency and |cos| fit parameters
        live in ``.attrs`` (``f_c``, ``abscos_*``, ``has_spectrum``, ``has_fit``).
        """
        charge_gates = np.asarray(results['charge_gates'], dtype=float)
        idle_time = dataset.coords['idle_time'].values

        coords: Dict[str, Any] = {
            'charge_gate': charge_gates,
            'idle_time': idle_time,
        }
        data_vars: Dict[str, Any] = {
            'raw_signal': (['charge_gate', 'idle_time'], np.asarray(dataset['signal'].values)),
            'f_1': ('charge_gate', np.asarray(results['f_1'], dtype=float)),
            'f_2': ('charge_gate', np.asarray(results['f_2'], dtype=float)),
        }
        f_c = float(results['f_c'])
        attrs: Dict[str, Any] = {'f_c': f_c}

        spectrum_ds = results.get('spectrum_dataset')
        if spectrum_ds is not None:
            coords['frequency'] = spectrum_ds.coords['frequency'].values
            data_vars['spectrum'] = (
                ['charge_gate', 'frequency'], spectrum_ds['spectrum'].values
            )
            attrs['has_spectrum'] = 1
        else:
            attrs['has_spectrum'] = 0

        abscos = results.get('abscos_params')
        if abscos is not None and abscos.get('success', False):
            amp = float(abscos['amplitude'])
            fit_freq = float(abscos['frequency'])
            phase = float(abscos['phase'])
            cg_fine = np.linspace(charge_gates.min(), charge_gates.max(), 200)
            curve = amp * np.cos(2 * np.pi * fit_freq * (cg_fine - phase))
            coords['cg_fine'] = cg_fine
            data_vars['fit_curve_even'] = ('cg_fine', f_c + curve)
            data_vars['fit_curve_odd'] = ('cg_fine', f_c - curve)
            data_vars['fit_abscos'] = ('cg_fine', np.abs(curve))
            attrs.update({
                'has_fit': 1,
                'abscos_amplitude': amp,
                'abscos_frequency': fit_freq,
                'abscos_phase': phase,
                'abscos_redchi': float(abscos.get('redchi', np.nan)),
            })
        else:
            attrs['has_fit'] = 0

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        # Draw strictly from plot_data so the figures stay reconstructable
        # downstream; rebuild it only when called outside analyze().
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        figs: Dict[str, plt.Figure] = {}
        figs['raw_colormap'] = plot_raw_2d_colormap(plot_data)
        figs['fft_spectrum'] = plot_2d_spectrum(plot_data)
        figs['fft_spectrum_with_fit'] = plot_2d_spectrum_with_fit(plot_data)
        figs['freq_vs_charge_gate'] = plot_1d_frequencies(plot_data)
        return figs

    # ------------------------------------------------------------------
    # Payload export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_plot_payload(results: Dict[str, Any], n_points: int = 200) -> Dict[str, np.ndarray]:
        """
        Extract the minimal numeric arrays needed to reconstruct the
        2D spectrum-with-fit figure without any re-calculation.

        Returns a dict suitable for ``np.savez_compressed``:
            spectrum_2d       (n_cg, n_freq)  – FFT amplitude 2-D array
            charge_gate_axis  (n_cg,)         – charge-gate coordinates
            frequency_axis    (n_freq,)       – frequency coordinates
            cg_fine           (n_points,)     – fine charge-gate grid for curves
            fit_curve_even    (n_points,)     – f_c + |cos| fit
            fit_curve_odd     (n_points,)     – f_c − |cos| fit
        """
        spectrum_ds = results.get('spectrum_dataset')
        if spectrum_ds is None:
            raise ValueError("results['spectrum_dataset'] is None; cannot build plot payload.")

        charge_gate_axis = spectrum_ds.coords['charge_gate'].values
        frequency_axis = spectrum_ds.coords['frequency'].values
        spectrum_2d = spectrum_ds['spectrum'].values  # (n_cg, n_freq)

        cg_fine = np.linspace(charge_gate_axis.min(), charge_gate_axis.max(), n_points)

        abscos_params = results.get('abscos_params')
        f_c = float(results['f_c'])
        if abscos_params is not None and abscos_params.get('success', False):
            amp = abscos_params['amplitude']
            fit_freq = abscos_params['frequency']
            phase = abscos_params['phase']
            curve = amp * np.cos(2 * np.pi * fit_freq * (cg_fine - phase))
            fit_curve_even = f_c + curve
            fit_curve_odd = f_c - curve
        else:
            fit_curve_even = np.full(n_points, np.nan)
            fit_curve_odd = np.full(n_points, np.nan)

        return {
            'spectrum_2d': spectrum_2d,
            'charge_gate_axis': charge_gate_axis,
            'frequency_axis': frequency_axis,
            'cg_fine': cg_fine,
            'fit_curve_even': fit_curve_even,
            'fit_curve_odd': fit_curve_odd,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spectrum_dataset(fft_spectra, fft_freqs, charge_gates):
        """Assemble per-gate FFT amplitudes into an xr.Dataset."""
        if fft_freqs is None:
            return None

        n_freq = len(fft_freqs)
        spectra = []
        for amp in fft_spectra:
            if amp is not None and len(amp) == n_freq:
                spectra.append(amp)
            else:
                spectra.append(np.full(n_freq, np.nan))

        return xr.Dataset(
            {'spectrum': (['charge_gate', 'frequency'], np.array(spectra))},
            coords={
                'charge_gate': charge_gates,
                'frequency': fft_freqs,
            },
        )

    @staticmethod
    def _fit_abscos_dispersion(charge_gates, f_1, f_2, f_c, frequency_hint=None, frequency_fixed=None, phase_bounds=None):
        """
        Merge |f_1 − f_c| and |f_2 − f_c| and fit with FitAbsCos.

        Parameters
        ----------
        phase_bounds : tuple[float, float] | None
            (min, max) bounds for the phase parameter.

        Returns (ModelResult | None, params dict | None).
        """
        cg_list = []
        freq_diff_list = []

        valid_f1 = ~np.isnan(f_1)
        if np.any(valid_f1):
            cg_list.append(charge_gates[valid_f1])
            freq_diff_list.append(np.abs(f_1[valid_f1] - f_c))

        valid_f2 = ~np.isnan(f_2)
        if np.any(valid_f2):
            cg_list.append(charge_gates[valid_f2])
            freq_diff_list.append(np.abs(f_2[valid_f2] - f_c))

        if len(freq_diff_list) == 0:
            return None, None

        merged_cg = np.concatenate(cg_list)
        merged_freq = np.concatenate(freq_diff_list)

        if len(merged_freq) < 3:
            return None, None

        # Sort by charge_gate for cleaner fitting
        order = np.argsort(merged_cg)
        merged_cg = merged_cg[order]
        merged_freq = merged_freq[order]

        fit_da = xr.DataArray(merged_freq, coords={'x': merged_cg}, dims='x')
        fitter = FitAbsCos(fit_da)
        fitter.guess()

        # Fix frequency if user supplies a hard value (not varied in fit)
        if frequency_fixed is not None:
            fitter.params['frequency'].set(value=frequency_fixed, vary=False)
        elif frequency_hint is not None:
            fitter.params['frequency'].set(value=frequency_hint)

        if phase_bounds is not None:
            fitter.params['phase'].set(min=phase_bounds[0], max=phase_bounds[1])

        try:
            fit_result = fitter.fit()
            params = {k: v.value for k, v in fit_result.params.items()}
            params['redchi'] = fit_result.redchi
            params['success'] = fit_result.success
            return fit_result, params
        except Exception:
            return None, None
