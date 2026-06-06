from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.state_discrimination import StateDiscriminationAnalyzer
from scqat.protocols.readout_fidelity.visualization import (
    plot_outlier_vs_sweep,
    plot_std_vs_sweep,
    plot_mean_distance_vs_sweep,
    plot_norm_res_vs_sweep,
    plot_fidelity_vs_sweep,
    plot_means_on_iq_plane,
)


class ReadoutFidelityAnalyzer(BaseAnalyzer):
    """
    Readout-fidelity sweep: run state discrimination at every point of a swept
    readout parameter and summarise how the discrimination quality evolves.

    Expects an xarray.Dataset with:
        - Variables:   ``I``, ``Q``
        - Coordinates: ``shot_idx``, ``prepared_state`` (required by the inner
          :class:`StateDiscriminationAnalyzer`)
        - Coordinate:  the swept axis named by :attr:`sweep_coord`

    For each value of ``sweep_coord`` the data is sliced and handed to a
    :class:`StateDiscriminationAnalyzer`; the per-slice trained GMM std/means,
    outlier probability, normalised residue, Gaussian norms and direct counts are
    collected as a function of the sweep.

    This unifies qcat's near-duplicate ``ROFidelityPower`` (``amp_prefactor``) and
    ``ROFidelityFreq`` (``frequency``); use the :class:`ReadoutPowerFidelityAnalyzer`
    / :class:`ReadoutFreqFidelityAnalyzer` subclasses, or set ``sweep_coord``
    directly. (The power-specific linear mean-drift refit from qcat is not ported
    here — see MIGRATION.md.)
    """

    protocol_name = "readout_fidelity"
    sweep_coord: Optional[str] = None  # subclasses set this; or pass sweep_coord kwarg

    # ------------------------------------------------------------------
    def _resolve_coord(self, kwargs: Dict[str, Any]) -> str:
        coord = kwargs.get('sweep_coord', self.sweep_coord)
        if coord is None:
            raise ValueError(
                "ReadoutFidelityAnalyzer needs a sweep coordinate: set the "
                "'sweep_coord' class attribute (or use a subclass) or pass "
                "sweep_coord=... ."
            )
        return coord

    def _check_data(self, dataset: xr.Dataset) -> None:
        for var in ("I", "Q"):
            if var not in dataset:
                raise ValueError(f"Readout fidelity requires a '{var}' variable.")
        for coord in ("shot_idx", "prepared_state"):
            if coord not in dataset.coords:
                raise ValueError(f"Readout fidelity requires a '{coord}' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run state discrimination per sweep point and collect the summary curves.

        Kwargs:
            sweep_coord (str): Override the swept coordinate name.
            user_std (float): Forwarded to StateDiscriminationAnalyzer.
            outlier_sigma (float): Forwarded to StateDiscriminationAnalyzer.

        Returns a dict with the sweep axis and per-sweep arrays:
            sweep_coord (name), sweep_values,
            std (S,), mean (S, center, iq), p_outlier (S, prepared_state),
            norm_res (S, prepared_state), gaussian_norms (S, prepared_state, gauss),
            direct_counts (S, prepared_state, count).
        """
        coord = self._resolve_coord(kwargs)
        sd_kwargs = {k: kwargs[k] for k in ('user_std', 'user_mean', 'outlier_sigma') if k in kwargs}

        sweep_values = np.asarray(dataset.coords[coord].values)
        sd = StateDiscriminationAnalyzer()

        std_list, mean_list, p_outlier_list, norm_res_list = [], [], [], []
        gaussian_norms_list, direct_counts_list = [], []

        for val in sweep_values:
            subdata = dataset.sel({coord: val})
            try:
                res = sd.extract_parameters(subdata, **sd_kwargs)
                tp = res['trained_paras']
                std_list.append(float(tp['std']))
                mean_list.append(np.asarray(tp['mean'], dtype=float))
                p_outlier_list.append(np.asarray(res['outlier_probability'], dtype=float))
                norm_res_list.append(np.asarray(res['norm_res'], dtype=float))
                gaussian_norms_list.append(np.asarray(res['gaussian_norms'], dtype=float))
                direct_counts_list.append(np.asarray(res['direct_counts'], dtype=float))
            except Exception:
                std_list.append(np.nan)
                mean_list.append(None)
                p_outlier_list.append(None)
                norm_res_list.append(None)
                gaussian_norms_list.append(None)
                direct_counts_list.append(None)

        # Determine common shapes from the first successful slice, then stack
        # (filling failed slices with NaN of the right shape).
        mean = self._stack(mean_list)
        p_outlier = self._stack(p_outlier_list)
        norm_res = self._stack(norm_res_list)
        gaussian_norms = self._stack(gaussian_norms_list)
        direct_counts = self._stack(direct_counts_list)

        return {
            'sweep_coord': coord,
            'sweep_values': sweep_values,
            'std': np.asarray(std_list, dtype=float),
            'mean': mean,
            'p_outlier': p_outlier,
            'norm_res': norm_res,
            'gaussian_norms': gaussian_norms,
            'direct_counts': direct_counts,
        }

    @staticmethod
    def _stack(items):
        """Stack a list of equal-shaped arrays, substituting NaN arrays for any
        ``None`` (failed slice). Returns ``None`` if every entry failed."""
        shape = next((a.shape for a in items if a is not None), None)
        if shape is None:
            return None
        filled = [a if a is not None else np.full(shape, np.nan) for a in items]
        return np.stack(filled, axis=0)

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Assemble the per-sweep summary curves into one self-sufficient Dataset
        so the figures redraw from the saved ``*_plotdata.nc`` alone. The swept
        coordinate name is preserved both as the dimension name and in
        ``attrs['sweep_coord']``."""
        coord = results['sweep_coord']
        sweep = np.asarray(results['sweep_values'])

        data_vars: Dict[str, Any] = {'std': (coord, np.asarray(results['std'], dtype=float))}
        coords: Dict[str, Any] = {coord: sweep}

        mean = results.get('mean')
        if mean is not None:
            coords['center'] = np.arange(mean.shape[1])
            coords['iq'] = ['I', 'Q']
            data_vars['mean'] = ([coord, 'center', 'iq'], mean)

        p_outlier = results.get('p_outlier')
        if p_outlier is not None:
            coords['prepared_state'] = np.arange(p_outlier.shape[1])
            data_vars['p_outlier'] = ([coord, 'prepared_state'], p_outlier)

        norm_res = results.get('norm_res')
        if norm_res is not None:
            coords.setdefault('prepared_state', np.arange(norm_res.shape[1]))
            data_vars['norm_res'] = ([coord, 'prepared_state'], norm_res)

        gnorms = results.get('gaussian_norms')
        if gnorms is not None:
            coords.setdefault('prepared_state', np.arange(gnorms.shape[1]))
            coords['gauss'] = np.arange(gnorms.shape[2])
            data_vars['gaussian_norms'] = ([coord, 'prepared_state', 'gauss'], gnorms)

        dcounts = results.get('direct_counts')
        if dcounts is not None:
            coords.setdefault('prepared_state', np.arange(dcounts.shape[1]))
            coords['count'] = np.arange(dcounts.shape[2])
            data_vars['direct_counts'] = ([coord, 'prepared_state', 'count'], dcounts)

        return xr.Dataset(data_vars, coords=coords, attrs={'sweep_coord': coord})

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the readout-fidelity sweep figures, drawing only from
        ``plot_data``; rebuild it when called outside ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)

        figs: Dict[str, plt.Figure] = {
            'std': plot_std_vs_sweep(plot_data),
        }
        if 'p_outlier' in plot_data:
            figs['outlier'] = plot_outlier_vs_sweep(plot_data)
        if 'mean' in plot_data:
            figs['mean_distance'] = plot_mean_distance_vs_sweep(plot_data)
            figs['means_on_IQ'] = plot_means_on_iq_plane(plot_data)
        if 'norm_res' in plot_data:
            figs['norm_res'] = plot_norm_res_vs_sweep(plot_data)
        if 'direct_counts' in plot_data or 'gaussian_norms' in plot_data:
            figs['fidelity'] = plot_fidelity_vs_sweep(plot_data)
        return figs


class ReadoutPowerFidelityAnalyzer(ReadoutFidelityAnalyzer):
    """Readout fidelity swept over readout amplitude (``amp_prefactor``).
    Ported from qcat ``readout_power.ROFidelityPower``."""
    protocol_name = "readout_power_fidelity"
    sweep_coord = "amp_prefactor"


class ReadoutFreqFidelityAnalyzer(ReadoutFidelityAnalyzer):
    """Readout fidelity swept over readout frequency (``frequency``).
    Ported from qcat ``readout_freq.ROFidelityFreq``."""
    protocol_name = "readout_freq_fidelity"
    sweep_coord = "frequency"
