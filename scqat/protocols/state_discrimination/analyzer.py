from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_gaussian2d import FitMultiGaussian2D
from scqat.protocols.state_discrimination.visualization import (
    plot_prepared_state_scatter,
    plot_2d_histogram,
    plot_outliers,
    plot_2d_fit_residue,
    axis_formatter,
)

class StateDiscriminationAnalyzer(BaseAnalyzer):
    """
    Analyzes I/Q plane data for superconducting qubit state discrimination 
    using 2D Multi-Gaussian Mixture Models.
    """

    protocol_name = "state_discrimination"

    def _check_data(self, dataset: xr.Dataset) -> None:
        """Ensures the dataset has the required coordinates for this protocol."""
        for coords_name in ["shot_idx", "prepared_state"]:
            if coords_name not in dataset.coords:
                raise ValueError(f"State Discrimination requires '{coords_name}' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Executes the GMM fitting and population counting.
        
        Kwargs:
            user_mean (list): Optional initial guess for GMM centers.
            user_std (float): Optional initial guess for Gaussian std dev.
            outlier_sigma (float): Threshold for outlier detection (default: 3).
        """
        user_mean = kwargs.get('user_mean', None)
        user_std = kwargs.get('user_std', None)
        outlier_sigma = kwargs.get('outlier_sigma', 3)

        # 1. Preprocess raw data into 2D Histograms
        hist_dataset, mean_init, std_init = self._preprocess_data(dataset, user_std)

        # 2. Train the global GMM Model
        trained_paras = self._train_global_model(hist_dataset, mean_init, std_init, user_mean, user_std)

        # 3. Fit individual prepared states
        fit_results, fit_residues, norm_res = self._fit_individual_states(hist_dataset, trained_paras)

        # 4. Calculate distances, labels, and outliers
        distance_dataset = self._calc_distances(dataset, trained_paras['mean'])
        state_label = distance_dataset['distance'].argmin(dim='center')
        
        max_label = int(state_label.max().item())
        counts = xr.apply_ufunc(
            lambda arr: np.bincount(arr, minlength=max_label + 1),
            state_label,
            input_core_dims=[['idx_shot']],
            output_core_dims=[['count']],
            vectorize=True,
            output_dtypes=[int]
        )

        gaussian_norms = np.array([res['amp'] for res in fit_results])
        gaussian_norms = gaussian_norms / np.sum(gaussian_norms, axis=1, keepdims=True)

        outlier_mask = distance_dataset['distance'].min(dim='center') > (outlier_sigma * np.mean(trained_paras['std']))
        p_outlier = np.count_nonzero(outlier_mask, axis=1) / dataset['shot_idx'].size

        # 5. Pack everything into the results dictionary
        return {
            'trained_paras': trained_paras,
            'fitted_paras': fit_results,
            'gaussian_norms': gaussian_norms,
            'direct_counts': counts.values / dataset['shot_idx'].size,
            'state_label': state_label.values,
            'outlier_mask': outlier_mask.values,
            'outlier_probability': p_outlier,
            'norm_res': norm_res,
            'fit_residues': fit_residues,
            'hist_dataset': hist_dataset # Saving the binned data for plotting later
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the trained/fitted GMM parameters and per-state summaries; drop
        the per-shot arrays, residue grids, and binned histogram Dataset."""
        keep = (
            'trained_paras', 'fitted_paras', 'gaussian_norms',
            'direct_counts', 'outlier_probability', 'norm_res',
        )
        return {k: results[k] for k in keep}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle everything the four figures need into one Dataset: raw I/Q with
        per-shot state labels and outlier masks, the binned density and fit
        residue per state, the trained GMM mean/std, and the per-state count /
        norm / outlier summaries. Shared I/Q axis limits (5σ) live in ``.attrs``.
        """
        hist = results['hist_dataset']
        states = hist['prepared_state'].values
        n_state = len(states)

        I_arr = np.stack([dataset['I'].sel(prepared_state=s).values.ravel() for s in states])
        Q_arr = np.stack([dataset['Q'].sel(prepared_state=s).values.ravel() for s in states])
        state_label = np.asarray(results['state_label'])
        outlier_mask = np.asarray(results['outlier_mask']).astype(np.int8)
        density = hist['density'].values
        fit_residue = results['fit_residues'].values
        trained = results['trained_paras']
        mean = np.asarray(trained['mean'], dtype=float)
        direct_counts = np.asarray(results['direct_counts'])
        gaussian_norms = np.asarray(results['gaussian_norms'])
        outlier_prob = np.asarray(results['outlier_probability'], dtype=float)
        norm_res = np.asarray(results['norm_res'], dtype=float)

        all_I, all_Q = I_arr.ravel(), Q_arr.ravel()
        lim_I = (float(all_I.mean() - 5 * all_I.std()), float(all_I.mean() + 5 * all_I.std()))
        lim_Q = (float(all_Q.mean() - 5 * all_Q.std()), float(all_Q.mean() + 5 * all_Q.std()))

        return xr.Dataset(
            {
                'I': (['prepared_state', 'idx_shot'], I_arr),
                'Q': (['prepared_state', 'idx_shot'], Q_arr),
                'state_label': (['prepared_state', 'idx_shot'], state_label),
                'outlier_mask': (['prepared_state', 'idx_shot'], outlier_mask),
                'density': (['prepared_state', 'y', 'x'], density),
                'fit_residue': (['prepared_state', 'y', 'x'], fit_residue),
                'trained_mean': (['center', 'comp'], mean),
                'trained_amp': ('center', np.asarray(trained['amp'], dtype=float)),
                'direct_counts': (['prepared_state', 'count'], direct_counts),
                'gaussian_norms': (['prepared_state', 'gauss'], gaussian_norms),
                'outlier_probability': ('prepared_state', outlier_prob),
                'norm_res': ('prepared_state', norm_res),
            },
            coords={
                'prepared_state': np.arange(n_state),
                'idx_shot': np.arange(I_arr.shape[1]),
                'x': hist['x'].values,
                'y': hist['y'].values,
                'center': np.arange(mean.shape[0]),
                'comp': np.arange(2),
                'count': np.arange(direct_counts.shape[1]),
                'gauss': np.arange(gaussian_norms.shape[1]),
            },
            attrs={
                'trained_std': float(trained['std']),
                'trained_covariance': float(trained['covariance']),
                'lim_I_low': lim_I[0], 'lim_I_high': lim_I[1],
                'lim_Q_low': lim_Q[0], 'lim_Q_high': lim_Q[1],
            },
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate all 4 diagnostic plots, drawing entirely from plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)

        fig_raw, axes_raw = plot_prepared_state_scatter(plot_data)
        fig_2Dhist, axes_2Dhist = plot_2d_histogram(plot_data)
        fig_outliers, axes_outliers = plot_outliers(plot_data)
        fig_residue, axes_residue = plot_2d_fit_residue(plot_data)

        lim_I = (plot_data.attrs['lim_I_low'], plot_data.attrs['lim_I_high'])
        lim_Q = (plot_data.attrs['lim_Q_low'], plot_data.attrs['lim_Q_high'])

        for i in range(plot_data.sizes['prepared_state']):
            axis_formatter(axes_raw[i], lim_I, lim_Q, i)
            axis_formatter(axes_2Dhist[i], lim_I, lim_Q, i)
            axis_formatter(axes_outliers[i], lim_I, lim_Q, i)
            axis_formatter(axes_residue[i], lim_I, lim_Q, i)

        return {
            "raw": fig_raw,
            "2DHist": fig_2Dhist,
            "outliers": fig_outliers,
            "fit_residue": fig_residue,
        }

    # ==========================================
    # STATELESS HELPER METHODS
    # ==========================================
    
    def _preprocess_data(self, dataset: xr.Dataset, user_std=None):
        """Stateless helper to bin the raw data into 2D histograms."""
        prepared_states = dataset.coords['prepared_state'].values
        
        # Calculate standard deviations for bin sizing
        std_I = dataset['I'].std(dim='shot_idx').values
        std_Q = dataset['Q'].std(dim='shot_idx').values
        std_init = np.min([np.array([std_I[i], std_Q[i]]) for i in range(len(prepared_states))])
        
        step = (user_std if user_std else std_init) / 3
        # step = max(step, 1e-3)
        
        I_all, Q_all = dataset['I'].values.ravel(), dataset['Q'].values.ravel()
        xedges = np.arange(I_all.min(), I_all.max() + step, step)
        yedges = np.arange(Q_all.min(), Q_all.max() + step, step)
        if len(xedges) < 2: xedges = np.linspace(I_all.min(), I_all.max(), 2)
        if len(yedges) < 2: yedges = np.linspace(Q_all.min(), Q_all.max(), 2)
        
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        
        density_arr = np.zeros((len(prepared_states), len(ycenters), len(xcenters)))
        mean_init = []
        
        for i, state in enumerate(prepared_states):
            I, Q = dataset['I'].sel(prepared_state=state).values, dataset['Q'].sel(prepared_state=state).values
            H, _, _ = np.histogram2d(I, Q, bins=[xedges, yedges], density=True)
            density_arr[i, :, :] = H.T
            
            max_idx = np.unravel_index(np.argmax(H), H.shape)
            mean_init.append(np.array([xcenters[max_idx[0]], ycenters[max_idx[1]]]))

        hist_dataset = xr.Dataset(
            {'density': (['prepared_state', 'y', 'x'], density_arr)},
            coords={'prepared_state': prepared_states, 'x': xcenters, 'y': ycenters}
        )
        return hist_dataset, np.array(mean_init), std_init

    def _train_global_model(self, hist_dataset, mean_init, std_init, user_mean, user_std):
        """Stateless helper for global GMM fitting."""
        if user_mean is not None and user_std is not None:
            return {
                'mean': np.array(user_mean), 'std': user_std, 
                'covariance': user_std**2, 'amp': np.ones(len(user_mean))
            }

        density_all = np.sum(hist_dataset['density'].values, axis=0)
        x, y = hist_dataset['x'].values, hist_dataset['y'].values
        
        fit_result = self._do_lmfit_2dgaussian(density_all, x, y, user_mean or mean_init, user_std or std_init)
        return self._extract_params(fit_result, len(mean_init))

    def _fit_individual_states(self, hist_dataset, trained_paras):
        """Stateless helper to apply the global model to individual states."""
        fit_results, fit_residues_list, norm_res = [], [], []
        x, y = hist_dataset['x'].values, hist_dataset['y'].values
        
        for state in hist_dataset['prepared_state'].values:
            density = hist_dataset['density'].sel(prepared_state=state).values
            fit_result = self._do_lmfit_2dgaussian(density, x, y, trained_paras['mean'], trained_paras['std'])
            fit_results.append(self._extract_params(fit_result, len(trained_paras['mean'])))
            
            best_fit = fit_result.best_fit.reshape(density.shape)
            residue = density - best_fit
            fit_residues_list.append(residue)
            norm_res.append(np.nansum(residue) / np.nansum(density) if np.nansum(density) != 0 else np.nan)

        fit_residues = xr.DataArray(
            np.stack(fit_residues_list, axis=0),
            dims=["prepared_state", "y", "x"],
            coords={
                "prepared_state": hist_dataset["prepared_state"].values,
                "y": y, "x": x,
            },
        )
        return fit_results, fit_residues, norm_res

    def _calc_distances(self, dataset, mean_trained):
        """Stateless helper to map point distances to GMM centers."""
        prepared_states = dataset.coords['prepared_state'].values
        n_center, n_state, n_shot = len(mean_trained), len(prepared_states), dataset.sizes['shot_idx']
        dist_arr = np.zeros((n_center, n_state, n_shot))
        
        for i_center, mean in enumerate(mean_trained):
            for i_state, state in enumerate(prepared_states):
                I = dataset['I'].sel(prepared_state=state).values.ravel()
                Q = dataset['Q'].sel(prepared_state=state).values.ravel()
                dist_arr[i_center, i_state, :] = np.sqrt((I - mean[0])**2 + (Q - mean[1])**2)

        return xr.Dataset(
            {'distance': (['center', 'prepared_state', 'idx_shot'], dist_arr)},
            coords={'center': np.arange(n_center), 'prepared_state': prepared_states, 'idx_shot': np.arange(n_shot)}
        )

    def _do_lmfit_2dgaussian(self, density, x, y, mean, std):
        """Wrapper for the FitMultiGaussian2D utility."""
        n_gauss = len(mean)
        fitter = FitMultiGaussian2D(density, x, y, n_gauss=n_gauss)
        fitter.params['offset'].set(value=0, vary=False)
        for i in range(n_gauss):
            fitter.params[f'g{i}_x0'].set(value=mean[i][0], vary=False)
            fitter.params[f'g{i}_y0'].set(value=mean[i][1], vary=False)
            if i == 0:
                fitter.params[f'g{i}_sigma_x'].set(value=std, vary=False)
            else:
                fitter.params[f'g{i}_sigma_x'].set(expr='g0_sigma_x')
            fitter.params[f'g{i}_sigma_y'].set(expr='g0_sigma_x')
        return fitter.fit()

    def _extract_params(self, fit_result, n_gauss):
        """Unpacks lmfit results into a dictionary."""
        mean, amp = [], []
        for i in range(n_gauss):
            mean.append(np.array([fit_result.params[f'g{i}_x0'].value, fit_result.params[f'g{i}_y0'].value]))
            amp.append(fit_result.params[f'g{i}_amp'].value)
        std = fit_result.params['g0_sigma_x'].value
        return {'mean': np.array(mean), 'std': std, 'covariance': std**2, 'amp': np.array(amp)}