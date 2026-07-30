from typing import Any, Dict, Optional, Tuple

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.tools.discriminate import discriminate_states, validate_discriminate_kwargs
from scqat.estimators.state_discrimination.visualization import (
    plot_prepared_state_scatter,
    plot_2d_histogram,
    plot_outliers,
    plot_2d_fit_residue,
    axis_formatter,
)


def state_iq_arrays(dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve a discrimination Dataset into the ``(n_prepared_state, n_shot)``
    I/Q arrays :func:`scqat.tools.discriminate.discriminate_states` consumes
    (row order = the dataset's ``prepared_state`` order). Shared by every
    estimator in the discrimination family."""
    states = dataset.coords["prepared_state"].values
    I = np.stack([dataset["I"].sel(prepared_state=s).values.ravel() for s in states])
    Q = np.stack([dataset["Q"].sel(prepared_state=s).values.ravel() for s in states])
    return I, Q


def _axis_limits(values: np.ndarray, centres: np.ndarray,
                 margin: float) -> Tuple[float, float]:
    """One shared plot axis: the pooled 5-sigma shot spread, WIDENED to hold
    every trained centre plus ``margin``.

    The pooled spread alone is dominated by the blob SEPARATION only while the
    prepared states are comparably populated. When one state holds nearly every
    shot — a ground-state-only thermal-population run, where the minority blob
    is the whole measurement — it collapses to the blob WIDTH and crops that
    blob out of the figure, hiding the evidence for the reported population.
    ``margin`` should cover the n-sigma circles the figures draw (3 sigma).

    Limits only ever GROW here, so a balanced run keeps the window it had.
    """
    low = float(values.mean() - 5 * values.std())
    high = float(values.mean() + 5 * values.std())
    if len(centres):
        low = min(low, float(np.min(centres) - margin))
        high = max(high, float(np.max(centres) + margin))
    return low, high


class StateDiscriminationEstimator(BaseEstimator):
    """
    Analyzes I/Q plane data for superconducting qubit state discrimination
    using 2D Multi-Gaussian Mixture Models.

    The heavy lifting is the family-shared reduction
    :func:`scqat.tools.discriminate.discriminate_states`; this estimator only
    resolves the dataset into arrays, forwards its flat kwarg surface, and owns
    the artifacts (metadata / plot data / figures).
    """

    estimator_name = "state_discrimination"

    def _check_data(self, dataset: xr.Dataset) -> None:
        """Ensures the dataset has the required coordinates and variables."""
        for coords_name in ["shot_idx", "prepared_state"]:
            if coords_name not in dataset.coords:
                raise ValueError(f"State Discrimination requires '{coords_name}' coordinate in the dataset.")
        for var in ("I", "Q"):
            if var not in dataset:
                raise ValueError(f"State Discrimination requires an '{var}' variable in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Executes the GMM fitting and population counting.

        Kwargs — flat and fully owned; unknown names raise:
            user_mean (list): Optional initial guess for GMM centers.
            user_std (float): Optional initial guess for Gaussian std dev.
            outlier_sigma (float): Threshold for outlier detection (default: 3).
        """
        validate_discriminate_kwargs(kwargs)
        I, Q = state_iq_arrays(dataset)
        res = discriminate_states(I, Q, **kwargs)

        # Re-wrap the array outputs in the xarray shapes this estimator's
        # artifacts (plot data / figures) are built from.
        states = dataset.coords["prepared_state"].values
        hist_dataset = xr.Dataset(
            {"density": (["prepared_state", "y", "x"], res["density"])},
            coords={"prepared_state": states, "x": res["hist_x"], "y": res["hist_y"]},
        )
        fit_residues = xr.DataArray(
            res["fit_residues"],
            dims=["prepared_state", "y", "x"],
            coords={"prepared_state": states, "y": res["hist_y"], "x": res["hist_x"]},
        )

        return {
            'trained_paras': res["trained_paras"],
            'fitted_paras': res["fitted_paras"],
            'gaussian_norms': res["gaussian_norms"],
            'direct_counts': res["direct_counts"],
            'state_label': res["state_label"],
            'outlier_mask': res["outlier_mask"],
            'outlier_probability': res["outlier_probability"],
            'norm_res': res["norm_res"],
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
        norm / outlier summaries. Shared I/Q axis limits live in ``.attrs``
        (:func:`_axis_limits` — the 5σ shot spread, widened to keep every trained
        centre and its circles in frame).
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

        lim_I = _axis_limits(I_arr.ravel(), mean[:, 0], 4 * float(trained['std']))
        lim_Q = _axis_limits(Q_arr.ravel(), mean[:, 1], 4 * float(trained['std']))

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

