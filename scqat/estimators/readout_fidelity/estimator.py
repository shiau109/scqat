import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.state_discrimination import StateDiscriminationEstimator
from scqat.estimators.readout_fidelity.visualization import (
    plot_outlier_vs_sweep,
    plot_std_vs_sweep,
    plot_snr_vs_sweep,
    plot_mean_distance_vs_sweep,
    plot_mean_i_vs_sweep,
    plot_mean_q_vs_sweep,
    plot_norm_res_vs_sweep,
    plot_fidelity_vs_sweep,
    plot_means_on_iq_plane,
)


class ReadoutFidelityEstimator(BaseEstimator):
    """
    Readout-fidelity sweep: run state discrimination at every point of a swept
    readout parameter, summarise how the discrimination quality evolves, and
    report the sweep value that maximises the readout fidelity.

    Expects an xarray.Dataset with:
        - Variables:   ``I``, ``Q``
        - Coordinates: ``shot_idx``, ``prepared_state`` (required by the inner
          :class:`StateDiscriminationEstimator`)
        - Coordinate:  the swept axis named by :attr:`sweep_coord`

    For each value of ``sweep_coord`` the data is sliced and handed to a
    :class:`StateDiscriminationEstimator`; the per-slice trained GMM std/means,
    outlier probability, normalised residue, Gaussian norms and direct counts are
    collected as a function of the sweep. The **fidelity** at each point is the
    mean of the confusion-matrix diagonal (``direct_counts[k, k]`` = fraction of
    prepared-state-k shots assigned to label k), and the reported answer
    (``best_sweep_value`` / ``best_fidelity``) is the point that maximises it.

    This unifies qcat's near-duplicate ``ROFidelityPower`` (``amp_prefactor``) and
    ``ROFidelityFreq`` (``frequency``); use the :class:`ReadoutPowerFidelityEstimator`
    / :class:`ReadoutFreqFidelityEstimator` subclasses, or set ``sweep_coord``
    directly. (The power-specific linear mean-drift refit from qcat is not ported
    here — see MIGRATION.md.)
    """

    estimator_name = "readout_fidelity"
    sweep_coord: Optional[str] = None  # subclasses set this; or pass sweep_coord kwarg
    fidelity_floor: float = 0.5  # below this the best point is flagged unsuccessful

    # ------------------------------------------------------------------
    def _resolve_coord(self, kwargs: Dict[str, Any]) -> str:
        coord = kwargs.get('sweep_coord', self.sweep_coord)
        if coord is None:
            raise ValueError(
                "ReadoutFidelityEstimator needs a sweep coordinate: set the "
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
        Run state discrimination per sweep point, collect the summary curves, and
        select the fidelity-optimal sweep value.

        Kwargs:
            sweep_coord (str): Override the swept coordinate name.
            user_std / user_mean / outlier_sigma: Forwarded to
                :class:`StateDiscriminationEstimator`.
            (subclasses may consume further kwargs, e.g. ``outliers_threshold``.)

        Returns the sweep axis, per-sweep arrays, and the best point:
            sweep_coord (name), sweep_values (S,),
            std (S,), mean (S, center, iq), p_outlier (S, prepared_state),
            norm_res (S, prepared_state), gaussian_norms (S, prepared_state, gauss),
            direct_counts (S, prepared_state, count), fidelity (S,), snr (S,),
            failed (S,), best_index, best_sweep_value, best_fidelity, success.
        """
        coord = self._resolve_coord(kwargs)
        sd_kwargs = {k: kwargs[k] for k in ('user_std', 'user_mean', 'outlier_sigma') if k in kwargs}

        sweep_values = np.asarray(dataset.coords[coord].values)
        sd = StateDiscriminationEstimator()

        std_list, mean_list, p_outlier_list, norm_res_list = [], [], [], []
        gaussian_norms_list, direct_counts_list, failed_list = [], [], []

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
                failed_list.append(False)
            except Exception:
                std_list.append(np.nan)
                mean_list.append(None)
                p_outlier_list.append(None)
                norm_res_list.append(None)
                gaussian_norms_list.append(None)
                direct_counts_list.append(None)
                failed_list.append(True)

        # Determine common shapes from the first successful slice, then stack
        # (filling failed slices with NaN of the right shape).
        direct_counts = self._stack(direct_counts_list)
        results: Dict[str, Any] = {
            'sweep_coord': coord,
            'sweep_values': sweep_values,
            'std': np.asarray(std_list, dtype=float),
            'mean': self._stack(mean_list),
            'p_outlier': self._stack(p_outlier_list),
            'norm_res': self._stack(norm_res_list),
            'gaussian_norms': self._stack(gaussian_norms_list),
            'direct_counts': direct_counts,
            'fidelity': self._fidelity_curve(direct_counts),
            'failed': np.asarray(failed_list, dtype=bool),
        }
        results['snr'] = self._snr_curve(results['mean'], results['std'])
        self._set_best(results, **kwargs)
        return results

    @staticmethod
    def _stack(items: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """Stack a list of equal-shaped arrays, substituting NaN arrays for any
        ``None`` (failed slice). Returns ``None`` if every entry failed."""
        shape = next((a.shape for a in items if a is not None), None)
        if shape is None:
            return None
        filled = [a if a is not None else np.full(shape, np.nan) for a in items]
        return np.stack(filled, axis=0)

    @staticmethod
    def _fidelity_curve(direct_counts: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Per-sweep readout fidelity: mean of the confusion-matrix diagonal
        ``direct_counts[s, k, k]`` over the available states. Failed slices (all
        NaN) yield NaN."""
        if direct_counts is None:
            return None
        _, n_state, n_count = direct_counts.shape
        n = min(n_state, n_count)
        diag = direct_counts[:, np.arange(n), np.arange(n)]  # (S, n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows -> NaN
            return np.nanmean(diag, axis=1)

    @staticmethod
    def _snr_curve(mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Per-sweep readout SNR: the |center₁ − center₀| separation in units of one
        blob's GMM std. ``None`` when centers are unavailable or there are fewer than two;
        failed slices (NaN std/mean) yield NaN."""
        if mean is None or std is None or mean.shape[1] < 2:
            return None
        sep = np.linalg.norm(mean[:, 0, :] - mean[:, 1, :], axis=1)  # (S,)
        with np.errstate(divide='ignore', invalid='ignore'):
            return sep / np.asarray(std, dtype=float)

    # --- best-point selection (overridable) ---------------------------
    def _set_best(self, results: Dict[str, Any], **kwargs) -> None:
        """Populate best_index / best_sweep_value / best_fidelity / success."""
        idx = self._select_best_index(results, **kwargs)
        if idx is None:
            results.update(best_index=None, best_sweep_value=None,
                           best_fidelity=None, success=False)
            return
        best_fid = float(results['fidelity'][idx])
        ok = self._selection_ok(results, idx, **kwargs)
        results.update(
            best_index=int(idx),
            best_sweep_value=float(results['sweep_values'][idx]),
            best_fidelity=best_fid,
            success=bool(ok and np.isfinite(best_fid) and best_fid >= self.fidelity_floor),
        )

    def _select_best_index(self, results: Dict[str, Any], **kwargs) -> Optional[int]:
        """Index of the fidelity-maximising sweep point (NaN-safe), or ``None``
        when no point yielded a finite fidelity."""
        fidelity = results.get('fidelity')
        if fidelity is None or not np.any(np.isfinite(fidelity)):
            return None
        return int(np.nanargmax(fidelity))

    def _selection_ok(self, results: Dict[str, Any], idx: int, **kwargs) -> bool:
        """Whether the selected point satisfies any subclass constraint. Base
        imposes none (always True)."""
        return True

    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the small answer — the sweep axis, the fidelity curve, and the
        chosen best point — and drop the bulky per-slice arrays (mean, p_outlier,
        norm_res, gaussian_norms, direct_counts), which live in the plot data."""
        keep = (
            'sweep_coord', 'sweep_values', 'fidelity', 'snr',
            'best_index', 'best_sweep_value', 'best_fidelity', 'success',
        )
        return {k: results.get(k) for k in keep}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Assemble the per-sweep summary curves into one self-sufficient Dataset
        so the figures redraw from the saved ``*_plotdata.nc`` alone. The swept
        coordinate name is preserved both as the dimension name and in
        ``attrs['sweep_coord']``; the chosen best point rides along in attrs."""
        coord = results['sweep_coord']
        sweep = np.asarray(results['sweep_values'])

        data_vars: Dict[str, Any] = {'std': (coord, np.asarray(results['std'], dtype=float))}
        coords: Dict[str, Any] = {coord: sweep}

        fidelity = results.get('fidelity')
        if fidelity is not None:
            data_vars['fidelity'] = (coord, np.asarray(fidelity, dtype=float))

        snr = results.get('snr')
        if snr is not None:
            data_vars['snr'] = (coord, np.asarray(snr, dtype=float))

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

        attrs: Dict[str, Any] = {'sweep_coord': coord}
        if results.get('best_sweep_value') is not None:
            attrs['best_sweep_value'] = float(results['best_sweep_value'])
            attrs['best_fidelity'] = float(results['best_fidelity'])
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

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
        if 'snr' in plot_data:
            figs['snr'] = plot_snr_vs_sweep(plot_data)
        if 'p_outlier' in plot_data:
            figs['outlier'] = plot_outlier_vs_sweep(plot_data)
        if 'mean' in plot_data:
            figs['mean_distance'] = plot_mean_distance_vs_sweep(plot_data)
            figs['mean_I'] = plot_mean_i_vs_sweep(plot_data)
            figs['mean_Q'] = plot_mean_q_vs_sweep(plot_data)
            figs['means_on_IQ'] = plot_means_on_iq_plane(plot_data)
        if 'norm_res' in plot_data:
            figs['norm_res'] = plot_norm_res_vs_sweep(plot_data)
        if any(v in plot_data for v in ('fidelity', 'direct_counts', 'gaussian_norms')):
            figs['fidelity'] = plot_fidelity_vs_sweep(plot_data)
        return figs


class ReadoutPowerFidelityEstimator(ReadoutFidelityEstimator):
    """Readout fidelity swept over readout amplitude (``amp_prefactor``).

    ``best_sweep_value`` is the optimal **amp_prefactor** — a multiplier on the
    current readout-pulse amplitude. The optimum is the fidelity-maximising point
    among amplitudes that keep the outlier population in check: pass
    ``outliers_threshold`` (e.g. 0.98) and only points whose in-distribution
    fraction ``1 - max_k p_outlier`` meets it are eligible; if none qualify the
    global fidelity maximum is returned with ``success=False``.

    Ported from qcat ``readout_power.ROFidelityPower`` (its linear mean-drift refit
    ``fit_means_vs_amp_prefactor`` is intentionally not ported — see MIGRATION.md).
    """
    estimator_name = "readout_power_fidelity"
    sweep_coord = "amp_prefactor"

    def _candidate_mask(self, results: Dict[str, Any], **kwargs) -> np.ndarray:
        """Boolean mask of sweep points allowed by ``outliers_threshold``: a point
        qualifies when its in-distribution fraction ``1 - max_k p_outlier`` is at
        least the threshold. With no threshold, all finite-fidelity points pass."""
        fidelity = results.get('fidelity')
        finite = np.isfinite(fidelity)
        threshold = kwargs.get('outliers_threshold')
        p_outlier = results.get('p_outlier')
        if threshold is None or p_outlier is None:
            return finite
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN rows -> NaN
            in_dist = 1.0 - np.nanmax(p_outlier, axis=1)  # (S,)
        return finite & (in_dist >= threshold)

    def _select_best_index(self, results: Dict[str, Any], **kwargs) -> Optional[int]:
        fidelity = results.get('fidelity')
        if fidelity is None or not np.any(np.isfinite(fidelity)):
            return None
        mask = self._candidate_mask(results, **kwargs)
        if np.any(mask):
            return int(np.argmax(np.where(mask, fidelity, -np.inf)))
        return int(np.nanargmax(fidelity))  # constraint unmet -> global best

    def _selection_ok(self, results: Dict[str, Any], idx: int, **kwargs) -> bool:
        return bool(self._candidate_mask(results, **kwargs)[idx])


class ReadoutFreqFidelityEstimator(ReadoutFidelityEstimator):
    """Readout fidelity swept over readout frequency (``frequency``).

    ``best_sweep_value`` is the optimal **detuning** (Hz), expressed relative to the
    current readout IF the sweep was centred on; the consuming node maps it onto an
    absolute readout frequency. The optimum is simply the fidelity-maximising point.

    Ported from qcat ``readout_freq.ROFidelityFreq``.
    """
    estimator_name = "readout_freq_fidelity"
    sweep_coord = "frequency"
