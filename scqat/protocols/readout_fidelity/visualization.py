"""
Readout-fidelity sweep plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ReadoutFidelityAnalyzer.build_plot_data`` and draws without any recalculation,
so the figures reproduce from the saved ``*_plotdata.nc`` alone. The swept axis
name is taken from ``plot_data.attrs['sweep_coord']`` (e.g. ``amp_prefactor`` or
``frequency``).

plot_data layout
----------------
coords : <sweep_coord>, ``center``, ``iq``, ``prepared_state``, ``gauss``, ``count``
vars   : ``std`` (sweep), ``mean`` (sweep, center, iq),
         ``p_outlier``/``norm_res`` (sweep, prepared_state),
         ``gaussian_norms`` (sweep, prepared_state, gauss),
         ``direct_counts`` (sweep, prepared_state, count)
attrs  : ``sweep_coord``
"""

import numpy as np
import matplotlib.pyplot as plt


def _sweep(plot_data):
    coord = plot_data.attrs['sweep_coord']
    return coord, plot_data.coords[coord].values


def plot_std_vs_sweep(plot_data):
    """Trained GMM standard deviation as a function of the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(sweep, plot_data['std'].values, 'o-')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('GMM std', fontsize=14)
    ax.set_title('State-discrimination std vs sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_outlier_vs_sweep(plot_data):
    """Outlier probability per prepared_state vs the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    p_out = plot_data['p_outlier'].values
    for p in range(p_out.shape[1]):
        ax.plot(sweep, p_out[:, p], 'o-', label=f'prepared_state={p}')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('Outlier probability', fontsize=14)
    ax.set_title('Outlier probability vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_mean_distance_vs_sweep(plot_data):
    """Euclidean distance between the two trained GMM centers vs the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    mean = plot_data['mean'].values  # (sweep, center, iq)
    if mean.shape[1] >= 2:
        dist = np.sqrt(np.sum((mean[:, 0, :] - mean[:, 1, :]) ** 2, axis=1))
        ax.plot(sweep, dist, 'o-')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('|center₀ − center₁|', fontsize=14)
    ax.set_title('GMM center separation vs sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_norm_res_vs_sweep(plot_data):
    """Normalised fit residue per prepared_state vs the sweep."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    nr = plot_data['norm_res'].values
    for p in range(nr.shape[1]):
        ax.plot(sweep, nr[:, p], 'o-', label=f'prepared_state={p}')
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('res / density', fontsize=14)
    ax.set_title('Normalised residue vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_fidelity_vs_sweep(plot_data):
    """Correct-assignment fidelity vs the sweep: the diagonal of ``direct_counts``
    (fraction of shots of prepared_state k assigned to label k), with the diagonal
    of ``gaussian_norms`` overlaid dashed when available."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    if 'direct_counts' in plot_data:
        dc = plot_data['direct_counts'].values  # (sweep, prepared_state, count)
        n = min(dc.shape[1], dc.shape[2])
        for k in range(n):
            ax.plot(sweep, dc[:, k, k], 'o-', label=f'direct counts state {k}')

    if 'gaussian_norms' in plot_data:
        gn = plot_data['gaussian_norms'].values  # (sweep, prepared_state, gauss)
        n = min(gn.shape[1], gn.shape[2])
        for k in range(n):
            ax.plot(sweep, gn[:, k, k], '--', alpha=0.7, label=f'gaussian norm state {k}')

    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel('Fidelity (correct assignment)', fontsize=14)
    ax.set_title('Readout fidelity vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_means_on_iq_plane(plot_data):
    """Trained GMM centers in the I/Q plane, coloured by the sweep value."""
    coord, sweep = _sweep(plot_data)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)
    mean = plot_data['mean'].values  # (sweep, center, iq)
    sc = None
    for c in range(mean.shape[1]):
        sc = ax.scatter(mean[:, c, 0], mean[:, c, 1], c=sweep, cmap='viridis',
                        s=40, marker='o' if c == 0 else '^', label=f'center {c}')
    if sc is not None:
        fig.colorbar(sc, ax=ax, label=coord)
    ax.set_xlabel('I', fontsize=14)
    ax.set_ylabel('Q', fontsize=14)
    ax.set_title('GMM centers on IQ plane vs sweep')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
