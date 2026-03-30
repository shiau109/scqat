import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_outliers(data, outlier_mask, analysis_result=None):
    """
    Plot scatter plots of I vs Q for each prepared_state, showing only the outlier points as defined by outlier_mask.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        outlier_mask (dict): Dictionary mapping prepared_state index to boolean mask array (same length as shot_idx for that state).
    Returns:
        fig: matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)

    for i in range(2):
        mask = outlier_mask[i]

        # Extract I and Q for both prepared states
        I_vals = data['I'].sel(prepared_state=i).values[mask]
        Q_vals = data['Q'].sel(prepared_state=i).values[mask]
        axes[i].scatter(I_vals, Q_vals, s=10, alpha=0.8, color='orange', marker='o', edgecolor='none', label='Outlier')

        # Optionally plot mean as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'std' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'outlier_probability' in analysis_result:
                outlier_prob = analysis_result['outlier_probability']
                text_msg = f"Outlier prob.: {outlier_prob[i]:.3e}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    fig.tight_layout()
    plt.close(fig)
    return fig, axes

def plot_prepared_state_scatter(data, analysis_result=None):
    """
    Plot two scatter plots of I vs Q for prepared_state=0 and prepared_state=1, sharing the same axis limits.
    Optionally plot GMM mean as black dots if analysis_result is provided.
    Args:
        data (xr.Dataset): Dataset with variables 'I', 'Q', coords 'shot_idx', 'prepared_state'.
        analysis_result (dict, optional): Dictionary with GMM parameters (expects 'mean').
    Returns:
        fig: matplotlib Figure
    """
    I_list = []
    Q_list = []
    for i in range(2):
        # Extract I and Q for both prepared states
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150)
    for i in range(2):
        # Default color is blue, but if state_label is provided, use it for coloring
        if analysis_result is not None and 'state_label' in analysis_result:
            labels = np.array(analysis_result['state_label'][i])
            # Use a colormap for 2 classes
            cmap = plt.get_cmap('coolwarm')
            colors = cmap(labels / (labels.max() if labels.max() > 0 else 1))
            axes[i].scatter(I_list[i], Q_list[i], s=6, alpha=0.7, c=colors, marker='o', edgecolor='none')
        else:
            axes[i].scatter(I_list[i], Q_list[i], s=1, alpha=0.5, color='blue')

        # Optionally plot GMM mean as black dots
        if analysis_result is not None:

            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'covariance' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    fig.tight_layout()
    plt.close(fig)
    return fig, axes

def plot_2d_histogram(hist_dataset, analysis_result=None):
    """
    Plot 2D histogram (density) for each prepared_state from hist_dataset.
    If analysis_result is provided, overlay GMM mean and covariance.
    Args:
        hist_dataset (xr.Dataset): Dataset with variable 'density' and coords 'prepared_state', 'x', 'y'.
        analysis_result (dict, optional): GMM fit results to overlay.
    Returns:
        fig: matplotlib Figure
    """
    from matplotlib.colors import LogNorm
    n_states = hist_dataset.sizes['prepared_state']
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5), dpi=150)
    if n_states == 1:
        axes = [axes]

    x = hist_dataset['x'].values
    y = hist_dataset['y'].values
    xedges = np.concatenate([x - (x[1] - x[0])/2, [x[-1] + (x[1] - x[0])/2]]) if len(x) > 1 else np.array([x[0]-0.5, x[0]+0.5])
    yedges = np.concatenate([y - (y[1] - y[0])/2, [y[-1] + (y[1] - y[0])/2]]) if len(y) > 1 else np.array([y[0]-0.5, y[0]+0.5])
    for i, state in enumerate(hist_dataset.coords['prepared_state'].values):
        density = hist_dataset['density'].sel(prepared_state=state).values  # shape (len(y), len(x))
        # Mask zero values so they are not shown in the log plot
        density_masked = np.ma.masked_where(density <= 0, density)
        pcm = axes[i].pcolormesh(xedges, yedges, density_masked, shading='auto', cmap='viridis', norm=LogNorm())
        axes[i].set_title(f"prepared_state={state}")
        axes[i].set_xlabel('I')
        axes[i].set_ylabel('Q')

        # Optionally plot GMM mean as black dots
        if analysis_result is not None:
            trained_paras = analysis_result.get('trained_paras', None)
            if trained_paras is not None and 'mean' in trained_paras:
                mean = trained_paras['mean']
                plot_gmm_mean_on_axes(axes[i], mean)
            if trained_paras is not None and 'covariance' in trained_paras:
                plot_gmm_circles_on_axis(axes[i], trained_paras)

            y_offset = 0.98
            if 'direct_counts' in analysis_result:
                direct_counts = analysis_result['direct_counts']
                text_msg = f"Direct counts:\n{direct_counts[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )
                y_offset -= 0.18  # Move next box down
            if 'gaussian_norms' in analysis_result:
                gaussian_norms = analysis_result['gaussian_norms']
                text_msg = f"Gaussian norms:\n{gaussian_norms[i]}"
                axes[i].text(
                    0.02, y_offset, text_msg,
                    transform=axes[i].transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
                )

    plt.tight_layout()
    plt.close(fig)
    return fig, axes


def plot_gmm_mean_on_axes(axes, mean):
    """
    Plot GMM mean as black dots on each axis in axes.
    Args:
        axes: list of matplotlib Axes
        mean: array-like, shape (2, 2), GMM mean for two components
    """
    axes.scatter(mean[0][0], mean[0][1], c='k', s=40, marker='o')
    axes.scatter(mean[1][0], mean[1][1], c='k', s=40, marker='o')

def plot_gmm_circles_on_axis(axes, analysis_result, n_std=[1,2,3], **circle_kwargs):
    """
    Plot GMM mean as centers and covariance as radii (dashed circles) on a given axis.
    Args:
        axes: matplotlib Axes
        analysis_result: dict, expects 'mean' (N,2) and 'covariance' (N,) from GMM
        n_std: list or float, number(s) of standard deviations for the radius
        circle_kwargs: additional kwargs for Ellipse
    """
    mean = analysis_result['mean']
    std = analysis_result['std']
    # Accept n_std as a list or a single float
    if isinstance(n_std, (int, float)):
        n_std_list = [n_std]
    else:
        n_std_list = list(n_std)
    for i in range(mean.shape[0]):
        center = mean[i]
        for n in n_std_list:
            radius = std * n
            circle = Ellipse(xy=center, width=2*radius, height=2*radius, angle=0,
                             edgecolor='k', facecolor='none', linestyle='--', linewidth=1.5, **circle_kwargs)
            axes.add_patch(circle)

def axis_formatter(axes, lim_I, lim_Q, i):
    from matplotlib.ticker import ScalarFormatter
    axes.set_xlim(lim_I)
    axes.set_ylim(lim_Q)
    axes.set_aspect('equal')
    # Use ScalarFormatter for scientific notation
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    # Force scientific notation if needed
    axes.ticklabel_format(style='sci', axis='both', scilimits=(-2,2))
    # Get offset text (exponential part)
    axes.xaxis.offsetText.set_visible(True)
    axes.yaxis.offsetText.set_visible(True)
    # Set axis labels with exponent if present
    xlabel = r"$I$"
    ylabel = r"$Q$"
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(f'prepared_state={i}')

def compute_shared_axis_limits(data, n_std=5):
    """
    Compute shared axis limits for I and Q from a dataset with 'prepared_state' axis.
    Args:
        data: xarray Dataset with variables 'I', 'Q', and 'prepared_state' coordinate
        n_std: number of standard deviations for the axis limits (default 5)
    Returns:
        lim_I: tuple (min, max) for I axis
        lim_Q: tuple (min, max) for Q axis
    """
    I_list = []
    Q_list = []
    for i in range(2):
        I_list.append(data['I'].sel(prepared_state=i).values)
        Q_list.append(data['Q'].sel(prepared_state=i).values)
    all_I = np.concatenate(I_list)
    all_Q = np.concatenate(Q_list)
    I_mean, Q_mean = np.mean(all_I), np.mean(all_Q)
    I_std, Q_std = np.std(all_I), np.std(all_Q)
    lim_I = (I_mean - n_std*I_std, I_mean + n_std*I_std)
    lim_Q = (Q_mean - n_std*Q_std, Q_mean + n_std*Q_std)
    return lim_I, lim_Q

def plot_2d_fit_residue(fit_residues, norm_res):
    """
    Plot 2D fit residue (difference between density and best fit) for each prepared_state.
    Args:
        fit_residues (xr.DataArray): Residue arrays with dims (prepared_state, y, x).
        norm_res (list or array): Normalized residue values for each prepared_state.
    Returns:
        fig: matplotlib Figure
        axes: matplotlib Axes
    """
    n_states = fit_residues.sizes['prepared_state']
    fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5), dpi=150)
    if n_states == 1:
        axes = [axes]
    x = fit_residues['x'].values
    y = fit_residues['y'].values
    xedges = np.concatenate([x - (x[1] - x[0])/2, [x[-1] + (x[1] - x[0])/2]]) if len(x) > 1 else np.array([x[0]-0.5, x[0]+0.5])
    yedges = np.concatenate([y - (y[1] - y[0])/2, [y[-1] + (y[1] - y[0])/2]]) if len(y) > 1 else np.array([y[0]-0.5, y[0]+0.5])
    vmin = float(np.nanmin(fit_residues.values))
    vmax = float(np.nanmax(fit_residues.values))
    absmax = max(abs(vmin), abs(vmax))
    for i, state in enumerate(fit_residues['prepared_state'].values):
        residue = fit_residues.sel(prepared_state=state).values
        pcm = axes[i].pcolormesh(xedges, yedges, residue, shading='auto', cmap='bwr', vmin=-absmax, vmax=absmax)
        axes[i].set_title(f"Fit Residue prepared_state={state}")
        axes[i].set_xlabel('I')
        axes[i].set_ylabel('Q')
        fig.colorbar(pcm, ax=axes[i], label='Residue (density - fit)')
        # Show normalized sum of absolute residues over density as a text box
        axes[i].text(
            0.02, 0.98,
            f"res/density: {norm_res[i]:.3e}",
            transform=axes[i].transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    plt.tight_layout()
    plt.close(fig)
    return fig, axes
