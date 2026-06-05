"""
State-discrimination plotting helpers.

Every function consumes the **plot_data** Dataset built by
``StateDiscriminationAnalyzer.build_plot_data`` and draws without any
recalculation, so the four figures can be reproduced from the saved
``*_plotdata.nc`` file alone.

plot_data layout
----------------
coords : ``prepared_state``, ``idx_shot``, ``x``, ``y``, ``center``, ``comp``,
         ``count``, ``gauss``
vars   : ``I``/``Q``/``state_label``/``outlier_mask`` (prepared_state, idx_shot),
         ``density``/``fit_residue`` (prepared_state, y, x),
         ``trained_mean`` (center, comp), ``trained_amp`` (center),
         ``direct_counts`` (prepared_state, count),
         ``gaussian_norms`` (prepared_state, gauss),
         ``outlier_probability``/``norm_res`` (prepared_state)
attrs  : ``trained_std``, ``trained_covariance``,
         ``lim_I_low``/``lim_I_high``/``lim_Q_low``/``lim_Q_high``
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def _trained(plot_data):
    """Reconstruct the trained-GMM dict the overlay helpers expect."""
    return {
        "mean": plot_data["trained_mean"].values,
        "std": plot_data.attrs["trained_std"],
    }


def _as_axes_list(axes, n):
    """Normalize plt.subplots return into an indexable list."""
    return [axes] if n == 1 else list(axes)


def plot_prepared_state_scatter(plot_data):
    """Scatter of I vs Q per prepared_state, coloured by assigned state label."""
    n_state = plot_data.sizes["prepared_state"]
    fig, axes = plt.subplots(1, n_state, figsize=(4 * n_state, 4), dpi=150)
    axes = _as_axes_list(axes, n_state)

    trained = _trained(plot_data)
    mean = trained["mean"]
    cmap = plt.get_cmap("coolwarm")

    for i in range(n_state):
        I = plot_data["I"].isel(prepared_state=i).values
        Q = plot_data["Q"].isel(prepared_state=i).values
        labels = plot_data["state_label"].isel(prepared_state=i).values
        colors = cmap(labels / (labels.max() if labels.max() > 0 else 1))
        axes[i].scatter(I, Q, s=6, alpha=0.7, c=colors, marker="o", edgecolor="none")

        plot_gmm_mean_on_axes(axes[i], mean)
        plot_gmm_circles_on_axis(axes[i], trained)

        y_offset = 0.98
        axes[i].text(
            0.02, y_offset, f"Direct counts:\n{plot_data['direct_counts'].isel(prepared_state=i).values}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        axes[i].text(
            0.02, y_offset - 0.18, f"Gaussian norms:\n{plot_data['gaussian_norms'].isel(prepared_state=i).values}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()
    plt.close(fig)
    return fig, axes


def plot_outliers(plot_data):
    """Scatter of only the outlier I/Q points per prepared_state."""
    n_state = plot_data.sizes["prepared_state"]
    fig, axes = plt.subplots(1, n_state, figsize=(4 * n_state, 4), dpi=150)
    axes = _as_axes_list(axes, n_state)

    trained = _trained(plot_data)
    mean = trained["mean"]

    for i in range(n_state):
        mask = plot_data["outlier_mask"].isel(prepared_state=i).values.astype(bool)
        I = plot_data["I"].isel(prepared_state=i).values[mask]
        Q = plot_data["Q"].isel(prepared_state=i).values[mask]
        axes[i].scatter(I, Q, s=10, alpha=0.8, color="orange", marker="o",
                        edgecolor="none", label="Outlier")

        plot_gmm_mean_on_axes(axes[i], mean)
        plot_gmm_circles_on_axis(axes[i], trained)

        p_out = float(plot_data["outlier_probability"].isel(prepared_state=i).values)
        axes[i].text(
            0.02, 0.98, f"Outlier prob.: {p_out:.3e}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()
    plt.close(fig)
    return fig, axes


def plot_2d_histogram(plot_data):
    """2D histogram (density) per prepared_state with GMM mean/covariance overlay."""
    from matplotlib.colors import LogNorm

    n_state = plot_data.sizes["prepared_state"]
    fig, axes = plt.subplots(1, n_state, figsize=(6 * n_state, 5), dpi=150)
    axes = _as_axes_list(axes, n_state)

    x = plot_data["x"].values
    y = plot_data["y"].values
    xedges = (np.concatenate([x - (x[1] - x[0]) / 2, [x[-1] + (x[1] - x[0]) / 2]])
              if len(x) > 1 else np.array([x[0] - 0.5, x[0] + 0.5]))
    yedges = (np.concatenate([y - (y[1] - y[0]) / 2, [y[-1] + (y[1] - y[0]) / 2]])
              if len(y) > 1 else np.array([y[0] - 0.5, y[0] + 0.5]))

    trained = _trained(plot_data)
    mean = trained["mean"]

    for i in range(n_state):
        density = plot_data["density"].isel(prepared_state=i).values
        density_masked = np.ma.masked_where(density <= 0, density)
        axes[i].pcolormesh(xedges, yedges, density_masked, shading="auto",
                           cmap="viridis", norm=LogNorm())
        axes[i].set_title(f"prepared_state={i}")
        axes[i].set_xlabel("I")
        axes[i].set_ylabel("Q")

        plot_gmm_mean_on_axes(axes[i], mean)
        plot_gmm_circles_on_axis(axes[i], trained)

        y_offset = 0.98
        axes[i].text(
            0.02, y_offset, f"Direct counts:\n{plot_data['direct_counts'].isel(prepared_state=i).values}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        axes[i].text(
            0.02, y_offset - 0.18, f"Gaussian norms:\n{plot_data['gaussian_norms'].isel(prepared_state=i).values}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.close(fig)
    return fig, axes


def plot_2d_fit_residue(plot_data):
    """2D fit residue (density - best fit) per prepared_state."""
    n_state = plot_data.sizes["prepared_state"]
    fig, axes = plt.subplots(1, n_state, figsize=(6 * n_state, 5), dpi=150)
    axes = _as_axes_list(axes, n_state)

    x = plot_data["x"].values
    y = plot_data["y"].values
    xedges = (np.concatenate([x - (x[1] - x[0]) / 2, [x[-1] + (x[1] - x[0]) / 2]])
              if len(x) > 1 else np.array([x[0] - 0.5, x[0] + 0.5]))
    yedges = (np.concatenate([y - (y[1] - y[0]) / 2, [y[-1] + (y[1] - y[0]) / 2]])
              if len(y) > 1 else np.array([y[0] - 0.5, y[0] + 0.5]))

    residues = plot_data["fit_residue"].values
    absmax = max(abs(float(np.nanmin(residues))), abs(float(np.nanmax(residues))))

    for i in range(n_state):
        residue = plot_data["fit_residue"].isel(prepared_state=i).values
        pcm = axes[i].pcolormesh(xedges, yedges, residue, shading="auto",
                                 cmap="bwr", vmin=-absmax, vmax=absmax)
        axes[i].set_title(f"Fit Residue prepared_state={i}")
        axes[i].set_xlabel("I")
        axes[i].set_ylabel("Q")
        fig.colorbar(pcm, ax=axes[i], label="Residue (density - fit)")
        norm_res = float(plot_data["norm_res"].isel(prepared_state=i).values)
        axes[i].text(
            0.02, 0.98, f"res/density: {norm_res:.3e}",
            transform=axes[i].transAxes, fontsize=10, verticalalignment="top",
            horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.close(fig)
    return fig, axes


def plot_gmm_mean_on_axes(axes, mean):
    """Plot GMM means as black dots on the given axis."""
    for c in range(mean.shape[0]):
        axes.scatter(mean[c][0], mean[c][1], c="k", s=40, marker="o")


def plot_gmm_circles_on_axis(axes, trained, n_std=(1, 2, 3), **circle_kwargs):
    """Plot dashed n-sigma circles around each GMM mean."""
    mean = trained["mean"]
    std = trained["std"]
    n_std_list = [n_std] if isinstance(n_std, (int, float)) else list(n_std)
    for i in range(mean.shape[0]):
        center = mean[i]
        for n in n_std_list:
            radius = std * n
            circle = Ellipse(xy=center, width=2 * radius, height=2 * radius, angle=0,
                             edgecolor="k", facecolor="none", linestyle="--",
                             linewidth=1.5, **circle_kwargs)
            axes.add_patch(circle)


def axis_formatter(axes, lim_I, lim_Q, i):
    from matplotlib.ticker import ScalarFormatter
    axes.set_xlim(lim_I)
    axes.set_ylim(lim_Q)
    axes.set_aspect("equal")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axes.xaxis.set_major_formatter(formatter)
    axes.yaxis.set_major_formatter(formatter)
    axes.ticklabel_format(style="sci", axis="both", scilimits=(-2, 2))
    axes.xaxis.offsetText.set_visible(True)
    axes.yaxis.offsetText.set_visible(True)
    axes.set_xlabel(r"$I$")
    axes.set_ylabel(r"$Q$")
    axes.set_title(f"prepared_state={i}")
