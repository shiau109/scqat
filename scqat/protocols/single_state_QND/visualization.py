import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_2d_histogram_single(hist_dataset, analysis_result=None):
    """
    Plot a single-panel 2D histogram (density) from hist_dataset.
    Optionally overlay Gaussian mean and std circles.

    Args:
        hist_dataset (xr.Dataset): Dataset with variable 'density' and coords 'x', 'y'.
        analysis_result (dict, optional): Fit results with 'fitted_paras'.
    Returns:
        fig, ax
    """
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    x = hist_dataset["x"].values
    y = hist_dataset["y"].values
    xedges = (
        np.concatenate([x - (x[1] - x[0]) / 2, [x[-1] + (x[1] - x[0]) / 2]])
        if len(x) > 1
        else np.array([x[0] - 0.5, x[0] + 0.5])
    )
    yedges = (
        np.concatenate([y - (y[1] - y[0]) / 2, [y[-1] + (y[1] - y[0]) / 2]])
        if len(y) > 1
        else np.array([y[0] - 0.5, y[0] + 0.5])
    )

    density = hist_dataset["density"].values
    density_masked = np.ma.masked_where(density <= 0, density)
    ax.pcolormesh(xedges, yedges, density_masked, shading="auto", cmap="viridis", norm=LogNorm())
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("2D Histogram")
    ax.set_aspect("equal")

    if analysis_result is not None:
        fitted_paras = analysis_result.get("fitted_paras", None)
        if fitted_paras is not None:
            _plot_mean_and_circles(ax, fitted_paras)

        y_offset = 0.98
        if "outlier_probability" in analysis_result:
            text_msg = f"Outlier prob.: {analysis_result['outlier_probability']:.3e}"
            ax.text(
                0.02, y_offset, text_msg,
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_outliers_single(data, outlier_mask, analysis_result=None):
    """
    Plot scatter of outlier I/Q points for a single-state dataset.

    Args:
        data (xr.Dataset): Dataset with 'I', 'Q', coord 'shot_idx'.
        outlier_mask (array): Boolean mask (same length as shot_idx).
        analysis_result (dict, optional): Fit results with 'fitted_paras'.
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

    I_vals = data["I"].values.ravel()
    Q_vals = data["Q"].values.ravel()

    # Plot all points faintly
    ax.scatter(I_vals, Q_vals, s=3, alpha=0.3, color="blue", edgecolor="none", label="Inlier")
    # Overlay outliers
    ax.scatter(
        I_vals[outlier_mask], Q_vals[outlier_mask],
        s=10, alpha=0.8, color="orange", marker="o", edgecolor="none", label="Outlier",
    )

    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title("Outlier Detection")
    ax.set_aspect("equal")
    ax.legend(markerscale=2)

    if analysis_result is not None:
        fitted_paras = analysis_result.get("fitted_paras", None)
        if fitted_paras is not None:
            _plot_mean_and_circles(ax, fitted_paras)

        y_offset = 0.98
        if "outlier_probability" in analysis_result:
            text_msg = f"Outlier prob.: {analysis_result['outlier_probability']:.3e}"
            ax.text(
                0.02, y_offset, text_msg,
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def plot_distance_vs_shot(data, analysis_result):
    """
    Plot the Mahalanobis-like distance (normalised by fitted std) of each
    shot from the fitted Gaussian centre, as a function of shot_idx.

    Horizontal reference lines at 1σ, 2σ, 3σ correspond to the circles
    drawn on the 2-D histogram.

    Args:
        data (xr.Dataset): Dataset with 'I', 'Q', coord 'shot_idx'.
        analysis_result (dict): Must contain 'fitted_paras' with 'mean' and 'std'.
    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    I_vals = data["I"].values.ravel()
    Q_vals = data["Q"].values.ravel()
    shot_idx = np.arange(len(I_vals))

    mean = analysis_result["fitted_paras"]["mean"][0]
    std = analysis_result["fitted_paras"]["std"]

    distances = np.sqrt((I_vals - mean[0]) ** 2 + (Q_vals - mean[1]) ** 2)
    norm_distances = distances / std

    ax.scatter(shot_idx, norm_distances, s=2, alpha=0.5, color="steelblue", edgecolor="none")

    for n in (1, 2, 3):
        ax.axhline(n, color="k", linestyle="--", linewidth=1, alpha=0.6)
        ax.text(len(shot_idx) * 0.99, n + 0.05, f"{n}σ",
                ha="right", va="bottom", fontsize=9, color="k")

    ax.set_xlabel("shot_idx")
    ax.set_ylabel("distance / σ")
    ax.set_title("Normalised distance from fitted peak vs shot")
    ax.set_xlim(shot_idx[0], shot_idx[-1])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plt.close(fig)
    return fig, ax


def _plot_mean_and_circles(ax, fitted_paras, n_std_list=(1, 2, 3)):
    """Overlay Gaussian center and n-sigma circles on an axis."""
    mean = fitted_paras["mean"][0]
    std = fitted_paras["std"]
    ax.scatter(mean[0], mean[1], c="k", s=40, marker="o", zorder=5)
    for n in n_std_list:
        radius = std * n
        circle = Ellipse(
            xy=mean, width=2 * radius, height=2 * radius, angle=0,
            edgecolor="k", facecolor="none", linestyle="--", linewidth=1.5,
        )
        ax.add_patch(circle)
