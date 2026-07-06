"""Plotting for the T1 relaxation estimator — draws ONLY from plot_data."""

import matplotlib.pyplot as plt
import xarray as xr


def plot_decay(plot_data: xr.Dataset) -> plt.Figure:
    """Signal + exponential best fit vs wait time, T1 annotated."""
    t_us = plot_data["wait_time"].values * 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, plot_data["signal"].values, ".", label="data", alpha=0.7)
    ax.plot(t_us, plot_data["best_fit"].values, "-", label="fit")
    t1_us = plot_data.attrs["t1"] * 1e6
    ax.set_xlabel("wait time (us)")
    ax.set_ylabel("signal")
    ax.set_title(f"T1 relaxation: T1 = {t1_us:.2f} us")
    ax.legend()
    fig.tight_layout()
    return fig
