"""Plotting for the T2 echo estimator — draws ONLY from plot_data."""

import matplotlib.pyplot as plt
import xarray as xr


def plot_decay(plot_data: xr.Dataset) -> plt.Figure:
    """Echo signal + exponential best fit vs idle time, T2_echo annotated."""
    t_us = plot_data["idle_time"].values * 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, plot_data["signal"].values, ".", label="data", alpha=0.7)
    ax.plot(t_us, plot_data["best_fit"].values, "-", label="fit")
    t2e_us = plot_data.attrs["t2_echo"] * 1e6
    ax.set_xlabel("echo idle time (us)")
    ax.set_ylabel("signal")
    ax.set_title(f"Hahn echo: T2_echo = {t2e_us:.2f} us")
    ax.legend()
    fig.tight_layout()
    return fig
