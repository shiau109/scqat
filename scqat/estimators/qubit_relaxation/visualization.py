"""Plotting for the T1 relaxation estimator — draws ONLY from plot_data."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_decay(plot_data: xr.Dataset) -> plt.Figure:
    """Signal + exponential best fit vs wait time, T1 annotated.

    The raw signal is drawn UNCONDITIONALLY; the fit overlay and the T1 in the
    title appear only when the fit produced a finite curve. A failed fit (all-NaN
    ``best_fit`` / NaN ``t1``) therefore yields a clean raw-data-only figure
    instead of a crash or a misleading flat line.
    """
    t_us = plot_data["wait_time"].values * 1e6
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_us, plot_data["signal"].values, ".", label="data", alpha=0.7)

    best_fit = plot_data["best_fit"].values
    t1_us = plot_data.attrs["t1"] * 1e6
    if np.any(np.isfinite(best_fit)) and np.isfinite(t1_us):
        ax.plot(t_us, best_fit, "-", label="fit")
        title = f"T1 relaxation: T1 = {t1_us:.2f} us"
    else:
        title = "T1 relaxation: fit failed (raw data)"

    ax.set_xlabel("wait time (us)")
    ax.set_ylabel("signal")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
