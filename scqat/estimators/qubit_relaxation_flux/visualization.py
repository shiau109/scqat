import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_relaxation_flux(plot_data: xr.Dataset) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    
    # 2D colorplot of decay signal
    flux = plot_data["flux_amp"].values
    wait = plot_data["wait_time"].values * 1e6 # in us
    signal = plot_data["signal"].values # shape (flux_amp, wait_time)
    
    X, Y = np.meshgrid(flux, wait)
    # Transpose signal to (wait_time, flux_amp) for pcolormesh matching meshgrid
    im = ax1.pcolormesh(X, Y, signal.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax1, label="Signal (a.u.)")
    ax1.set_title("Relaxation Decay vs Flux Amplitude")
    ax1.set_ylabel("Wait Time (us)")
    ax1.set_xlabel("Flux Pulse Amplitude (V)")
    
    # 1D plot of T1 vs Flux
    t1_us = plot_data["t1"].values * 1e6 # convert to us
    ax2.plot(flux, t1_us, "o-", color="darkorange", label="T1 fit")
    ax2.set_title("T1 Spectrum")
    ax2.set_xlabel("Flux Pulse Amplitude (V)")
    ax2.set_ylabel("T1 Time (us)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig
