import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_echo_flux(plot_data: xr.Dataset) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=False)
    
    flux = plot_data["flux_amp"].values
    wait = plot_data["wait_time"].values * 1e6 # in us
    signal = plot_data["signal"].values # shape (flux_amp, wait_time)
    
    X, Y = np.meshgrid(flux, wait)
    im = ax1.pcolormesh(X, Y, signal.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax1, label="Signal (a.u.)")
    ax1.set_title("Hahn Echo Decay vs Flux Amplitude")
    ax1.set_ylabel("Total Wait Time (us)")
    ax1.set_xlabel("Flux Pulse Amplitude (V)")
    
    t2_echo_us = plot_data["t2_echo"].values * 1e6 # convert to us
    ax2.plot(flux, t2_echo_us, "o-", color="royalblue", label="T2 Echo fit")
    ax2.set_title("T2 Echo Spectrum")
    ax2.set_xlabel("Flux Pulse Amplitude (V)")
    ax2.set_ylabel("T2 Echo Time (us)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    return fig
