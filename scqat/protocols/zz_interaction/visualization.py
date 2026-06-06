"""
ZZ-interaction echo plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ZZInteractionEchoAnalyzer.build_plot_data`` and draws without any
recalculation, so the figures reproduce from the saved ``*_plotdata.nc`` alone.

plot_data layout
----------------
coords : ``flux``, ``time``
vars   : ``signal`` (flux, time), ``f`` (flux), ``tau`` (flux)
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_with_overlay(plot_data: xr.Dataset) -> plt.Figure:
    """Raw signal as a 2D colour-map (flux vs time) with the ZZ period (1/f) and
    echo decay (tau) overlaid as a function of flux."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    flux = plot_data.coords['flux'].values
    time = plot_data.coords['time'].values
    signal = plot_data['signal'].values  # (flux, time)

    X, Y = np.meshgrid(flux, time)
    im = ax.pcolormesh(X, Y, signal.T, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Signal')

    f = plot_data['f'].values
    tau = plot_data['tau'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        period = np.where(f != 0, 1.0 / f, np.nan)

    ax.plot(flux, period, color='blue', label='ZZ period (1/f)')
    ax.plot(flux, tau, color='red', label='T2 (τ)')

    ax.set_xlabel('Flux', fontsize=20)
    ax.set_ylabel('Free evolution time', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.locator_params(axis='x', nbins=7)
    ax.legend(fontsize=14, loc='best', frameon=True)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_zz_value(plot_data: xr.Dataset) -> plt.Figure:
    """ZZ strength (oscillation frequency f) and 1/T2 (1/tau) vs flux."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    flux = plot_data.coords['flux'].values
    f = plot_data['f'].values
    tau = plot_data['tau'].values
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_t2 = np.where(tau != 0, 1.0 / tau, np.nan)

    ax.plot(flux, f, color='blue', label='ZZ strength (f)')
    ax.plot(flux, inv_t2, color='red', label='1/T2')

    ax.set_xlabel('Flux', fontsize=20)
    ax.set_ylabel('ZZ (MHz)', fontsize=20)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.locator_params(axis='x', nbins=7)
    ax.legend(fontsize=14, loc='best', frameon=True)
    fig.tight_layout()
    plt.close(fig)
    return fig
