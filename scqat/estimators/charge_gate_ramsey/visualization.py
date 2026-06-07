"""
Charge-gate Ramsey plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ChargeGateRamseyEstimator.build_plot_data`` and draws without any
recalculation, so the figures can be reproduced from the saved
``*_plotdata.nc`` file alone.

plot_data layout
----------------
coords : ``charge_gate``, ``idle_time``, ``frequency`` (if ``has_spectrum``),
         ``cg_fine`` (if ``has_fit``)
vars   : ``raw_signal`` (charge_gate, idle_time), ``spectrum``
         (charge_gate, frequency), ``f_1``/``f_2`` (charge_gate),
         ``fit_curve_even``/``fit_curve_odd``/``fit_abscos`` (cg_fine)
attrs  : ``f_c``, ``has_spectrum``, ``has_fit``, ``abscos_amplitude``,
         ``abscos_frequency``, ``abscos_phase``, ``abscos_redchi``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_2d_colormap(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the raw signal as a 2D colour-map (charge_gate vs idle_time)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    idle_time = plot_data.coords['idle_time'].values
    charge_gate = plot_data.coords['charge_gate'].values

    X, Y = np.meshgrid(idle_time, charge_gate)
    im = ax.pcolormesh(X, Y, plot_data['raw_signal'].values, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel('Idle Time')
    ax.set_ylabel('Charge Gate (V)')
    ax.set_title('Charge Gate Ramsey – Raw Signal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_2d_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the FFT spectrum as a 2D colour-map with f_1/f_2 overlays."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    if not plot_data.attrs.get('has_spectrum', 0):
        ax.text(0.5, 0.5, 'No spectrum data', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = plot_data.coords['frequency'].values
    charge_gate = plot_data.coords['charge_gate'].values
    X, Y = np.meshgrid(freq, charge_gate)

    im = ax.pcolormesh(X, Y, plot_data['spectrum'].values, shading='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='Spectrum Amplitude')

    # Overlay f_1 / f_2 markers
    f_1 = plot_data['f_1'].values
    f_2 = plot_data['f_2'].values
    cg = charge_gate

    valid = ~np.isnan(f_1)
    if np.any(valid):
        ax.plot(f_1[valid], cg[valid], 'bo', markersize=4, label='f₁', alpha=0.8)

    valid = ~np.isnan(f_2)
    if np.any(valid):
        ax.plot(f_2[valid], cg[valid], 'ro', markersize=4, label='f₂', alpha=0.8)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Charge Gate (V)')
    ax.set_title('FFT Spectrum vs Charge Gate')
    if len(freq) > 1:
        ax.set_xlim(freq[1], freq[-1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_2d_spectrum_with_fit(plot_data: xr.Dataset) -> plt.Figure:
    """
    Plot the FFT spectrum as a 2D colour-map with the |cos| fit overlay.

    Like plot_2d_spectrum but replaces the f_1/f_2 scatter dots with the
    pre-evaluated fit curves (f_c ± fit_curve) vs cg_fine.
    """
    _FIG_WIDTH = 3.0
    _FIG_HEIGHT = 2.0
    _rc = {
        "figure.figsize": (_FIG_WIDTH, _FIG_HEIGHT),
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
    }
    with plt.rc_context(_rc):
        return _plot_2d_spectrum_with_fit_inner(plot_data)


def _plot_2d_spectrum_with_fit_inner(plot_data: xr.Dataset) -> plt.Figure:
    fig, ax = plt.subplots()

    if not plot_data.attrs.get('has_spectrum', 0):
        ax.text(0.5, 0.5, 'No spectrum data', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = plot_data.coords['frequency'].values
    charge_gate = plot_data.coords['charge_gate'].values
    X, Y = np.meshgrid(charge_gate, freq)

    im = ax.pcolormesh(X, Y, plot_data['spectrum'].values.T, shading='auto', cmap='Purples')
    # plt.colorbar(im, ax=ax, label='Spectrum Amplitude')

    # Overlay the pre-evaluated |cos| fit as f_c ± fit_curve lines
    if plot_data.attrs.get('has_fit', 0):
        cg_fine = plot_data.coords['cg_fine'].values
        ax.plot(cg_fine, plot_data['fit_curve_even'].values, 'r--',
                linewidth=1, alpha=0.7, label='Even')
        ax.plot(cg_fine, plot_data['fit_curve_odd'].values, 'b--',
                linewidth=1, alpha=0.7, label='Odd')

    ax.set_xlabel('$n_g$ (2e)', fontfamily='serif')
    ax.set_ylabel('Detuning (MHz)', fontfamily='serif')
    if len(freq) > 1:
        ax.set_ylim(0, freq[-1] / 1.5)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_1d_frequencies(plot_data: xr.Dataset) -> plt.Figure:
    """Plot |f_1 − f_c| and |f_2 − f_c| vs charge_gate with |cos| fit overlay."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    cg = plot_data.coords['charge_gate'].values
    f_1 = plot_data['f_1'].values
    f_2 = plot_data['f_2'].values
    f_c = plot_data.attrs['f_c']

    valid_f1 = ~np.isnan(f_1)
    if np.any(valid_f1):
        ax.scatter(cg[valid_f1], np.abs(f_1[valid_f1] - f_c),
                   c='blue', s=50, label='|f₁ − f_c|', alpha=0.8)

    valid_f2 = ~np.isnan(f_2)
    if np.any(valid_f2):
        ax.scatter(cg[valid_f2], np.abs(f_2[valid_f2] - f_c),
                   c='red', s=50, label='|f₂ − f_c|', alpha=0.8)

    # Overlay the pre-evaluated |cos| fit curve
    if plot_data.attrs.get('has_fit', 0):
        cg_fine = plot_data.coords['cg_fine'].values
        ax.plot(cg_fine, plot_data['fit_abscos'].values, 'g-',
                linewidth=2, label='|cos| fit', alpha=0.8)

        textstr = (
            f"|cos| fit:\n"
            f"  amplitude = {plot_data.attrs['abscos_amplitude']:.4g}\n"
            f"  frequency = {plot_data.attrs['abscos_frequency']:.4g} V⁻¹\n"
            f"  phase = {plot_data.attrs['abscos_phase']:.4g} V\n"
            f"  χ²/dof = {plot_data.attrs.get('abscos_redchi', np.nan):.4g}"
        )
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Charge Gate (V)')
    ax.set_ylabel('Frequency Difference')
    ax.set_title('Ramsey Frequency Dispersion vs Charge Gate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
