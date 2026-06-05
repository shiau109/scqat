"""
Ramsey plotting helpers.

Both functions consume the **plot_data** Dataset built by
``RamseyAnalyzer.build_plot_data`` and draw without any recalculation.

plot_data layout
----------------
coords : ``idle_time``, ``fft_freq``
vars   : ``signal`` (idle_time), ``best_fit`` (idle_time), ``fft_amp`` (fft_freq)
attrs  : ``model_type``, ``a_1``, ``kappa_1``, ``f_1``, ``phi_1``, ``c`` and,
         for the beat model, ``a_2``, ``kappa_2``, ``f_2``, ``phi_2``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_time_domain(plot_data: xr.Dataset) -> plt.Figure:
    """Plot raw Ramsey signal and fit curve with parameter annotations."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    idle_time = plot_data.coords['idle_time'].values
    ax.plot(idle_time, plot_data['signal'].values, '.', label='Raw Data', markersize=3)

    if 'best_fit' in plot_data:
        ax.plot(idle_time, plot_data['best_fit'].values, '-', label='Fit', linewidth=2)

    ax.legend()

    textstr = _build_param_text(plot_data.attrs)
    ax.text(
        0.98, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )

    ax.set_xlabel('Idle time', fontsize=16)
    ax.set_ylabel('Signal', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_fft(plot_data: xr.Dataset) -> plt.Figure:
    """Plot the FFT amplitude spectrum of the Ramsey signal."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.plot(plot_data.coords['fft_freq'].values, plot_data['fft_amp'].values,
            label='FFT (positive freq)')
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def _build_param_text(attrs) -> str:
    """Build a formatted string of fit parameters for annotation."""
    model_type = attrs.get('model_type', 'single')

    k1 = attrs.get('kappa_1', float('nan'))
    tau1 = 1.0 / k1 if k1 != 0 else float('nan')

    lines = [
        f"κ₁ = {k1:.4g}  (τ₁ = {tau1:.4g})",
        f"a₁ = {attrs.get('a_1', float('nan')):.4g}",
        f"f₁ = {attrs.get('f_1', float('nan')):.4g}",
        f"ϕ₁ = {attrs.get('phi_1', float('nan')):.4g}",
    ]

    if model_type == 'beat':
        k2 = attrs.get('kappa_2', float('nan'))
        tau2 = 1.0 / k2 if k2 != 0 else float('nan')
        f1 = attrs.get('f_1', float('nan'))
        f2 = attrs.get('f_2', float('nan'))
        lines += [
            f"κ₂ = {k2:.4g}  (τ₂ = {tau2:.4g})",
            f"a₂ = {attrs.get('a_2', float('nan')):.4g}",
            f"f₂ = {f2:.4g}",
            f"ϕ₂ = {attrs.get('phi_2', float('nan')):.4g}",
            f"f±δf = {(f1 + f2) / 2:.4g} ± {abs(f1 - f2) / 2:.4g}",
        ]

    return "\n".join(lines)
