import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_time_domain(dataset: xr.Dataset, results: dict) -> plt.Figure:
    """Plot raw Ramsey signal and fit curve with parameter annotations."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    idle_time = dataset.coords['idle_time'].values
    signal = dataset['signal'].values

    ax.plot(idle_time, signal, '.', label='Raw Data', markersize=3)

    if 'best_fit' in results:
        ax.plot(idle_time, results['best_fit'], '-', label='Fit', linewidth=2)

    ax.legend()

    # Build parameter textbox
    textstr = _build_param_text(results)
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


def plot_fft(results: dict) -> plt.Figure:
    """Plot the FFT amplitude spectrum of the Ramsey signal."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    freq = results['fft_freq']
    amp = results['fft_amp']

    ax.plot(freq, amp, label='FFT (positive freq)')
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def _build_param_text(results: dict) -> str:
    """Build a formatted string of fit parameters for annotation."""
    model_type = results.get('model_type', 'single')

    k1 = results.get('kappa_1', float('nan'))
    tau1 = 1.0 / k1 if k1 != 0 else float('nan')

    lines = [
        f"κ₁ = {k1:.4g}  (τ₁ = {tau1:.4g})",
        f"a₁ = {results.get('a_1', float('nan')):.4g}",
        f"f₁ = {results.get('f_1', float('nan')):.4g}",
        f"ϕ₁ = {results.get('phi_1', float('nan')):.4g}",
    ]

    if model_type == 'beat':
        k2 = results.get('kappa_2', float('nan'))
        tau2 = 1.0 / k2 if k2 != 0 else float('nan')
        f1 = results.get('f_1', float('nan'))
        f2 = results.get('f_2', float('nan'))
        lines += [
            f"κ₂ = {k2:.4g}  (τ₂ = {tau2:.4g})",
            f"a₂ = {results.get('a_2', float('nan')):.4g}",
            f"f₂ = {f2:.4g}",
            f"ϕ₂ = {results.get('phi_2', float('nan')):.4g}",
            f"f±δf = {(f1 + f2) / 2:.4g} ± {abs(f1 - f2) / 2:.4g}",
        ]

    return "\n".join(lines)
