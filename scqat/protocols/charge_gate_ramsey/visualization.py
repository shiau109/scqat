import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_2d_colormap(dataset: xr.Dataset) -> plt.Figure:
    """Plot the raw signal as a 2D colour-map (charge_gate vs idle_time)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    idle_time = dataset.coords['idle_time'].values
    charge_gate = dataset.coords['charge_gate'].values

    X, Y = np.meshgrid(idle_time, charge_gate)
    im = ax.pcolormesh(X, Y, dataset['signal'].values, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel('Idle Time')
    ax.set_ylabel('Charge Gate (V)')
    ax.set_title('Charge Gate Ramsey – Raw Signal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_2d_spectrum(results: dict) -> plt.Figure:
    """
    Plot the FFT spectrum as a 2D colour-map with optional f_1/f_2 overlays.

    Parameters
    ----------
    results : dict
        Output of ChargeGateRamseyAnalyzer.extract_parameters.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    spectrum_ds = results.get('spectrum_dataset')
    if spectrum_ds is None:
        ax.text(0.5, 0.5, 'No spectrum data', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = spectrum_ds.coords['frequency'].values
    charge_gate = spectrum_ds.coords['charge_gate'].values
    X, Y = np.meshgrid(freq, charge_gate)

    im = ax.pcolormesh(X, Y, spectrum_ds['spectrum'].values, shading='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='Spectrum Amplitude')

    # Overlay f_1 / f_2 markers
    f_1, f_2 = results['f_1'], results['f_2']
    cg = results['charge_gates']

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


def plot_1d_frequencies(results: dict) -> plt.Figure:
    """
    Plot |f_1 − f_c| and |f_2 − f_c| vs charge_gate with |cos| fit overlay.

    Parameters
    ----------
    results : dict
        Output of ChargeGateRamseyAnalyzer.extract_parameters.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    cg = results['charge_gates']
    f_1, f_2 = results['f_1'], results['f_2']
    f_c = results['f_c']

    valid_f1 = ~np.isnan(f_1)
    if np.any(valid_f1):
        ax.scatter(cg[valid_f1], np.abs(f_1[valid_f1] - f_c),
                   c='blue', s=50, label='|f₁ − f_c|', alpha=0.8)

    valid_f2 = ~np.isnan(f_2)
    if np.any(valid_f2):
        ax.scatter(cg[valid_f2], np.abs(f_2[valid_f2] - f_c),
                   c='red', s=50, label='|f₂ − f_c|', alpha=0.8)

    # Overlay |cos| fit curve
    abscos_params = results.get('abscos_params')
    if abscos_params is not None and abscos_params.get('success', False):
        amp = abscos_params['amplitude']
        freq = abscos_params['frequency']
        phase = abscos_params['phase']

        cg_fine = np.linspace(cg.min(), cg.max(), 200)
        fit_curve = amp * np.abs(np.cos(2 * np.pi * freq * (cg_fine - phase)))
        ax.plot(cg_fine, fit_curve, 'g-', linewidth=2, label='|cos| fit', alpha=0.8)

        textstr = (
            f"|cos| fit:\n"
            f"  amplitude = {amp:.4g}\n"
            f"  frequency = {freq:.4g} V⁻¹\n"
            f"  phase = {phase:.4g} V\n"
            f"  χ²/dof = {abscos_params.get('redchi', np.nan):.4g}"
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
