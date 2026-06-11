"""
Parametric-drive resonance plotting helper.

``plot_parametric_map`` consumes the **plot_data** Dataset built by
``ParametricDriveResonanceEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``amplitude_ratio``, ``driving_frequency``, ``peak``
vars   : ``amplitude`` (amplitude_ratio, driving_frequency); per-peak
         ``peak_amp_ratio`` / ``peak_frequency`` / ``peak_fwhm`` /
         ``peak_amplitude`` / ``good`` / ``outlier``
attrs  : ``n_amp``, ``n_peaks``, ``n_good``, ``n_outlier``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_parametric_map(plot_data: xr.Dataset) -> plt.Figure:
    """The 2-D signal map over (amplitude_ratio, driving_frequency) with every
    kept resonance peak overlaid and outliers marked, drawn entirely from
    ``plot_data``."""
    amp = plot_data.coords["amplitude_ratio"].values.astype(float)
    freq_mhz = plot_data.coords["driving_frequency"].values.astype(float) / 1e6
    amplitude = plot_data["amplitude"].values  # (amplitude_ratio, driving_frequency)

    peak_amp = plot_data["peak_amp_ratio"].values.astype(float)
    peak_freq = plot_data["peak_frequency"].values.astype(float) / 1e6
    good = plot_data["good"].values.astype(bool)
    outlier = plot_data["outlier"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    pcm = ax.pcolormesh(amp, freq_mhz, amplitude.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Signal (arb. u.)")
    if good.any():
        ax.plot(peak_amp[good], peak_freq[good], "o", color="white", ms=4, mec="black",
                mew=0.4, label="peaks (kept)")
    if outlier.any():
        ax.plot(peak_amp[outlier], peak_freq[outlier], "x", color="red", ms=7, mew=1.5,
                label="rejected (outlier)")
    ax.set_xlabel("Amplitude ratio (arb. u.)")
    ax.set_ylabel("Driving frequency (MHz)")
    n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
    n_peaks = int(plot_data.attrs.get("n_peaks", peak_amp.size))
    ax.set_title(f"Parametric-drive resonance (kept {n_good}/{n_peaks} peaks)")
    if good.any() or outlier.any():
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
