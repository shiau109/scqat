"""
Resonator-spectroscopy-vs-power plotting helper.

``plot_power_map`` consumes the **plot_data** Dataset built by
``ResonatorSpectroscopyPowerEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``power``, ``detuning`` (+ optional ``full_freq``)
vars   : ``amplitude`` + power-normalized ``amplitude_db`` (power, detuning);
         per-power ``center_detuning`` / ``fwhm`` / ``dip_amplitude`` /
         ``success`` / ``good`` / ``outlier`` (+ optional ``center_full_freq``)
attrs  : ``n_power``, ``n_success``, ``n_good``, ``n_outlier``, ``has_full_freq``,
         ``optimal_power``, ``frequency_shift`` (+ optional ``resonator_frequency``)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_power_map(plot_data: xr.Dataset) -> plt.Figure:
    """The 2-D power-normalized ``20*log10|IQ| - power`` map over (power,
    frequency) with the fitted resonator-centre trace and optimal-power marker
    overlaid, drawn entirely from ``plot_data``."""
    power = plot_data.coords["power"].values.astype(float)
    if "amplitude_db" in plot_data:
        amplitude_db = plot_data["amplitude_db"].values  # (power, detuning)
    else:
        # Older plotdata files carry only the linear map; normalize here.
        amplitude = plot_data["amplitude"].values
        amplitude_db = (
            20.0 * np.log10(np.maximum(amplitude, np.finfo(float).tiny)) - power[:, None]
        )

    use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
    if use_full:
        yvals = plot_data["full_freq"].values.astype(float) / 1e9
        center = plot_data["center_full_freq"].values / 1e9
        ylabel = "RF frequency (GHz)"
    else:
        yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
        center = plot_data["center_detuning"].values / 1e6
        ylabel = "Detuning (MHz)"

    good = plot_data["good"].values.astype(bool)
    outlier = plot_data["outlier"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    pcm = ax.pcolormesh(power, yvals, amplitude_db.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="20 log10 |IQ| - power (dB)")
    # Kept centres form the trace; rejected (out-of-window / outlier) shown as red x.
    ax.plot(power[good], center[good], ".-", color="C3", ms=5, lw=1.0, label="centre (kept)")
    if outlier.any():
        ax.plot(power[outlier], center[outlier], "x", color="red", ms=7, mew=1.5,
                label="rejected (outlier)")

    # Optimal-power marker (vertical line) when the pick succeeded.
    optimal_power = float(plot_data.attrs.get("optimal_power", np.nan))
    if bool(plot_data.attrs.get("optimal_success", 0)) and np.isfinite(optimal_power):
        ax.axvline(optimal_power, color="magenta", ls="--", lw=1.5,
                   label=f"optimal power = {optimal_power:.1f} dB")

    ax.set_xlabel("Readout power (dB)")
    ax.set_ylabel(ylabel)
    n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
    n_power = int(plot_data.attrs.get("n_power", len(power)))
    ax.set_title(f"Resonator spectroscopy vs power (kept {n_good}/{n_power})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
