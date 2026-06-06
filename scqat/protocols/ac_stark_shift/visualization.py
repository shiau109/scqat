"""
AC-Stark shift plotting helpers.

Every function consumes the **plot_data** Dataset built by
``AcStarkShiftAnalyzer.build_plot_data`` and draws without any recalculation, so
the figures reproduce from the saved ``*_plotdata.nc`` alone. The readout-power
axis name is taken from ``plot_data.attrs['power_coord']``; when
``attrs['has_photon']`` is set the detuning axis is shown as photon number
(``detuning/1e6 / chi_eff``), matching ``notebooks/ac_stark_spectroscopy.ipynb``.

plot_data layout
----------------
coords : <power_coord>, ``detuning``, ``P_fine`` (when a fit exists)
vars   : ``raw_signal`` (power, detuning), ``peak_detuning``/``amp_squared``/``y_value`` (power),
         ``fit_y`` (P_fine)
attrs  : ``power_coord``, ``has_photon``, ``chi_eff``, ``amp_ref``,
         ``fit_slope``, ``fit_intercept``
"""

import numpy as np
import matplotlib.pyplot as plt


def _det_axis(plot_data):
    """Return (axis values in MHz-or-photon, label) for the detuning axis."""
    detuning = plot_data.coords["detuning"].values
    if plot_data.attrs.get("has_photon", 0):
        chi_eff = plot_data.attrs["chi_eff"]
        return (detuning / 1e6) / chi_eff, "Photon number"
    return detuning / 1e6, "Detuning (MHz)"


def plot_raw_2d_with_peaks(plot_data):
    """Raw |IQ| map (detuning/photon vs power) with fitted peaks overlaid."""
    coord = plot_data.attrs["power_coord"]
    power = plot_data.coords[coord].values
    det_axis, det_label = _det_axis(plot_data)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    X, Y = np.meshgrid(det_axis, power)
    im = ax.pcolormesh(X, Y, plot_data["raw_signal"].values, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Signal (arb. u.)")

    peak_det = plot_data["peak_detuning"].values
    if plot_data.attrs.get("has_photon", 0):
        peak_axis = (peak_det / 1e6) / plot_data.attrs["chi_eff"]
    else:
        peak_axis = peak_det / 1e6
    m = np.isfinite(peak_axis)
    ax.plot(peak_axis[m], power[m], "rx-", ms=6, lw=1.2, label="Fitted peak")

    ax.set_xlabel(det_label)
    ax.set_ylabel(coord)
    ax.set_title("AC-Stark shift — spectroscopy vs readout amplitude")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_shift_vs_power(plot_data):
    """Peak position (photon number or detuning) vs amp², with the linear fit."""
    coord = plot_data.attrs["power_coord"]
    has_photon = plot_data.attrs.get("has_photon", 0)
    scale = 1.0 if has_photon else 1e6  # y_value is photons or Hz
    ylabel = "Photon number" if has_photon else "Detuning (MHz)"

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    amp_sq = plot_data["amp_squared"].values
    y = plot_data["y_value"].values / scale
    m = np.isfinite(amp_sq) & np.isfinite(y)
    ax.plot(amp_sq[m], y[m], "o", ms=5, label="Data")

    if "fit_y" in plot_data:
        a = plot_data.attrs["fit_slope"]
        b = plot_data.attrs["fit_intercept"]
        ax.plot(plot_data.coords["P_fine"].values, plot_data["fit_y"].values / scale,
                "-", lw=1.5, color="C1",
                label=f"fit: y = {a:.4g}·A² + {b:.4g}")

    ax.set_xlabel(f"{coord}²")
    ax.set_ylabel(ylabel)
    ax.set_title("AC-Stark shift — linear fit")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
