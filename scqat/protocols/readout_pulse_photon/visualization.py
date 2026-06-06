"""
Time-dependent readout-photon plotting helpers.

Every function consumes the **plot_data** Dataset built by
``ReadoutPulsePhotonAnalyzer.build_plot_data`` and draws without any
recalculation, so the figures reproduce from the saved ``*_plotdata.nc`` alone.
The delay axis name is taken from ``plot_data.attrs['time_coord']``; when
``attrs['has_photon']`` is set the detuning axis is shown as photon number
(``detuning/1e6 / chi_eff``), matching ``notebooks/ac_stark_readout.ipynb``.

plot_data layout
----------------
coords : <time_coord>, ``detuning``
vars   : ``raw_signal`` (delay, detuning), ``peak_detuning``/``y_value`` (delay)
attrs  : ``time_coord``, ``has_photon``, ``chi_eff``, ``steady_state_value``,
         ``steady_state_lo``/``steady_state_hi`` (when a window was given)
"""

import numpy as np
import matplotlib.pyplot as plt


def _det_axis(plot_data):
    detuning = plot_data.coords["detuning"].values
    if plot_data.attrs.get("has_photon", 0):
        chi_eff = plot_data.attrs["chi_eff"]
        return (detuning / 1e6) / chi_eff, "Photon number"
    return detuning / 1e6, "Detuning (MHz)"


def plot_raw_2d_with_peaks(plot_data):
    """Raw |IQ| map (delay vs detuning/photon) with the fitted peak overlaid."""
    coord = plot_data.attrs["time_coord"]
    delays = plot_data.coords[coord].values
    det_axis, det_label = _det_axis(plot_data)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    X, Y = np.meshgrid(delays, det_axis)
    im = ax.pcolormesh(X, Y, plot_data["raw_signal"].values.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Signal (arb. u.)")

    peak_det = plot_data["peak_detuning"].values
    if plot_data.attrs.get("has_photon", 0):
        peak_axis = (peak_det / 1e6) / plot_data.attrs["chi_eff"]
    else:
        peak_axis = peak_det / 1e6
    m = np.isfinite(peak_axis)
    ax.plot(delays[m], peak_axis[m], "r-", lw=1.2, label="Fitted peak")

    ax.set_xlabel(coord)
    ax.set_ylabel(det_label)
    ax.set_title("Time-dep RR photon — peak positions")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_photon_vs_delay(plot_data):
    """Photon number (or detuning) vs delay, with the steady-state average."""
    coord = plot_data.attrs["time_coord"]
    has_photon = plot_data.attrs.get("has_photon", 0)
    scale = 1.0 if has_photon else 1e6
    ylabel = "Photon number" if has_photon else "Detuning (MHz)"

    delays = plot_data.coords[coord].values
    y = plot_data["y_value"].values / scale

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    m = np.isfinite(y)
    ax.plot(delays[m], y[m], "o-", ms=4, lw=1.2, label="Data")

    steady = plot_data.attrs.get("steady_state_value", np.nan)
    if np.isfinite(steady):
        ax.axhline(steady / scale, color="orange", ls=":", lw=1.5,
                   label=f"steady-state = {steady / scale:.3g}")
    if "steady_state_lo" in plot_data.attrs:
        ax.axvspan(plot_data.attrs["steady_state_lo"], plot_data.attrs["steady_state_hi"],
                   color="orange", alpha=0.12)

    ax.set_xlabel(coord)
    ax.set_ylabel(ylabel)
    ax.set_title("Time-dep RR photon — vs delay")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
