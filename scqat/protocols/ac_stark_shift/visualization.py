"""
AC-Stark shift plotting helpers.

Every function consumes the **plot_data** Dataset built by
``AcStarkShiftAnalyzer.build_plot_data`` and draws without any recalculation, so
the figures reproduce from the saved ``*_plotdata.nc`` alone. The readout-power
axis name is taken from ``plot_data.attrs['power_coord']``.

plot_data layout
----------------
coords : <power_coord>, ``detuning``, ``P_fine`` (only when ``has_fit``)
vars   : ``raw_signal`` (power, detuning), ``f01``/``P``/``photon_number`` (power),
         ``stark_fit`` (P_fine, only when ``has_fit``)
attrs  : ``power_coord``, ``has_fit``, ``X_eff``, ``coeff_A``, ``f01_bare``
"""

import numpy as np
import matplotlib.pyplot as plt


def _power(plot_data):
    coord = plot_data.attrs["power_coord"]
    return coord, plot_data.coords[coord].values


def plot_raw_2d_with_f01(plot_data):
    """Raw spectroscopy map (detuning vs power) with the fitted f01 overlaid."""
    coord, power = _power(plot_data)
    detuning = plot_data.coords["detuning"].values
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    X, Y = np.meshgrid(power, detuning / 1e6)
    im = ax.pcolormesh(X, Y, plot_data["raw_signal"].values.T, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Signal")

    f01 = plot_data["f01"].values / 1e6
    ax.plot(power, f01, "o", color="red", ms=4, label=r"$f_{01}$")
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel("Detuning (MHz)", fontsize=14)
    ax.set_title("AC-Stark shift — spectroscopy vs readout power")
    ax.legend()
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_stark_fit(plot_data):
    """f01 vs readout-output voltage (sqrt power) with the AC-Stark fit overlaid."""
    coord, power = _power(plot_data)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

    P = plot_data["P"].values
    voltage = np.sqrt(np.clip(P, 0, None))
    ax.plot(voltage, plot_data["f01"].values / 1e6, "o", color="blue", alpha=0.6, ms=5, label="data")

    if plot_data.attrs.get("has_fit", 0):
        P_fine = plot_data.coords["P_fine"].values
        ax.plot(np.sqrt(np.clip(P_fine, 0, None)), plot_data["stark_fit"].values / 1e6,
                "-", color="red", lw=2, label="Stark fit")
        txt = (f"A = {plot_data.attrs['coeff_A']:.4g}\n"
               f"$f_{{01}}^{{bare}}$ = {plot_data.attrs['f01_bare'] / 1e6:.4g} MHz\n"
               f"$\\chi_{{eff}}$ = {plot_data.attrs['X_eff'] / 1e6:.4g} MHz")
        ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=10, va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.set_xlabel(r"Readout output voltage ($\sqrt{P}$)", fontsize=14)
    ax.set_ylabel(r"$f_{01}$ (MHz)", fontsize=14)
    ax.set_title("AC-Stark shift fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_photon_number(plot_data):
    """Predicted readout photon number n = A*P vs power."""
    coord, power = _power(plot_data)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    ax.plot(power, plot_data["photon_number"].values, "o-", color="green")
    ax.set_xlabel(coord, fontsize=14)
    ax.set_ylabel(r"$\bar{n}$", fontsize=14)
    ax.set_title("Estimated readout photon number")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
