"""
Parametric-drive decoherence plotting helpers.

Both functions consume the **plot_data** Dataset built by
``ParametricDriveDecoherenceEstimator.build_plot_data`` and draw without any
recalculation.

plot_data layout
----------------
coords : ``driving_frequency`` (Hz), ``driving_time`` (ns)
vars   : per-frequency scalars ``gamma`` / ``gamma_err`` / ``lambda_`` /
         ``lambda_err`` / ``Delta`` / ``Delta_err`` / ``rho_0`` / ``rho_0_err`` /
         ``chisqr`` / ``ep_metric`` / ``success``; 2-D maps ``rho11_data`` /
         ``rho11_fit`` (driving_frequency, driving_time)
attrs  : ``has_tomography``, ``n_freq``, ``n_decoh_ok``
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_decoherence_params(plot_data: xr.Dataset) -> plt.Figure:
    """4-panel summary of the fitted decoherence parameters vs driving frequency:
    γ, λ, |Δ| (with error bars) and the EP figure of merit 8λ²/γ²."""
    f_mhz = plot_data.coords["driving_frequency"].values.astype(float) / 1e6
    gamma = plot_data["gamma"].values
    lam = plot_data["lambda_"].values
    delta = np.abs(plot_data["Delta"].values)
    ep = plot_data["ep_metric"].values

    g_err = plot_data["gamma_err"].values
    l_err = plot_data["lambda_err"].values
    d_err = plot_data["Delta_err"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=120)
    panels = [
        (axes[0, 0], gamma, g_err, r"$\gamma$ (1/ns)", r"relaxation rate $\gamma$"),
        (axes[0, 1], lam, l_err, r"$\lambda$ (1/ns)", r"coupling $\lambda$"),
        (axes[1, 0], delta, d_err, r"$|\Delta|$ (1/ns)", r"detuning $|\Delta|$"),
    ]
    for ax, y, yerr, ylabel, title in panels:
        ax.errorbar(f_mhz, y, yerr=yerr, fmt="o-", ms=4, capsize=2)
        ax.set_xlabel("Driving frequency (MHz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(f_mhz, ep, "s-", ms=4, color="C3")
    ax.axhline(1.0, color="k", ls="--", lw=0.8, label="EP ($8\\lambda^2/\\gamma^2=1$)")
    ax.set_xlabel("Driving frequency (MHz)")
    ax.set_ylabel(r"$8\lambda^2/\gamma^2$")
    ax.set_title("exceptional-point figure of merit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    n_ok = int(plot_data.attrs.get("n_decoh_ok", int(np.isfinite(gamma).sum())))
    n_freq = int(plot_data.attrs.get("n_freq", f_mhz.size))
    kind = "tomography" if plot_data.attrs.get("has_tomography", 0) else r"$\rho_{11}$-only"
    fig.suptitle(f"Parametric-drive decoherence [{kind}] — fitted {n_ok}/{n_freq} frequencies")
    fig.tight_layout()
    return fig


def plot_rho11_fits(plot_data: xr.Dataset) -> plt.Figure:
    """ρ₁₁(t) data (dots) and decoherence fit (lines), one trace per driving
    frequency, coloured by frequency with a shared colorbar."""
    t = plot_data.coords["driving_time"].values.astype(float)
    freqs = plot_data.coords["driving_frequency"].values.astype(float)
    data = plot_data["rho11_data"].values  # (freq, time)
    fit = plot_data["rho11_fit"].values

    f_mhz = freqs / 1e6
    fmin, fmax = float(f_mhz.min()), float(f_mhz.max())
    norm = plt.Normalize(vmin=fmin, vmax=fmax if fmax > fmin else fmin + 1.0)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for i in range(freqs.size):
        color = cmap(norm(f_mhz[i]))
        ax.plot(t, data[i], "o", ms=2.5, alpha=0.5, color=color)
        if np.isfinite(fit[i]).any():
            ax.plot(t, fit[i], "-", lw=1.2, color=color)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Driving frequency (MHz)")
    ax.set_xlabel("Driving time (ns)")
    ax.set_ylabel(r"$\rho_{11}$")
    ax.set_title(r"$\rho_{11}(t)$ data (dots) and non-Markovian decoherence fit (lines)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
