"""Figures for qubit_ramsey_flux_pulse - drawn from plot_data only.

Both plotters sort by the flux axis for DISPLAY only; the stored arrays keep the
realized sweep order. The raw fringe map is drawn unconditionally; the fit overlay
and the apex/park markers only when finite.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_fringe_map(plot_data: xr.Dataset) -> plt.Figure:
    """The raw signal over (flux, idle time)."""
    x = plot_data["flux_bias"].values
    t = plot_data["idle_time"].values
    sig = plot_data["signal"].transpose("flux_bias", "idle_time").values
    ox, ot = np.argsort(x, kind="stable"), np.argsort(t, kind="stable")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mesh = ax.pcolormesh(t[ot] * 1e6, x[ox] * 1e3, sig[np.ix_(ox, ot)], shading="nearest")
    fig.colorbar(mesh, ax=ax, label="signal")
    ax.set_xlabel("idle time (us)")
    ax.set_ylabel("flux (mV)")
    ax.set_title(f"Ramsey fringe vs flux (detuning {plot_data.attrs.get('ramp_detuning_hz', np.nan) / 1e6:+.3f} MHz)")
    fig.tight_layout()
    return fig


def plot_flux_curve(plot_data: xr.Dataset) -> plt.Figure:
    """delta_f = f_q - f_drive vs flux, the quadratic, and the apex / park point."""
    x = plot_data["flux_bias"].values * 1e3
    d = plot_data["delta_f_hz"].values / 1e3
    e = plot_data["delta_f_stderr_hz"].values / 1e3
    valid = plot_data["valid"].values.astype(bool)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x[valid], d[valid], yerr=e[valid], fmt="o", label="fringe")
    if (~valid).any():
        ax.plot(x[~valid], d[~valid], "x", color="0.5", label="rejected")
    fit = plot_data["fit_delta_f_hz"].values
    if np.isfinite(fit).any():
        ax.plot(plot_data["flux_fine"].values * 1e3, fit / 1e3, "-", label="quadratic")
    attrs = plot_data.attrs
    apex = attrs.get("apex_flux", np.nan)
    if np.isfinite(apex) and not attrs.get("apex_not_bracketed", 1):
        ax.axvline(apex * 1e3, ls="--", color="C2",
                   label=f"apex {apex * 1e3:+.3f} mV")
    park = attrs.get("park_flux", np.nan)
    if np.isfinite(park):
        ax.axvline(park * 1e3, ls=":", color="C3", label=f"park {park * 1e3:+.3f} mV")
    ax.set_xlabel("flux (mV)")
    ax.set_ylabel("f_q - f_drive (kHz)")
    title = f"{attrs.get('question', '')}: " + ("ok" if attrs.get("success") else "FAILED")
    if attrs.get("fold_suspected"):
        title += " (fold suspected)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
