"""Deterministic Benchmarking Plotting Functions."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators._twin_axis import add_twin_axis


def plot_deterministic_benchmarking(plot_data: xr.Dataset) -> plt.Figure:
    """Plot Deterministic Benchmarking trajectories & frequency vs amp factor fit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    reps = plot_data["repetition"].values
    amp_prefactors = plot_data["amp_prefactor"].values
    num_amps = len(amp_prefactors)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_amps))

    # Left plot: Trajectories P0(N) vs N
    for i, a_val in enumerate(amp_prefactors):
        pz_curve = plot_data["pz"].sel(amp_prefactor=a_val).values
        ax1.plot(reps, pz_curve, "o", color=colors[i], label=f"amp factor = {a_val:.3f}")
        if "fit_pz" in plot_data:
            fit_curve = plot_data["fit_pz"].sel(amp_prefactor=a_val).values
            reps_fine = plot_data["repetition_fine"].values if "repetition_fine" in plot_data else reps
            ax1.plot(reps_fine, fit_curve, "-", color=colors[i], alpha=0.8)

    unit = plot_data.attrs.get("unit", "P0")
    ax1.set_xlabel("Number of Gate Repetitions N")
    ax1.set_ylabel(f"Ground State Population ({unit})")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("DB Trajectories")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize="small")

    # Right plot: Frequency omega vs amp factor
    omegas = plot_data["omega"].values
    a_opt = float(plot_data.attrs.get("opt_factor", 1.0))

    ax2.plot(amp_prefactors, omegas, "o", color="#1f77b4", markersize=7, label="Measured \u03c9")
    if "fit_omega_fine" in plot_data:
        a_fine = plot_data["amp_prefactor_fine"].values
        fit_w = plot_data["fit_omega_fine"].values
        ax2.plot(a_fine, fit_w, "--", color="gray", alpha=0.7, label="Linear Fit |\u03a9(a)|")

    opt_twin = plot_data.attrs.get("opt_twin_value")
    opt_label = f"Opt Factor ({a_opt:.4f})"
    if opt_twin is not None:  # state the answer in both frames
        opt_label += f" = {opt_twin:.6f} abs"
    ax2.axvline(a_opt, color="crimson", linestyle=":", label=opt_label)
    ax2.axhline(0.0, color="black", linestyle="-", alpha=0.3)
    ax2.plot(a_opt, 0.0, "*", color="crimson", markersize=12, label=f"Optimum a_opt={a_opt:.4f}")
    ax2.set_xlabel("Amplitude Scaling Factor a")
    ax2.set_ylabel("Oscillation Frequency \u03c9 [rad / gate count]")
    ax2.set_title(f"Frequency vs Amp Factor\nOptimal Scale Factor a_opt = {a_opt:.4f}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize="small")

    # the same amplitude axis in the caller's companion scale
    if "twin" in plot_data:
        add_twin_axis(ax2, amp_prefactors, plot_data["twin"].values,
                      str(plot_data.attrs.get("twin_label", "")))

    fig.tight_layout()
    return fig
