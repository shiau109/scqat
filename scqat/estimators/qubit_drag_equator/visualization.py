import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_drag_equator(plot_data: xr.Dataset) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    beta = plot_data["beta"].values
    y0 = plot_data["seq0"].values
    y1 = plot_data["seq1"].values
    
    ax.plot(beta, y0, "o", color="#1f77b4", label="Rx(pi) - Ry(pi/2)")
    ax.plot(beta, y1, "o", color="#ff7f0e", label="Ry(pi) - Rx(pi/2)")
    
    if "fit_seq0" in plot_data:
        ax.plot(beta, plot_data["fit_seq0"].values, "k-", linewidth=1.5)
    if "fit_seq1" in plot_data:
        ax.plot(beta, plot_data["fit_seq1"].values, "k-", linewidth=1.5)
    
    opt_beta = float(plot_data.attrs.get("opt_beta", np.nan))
    if np.isfinite(opt_beta):
        ax.axvline(opt_beta, linestyle="--", color="black", label=f"DRAG: {opt_beta:.2e}")
        
    ax.set_title("cosine_pulse DRAG Calibration", fontsize=12)
    ax.set_xlabel("DRAG coefficient", fontsize=11)
    ax.grid(True, linestyle="-", color="lightgray")
    ax.legend(frameon=True)
    
    plt.tight_layout()
    return fig
