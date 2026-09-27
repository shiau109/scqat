"""Figures for ``pair_coupler_spectroscopy_swap``.

Consumes the **plot_data** Dataset built by
``PairCouplerSpectroscopySwapEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``tone_freq_hz``, ``ramp_played`` (0 = reference, 1 = ramp),
         ``joint_state``, ``member`` (``high``/``low``)
vars   : ``joint_population`` (joint_state, ramp_played, tone_freq_hz);
         ``marginal`` (member, ramp_played, tone_freq_hz); ``difference`` and
         ``fit_curve`` (tone_freq_hz; the fit is all-NaN without a peak)
attrs  : every scalar of the results (``probe``, ``f_c_hz``, ``fwhm_hz``, the
         flags, ``high_name``/``low_name``) plus ``other_peaks_hz`` and
         ``reference_peaks_hz``

The raw populations are drawn UNCONDITIONALLY; the fit overlay and markers are
guarded, so a run with no peak still renders.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

__all__ = ["plot_spectrum"]

AXIS = "tone_freq_hz"


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def plot_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Top: each member's excited population in both arms (the probe solid, the
    other member faint). Bottom: the probe's ramp-minus-reference difference, its
    fitted line, f_c and every other peak."""
    f_ghz = np.asarray(plot_data[AXIS].values, dtype=float) / 1e9
    probe = str(plot_data.attrs.get("probe", "high"))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for member in plot_data["member"].values:
        member = str(member)
        name = plot_data.attrs.get(f"{member}_name", member)
        is_probe = member == probe
        for arm, style, what in ((1, "-", "ramp"), (0, "--", "reference")):
            y = plot_data["marginal"].sel(member=member, ramp_played=arm).values
            top.plot(f_ghz, y, style, lw=1.0 if is_probe else 0.7,
                     alpha=1.0 if is_probe else 0.4,
                     color="tab:red" if arm else "tab:gray",
                     label=f"{name}{' (probe)' if is_probe else ''} {what}")
    top.set_ylabel("P(excited)")
    top.set_title("coupler spectroscopy by swap")
    top.legend(loc="best", fontsize=8)

    bottom.plot(f_ghz, plot_data["difference"].values, lw=1.0, color="k",
                label="probe: ramp - reference")
    fit = np.asarray(plot_data["fit_curve"].values, dtype=float)
    if np.isfinite(fit).any():
        bottom.plot(f_ghz, fit, color="tab:blue", lw=1.2, label="Lorentzian fit")
    f_c = plot_data.attrs.get("f_c_hz")
    if _finite(f_c):
        bottom.axvline(float(f_c) / 1e9, color="tab:blue", ls="--", lw=1.0,
                       label=f"f_c = {float(f_c) / 1e9:.4f} GHz")
    for f in np.atleast_1d(plot_data.attrs.get("other_peaks_hz", [])):
        if _finite(f):
            bottom.axvline(float(f) / 1e9, color="tab:orange", ls=":", lw=1.0)
    for f in np.atleast_1d(plot_data.attrs.get("reference_peaks_hz", [])):
        if _finite(f):
            bottom.axvline(float(f) / 1e9, color="tab:gray", ls="-.", lw=1.0)
    flags = [k for k in ("no_peak", "reference_feature", "peak_at_edge", "multiple_peaks")
             if int(plot_data.attrs.get(k, 0))]
    if flags:
        bottom.text(0.01, 0.95, "flags: " + ", ".join(flags), transform=bottom.transAxes,
                    va="top", fontsize=8, color="tab:red")
    bottom.set_xlabel("tone frequency (GHz)")
    bottom.set_ylabel("difference")
    bottom.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
