"""Figures for ``pair_coupler_spectroscopy_swap``.

Consumes the **plot_data** Dataset built by
``PairCouplerSpectroscopySwapEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``tone_freq_hz``, ``ramp_played`` (0 = reference, 1 = ramp),
         ``joint_state``, ``member`` (``high``/``low``)
vars   : ``joint_population`` (joint_state, ramp_played, tone_freq_hz);
         ``marginal`` (member, ramp_played, tone_freq_hz); ``total`` (ramp_played,
         tone_freq_hz) = 1 - P00; ``difference`` (tone_freq_hz) = ramp - reference
         total; ``fit_curve`` (tone_freq_hz; the fitted lines on the ramp total,
         all-NaN without a line)
attrs  : every scalar of the results (``probe``, ``f_c_hz``, ``alpha_hz``,
         ``f02_half_hz``, ``f03_third_hz``, the flags, ``high_name``/``low_name``)
         plus ``coupler_lines_hz``, ``probe_lines_hz`` and ``unexplained_lines_hz``

The raw populations are drawn UNCONDITIONALLY; the fit overlay and markers are
guarded, so a run with no line still renders.
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


def _marks(ax, attrs: dict) -> None:
    """f01 and its ladder, then every other line, as vertical markers."""
    f_c = attrs.get("f_c_hz")
    if _finite(f_c):
        ax.axvline(float(f_c) / 1e9, color="tab:blue", ls="--", lw=1.0,
                   label=f"f01 = {float(f_c) / 1e9:.4f} GHz")
    for key, label in (("f02_half_hz", "f02/2"), ("f03_third_hz", "f03/3")):
        f = attrs.get(key)
        if _finite(f):
            ax.axvline(float(f) / 1e9, color="tab:cyan", ls="--", lw=0.9,
                       label=f"{label} = {float(f) / 1e9:.4f} GHz")
    for key, color, label in (("unexplained_lines_hz", "tab:red", "off the ladder"),
                              ("probe_lines_hz", "tab:gray", "not changed by the ramp")):
        values = [f for f in np.atleast_1d(attrs.get(key, [])) if _finite(f)]
        for i, f in enumerate(values):
            ax.axvline(float(f) / 1e9, color=color, ls=":", lw=1.0,
                       label=label if i == 0 else None)


def plot_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Top: the total excitation 1 - P00 of both arms, the fitted lines, f01 and its
    ladder. Middle: each member's excited population, ramp minus reference - where
    the swapped excitation landed. Bottom: the total's ramp-minus-reference
    difference, which decides what is a coupler line."""
    attrs = dict(plot_data.attrs)
    f_ghz = np.asarray(plot_data[AXIS].values, dtype=float) / 1e9
    probe = str(attrs.get("probe", ""))
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})

    for arm, color, what in ((1, "tab:red", "ramp"), (0, "tab:gray", "reference")):
        top.plot(f_ghz, plot_data["total"].sel(ramp_played=arm).values, lw=1.0,
                 color=color, label=f"1 - P00, {what}")
    fit = np.asarray(plot_data["fit_curve"].values, dtype=float)
    if np.isfinite(fit).any():
        top.plot(f_ghz, fit, color="tab:blue", lw=1.0, alpha=0.8, label="fitted lines")
    _marks(top, attrs)
    title = "coupler spectroscopy by swap"
    if probe:
        title += f" (tone on {attrs.get(f'{probe}_name', probe)}'s line)"
    top.set_title(title)
    top.set_ylabel("total excitation")
    top.legend(loc="best", fontsize=7)

    for member, color in zip(plot_data["member"].values, ("tab:purple", "tab:green")):
        member = str(member)
        name = attrs.get(f"{member}_name", member)
        marg = plot_data["marginal"].sel(member=member)
        middle.plot(f_ghz, (marg.sel(ramp_played=1) - marg.sel(ramp_played=0)).values,
                    lw=0.9, color=color, label=f"{name}: ramp - reference")
    middle.axhline(0.0, color="k", lw=0.5)
    middle.set_ylabel("P(excited) change")
    middle.legend(loc="best", fontsize=7)

    bottom.plot(f_ghz, plot_data["difference"].values, lw=1.0, color="k",
                label="1 - P00: ramp - reference")
    bottom.axhline(0.0, color="k", lw=0.5)
    _marks(bottom, attrs)
    flags = [k for k in ("no_line", "unexplained_lines", "peak_at_edge")
             if int(attrs.get(k, 0))]
    if flags:
        bottom.text(0.01, 0.95, "flags: " + ", ".join(flags), transform=bottom.transAxes,
                    va="top", fontsize=8, color="tab:red")
    bottom.set_xlabel("tone frequency (GHz)")
    bottom.set_ylabel("difference")
    fig.tight_layout()
    return fig
