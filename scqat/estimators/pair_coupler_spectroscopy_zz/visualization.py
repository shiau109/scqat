"""Figures for ``pair_coupler_spectroscopy_zz``.

Consumes the **plot_data** Dataset built by
``PairCouplerSpectroscopyZZEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``tone_freq_hz``, ``pi_played`` (0 = reference, 1 = pi), ``joint_state``,
         ``member`` (``high``/``low``)
vars   : ``joint_population`` (joint_state, pi_played, tone_freq_hz);
         ``marginal`` (member, pi_played, tone_freq_hz); ``difference``
         (tone_freq_hz) = the pi member's pi minus reference; ``fit_curve``
         (tone_freq_hz; the fitted dips, all-NaN without one)
attrs  : every scalar of the results (``pi_member``, ``f_c_hz``, ``alpha_hz``,
         ``f02_half_hz``, ``f03_third_hz``, ``pi_contrast``, the flags,
         ``high_name``/``low_name``) plus ``lines_hz`` and ``unexplained_lines_hz``

The raw populations are drawn UNCONDITIONALLY; the fit overlay and markers are
guarded, so a run with no dip still renders.
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
    """f01 and its ladder, then every line off it, as vertical markers."""
    f_c = attrs.get("f_c_hz")
    if _finite(f_c):
        ax.axvline(float(f_c) / 1e9, color="tab:blue", ls="--", lw=1.0,
                   label=f"f01 = {float(f_c) / 1e9:.4f} GHz")
    for key, label in (("f02_half_hz", "f02/2"), ("f03_third_hz", "f03/3")):
        f = attrs.get(key)
        if _finite(f):
            ax.axvline(float(f) / 1e9, color="tab:cyan", ls="--", lw=0.9,
                       label=f"{label} = {float(f) / 1e9:.4f} GHz")
    values = [f for f in np.atleast_1d(attrs.get("unexplained_lines_hz", [])) if _finite(f)]
    for i, f in enumerate(values):
        ax.axvline(float(f) / 1e9, color="tab:red", ls=":", lw=1.0,
                   label="off the ladder" if i == 0 else None)


def plot_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Top: each member's excited population in both arms (the pi member solid).
    Bottom: the pi member's pi-minus-reference difference, its fitted dips, f01 and
    its ladder."""
    attrs = dict(plot_data.attrs)
    f_ghz = np.asarray(plot_data[AXIS].values, dtype=float) / 1e9
    pi_member = str(attrs.get("pi_member", "high"))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for member in plot_data["member"].values:
        member = str(member)
        name = attrs.get(f"{member}_name", member)
        mine = member == pi_member
        for arm, color, what in ((1, "tab:red", "pi"), (0, "tab:gray", "reference")):
            y = plot_data["marginal"].sel(member=member, pi_played=arm).values
            top.plot(f_ghz, y, lw=1.0 if mine else 0.7, alpha=1.0 if mine else 0.4,
                     color=color if mine else ("tab:purple" if arm else "tab:green"),
                     label=f"{name}{' (pi member)' if mine else ' (tone line)'} {what}")
    top.set_ylabel("P(excited)")
    top.set_title("coupler spectroscopy by ZZ")
    top.legend(loc="best", fontsize=7)

    bottom.plot(f_ghz, plot_data["difference"].values, lw=1.0, color="k",
                label="pi member: pi - reference")
    fit = np.asarray(plot_data["fit_curve"].values, dtype=float)
    if np.isfinite(fit).any():
        bottom.plot(f_ghz, fit, color="tab:blue", lw=1.0, alpha=0.8, label="fitted dips")
    _marks(bottom, attrs)
    notes = []
    if _finite(attrs.get("pi_contrast")):
        notes.append(f"pi contrast {float(attrs['pi_contrast']):.2f}")
    flags = [k for k in ("no_line", "unexplained_lines", "peak_at_edge")
             if int(attrs.get(k, 0))]
    if flags:
        notes.append("flags: " + ", ".join(flags))
    if notes:
        bottom.text(0.01, 0.05, "; ".join(notes), transform=bottom.transAxes,
                    va="bottom", fontsize=8, color="tab:red" if flags else "k")
    bottom.set_xlabel("tone frequency (GHz)")
    bottom.set_ylabel("difference")
    bottom.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig
