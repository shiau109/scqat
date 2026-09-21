"""
AC-Stark shift plotting helpers.

Every function consumes the **plot_data** Dataset built by
``AcStarkShiftEstimator.build_plot_data`` and recomputes nothing (only unit
conversion), so the figures redraw from the saved ``*_plotdata.nc`` alone. Each
draws its measured data unconditionally and guards every fit overlay, so a
failed fit still leaves a figure.

plot_data layout
----------------
coords : ``amp_prefactor``, ``detuning``, ``amp_dense``
vars   : ``raw_signal`` (amp_prefactor, detuning) — ``|IQ - ref|``;
         ``peak_detuning`` / ``peak_detuning_err`` / ``peak_fwhm`` /
         ``peak_fwhm_err`` / ``accepted`` / ``in_fit`` / ``photon_number``
         (amp_prefactor); ``fit_detuning`` / ``fit_fwhm`` (amp_dense);
         optional ``twin`` (amp_prefactor) — the absolute amplitude
attrs  : ``success``, ``stark_slope_hz``, ``intercept_detuning_hz``,
         ``broadening_slope_hz``, ``fwhm_intercept_hz``, ``chi_hz``,
         ``has_photon``, ``photons_per_prefactor2``, ``amp_ref``,
         ``n_rows_fit``, ``rms_residual_hz``, ``ref_source``, ``twin_label``
"""

import numpy as np
import matplotlib.pyplot as plt

from scqat.estimators._twin_axis import add_twin_axis


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _squared_twin(amp: np.ndarray, twin: np.ndarray):
    """The twin scale squared, when the primary axis is ``amp_prefactor**2``.

    Only drawable while both squares stay strictly monotone, i.e. the swept
    factors do not straddle zero (a negative factor folds onto its positive
    twin). ``None`` otherwise — the figure then keeps its primary axis only.
    """
    if amp.size < 2 or np.any(amp < 0) or np.any(twin < 0):
        return None
    sq_amp, sq_twin = amp ** 2, twin ** 2
    step, twin_step = np.diff(sq_amp), np.diff(sq_twin)
    if not ((step > 0).all() or (step < 0).all()):
        return None
    if not ((twin_step > 0).all() or (twin_step < 0).all()):
        return None
    return sq_amp, sq_twin


def _markers(ax, x, y, yerr, accepted, in_fit, scale):
    """Rows in the fit (filled), accepted but clipped (open), rejected (x)."""
    fit_rows = in_fit & np.isfinite(y)
    clipped = accepted & ~in_fit & np.isfinite(y)
    rejected = ~accepted & np.isfinite(y)
    err = np.where(np.isfinite(yerr), yerr, 0.0) / scale
    if fit_rows.any():
        ax.errorbar(x[fit_rows], y[fit_rows] / scale, yerr=err[fit_rows], fmt="o",
                    color="C0", ms=5, capsize=2, label="rows in the fit")
    if clipped.any():
        ax.plot(x[clipped], y[clipped] / scale, "o", mfc="none", color="C0", ms=6,
                label="accepted, clipped as outlier")
    if rejected.any():
        ax.plot(x[rejected], y[rejected] / scale, "x", color="C3", ms=6,
                label="rejected")


def plot_stark_map(plot_data):
    """The raw ``|IQ - ref|`` map over (amplitude factor, detuning), with the
    line position of every row and the fitted parabola in the factor."""
    amp = plot_data.coords["amp_prefactor"].values
    detuning = plot_data.coords["detuning"].values
    raw = plot_data["raw_signal"].values  # (amp_prefactor, detuning)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    mesh = ax.pcolormesh(amp, detuning / 1e6, raw.T, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="|IQ - ref| (arb. u.)")

    peak = plot_data["peak_detuning"].values
    accepted = plot_data["accepted"].values.astype(bool)
    in_fit = plot_data["in_fit"].values.astype(bool)
    kept = in_fit & np.isfinite(peak)
    others = ~in_fit & np.isfinite(peak)
    if kept.any():
        ax.plot(amp[kept], peak[kept] / 1e6, "o", color="white", mec="k", ms=5,
                label="line (in the fit)")
    if others.any():
        ax.plot(amp[others], peak[others] / 1e6, "x", color="C3", ms=6,
                label="line (not in the fit)")
    fit = plot_data["fit_detuning"].values
    if np.isfinite(fit).any():
        ax.plot(plot_data.coords["amp_dense"].values, fit / 1e6, "-", color="C1", lw=1.5,
                label="f0 + s·a²")
    if kept.any() or others.any() or np.isfinite(fit).any():
        ax.legend(fontsize=8, loc="best")

    ax.set_xlabel("Stark-tone amplitude factor (amp_prefactor)")
    ax.set_ylabel("Drive detuning (MHz)")
    ax.set_title("AC-Stark shift — qubit line vs Stark-tone amplitude")
    if "twin" in plot_data:
        add_twin_axis(ax, amp, plot_data["twin"].values,
                      str(plot_data.attrs.get("twin_label", "")))
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_shift(plot_data):
    """Line position against ``amp_prefactor**2`` with the linear fit; a right
    axis in photon number when chi was supplied."""
    attrs = plot_data.attrs
    amp = plot_data.coords["amp_prefactor"].values
    x = amp ** 2
    peak = plot_data["peak_detuning"].values
    accepted = plot_data["accepted"].values.astype(bool)
    in_fit = plot_data["in_fit"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    _markers(ax, x, peak, plot_data["peak_detuning_err"].values, accepted, in_fit, 1e6)

    fit = plot_data["fit_detuning"].values
    slope, intercept = attrs.get("stark_slope_hz"), attrs.get("intercept_detuning_hz")
    if np.isfinite(fit).any() and _finite(slope):
        dense = plot_data.coords["amp_dense"].values
        ax.plot(dense ** 2, fit / 1e6, "-", color="C1", lw=1.5,
                label=f"s = {float(slope) / 1e6:.4g} MHz per factor²")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    ax.set_xlabel("amp_prefactor²")
    ax.set_ylabel("Line detuning (MHz)")
    ax.set_title("AC-Stark shift vs tone power")
    ax.grid(True, alpha=0.3)

    chi = attrs.get("chi_hz")
    if int(attrs.get("has_photon", 0)) and _finite(chi) and _finite(intercept):
        per_photon_mhz = -2.0 * float(chi) / 1e6
        zero_mhz = float(intercept) / 1e6
        ax.secondary_yaxis(
            "right",
            functions=(lambda y: (y - zero_mhz) / per_photon_mhz,
                       lambda n: zero_mhz + n * per_photon_mhz),
        ).set_ylabel("photon number")
    if "twin" in plot_data:
        squared = _squared_twin(amp, plot_data["twin"].values)
        if squared is not None:
            add_twin_axis(ax, squared[0], squared[1],
                          f"({plot_data.attrs.get('twin_label', '')})²")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_broadening(plot_data):
    """Line FWHM against ``amp_prefactor**2`` with the linear fit — the
    measurement-induced dephasing that grows with the photon number."""
    attrs = plot_data.attrs
    amp = plot_data.coords["amp_prefactor"].values
    x = amp ** 2
    fwhm = plot_data["peak_fwhm"].values
    accepted = plot_data["accepted"].values.astype(bool)
    in_fit = plot_data["in_fit"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    _markers(ax, x, fwhm, plot_data["peak_fwhm_err"].values, accepted, in_fit, 1e6)

    fit = plot_data["fit_fwhm"].values
    b = attrs.get("broadening_slope_hz")
    if np.isfinite(fit).any() and _finite(b):
        dense = plot_data.coords["amp_dense"].values
        ax.plot(dense ** 2, fit / 1e6, "-", color="C1", lw=1.5,
                label=f"b = {float(b) / 1e6:.4g} MHz per factor²")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8, loc="best")

    ax.set_xlabel("amp_prefactor²")
    ax.set_ylabel("Line FWHM (MHz)")
    ax.set_title("Measurement-induced broadening vs tone power")
    ax.grid(True, alpha=0.3)
    if "twin" in plot_data:
        squared = _squared_twin(amp, plot_data["twin"].values)
        if squared is not None:
            add_twin_axis(ax, squared[0], squared[1],
                          f"({plot_data.attrs.get('twin_label', '')})²")
    fig.tight_layout()
    plt.close(fig)
    return fig
