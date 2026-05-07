"""Plotting helpers for the EP tomography → ρ₁₁ pipeline.

These produce the same figures shown by ``notebooks/EP/view_single_raw.ipynb``
so they can be reused for batch processing (saving to disk per file).

The single entry point is :func:`make_figures`, which takes one entry from
``analyze_file`` (or the wrapped form used by ``batch_raw.ipynb``) and returns
``{name: matplotlib.figure.Figure}``. Callers decide whether to ``plt.show()``,
save, or close the figures.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.math_tools.fit_multi_damped_oscillation import multi_damped_osc_eval
from scqat.math_tools.fit_qubit_decoherence import rho11_model
from scqat.protocols.qubit_decoherence import QubitDecoherenceAnalyzer


# ---------------------------------------------------------------------------
# Per-qubit plot builders
# ---------------------------------------------------------------------------
def _heatmap_state(sq_data: xr.Dataset, qname: str) -> plt.Figure | None:
    """State heatmap vs driving_time × driving_frequency (basis=2)."""
    if "state" not in sq_data.data_vars:
        return None
    sd = sq_data.isel(basis=2) if "basis" in sq_data.coords else sq_data

    da = sd["state"]
    extra = [d for d in da.dims if d not in ("driving_time", "driving_frequency")]
    if extra:
        da = da.isel({d: 0 for d in extra})
    da = da.transpose("driving_frequency", "driving_time")

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(
        da["driving_time"].values,
        da["driving_frequency"].values,
        da.values,
        shading="auto",
        cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label="state")
    ax.set_xlabel("driving_time")
    ax.set_ylabel("driving_frequency")
    ax.set_title(f"{qname}: state")
    fig.tight_layout()
    return fig


def _hankel_summary(per_freq_hankel: dict[float, dict[str, Any]], qname: str) -> plt.Figure | None:
    items = sorted(per_freq_hankel.items())
    if not items:
        return None
    freqs = np.array([f for f, _ in items])
    lambda_seeds = np.array(
        [d["Lambda_seed"] if d["Lambda_seed"] is not None else np.nan for _, d in items],
        dtype=float,
    )
    n_modes_arr = np.array([d["n_modes"] for _, d in items], dtype=int)

    mode_data: dict[int, dict[str, list]] = {}
    for f_drive, diag in items:
        for k, mode in enumerate(diag.get("modes", [])):
            md = mode_data.setdefault(k, {"drive_f": [], "freq_hz": [], "decay": [], "amplitude": []})
            md["drive_f"].append(f_drive)
            md["freq_hz"].append(mode.get("freq_hz", np.nan))
            md["decay"].append(mode.get("decay_rate", np.nan))
            md["amplitude"].append(mode.get("amplitude", np.nan))

    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    axes[0].plot(freqs, lambda_seeds, "o-", color="C0")
    axes[0].set_ylabel(r"$\Lambda_\mathrm{seed}$ (= |decay rate mode 0|)")
    axes[0].set_title(rf"{qname}: Hankel mode 0 $\Lambda_\mathrm{{seed}}$ vs driving frequency")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(freqs, n_modes_arr, "s-", color="C1")
    axes[1].set_ylabel("Number of Hankel modes retained")
    axes[1].set_title(rf"{qname}: Hankel n_modes vs driving frequency")
    axes[1].yaxis.get_major_locator().set_params(integer=True)
    axes[1].grid(True, alpha=0.3)

    for k, md in sorted(mode_data.items()):
        color = f"C{k}"
        label = f"mode {k}" + (" (seed)" if k == 0 else "")
        size = 40 if k == 0 else 25
        zorder = 3 if k == 0 else 2
        alpha = 1.0 if k == 0 else 0.6
        axes[2].scatter(md["drive_f"], md["freq_hz"], s=size, color=color, zorder=zorder, alpha=alpha, label=label)
        axes[3].scatter(md["drive_f"], md["amplitude"], s=size, color=color, zorder=zorder, alpha=alpha, label=label)
        axes[4].scatter(md["drive_f"], md["decay"], s=size, color=color, zorder=zorder, alpha=alpha, label=label)

    axes[2].set_ylabel("freq_hz (Hz)")
    axes[2].set_title(rf"{qname}: Hankel mode freq_hz vs driving frequency")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].set_ylabel("amplitude (arb.)")
    axes[3].set_title(rf"{qname}: Hankel mode amplitude vs driving frequency")
    axes[3].legend(loc="best", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    axes[4].set_ylabel("decay_rate (1/time_unit)")
    axes[4].set_xlabel("driving_frequency (Hz)")
    axes[4].set_title(rf"{qname}: Hankel mode decay_rate vs driving frequency")
    axes[4].legend(loc="best", fontsize=8)
    axes[4].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _mdo_overlay(rho_ds: xr.Dataset, per_freq_mdo: dict, qname: str) -> plt.Figure | None:
    t = rho_ds.coords["driving_time"].values.astype(float)
    freqs_sorted = np.sort(np.array(list(per_freq_mdo.keys())))
    if len(freqs_sorted) == 0:
        return None

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=freqs_sorted.min(), vmax=freqs_sorted.max())

    fig, ax = plt.subplots(figsize=(10, 6))
    for f_val in freqs_sorted:
        color = cmap(norm(f_val))
        y_data = rho_ds["rho_11"].sel(driving_frequency=f_val).values
        ax.plot(t, y_data, ".", ms=3, color=color, alpha=0.6)
        res = per_freq_mdo.get(f_val)
        if res is not None:
            ax.plot(t, res["fit_curve"] + res.get("baseline", 0.0), "-", lw=1.2, color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("driving_frequency (Hz)")
    ax.set_xlabel("driving_time")
    ax.set_ylabel(r"$\rho_{11}$")
    ax.set_title(rf"{qname}: raw $\rho_{{11}}$ (dots) and multi-damped-osc fits (lines)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _mdo_param_summary(per_freq_mdo: dict, qname: str) -> plt.Figure | None:
    items = sorted([(f, r) for f, r in per_freq_mdo.items() if r is not None])
    if not items:
        return None
    mode_data: dict[int, dict[str, list]] = {}
    for f_drive, r in items:
        for k, m in enumerate(r["modes"]):
            md = mode_data.setdefault(k, {"drive_f": [], "a": [], "k_decay": [], "f_hz": []})
            md["drive_f"].append(f_drive)
            md["a"].append(m["a"])
            md["k_decay"].append(m["k"])
            md["f_hz"].append(m["f"])
    c_drive_f = [f for f, _ in items]
    c_vals = [r["c"] for _, r in items]

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    for k, md in sorted(mode_data.items()):
        color = f"C{k}"
        label = f"mode {k}"
        axes[0].scatter(md["drive_f"], md["a"], color=color, label=label)
        axes[1].scatter(md["drive_f"], md["k_decay"], color=color, label=label)
        axes[2].scatter(md["drive_f"], md["f_hz"], color=color, label=label)
    axes[3].scatter(c_drive_f, c_vals, color="C4")

    axes[0].set_ylabel("amplitude $a$")
    axes[0].set_title(f"{qname}: multi-damped-osc fit parameters vs driving_frequency")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("decay rate $k$ (1/time_unit)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel("frequency $f$ (Hz)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[3].set_ylabel("constant $c$")
    axes[3].set_xlabel("driving_frequency (Hz)")
    axes[3].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _mdo_best_inspection(rho_ds: xr.Dataset, per_freq_mdo: dict, qname: str) -> plt.Figure | None:
    valid = [(f, r) for f, r in per_freq_mdo.items() if r is not None and np.isfinite(r["chisqr"])]
    if not valid:
        return None
    f_val, res = min(valid, key=lambda x: x[1]["chisqr"])
    sub = rho_ds.sel(driving_frequency=f_val)
    t = sub.coords["driving_time"].values.astype(float)
    y_raw = sub["rho_11"].values.astype(float)
    baseline = res.get("baseline", 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(t, y_raw, ".", ms=4, color="C0", alpha=0.6, label="data")
    axes[0].plot(t, res["fit_curve"] + baseline, "-", lw=2, color="C1", label="MDO fit (total)")
    for k, mode in enumerate(res["modes"]):
        y_mode = multi_damped_osc_eval(
            t,
            [{"a": mode["a"], "k": mode["k"], "f": mode["f"], "phi": mode["phi"]}],
            c=0.0,
        )
        axes[0].plot(
            t, y_mode + baseline, "--", lw=1.2, alpha=0.8,
            label=f"mode {k}  (a={mode['a']:.3g}, k={mode['k']:.3g}, f={mode['f']:.3g} Hz)",
        )
    axes[0].set_ylabel(r"$\rho_{11}$")
    axes[0].set_title(rf"{qname}: best MDO fit at driving_frequency = {f_val:.6g} Hz "
                      rf"(chisqr={res['chisqr']:.4g})")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, res["residuals"], ".", ms=3, color="C2", alpha=0.7)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].set_xlabel("driving_time")
    axes[1].set_ylabel("residuals")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _mdo_seed_diff(per_freq_mdo: dict, hankel_diag: dict, qname: str) -> plt.Figure | None:
    items = sorted([(f, r) for f, r in per_freq_mdo.items() if r is not None])
    if not items:
        return None
    max_modes = max(r["n_modes"] for _, r in items)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for k in range(max_modes):
        color = f"C{k}"
        drive_f, dk_diff, df_diff, da_diff = [], [], [], []
        for f_drive, r in items:
            if k >= r["n_modes"]:
                continue
            seed_modes = hankel_diag.get(float(f_drive), {}).get("modes", [])
            if k >= len(seed_modes):
                continue
            drive_f.append(f_drive)
            dk_diff.append(seed_modes[k].get("decay_rate", np.nan) - r["modes"][k]["k"])
            df_diff.append(seed_modes[k].get("freq_hz", np.nan) - r["modes"][k]["f"])
            da_diff.append(seed_modes[k].get("amplitude", np.nan) / r["modes"][k]["a"])
        if drive_f:
            axes[0].scatter(drive_f, dk_diff, color=color, marker="o", s=30, label=f"mode {k}")
            axes[1].scatter(drive_f, df_diff, color=color, marker="o", s=30, label=f"mode {k}")
            axes[2].scatter(drive_f, da_diff, color=color, marker="o", s=30, label=f"mode {k}")

    for ax in axes[:2]:
        ax.axhline(0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel(r"$\Delta k$ (seed $-$ fit)  [1/time_unit]")
    axes[0].set_title(f"{qname}: decay rate difference (seed − fit) per mode")
    axes[0].set_ylim(-0.001, 0.001)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel(r"$\Delta f$ (seed $-$ fit)  [Hz]")
    axes[1].set_title(f"{qname}: frequency difference (seed − fit) per mode")
    axes[1].set_ylim(-0.001, 0.001)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].set_ylabel(r"$a_{seed}/a_{fit}$")
    axes[2].set_xlabel("driving_frequency (Hz)")
    axes[2].set_title(f"{qname}: amplitude ratio (seed / fit) per mode")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _decoh_summary(per_freq: dict, decoh_initial_guesses: dict, qname: str) -> plt.Figure | None:
    items = sorted([(f, r) for f, r in per_freq.items() if r is not None])
    if not items:
        return None
    freqs = np.array([f for f, _ in items])
    gamma = np.array([r["gamma"] for _, r in items], dtype=float)
    gamma_err = np.array([r["gamma_err"] for _, r in items], dtype=float)
    lam = np.array([r["lambda_"] for _, r in items], dtype=float)
    lam_err = np.array([r["lambda_err"] for _, r in items], dtype=float)
    delta = np.array([r["Delta"] for _, r in items], dtype=float)
    delta_err = np.array([r["Delta_err"] for _, r in items], dtype=float)
    abs_delta = np.abs(delta)

    guess_gamma = np.array([decoh_initial_guesses[f]["gamma"] for f, _ in items], dtype=float)
    guess_lam = np.array([decoh_initial_guesses[f]["lambda_"] for f, _ in items], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = 8 * lam ** 2 / gamma ** 2

    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    axes[0].errorbar(freqs, gamma, yerr=gamma_err, fmt="o-", capsize=3, label="fit")
    axes[0].plot(freqs, guess_gamma, "x--", color="C3", alpha=0.7, label="initial guess (gamma_seed)")
    axes[0].set_ylabel(r"$\gamma$")
    axes[0].set_title(rf"{qname}: $\gamma$ vs driving frequency")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(freqs, lam, yerr=lam_err, fmt="s-", color="C1", capsize=3, label="fit")
    axes[1].plot(freqs, guess_lam, "x--", color="C3", alpha=0.7, label="initial guess")
    axes[1].set_ylabel(r"$\lambda$")
    axes[1].set_title(rf"{qname}: $\lambda$ vs driving frequency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(freqs, delta, yerr=delta_err, fmt="^-", color="C2", capsize=3, label="fit")
    axes[2].axhline(0, color="k", lw=0.8, ls="--")
    axes[2].set_ylabel(r"$\Delta$")
    axes[2].set_title(rf"{qname}: $\Delta$ vs driving frequency")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].errorbar(freqs, abs_delta, yerr=delta_err, fmt="D-", color="C5", capsize=3, label="fit")
    axes[3].set_ylabel(r"$|\Delta|$")
    axes[3].set_title(rf"{qname}: $|\Delta|$ vs driving frequency")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    axes[4].scatter(freqs, ratio, color="C4")
    axes[4].set_xlabel("driving_frequency (Hz)")
    axes[4].set_ylabel(r"$8\lambda^2/\gamma^2$")
    axes[4].set_title(rf"{qname}: $8\lambda^2/\gamma^2$ vs driving frequency")
    axes[4].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _decoh_overlay(rho_ds: xr.Dataset, per_freq: dict, qname: str) -> plt.Figure | None:
    t = rho_ds.coords["driving_time"].values.astype(float)
    freqs_sorted = np.sort(np.array(list(per_freq.keys())))
    if len(freqs_sorted) == 0:
        return None
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=freqs_sorted.min(), vmax=freqs_sorted.max())

    fig, ax = plt.subplots(figsize=(10, 6))
    for f_val in freqs_sorted:
        color = cmap(norm(f_val))
        y_data = rho_ds["rho_11"].sel(driving_frequency=f_val).values
        ax.plot(t, y_data, ".", ms=3, color=color, alpha=0.6)
        res = per_freq.get(f_val)
        if res is not None:
            y_fit = rho11_model(t, res["gamma"], res["lambda_"], res["Delta"], res["rho_0"])
            ax.plot(t, y_fit, "-", lw=1.5, color=color)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("driving_frequency (Hz)")
    ax.set_xlabel("driving_time")
    ax.set_ylabel(r"$\rho_{11}$")
    ax.set_title(rf"{qname}: raw $\rho_{{11}}$ (dots) and decoherence fits (lines)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _decoh_best_inspection(
    rho_ds: xr.Dataset,
    per_freq: dict,
    decoh_initial_guesses: dict,
    qname: str,
) -> plt.Figure | None:
    valid = [(f, r) for f, r in per_freq.items() if r is not None and np.isfinite(r["chisqr"])]
    if not valid:
        return None
    f_val, _ = min(valid, key=lambda x: x[1]["chisqr"])

    sub = rho_ds.sel(driving_frequency=f_val).rename({"driving_time": "time"})
    fit_ds = sub[["rho_11"]]
    analyzer = QubitDecoherenceAnalyzer()
    _results, figs = analyzer.analyze(fit_ds)

    fig = figs.get("rho_11")
    # Drop any other figures the analyzer produced
    for key, f in figs.items():
        if key != "rho_11":
            plt.close(f)
    if fig is None:
        return None

    t_init = sub.coords["time"].values.astype(float)
    guess = decoh_initial_guesses[float(f_val)]
    g0, l0, d0, r0 = guess["gamma"], guess["lambda_"], guess.get("Delta", 0.0), guess["rho_0"]
    y_guess = rho11_model(t_init, g0, l0, d0, r0)
    ax_top = fig.axes[0]
    ax_top.plot(
        t_init, y_guess, "--", color="C2", lw=1.5,
        label=f"initial guess (γ={g0:.4g}, λ={l0:.4g}, Δ={d0:.4g})",
    )
    ax_top.legend()
    fig.suptitle(f"{qname}: best decoherence fit at f = {f_val:.6g} Hz")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def make_figures(per_qubit_result: dict[str, Any]) -> dict[str, plt.Figure]:
    """Build all view-style figures for a single qubit result.

    Parameters
    ----------
    per_qubit_result : dict
        One entry from ``analyze_file(...)``. Must contain
        ``qubit_name``, ``sq_data``, ``rho_ds``, ``hankel``, ``mdo``,
        ``decoh``, ``decoh_guesses``.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Mapping figure-name → Figure. Names are stable so callers can
        save them with predictable filenames.
    """
    qname = per_qubit_result["qubit_name"]
    sq_data = per_qubit_result["sq_data"]
    rho_ds = per_qubit_result["rho_ds"]
    hankel_diag = per_qubit_result["hankel"]
    mdo_res = per_qubit_result["mdo"]
    decoh_res = per_qubit_result["decoh"]
    decoh_guesses = per_qubit_result["decoh_guesses"]

    builders = {
        "01_state_heatmap":      lambda: _heatmap_state(sq_data, qname),
        "02_hankel_summary":     lambda: _hankel_summary(hankel_diag, qname),
        "03_mdo_overlay":        lambda: _mdo_overlay(rho_ds, mdo_res, qname),
        "04_mdo_param_summary":  lambda: _mdo_param_summary(mdo_res, qname),
        "05_mdo_best_fit":       lambda: _mdo_best_inspection(rho_ds, mdo_res, qname),
        "06_mdo_seed_diff":      lambda: _mdo_seed_diff(mdo_res, hankel_diag, qname),
        "07_decoh_summary":      lambda: _decoh_summary(decoh_res, decoh_guesses, qname),
        "08_decoh_overlay":      lambda: _decoh_overlay(rho_ds, decoh_res, qname),
        "09_decoh_best_fit":     lambda: _decoh_best_inspection(rho_ds, decoh_res, decoh_guesses, qname),
    }

    out: dict[str, plt.Figure] = {}
    for name, build in builders.items():
        fig = build()
        if fig is not None:
            out[f"{qname}_{name}"] = fig
    return out


def save_figures(figs: dict[str, plt.Figure], output_dir: str, *, ext: str = "png", dpi: int = 150) -> list[str]:
    """Save figures into ``output_dir``. Returns the list of written paths."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for name, fig in figs.items():
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    return written
