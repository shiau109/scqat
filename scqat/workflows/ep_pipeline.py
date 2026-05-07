"""EP tomography → ρ₁₁ decoherence pipeline.

This module packages the per-file analysis used in
``notebooks/EP/view_single_raw.ipynb`` so it can be reused across multiple
input files. It chains:

1. HDF5 load + qubit split (``scqat.parsers``).
2. Tomography → density-matrix construction (``rho_11``, ``rho_10``).
3. Per-``driving_frequency`` Hankel pre-analysis (``HankelAnalyzer``).
4. Per-``driving_frequency`` multi-damped-oscillation fit
   (``FitMultiDampedOscillation``), seeded from Hankel modes.
5. Per-``driving_frequency`` non-Markovian decoherence fit
   (``FitQubitDecoherence``), with γ seeded from Hankel mode 0.

Plotting is intentionally **not** included — callers (notebooks) are
expected to consume the returned dictionaries and do their own plotting.

Dataset contract for the input file
-----------------------------------
After ``load_xarray_h5`` the dataset must contain:

* coordinates ``driving_time``, ``driving_frequency``, ``basis``, and
  optionally ``qubit``;
* data variable ``state`` indexed by those coordinates, where
  ``state[basis=0,1,2]`` are the X, Y, Z tomography readouts and
  ``state`` is interpreted as :math:`P(|1\\rangle)` in each rotated frame.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.math_tools.fit_multi_damped_oscillation import (
    FitMultiDampedOscillation,
    multi_damped_osc_eval,
)
from scqat.math_tools.fit_qubit_decoherence import FitQubitDecoherence, rho11_model
from scqat.parsers.qualibrate_parser import repetition_data
from scqat.parsers.xarray_h5_parser import load_xarray_h5
from scqat.protocols.hankel_analysis import HankelAnalyzer

# ---------------------------------------------------------------------------
# Defaults (mirror the values previously hardcoded in view_single_raw.ipynb).
# ---------------------------------------------------------------------------
DEFAULT_BASIS_INDEX = 2          # Z-basis index used for sq_data preview
DEFAULT_RHO11_OFFSET = 0.045     # readout-zero subtraction
DEFAULT_RHO11_SCALE = 0.78       # readout contrast normalization
DEFAULT_TAIL_FRAC = 0.1          # fraction of tail used for baseline mean
DEFAULT_HANKEL_KWARGS: dict[str, Any] = {
    "mode_method": "relative",
    "recon_method": "mpm",
    "threshold": 3e-2,
    "eigval_threshold": 1e-3,
}


# ---------------------------------------------------------------------------
# I/O + tomography
# ---------------------------------------------------------------------------
def load_and_split(h5_path: str, repetition_dim: str = "qubit") -> list[xr.Dataset]:
    """Load an EP HDF5 file and split it into per-qubit datasets."""
    ds = load_xarray_h5(h5_path)
    if repetition_dim in ds.dims:
        return repetition_data(ds, repetition_dim=repetition_dim)
    return [ds]


def build_rho_dataset(
    sq_data: xr.Dataset,
    *,
    rho11_offset: float = DEFAULT_RHO11_OFFSET,
    rho11_scale: float = DEFAULT_RHO11_SCALE,
) -> xr.Dataset:
    """Construct a density-matrix dataset (rho_11, rho_10_re, rho_10_im) from
    tomography data with `basis = 0,1,2` for X, Y, Z readouts."""
    if "basis" not in sq_data.coords:
        raise ValueError("Dataset has no 'basis' coordinate; cannot build density matrix.")

    state = sq_data["state"]
    X = (1.0 - 2.0 * state.isel(basis=0)).drop_vars("basis", errors="ignore")
    Y = (1.0 - 2.0 * state.isel(basis=1)).drop_vars("basis", errors="ignore")
    rho_11 = (state.isel(basis=2).drop_vars("basis", errors="ignore") - rho11_offset) / rho11_scale
    rho_10_re = 0.5 * X
    rho_10_im = 0.5 * Y

    rho_ds = xr.Dataset(
        {
            "rho_11": rho_11.astype(float),
            "rho_10_re": rho_10_re.astype(float),
            "rho_10_im": rho_10_im.astype(float),
        },
        attrs=dict(sq_data.attrs),
    )
    return rho_ds.squeeze(drop=True)


# ---------------------------------------------------------------------------
# Per-frequency analysis stages
# ---------------------------------------------------------------------------
def _baseline_subtract(y: np.ndarray, tail_frac: float) -> tuple[np.ndarray, float]:
    tail = max(int(tail_frac * y.size), 1)
    baseline = float(np.mean(y[-tail:]))
    return y - baseline, baseline


def run_hankel_per_freq(
    rho_ds: xr.Dataset,
    *,
    tail_frac: float = DEFAULT_TAIL_FRAC,
    hankel_kwargs: dict[str, Any] | None = None,
    label: str = "",
) -> dict[float, dict[str, Any]]:
    """Run Hankel pre-analysis on tail-mean-subtracted ``rho_11`` per
    ``driving_frequency``. Returns ``{freq: {Lambda_seed, n_modes, modes}}``."""
    if "driving_frequency" not in rho_ds.coords or "driving_time" not in rho_ds.coords:
        raise ValueError("rho dataset must contain 'driving_frequency' and 'driving_time' coords.")

    hk = dict(DEFAULT_HANKEL_KWARGS)
    if hankel_kwargs:
        hk.update(hankel_kwargs)

    analyzer = HankelAnalyzer()
    out: dict[float, dict[str, Any]] = {}

    for f_val in rho_ds.coords["driving_frequency"].values:
        sub = rho_ds.sel(driving_frequency=f_val)
        t_arr = sub.coords["driving_time"].values.astype(float)
        y_arr = sub["rho_11"].values.astype(float)
        signal, _ = _baseline_subtract(y_arr, tail_frac)

        hankel_ds = xr.Dataset({"signal": ("time", signal)}, coords={"time": t_arr})

        Lambda_seed: float | None = None
        modes: list[dict[str, Any]] = []
        singular_values: np.ndarray | None = None
        reconstruction: np.ndarray | None = None
        try:
            h_results, h_figs = analyzer.analyze(hankel_ds, **hk)
            for _f in h_figs.values():
                plt.close(_f)
            modes = h_results.get("modes", [])
            singular_values = h_results.get("singular_values")
            reconstruction = h_results.get("reconstruction")
            if modes:
                Lambda_seed = float(abs(modes[0]["decay_rate"]))
        except Exception as exc:  # noqa: BLE001 - propagate as warning
            warnings.warn(
                f"[{label}] f={float(f_val):.3e} Hz: Hankel pre-analysis failed ({exc}); "
                "continuing without seed."
            )

        out[float(f_val)] = {
            "Lambda_seed": Lambda_seed,
            "n_modes": len(modes),
            "modes": modes,
            "singular_values": singular_values,
            "reconstruction": reconstruction,
        }

    return out


def _mdo_result_dict(fitter: FitMultiDampedOscillation, mr, t: np.ndarray) -> dict[str, Any] | None:
    if mr is None:
        return None
    modes_fit = fitter.unpack_modes(mr)
    c_fit = float(mr.params["c"].value)
    y_fit = multi_damped_osc_eval(
        t,
        [{"a": m["a"], "k": m["k"], "f": m["f"], "phi": m["phi"]} for m in modes_fit],
        c=c_fit,
    )
    return {
        "modes": modes_fit,
        "c": c_fit,
        "fit_curve": y_fit,
        "residuals": (mr.data - y_fit) if mr.data is not None else np.zeros_like(t),
        "chisqr": float(mr.chisqr) if mr.chisqr is not None else float("inf"),
        "success": bool(mr.success),
        "n_modes": len(modes_fit),
    }


def run_mdo_per_freq(
    rho_ds: xr.Dataset,
    hankel_diag: dict[float, dict[str, Any]],
    *,
    tail_frac: float = DEFAULT_TAIL_FRAC,
    label: str = "",
) -> dict[float, dict[str, Any] | None]:
    """Multi-damped-oscillation fit per ``driving_frequency``, seeded by
    Hankel modes from ``hankel_diag``."""
    out: dict[float, dict[str, Any] | None] = {}

    for f_val in rho_ds.coords["driving_frequency"].values:
        diag = hankel_diag.get(float(f_val), {})
        modes_seed = diag.get("modes", [])
        if not modes_seed:
            out[float(f_val)] = None
            continue

        sub = rho_ds.sel(driving_frequency=f_val)
        t_arr = sub.coords["driving_time"].values.astype(float)
        y_raw = sub["rho_11"].values.astype(float)
        y_sig, baseline = _baseline_subtract(y_raw, tail_frac)

        da = xr.DataArray(y_sig, coords={"x": t_arr}, dims="x")
        try:
            fitter = FitMultiDampedOscillation(da, modes=modes_seed)
            fitter.guess()
            mr = fitter.fit()
            res = _mdo_result_dict(fitter, mr, t_arr)
            if res is not None:
                res["baseline"] = baseline
            out[float(f_val)] = res
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{label}] f={float(f_val):.3e} Hz: multi-damped-osc fit failed ({exc})")
            out[float(f_val)] = None

    return out


def _decoh_result_dict(mr, t: np.ndarray) -> dict[str, Any]:
    p = mr.params
    gamma_fit = float(p["gamma"].value)
    lambda_fit = float(p["lambda_"].value)
    Delta_fit = float(p["Delta"].value)
    rho0_fit = float(p["rho_0"].value)

    def _err(name: str) -> float:
        e = p[name].stderr
        return float(e) if e is not None else float("nan")

    y_fit = rho11_model(t, gamma_fit, lambda_fit, Delta_fit, rho0_fit)
    d_sq = (gamma_fit / 2.0) ** 2 - 4.0 * lambda_fit ** 2
    if d_sq > 1e-20:
        regime = "overdamped"
    elif d_sq < -1e-20:
        regime = "underdamped"
    else:
        regime = "critical"

    return {
        "gamma": gamma_fit,
        "gamma_err": _err("gamma"),
        "lambda_": lambda_fit,
        "lambda_err": _err("lambda_"),
        "Delta": Delta_fit,
        "Delta_err": _err("Delta"),
        "d": complex(np.sqrt(np.complex128(d_sq))),
        "rho_0": rho0_fit,
        "rho_0_err": _err("rho_0"),
        "fit_curve": y_fit,
        "residuals": mr.data - y_fit if mr.data is not None else np.zeros_like(t),
        "chisqr": float(mr.chisqr) if mr.chisqr is not None else float("inf"),
        "regime": regime,
    }


def run_decoherence_per_freq(
    rho_ds: xr.Dataset,
    hankel_diag: dict[float, dict[str, Any]],
    *,
    label: str = "",
) -> tuple[dict[float, dict[str, Any] | None], dict[float, dict[str, float]]]:
    """Non-Markovian decoherence fit per ``driving_frequency``, with γ
    seeded from Hankel mode 0. Returns ``(fit_results, initial_guesses)``."""
    fit_out: dict[float, dict[str, Any] | None] = {}
    guess_out: dict[float, dict[str, float]] = {}

    for f_val in rho_ds.coords["driving_frequency"].values:
        sub = rho_ds.sel(driving_frequency=f_val)
        t_arr = sub.coords["driving_time"].values.astype(float)
        y_arr = sub["rho_11"].values.astype(float)

        diag = hankel_diag.get(float(f_val), {})
        Lambda_seed = diag.get("Lambda_seed")

        da = xr.DataArray(y_arr, coords={"x": t_arr}, dims="x")
        fitter = FitQubitDecoherence(da, component="rho_11")
        params = fitter.guess()
        gamma_min = params["gamma"].min
        gamma_max = params["gamma"].max
        rho_0_default = float(params["rho_0"].value)
        Delta_default = float(params["Delta"].value)

        if Lambda_seed is not None:
            gamma_seed = 2.0 * Lambda_seed
            if gamma_seed < gamma_min or gamma_seed > gamma_max:
                warnings.warn(
                    f"[{label}] f={float(f_val):.3e} Hz: Hankel-derived gamma seed "
                    f"{gamma_seed:.4g} outside bounds [{gamma_min:.4g}, {gamma_max:.4g}]; "
                    "clipping to bounds."
                )
                gamma_seed = float(np.clip(gamma_seed, gamma_min, gamma_max))
            params["gamma"].set(value=gamma_seed)
            params["lambda_"].set(value=gamma_seed / 4)
            fitter.params = params
            guess_out[float(f_val)] = {
                "gamma": gamma_seed,
                "lambda_": gamma_seed / 4,
                "Delta": Delta_default,
                "rho_0": rho_0_default,
            }
        else:
            guess_out[float(f_val)] = {
                "gamma": float(params["gamma"].value),
                "lambda_": float(params["lambda_"].value),
                "Delta": Delta_default,
                "rho_0": rho_0_default,
            }

        try:
            mr = fitter.fit()
            fit_out[float(f_val)] = _decoh_result_dict(mr, t_arr)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[{label}] f={float(f_val):.3e} Hz: decoherence fit failed ({exc})")
            fit_out[float(f_val)] = None

    return fit_out, guess_out


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def _qubit_label(sq_data: xr.Dataset, fallback: str) -> str:
    try:
        if "qubit" in sq_data:
            return str(sq_data["qubit"].values.item())
    except Exception:
        pass
    return fallback


def analyze_file(
    h5_path: str,
    *,
    rho11_offset: float = DEFAULT_RHO11_OFFSET,
    rho11_scale: float = DEFAULT_RHO11_SCALE,
    tail_frac: float = DEFAULT_TAIL_FRAC,
    hankel_kwargs: dict[str, Any] | None = None,
    repetition_dim: str = "qubit",
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the full EP tomography → ρ₁₁ decoherence pipeline on one HDF5 file.

    Returns a list (one entry per qubit) with keys:
    ``qubit_name``, ``sq_data``, ``rho_ds``, ``hankel``, ``mdo``,
    ``decoh``, ``decoh_guesses``.
    """
    qubit_datasets = load_and_split(h5_path, repetition_dim=repetition_dim)
    results: list[dict[str, Any]] = []

    for i, sq_data in enumerate(qubit_datasets):
        qname = _qubit_label(sq_data, fallback=f"Dataset_{i}")

        rho_ds = build_rho_dataset(
            sq_data,
            rho11_offset=rho11_offset,
            rho11_scale=rho11_scale,
        )

        hankel_diag = run_hankel_per_freq(
            rho_ds,
            tail_frac=tail_frac,
            hankel_kwargs=hankel_kwargs,
            label=qname,
        )
        mdo_res = run_mdo_per_freq(rho_ds, hankel_diag, tail_frac=tail_frac, label=qname)
        decoh_res, decoh_guesses = run_decoherence_per_freq(rho_ds, hankel_diag, label=qname)

        if verbose:
            n_freq = rho_ds.sizes.get("driving_frequency", 0)
            n_mdo_ok = sum(1 for v in mdo_res.values() if v is not None and v.get("success"))
            n_decoh_ok = sum(1 for v in decoh_res.values() if v is not None)
            print(
                f"[{qname}] freqs={n_freq}  hankel_seeded={sum(1 for d in hankel_diag.values() if d['Lambda_seed'] is not None)}"
                f"  mdo_ok={n_mdo_ok}  decoh_ok={n_decoh_ok}"
            )

        results.append(
            {
                "qubit_name": qname,
                "sq_data": sq_data,
                "rho_ds": rho_ds,
                "hankel": hankel_diag,
                "mdo": mdo_res,
                "decoh": decoh_res,
                "decoh_guesses": decoh_guesses,
            }
        )

    return results
