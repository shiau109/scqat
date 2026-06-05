"""EP tomography → ρ₁₁ decoherence pipeline.

This module packages the per-file analysis used in
``notebooks/EP/view_single_raw.ipynb`` so it can be reused across multiple
input files. It chains:

1. HDF5 load + qubit split (``scqat.parsers``).
2. Tomography → density-matrix construction (``rho_11``, ``rho_10``).
3. Per-``driving_frequency`` Hankel pre-analysis (``hankel_decompose``).
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

import numpy as np
import xarray as xr

from scqat.math_tools.fit_multi_damped_oscillation import (
    FitMultiDampedOscillation,
    multi_damped_osc_eval,
)
from scqat.math_tools.fit_qubit_decoherence import FitQubitDecoherence, rho11_model
from scqat.math_tools.hankel import hankel_decompose
from scqat.parsers.qualibrate_parser import repetition_data
from scqat.parsers.xarray_h5_parser import load_xarray_h5

# ---------------------------------------------------------------------------
# Defaults (mirror the values previously hardcoded in view_single_raw.ipynb).
# ---------------------------------------------------------------------------
DEFAULT_BASIS_INDEX = 2          # Z-basis index used for sq_data preview
DEFAULT_RHO11_OFFSET = 0.045     # readout-zero subtraction
DEFAULT_RHO11_SCALE = 0.78       # readout contrast normalization
DEFAULT_TAIL_FRAC = 0.1          # fraction of tail used for baseline mean
DEFAULT_HANKEL_KWARGS: dict[str, Any] = {
    "mode_method": "diff_ratio",
    "recon_method": "mpm",
    "threshold": 3,
    "eigval_threshold": 1e-3,
}

# For data
# DEFAULT_BASIS_INDEX = 2          # Z-basis index used for sq_data preview
# DEFAULT_RHO11_OFFSET = 0.045     # readout-zero subtraction
# DEFAULT_RHO11_SCALE = 0.78       # readout contrast normalization
# DEFAULT_TAIL_FRAC = 0.1          # fraction of tail used for baseline mean
# DEFAULT_HANKEL_KWARGS: dict[str, Any] = {
#     "mode_method": "diff_ratio",
#     "recon_method": "mpm",
#     "threshold": 1.5,
#     "eigval_threshold": 1e-3,
# }

# DEFAULT_HANKEL_KWARGS: dict[str, Any] = {
#     "mode_method": "relative",
#     "recon_method": "mpm",
#     "threshold": 3e-2,
#     "eigval_threshold": 1e-3,
# }
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


def _load_sim_rho_dataset(h5_path: str) -> xr.Dataset:
    """Load a QuTiP simulation HDF5 file and return a density-matrix dataset.

    Expected file layout
    --------------------
    * ``all_expect`` : shape ``(n_freq, n_time, 4)``

      - channel 0 : excited-state population ρ₁₁
      - channel 1 : ⟨σx⟩  →  rho_10_re = ⟨σx⟩ / 2
      - channel 2 : ⟨σy⟩  →  rho_10_im = ⟨σy⟩ / 2
      - channel 3 : resonator photon number ⟨a†a⟩

    * ``omega_flux_vals`` : shape ``(n_freq,)``  — driving-frequency axis
    * ``tlist``           : shape ``(n_time,)``  — time axis
    * root attrs          : simulation parameters (stored in dataset ``.attrs``)

    Returns an ``xr.Dataset`` with coords ``driving_frequency``, ``driving_time``
    and variables ``rho_11``, ``rho_10_re``, ``rho_10_im``, ``photon_num``.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        omega_flux_vals = f["sweep_vals"][()]/(2.0 * np.pi)*1e9  # convert from rad/s to Hz
        tlist = f["tlist"][()]
        all_expect = f["all_expect"][()]
        sim_attrs = {
            k: (v.decode() if isinstance(v, bytes) else v)
            for k, v in f.attrs.items()
        }

    return xr.Dataset(
        {
            "rho_11":     (["driving_frequency", "driving_time"], all_expect[:, :, 0]),
        },
        coords={
            "driving_frequency": omega_flux_vals,
            "driving_time":      tlist,
        },
        attrs=sim_attrs,
    )


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

    out: dict[float, dict[str, Any]] = {}

    for f_val in rho_ds.coords["driving_frequency"].values:
        sub = rho_ds.sel(driving_frequency=f_val)
        t_arr = sub.coords["driving_time"].values.astype(float)
        y_arr = sub["rho_11"].values.astype(float)
        signal, _ = _baseline_subtract(y_arr, tail_frac)

        Lambda_seed: float | None = None
        modes: list[dict[str, Any]] = []
        singular_values: np.ndarray | None = None
        reconstruction: np.ndarray | None = None
        n_modes_svd: int = 0
        try:
            h_results = hankel_decompose(signal, t_arr, **hk)
            modes = h_results.get("modes", [])
            n_modes_svd = h_results.get("n_modes", len(modes))
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
            "n_modes": len(modes),       # physical modes after negative-freq filtering
            "n_modes_svd": n_modes_svd,  # SVD rank from _select_n_modes (counts conjugate pairs)
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


def _run_analysis_stages(
    rho_ds: xr.Dataset,
    qname: str,
    *,
    tail_frac: float = DEFAULT_TAIL_FRAC,
    hankel_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Hankel → MDO → decoherence on a pre-built ``rho_ds``.

    Returns a dict with keys ``hankel``, ``mdo``, ``decoh``, ``decoh_guesses``.
    """
    if verbose:
        print(f"[{qname}] stage 1/3: Hankel pre-analysis ...")
    hankel_diag = run_hankel_per_freq(
        rho_ds, tail_frac=tail_frac, hankel_kwargs=hankel_kwargs, label=qname
    )
    if verbose:
        print(f"[{qname}] stage 1/3: Hankel done")
        print(f"[{qname}] stage 2/3: multi-damped-oscillation fit ...")
    mdo_res = run_mdo_per_freq(rho_ds, hankel_diag, tail_frac=tail_frac, label=qname)
    if verbose:
        print(f"[{qname}] stage 2/3: MDO done")
        print(f"[{qname}] stage 3/3: decoherence fit ...")
    decoh_res, decoh_guesses = run_decoherence_per_freq(rho_ds, hankel_diag, label=qname)
    if verbose:
        print(f"[{qname}] stage 3/3: decoherence done")

    if verbose:
        n_freq = rho_ds.sizes.get("driving_frequency", 0)
        n_mdo_ok = sum(1 for v in mdo_res.values() if v is not None and v.get("success"))
        n_decoh_ok = sum(1 for v in decoh_res.values() if v is not None)
        print(
            f"[{qname}] freqs={n_freq}"
            f"  hankel_seeded={sum(1 for d in hankel_diag.values() if d['Lambda_seed'] is not None)}"
            f"  mdo_ok={n_mdo_ok}  decoh_ok={n_decoh_ok}"
        )

    return {
        "hankel": hankel_diag,
        "mdo": mdo_res,
        "decoh": decoh_res,
        "decoh_guesses": decoh_guesses,
    }


def _prepare_entries_exp(
    h5_path: str,
    *,
    rho11_offset: float = DEFAULT_RHO11_OFFSET,
    rho11_scale: float = DEFAULT_RHO11_SCALE,
    repetition_dim: str = "qubit",
) -> list[dict[str, Any]]:
    """Load an experimental HDF5 file and return a list of ``{qubit_name, sq_data, rho_ds}`` dicts."""
    qubit_datasets = load_and_split(h5_path, repetition_dim=repetition_dim)
    return [
        {
            "qubit_name": _qubit_label(sq_data, fallback=f"Dataset_{i}"),
            "sq_data": sq_data,
            "rho_ds": build_rho_dataset(sq_data, rho11_offset=rho11_offset, rho11_scale=rho11_scale),
        }
        for i, sq_data in enumerate(qubit_datasets)
    ]


def _prepare_entries_sim(h5_path: str) -> list[dict[str, Any]]:
    """Load a simulation HDF5 file and return a single-entry list of ``{qubit_name, sq_data, rho_ds}``."""
    return [{"qubit_name": "simulation", "sq_data": None, "rho_ds": _load_sim_rho_dataset(h5_path)}]


_ENTRY_LOADERS = {
    "exp": _prepare_entries_exp,
    "sim": _prepare_entries_sim,
}


def analyze(
    h5_path: str,
    mode: str = "exp",
    *,
    rho11_offset: float = DEFAULT_RHO11_OFFSET,
    rho11_scale: float = DEFAULT_RHO11_SCALE,
    tail_frac: float = DEFAULT_TAIL_FRAC,
    hankel_kwargs: dict[str, Any] | None = None,
    repetition_dim: str = "qubit",
    time_stride: int = 1,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Run the full EP → ρ₁₁ decoherence pipeline on one HDF5 file.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    mode : {"exp", "sim"}
        ``"exp"`` — experimental tomography file (Qualibrate / xarray HDF5).
        ``"sim"`` — QuTiP simulation file with ``all_expect`` / ``omega_flux_vals``
        / ``tlist`` datasets.
    time_stride : int, optional
        Step size along the ``driving_time`` axis used to downsample before
        analysis.  A value of ``n`` keeps only indices 0, n, 2n, …  This is
        useful when the simulation time axis has >1000 points that would make
        the Hankel pre-analysis prohibitively slow.  Default ``1`` (no
        downsampling).

    Returns
    -------
    list[dict]
        One entry per qubit (``"exp"``) or a single entry (``"sim"``), each with
        keys ``qubit_name``, ``sq_data``, ``rho_ds``, ``hankel``, ``mdo``,
        ``decoh``, ``decoh_guesses``.
        ``sq_data`` is ``None`` for simulation data.
    """
    if mode not in _ENTRY_LOADERS:
        raise ValueError(f"Unknown mode {mode!r}. Expected 'exp' or 'sim'.")

    loader_kwargs: dict[str, Any] = {}
    if mode == "exp":
        loader_kwargs = {"rho11_offset": rho11_offset, "rho11_scale": rho11_scale, "repetition_dim": repetition_dim}
    if verbose:
        print(f"[loading] {h5_path} (mode={mode!r}) ...")
    entries = _ENTRY_LOADERS[mode](h5_path, **loader_kwargs)
    if verbose:
        print(f"[loading] done — {len(entries)} qubit(s) found")

    if time_stride > 1:
        for entry in entries:
            entry["rho_ds"] = entry["rho_ds"].isel(driving_time=slice(None, None, time_stride))
        if verbose:
            n_t = entries[0]["rho_ds"].sizes.get("driving_time", "?")
            print(f"[loading] time_stride={time_stride} → {n_t} driving_time points retained")

    return [
        {
            **entry,
            **_run_analysis_stages(
                entry["rho_ds"], entry["qubit_name"],
                tail_frac=tail_frac,
                hankel_kwargs=hankel_kwargs,
                verbose=verbose,
            ),
        }
        for entry in entries
    ]

