"""Repeated partial swaps read by two-qubit tomography.

One member of a pair is excited, the pair's swap is repeated N times with a
fixed AC-Stark tone between swaps, and the pair is then measured in the nine
two-qubit Pauli bases. Per stark amplitude and swap count this estimator
reconstructs the density matrix, and per stark amplitude it fits the swap
CHANNEL (``scqat.tools.swap_channel``) to the trajectory:

* ``theta`` - the exchange angle per step (what a chain's theory needs);
* ``phi`` - the total relative Z phase per step, i.e. what a compensation must
  cancel (stark + frame + any detuning + pulse-edge phases, inseparable);
* ``p_high`` / ``p_low`` - each member's T1 loss per step, ``lam`` - the
  dephasing of the single-excitation coherence per step, ``eps`` - the prep
  error, and the |11> population's growth per step.

Population data alone only see ``cos(theta_eff) = cos(phi/2) cos(theta)``; the
3D trajectory separates the two, at ANY stark amplitude. Across amplitudes the
phase's zero crossing is the compensating amplitude; theta must not depend on
the amplitude, so its spread is the model check. The reported ``theta_rad`` is
the fit at the amplitude whose phase is smallest (where theta is best
determined).

Dataset contract (one pair; dims in any order)
----------------------------------------------
* ``joint_population`` over ``joint_state`` (labels ``00 01 10 11``, digit 0 =
  the high member), ``stark_amp``, ``swap_count`` (non-negative integers, 0
  included) and ``basis`` (the nine labels of
  ``scqat.tools.two_qubit_tomography.BASIS_LABELS``).
* optional ``calibration_population`` over ``prepared_state`` (``00 .. 11``) x
  ``joint_state``: the in-run readout calibration. Without it the stored
  per-member fidelities (``fid_high`` / ``fid_low`` kwargs) correct the
  readout, and without those nothing does (``readout_correction`` says which).

kwargs: ``drive_side`` ("high" | "low": the member excited at N=0),
``high_name`` / ``low_name`` (figure labels), ``fid_high`` / ``fid_low``
(``(P(0|0), P(1|1))``), and for the T1/T2* comparison ``round_duration_ns``,
``t1_high_s``, ``t1_low_s``, ``t2_star_high_s``, ``t2_star_low_s``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.tools.sweep_order import ascending
from scqat.tools.swap_channel import features_of, fit_swap_channel, phase_root
from scqat.tools.two_qubit_tomography import (
    BASIS_LABELS,
    confusion_from_calibration,
    confusion_from_fidelities,
    correct_readout,
    dual_rail,
    pauli_expectations,
    purity,
    rho_from_paulis,
)

from .visualization import (
    FIG_BLOCH,
    FIG_COMPENSATION,
    FIG_COMPONENTS,
    plot_bloch,
    plot_compensation,
    plot_components,
)

AMP = "stark_amp"
COUNT = "swap_count"
JOINT_LABELS = ("00", "01", "10", "11")

#: per-amplitude fit columns carried into metadata and plot data
_COLUMNS = {
    "theta_per_amp": "theta", "theta_err_per_amp": "theta_err",
    "phase_per_step_rad": "phi", "phase_err_rad": "phi_err",
    "frame_offset_rad": "a_off",
    "t1_loss_high_per_amp": "p_high", "t1_loss_low_per_amp": "p_low",
    "dephasing_per_amp": "lam", "prep_error_per_amp": "eps",
    "fit_rms_per_amp": "rms",
}

#: the scalar metadata keys and their failed-fit defaults
_SCALARS = {
    "theta_rad": float("nan"), "theta_rad_err": float("nan"),
    "theta_spread_rad": float("nan"), "theta_consistency_sigma": float("nan"),
    "theta_stark_amp": float("nan"),
    "phase_at_theta_amp_rad": float("nan"),
    "compensating_stark_amp": float("nan"), "compensation_extrapolated": 0,
    "t1_loss_per_step_high": float("nan"), "t1_loss_per_step_low": float("nan"),
    "dephasing_per_step": float("nan"), "prep_error": float("nan"),
    "leak_to_11_per_step": float("nan"),
    "predicted_t1_loss_high": float("nan"), "predicted_t1_loss_low": float("nan"),
    "predicted_dephasing": float("nan"), "excess_dephasing_per_step": float("nan"),
    "fit_rms": float("nan"), "n_fit_ok": 0, "success": 0,
}


def _ordered(dataset: xr.Dataset) -> xr.Dataset:
    return ascending(dataset, AMP, COUNT)


def _confusion(dataset: xr.Dataset, fid_high, fid_low) -> tuple[np.ndarray, str]:
    if "calibration_population" in dataset.data_vars:
        cal = dataset["calibration_population"].sel(
            prepared_state=list(JOINT_LABELS), joint_state=list(JOINT_LABELS)
        ).transpose("prepared_state", "joint_state").values
        try:
            return confusion_from_calibration(cal), "calibration"
        except ValueError:
            pass
    if fid_high is not None and fid_low is not None:
        return confusion_from_fidelities(tuple(fid_high), tuple(fid_low)), "stored_product"
    return np.eye(4), "none"


def _reconstruct(dataset: xr.Dataset, confusion: np.ndarray) -> Dict[str, np.ndarray]:
    """rho over (amp, count) plus the dual-rail quantities and the fit features."""
    jp = dataset["joint_population"].sel(
        joint_state=list(JOINT_LABELS), basis=list(BASIS_LABELS)
    ).transpose(AMP, COUNT, "basis", "joint_state").values.astype(float)
    corrected = {b: correct_readout(jp[:, :, k, :], confusion)
                 for k, b in enumerate(BASIS_LABELS)}
    rho = rho_from_paulis(pauli_expectations(corrected))
    out = dual_rail(rho)
    out["purity"] = purity(rho)
    out["features"] = features_of(rho)
    return out


def _predicted(round_ns, t1_high, t1_low, t2_high, t2_low) -> Dict[str, float]:
    nan = float("nan")
    out = {"predicted_t1_loss_high": nan, "predicted_t1_loss_low": nan,
           "predicted_dephasing": nan}
    try:
        t = float(round_ns) * 1e-9
        t1h, t1l = float(t1_high), float(t1_low)
        out["predicted_t1_loss_high"] = 1.0 - np.exp(-t / t1h)
        out["predicted_t1_loss_low"] = 1.0 - np.exp(-t / t1l)
        gphi = (1.0 / float(t2_high) - 0.5 / t1h) + (1.0 / float(t2_low) - 0.5 / t1l)
        out["predicted_dephasing"] = 1.0 - np.exp(-t * gphi)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return {k: float(v) for k, v in out.items()}


class QcNSwapTomographyEstimator(BaseEstimator):
    """Fit the repeated-swap channel to two-qubit tomography, per stark amplitude."""

    estimator_name = "qc_n_swap_tomography"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError("qc_n_swap_tomography requires the joint_population variable "
                             f"(found data_vars: {list(dataset.data_vars)})")
        for axis in ("joint_state", AMP, COUNT, "basis"):
            if axis not in dataset.coords:
                raise ValueError(f"qc_n_swap_tomography requires a {axis!r} coordinate")
        labels = {str(b) for b in dataset["basis"].values}
        if labels != set(BASIS_LABELS):
            raise ValueError(f"basis must hold exactly {BASIS_LABELS}, got {sorted(labels)}")
        counts = np.asarray(dataset[COUNT].values)
        if counts.size < 4 or counts.min() < 0 or 0 not in counts:
            raise ValueError("swap_count must hold at least four non-negative counts including 0")

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "high",
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        fid_high=None, fid_low=None, round_duration_ns=None,
        t1_high_s=None, t1_low_s=None, t2_star_high_s=None, t2_star_low_s=None,
        **kwargs,
    ) -> Dict[str, Any]:
        ds = _ordered(dataset)
        confusion, how = _confusion(ds, fid_high, fid_low)
        rec = _reconstruct(ds, confusion)
        amps = np.asarray(ds[AMP].values, dtype=float)
        counts = np.asarray(ds[COUNT].values, dtype=int)
        excite_high = drive_side == "high"

        columns = {key: np.full(amps.size, np.nan) for key in _COLUMNS}
        ok = np.zeros(amps.size, dtype=bool)
        model = np.full((amps.size, counts.size, 5), np.nan)
        leak = np.full(amps.size, np.nan)
        for i in range(amps.size):
            fit = fit_swap_channel(counts, rec["features"][i], excite_high=excite_high)
            ok[i] = bool(fit["success"])
            model[i] = fit["model"]
            for key, name in _COLUMNS.items():
                columns[key][i] = fit.get(name, np.nan)
            p11 = rec["p11"][i]
            if np.isfinite(p11).sum() >= 2:
                leak[i] = float(np.polyfit(counts, p11, 1)[0])

        results: Dict[str, Any] = dict(_SCALARS)
        results["readout_correction"] = how
        results["drive_side"] = drive_side
        results["high_name"] = high_name or "high"
        results["low_name"] = low_name or "low"
        results["stark_amps"] = [float(a) for a in amps]
        results["swap_counts"] = [int(c) for c in counts]
        for key in _COLUMNS:
            results[key] = [float(v) for v in columns[key]]
        results["fit_success_per_amp"] = [int(v) for v in ok]
        results["leak_to_11_per_amp"] = [float(v) for v in leak]
        results.update(_predicted(round_duration_ns, t1_high_s, t1_low_s,
                                  t2_star_high_s, t2_star_low_s))
        results["n_fit_ok"] = int(ok.sum())

        if ok.any():
            phis = np.where(ok, columns["phase_per_step_rad"], np.nan)
            i_best = int(np.nanargmin(np.abs(phis)))
            thetas = columns["theta_per_amp"][ok]
            results.update({
                "theta_rad": float(columns["theta_per_amp"][i_best]),
                "theta_rad_err": float(columns["theta_err_per_amp"][i_best]),
                "theta_spread_rad": float(np.ptp(thetas)) if thetas.size > 1 else 0.0,
                "theta_stark_amp": float(amps[i_best]),
                "phase_at_theta_amp_rad": float(phis[i_best]),
                "t1_loss_per_step_high": float(columns["t1_loss_high_per_amp"][i_best]),
                "t1_loss_per_step_low": float(columns["t1_loss_low_per_amp"][i_best]),
                "dephasing_per_step": float(columns["dephasing_per_amp"][i_best]),
                "prep_error": float(columns["prep_error_per_amp"][i_best]),
                "leak_to_11_per_step": float(leak[i_best]),
                "fit_rms": float(columns["fit_rms_per_amp"][i_best]),
                "success": 1,
            })
            # The angle must not depend on the stark amplitude; judge that in
            # units of each fit's own error, since far from compensation the
            # angle is much less well determined than near it.
            others = ok.copy()
            others[i_best] = False
            if others.any():
                err = np.hypot(columns["theta_err_per_amp"][others],
                               columns["theta_err_per_amp"][i_best])
                dev = np.abs(columns["theta_per_amp"][others] - results["theta_rad"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    results["theta_consistency_sigma"] = float(np.nanmax(dev / err))
            else:
                results["theta_consistency_sigma"] = 0.0
            root, extrapolated = phase_root(amps[ok], phis[ok])
            results["compensating_stark_amp"] = float(root)
            results["compensation_extrapolated"] = int(extrapolated)
            results["excess_dephasing_per_step"] = float(
                results["dephasing_per_step"] - results["predicted_dephasing"])

        results["_rec"] = rec
        results["_model"] = model
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(self, dataset: xr.Dataset, results: Dict[str, Any],
                        **kwargs) -> Optional[xr.Dataset]:
        ds = _ordered(dataset)
        rec = results.get("_rec")
        if rec is None:
            confusion, _ = _confusion(ds, kwargs.get("fid_high"), kwargs.get("fid_low"))
            rec = _reconstruct(ds, confusion)
        amps = np.asarray(ds[AMP].values, dtype=float)
        counts = np.asarray(ds[COUNT].values, dtype=int)
        dims = (AMP, COUNT)
        out = xr.Dataset(coords={AMP: amps, COUNT: counts})
        for name in ("x", "y", "z", "p_sub", "p00", "p11", "purity"):
            out[name] = (dims, np.asarray(rec[name], dtype=float))
        model = results.get("_model")
        if model is None:
            model = np.full((amps.size, counts.size, 5), np.nan)
        model = np.asarray(model, dtype=float)
        out["fit_x"] = (dims, model[..., 0])
        out["fit_y"] = (dims, model[..., 1])
        out["fit_z"] = (dims, model[..., 2] - model[..., 3])
        out["fit_p_sub"] = (dims, model[..., 2] + model[..., 3])
        out["fit_p00"] = (dims, model[..., 4])
        for key in (*_COLUMNS, "leak_to_11_per_amp"):
            values = results.get(key)
            column = np.full(amps.size, np.nan) if values is None else np.asarray(values, float)
            out[key] = ((AMP,), column)
        out["fit_success_per_amp"] = ((AMP,), np.asarray(
            results.get("fit_success_per_amp", [0] * amps.size), dtype=int))
        out.attrs.update({key: type(default)(results.get(key, default))
                          for key, default in _SCALARS.items()})
        out.attrs["readout_correction"] = str(results.get("readout_correction", "none"))
        out.attrs["high_name"] = str(results.get("high_name", "high"))
        out.attrs["low_name"] = str(results.get("low_name", "low"))
        return out

    def generate_figures(self, dataset: xr.Dataset, results: Dict[str, Any],
                         plot_data: Optional[xr.Dataset] = None,
                         **kwargs) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return render_figures(
            {
                FIG_BLOCH: lambda: plot_bloch(plot_data),
                FIG_COMPONENTS: lambda: plot_components(plot_data),
                FIG_COMPENSATION: lambda: plot_compensation(plot_data),
            },
            label=self.estimator_name,
        )
