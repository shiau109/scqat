"""
Qubit Relaxation (T1) Estimator
===============================
Single-exponential fit of the excited-state signal after a pi pulse.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - wait_time : 1-D float array - delay after the pi pulse (s).
Data variables:
    - signal    : (wait_time,) - excited-state signal (e.g. rotated I quadrature
                  or population), decaying toward its offset.

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers``).
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, reduced_signal, with_iqdata
from scqat.tools.fit_exp_decay import FitExponentialDecay
from scqat.tools.iq_reduce import AXIAL_KNOBS, validate_iq_reduce_kwargs

from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators.qubit_relaxation.visualization import plot_decay


class QubitRelaxationEstimator(BaseEstimator):
    """Fit ``signal = a * exp(-t / t1) + c`` and report T1 (seconds).

    The signal is the signed axial projection of the complex IQ onto the |0>-|1>
    axis (or a pre-reduced real ``signal`` when the probe already discriminated)."""

    estimator_name = "qubit_relaxation"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "T1 estimator requires a 'signal' data variable, or complex 'IQdata', "
                "or both 'I' and 'Q'."
            )
        if "wait_time" not in dataset.coords:
            raise ValueError("T1 estimator requires a 'wait_time' coordinate (seconds)")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Fit the decay and report T1.

        Kwargs — ``angle`` / ``positions`` / ``pca_sign`` (see
        :func:`scqat.tools.iq_reduce.axial`); ignored when a real ``signal`` is present.
        """
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)
        sig = reduced_signal(dataset, **kwargs)
        da = sig.rename({"wait_time": "x"})
        fit_result = FitExponentialDecay(da).fit()
        t1 = float(fit_result.params["tau"].value)
        t_span = float(da["x"].values[-1] - da["x"].values[0])
        return {
            "t1": t1,
            "t1_stderr": float(fit_result.params["tau"].stderr or np.nan),
            "amplitude": float(fit_result.params["a"].value),
            "offset": float(fit_result.params["c"].value),
            "redchi": float(fit_result.redchi),
            # physical: converged, positive, and not absurdly beyond the swept window
            "success": bool(fit_result.success) and np.isfinite(t1) and 0 < t1 < 10 * t_span,
            "signal": np.asarray(sig.values, dtype=float),
            "reduction_method": sig.attrs.get("reduction_method"),
            "reduction_angle": sig.attrs.get("reduction_angle"),
            "best_fit": np.asarray(fit_result.best_fit, dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in results.items() if k not in {"best_fit", "signal"}}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        wait = np.asarray(dataset["wait_time"].values, dtype=float)
        data_vars = {
            "signal": ("wait_time", np.asarray(results["signal"], dtype=float)),
            "best_fit": ("wait_time", results["best_fit"]),
        }
        # the raw IQ cloud for the shared IQ-plane panel (absent on pre-reduced input)
        if "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars):
            iq = with_iqdata(dataset)["IQdata"].squeeze().values
            data_vars["iq_i"] = ("wait_time", np.real(iq).astype(float))
            data_vars["iq_q"] = ("wait_time", np.imag(iq).astype(float))
        return xr.Dataset(
            data_vars,
            coords={"wait_time": wait},
            attrs={
                "t1": results["t1"],
                "amplitude": results["amplitude"],
                "offset": results["offset"],
                "success": int(bool(results["success"])),
                "reduction_method": str(results.get("reduction_method", "signal")),
                # 0.0 is a legitimate angle (axis on I) — only None becomes NaN
                "reduction_angle": (float(results["reduction_angle"])
                                    if results.get("reduction_angle") is not None else float("nan")),
            },
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        # single-figure idiom: key == estimator_name -> saved as qubit_relaxation.png
        figs = {"qubit_relaxation": plot_decay(plot_data)}
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
