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

import warnings
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import POS_ATTRS, BaseEstimator, reduced_signal, with_iqdata
from scqat.core.figures import render_figures
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
        signal = np.asarray(sig.values, dtype=float)
        da = sig.rename({"wait_time": "x"})
        t_span = float(da["x"].values[-1] - da["x"].values[0])

        # A degenerate or non-converging fit must NEVER sink the raw-data
        # artifact. If the fit raises (e.g. flat data collapses the fitter's
        # bounds) degrade to a NaN fit with success=False, keeping the raw
        # signal so build_plot_data / the figure still draw the measured trace.
        try:
            fit_result = FitExponentialDecay(da).fit()
            t1 = float(fit_result.params["tau"].value)
            t1_stderr = float(fit_result.params["tau"].stderr or np.nan)
            amplitude = float(fit_result.params["a"].value)
            offset = float(fit_result.params["c"].value)
            redchi = float(fit_result.redchi)
            best_fit = np.asarray(fit_result.best_fit, dtype=float)
            converged = bool(fit_result.success)
        except Exception as err:  # noqa: BLE001 - a failed fit must not lose the raw data
            warnings.warn(
                f"{self.estimator_name}: exponential fit failed "
                f"({type(err).__name__}: {err}); reporting raw data with no fit",
                stacklevel=2,
            )
            t1 = t1_stderr = amplitude = offset = redchi = float("nan")
            best_fit = np.full_like(signal, np.nan)
            converged = False

        results = {
            "t1": t1,
            "t1_stderr": t1_stderr,
            "amplitude": amplitude,
            "offset": offset,
            "redchi": redchi,
            # physical: converged, positive, and not absurdly beyond the swept window
            "success": bool(converged and np.isfinite(t1) and 0 < t1 < 10 * t_span),
            "signal": signal,
            "reduction_method": sig.attrs.get("reduction_method"),
            "reduction_angle": sig.attrs.get("reduction_angle"),
            "best_fit": best_fit,
        }
        # the stored |0>/|1> centroids the axis came from (absent otherwise)
        for key in POS_ATTRS:
            if key in sig.attrs:
                results[key] = float(sig.attrs[key])
        return results

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
        attrs = {
            "t1": results["t1"],
            "amplitude": results["amplitude"],
            "offset": results["offset"],
            "success": int(bool(results["success"])),
            "reduction_method": str(results.get("reduction_method", "signal")),
            # 0.0 is a legitimate angle (axis on I) — only None becomes NaN
            "reduction_angle": (float(results["reduction_angle"])
                                if results.get("reduction_angle") is not None else float("nan")),
        }
        # the stored |0>/|1> centroids (drawn by the shared IQ-plane panel)
        for key in POS_ATTRS:
            if results.get(key) is not None:
                attrs[key] = float(results[key])
        return xr.Dataset(data_vars, coords={"wait_time": wait}, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        # Per-figure isolation: a failed fit (or a plotter that trips on an
        # all-NaN fit) must never take the raw-data figure down with it. The
        # single-figure idiom keeps key == estimator_name -> qubit_relaxation.png.
        builders = {self.estimator_name: lambda: plot_decay(plot_data)}
        if has_iq_plane(plot_data):
            builders["iq_plane"] = lambda: plot_iq_plane(plot_data)
        return render_figures(builders, label=self.estimator_name)
