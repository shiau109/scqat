from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import POS_ATTRS, BaseEstimator, reduced_signal, with_iqdata
from scqat.tools.fit_cosine import FitCosine
from scqat.tools.iq_reduce import AXIAL_KNOBS, validate_iq_reduce_kwargs
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators._twin_axis import TWIN_KNOBS, twin_at, twin_values
from scqat.estimators.power_rabi.visualization import plot_amplitude_fit


class PowerRabiEstimator(BaseEstimator):
    """
    Analyzes power-Rabi data to extract the drive-amplitude prefactor of a pi pulse.

    Expects an xarray.Dataset with:
        - Variables: complex ``IQdata`` (or both ``I`` and ``Q``) — reduced to the
          signed axial projection onto the |0>-|1> axis — OR a pre-reduced real
          ``signal`` (an already-discriminated state/population).
        - Coordinate: 'amp_prefactor'  (dimensionless multiplier of the current pulse
          amplitude, i.e. the QUA ``amplitude_scale`` sweep)

    Fits a non-decaying cosine ``a*cos(2*pi*f*x + phi) + c`` (via :class:`FitCosine`)
    and reports ``opt_amp_prefactor`` — the prefactor of the first oscillation extremum,
    which is the multiplier on the **current** pulse amplitude that yields a pi pulse.
    The estimator stays device-agnostic: the absolute amplitude is applied by the caller
    (the QM node multiplies this prefactor by the operation's current amplitude).

    A caller that also knows the ABSOLUTE amplitude behind each prefactor may pass it
    as a second coordinate over the same points (``twin_coord``); it is then drawn as a
    secondary axis and reported as ``opt_twin_value``. Purely additive — the fit, the
    answer and the success flag stay in the prefactor frame, and the estimator never
    learns which of the two scales is the ratio.
    """

    estimator_name = "power_rabi"

    #: optional companion scale for the swept axis — a coordinate name on the input
    #: dataset, and the label to draw it under. Overridable per call via kwargs.
    twin_coord: Optional[str] = None
    twin_label: Optional[str] = None

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Power-Rabi analysis requires a 'signal' variable, or complex 'IQdata', "
                "or both 'I' and 'Q'."
            )
        if "amp_prefactor" not in dataset.coords:
            raise ValueError("Power-Rabi analysis requires an 'amp_prefactor' coordinate in the dataset.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the power-Rabi oscillation and extract the pi-pulse amplitude prefactor.

        Kwargs — flat and fully owned; unknown names raise:
            angle, positions, pca_sign
                IQ->1-D axial-reduction knobs (see :func:`scqat.tools.iq_reduce.axial`);
                ignored when the dataset already carries a real ``signal``.
            twin_coord, twin_label
                optional companion scale for the swept axis (see
                :mod:`scqat.estimators._twin_axis`) — a coordinate over the same points
                plus its axis label. Absent/non-finite/non-monotone is simply not drawn.

        Returns a dict with:
            a, f, phi, c, opt_amp_prefactor, success, signal, reduction_method,
            reduction_angle, best_fit, fit_report — plus twin_values / twin_label /
            opt_twin_value when a drawable companion scale was supplied.
        """
        # popped BEFORE the reduction check: these are this estimator's own knobs, and
        # widening AXIAL_KNOBS would loosen the surface for all five families sharing
        # tools/iq_reduce.py.
        twin_coord = kwargs.pop("twin_coord", self.twin_coord)
        twin_label = kwargs.pop("twin_label", self.twin_label)
        validate_iq_reduce_kwargs(kwargs, allowed=AXIAL_KNOBS)
        # Prepare a DataArray with an 'x' coordinate for the FitCosine fitter.
        sig = reduced_signal(dataset, **kwargs)
        fit_data = sig.rename({"amp_prefactor": "x"})

        # FitCosine bounds the amplitude a >= 0, so a single phi=0 seed can get trapped
        # at a flat (a~0) fit for qubits whose readout *rises* from zero amplitude (which
        # needs phi ~ pi). Fit with both phase seeds and keep the lower-residual result so
        # the extraction is robust to the per-qubit readout sign.
        fit_result = None
        for phi_seed in (0.0, np.pi):
            fitter = FitCosine(fit_data)
            fitter.guess()
            fitter.params["phi"].set(value=phi_seed)
            res = fitter.fit()
            if fit_result is None or res.chisqr < fit_result.chisqr:
                fit_result = res
        p = {k: v.value for k, v in fit_result.params.items()}  # a, f, phi, c

        # pi pulse = the amplitude where the fitted readout deviates most from its
        # zero-amplitude (ground-state) value. The excited-state population rises
        # monotonically from 0 at x=0 to 1 at the pi pulse, so |signal - signal(0)| peaks
        # there — sign-independent (works for either readout sign) and correct even though
        # the sweep starts at an extremum.
        best_fit = np.asarray(fit_result.best_fit, dtype=float)
        amp = np.asarray(dataset.coords["amp_prefactor"].values, dtype=float)
        if p["f"] > 0 and best_fit.size == amp.size:
            opt_amp_prefactor = float(amp[int(np.argmax(np.abs(best_fit - best_fit[0])))])
        else:
            opt_amp_prefactor = float("nan")

        # Reject a degenerate (flat) fit: a real Rabi oscillation has a ~ half the signal
        # peak-to-peak, so guard against a ~ 0 falsely reporting success.
        ptp = float(np.ptp(np.asarray(fit_data.values, dtype=float)))
        contrast_ok = ptp > 0 and p["a"] > 0.05 * ptp
        # The optimum must sit STRICTLY INSIDE the swept window. `opt` is an argmax
        # over the swept array, so landing on an endpoint means the extremum was
        # never bracketed — the pi pulse is outside the window and the reported
        # value is just the edge. (This replaces a hardcoded `0 < opt < 2`, which
        # was the same test only for a 0..2 prefactor window and silently assumed
        # the axis was a RATIO — the estimator must stay frame-agnostic.)
        lo, hi = float(np.min(amp)), float(np.max(amp))
        bracketed = lo < opt_amp_prefactor < hi
        success = bool(
            bool(fit_result.success)
            and np.isfinite(opt_amp_prefactor)
            and bracketed
            and contrast_ok
        )

        results = {
            "a": p["a"],
            "f": p["f"],
            "phi": p["phi"],
            "c": p["c"],
            "opt_amp_prefactor": float(opt_amp_prefactor),
            "success": success,
            "signal": np.asarray(sig.values, dtype=float),
            "reduction_method": sig.attrs.get("reduction_method"),
            "reduction_angle": sig.attrs.get("reduction_angle"),
            "best_fit": fit_result.best_fit,
            "fit_report": fit_result.fit_report(),
        }
        # the stored |0>/|1> centroids the axis came from (absent otherwise)
        for key in POS_ATTRS:
            if key in sig.attrs:
                results[key] = float(sig.attrs[key])

        # the optional companion scale — keys are absent entirely when it is not
        # drawable, so downstream consumers test with `if key in results`
        twin = twin_values(dataset, "amp_prefactor", twin_coord)
        if twin is not None:
            results["twin_values"] = twin
            results["twin_label"] = str(twin_label or twin_coord)
            results["opt_twin_value"] = twin_at(amp, twin, opt_amp_prefactor)
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fit parameters and optimal prefactor; drop the diagnostic arrays.

        ``opt_twin_value`` is kept (a scalar answer in the companion scale) while
        ``twin_values`` is dropped with the other per-point arrays."""
        drop = {"best_fit", "fit_report", "signal", "twin_values"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """
        Bundle the raw signal + best-fit curve over ``amp_prefactor``; the fit
        parameters and ``opt_amp_prefactor`` live in ``.attrs`` so the figure needs
        no recomputation. A companion scale, when one was supplied, rides along as
        the ``twin`` variable so the saved plotdata redraws the secondary axis with
        no access to the device.
        """
        amp_prefactor = np.asarray(dataset.coords["amp_prefactor"].values, dtype=float)
        signal = np.asarray(results["signal"], dtype=float)
        best_fit = np.asarray(results["best_fit"], dtype=float)

        attrs = {
            "a": float(results["a"]),
            "f": float(results["f"]),
            "phi": float(results["phi"]),
            "c": float(results["c"]),
            "opt_amp_prefactor": float(results["opt_amp_prefactor"]),
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

        data_vars = {
            "signal": ("amp_prefactor", signal),
            "best_fit": ("amp_prefactor", best_fit),
        }
        # the optional companion scale + its label, so generate_figures can draw the
        # secondary axis from plot_data ALONE (the self-enforcing rule)
        if results.get("twin_values") is not None:
            data_vars["twin"] = ("amp_prefactor",
                                 np.asarray(results["twin_values"], dtype=float))
            attrs["twin_label"] = str(results.get("twin_label", ""))
            attrs["opt_twin_value"] = float(results.get("opt_twin_value", float("nan")))
        # the raw IQ cloud for the shared IQ-plane panel (absent on pre-reduced input)
        if "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars):
            iq = with_iqdata(dataset)["IQdata"].squeeze().values
            data_vars["iq_i"] = ("amp_prefactor", np.real(iq).astype(float))
            data_vars["iq_q"] = ("amp_prefactor", np.imag(iq).astype(float))

        return xr.Dataset(
            data_vars,
            coords={"amp_prefactor": amp_prefactor},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the amplitude-fit plot, drawing strictly from ``plot_data`` so the
        figure stays reconstructable downstream; rebuild it only when called outside
        ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        figs = {"amplitude": plot_amplitude_fit(plot_data)}
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
