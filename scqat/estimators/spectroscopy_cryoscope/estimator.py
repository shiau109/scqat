from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.tools.peak_fit import PEAK_KNOBS, fit_peaks, validate_peak_kwargs
from scqat.tools.step_response_fit import fit_step_response
from scqat.estimators.spectroscopy_cryoscope.visualization import (
    plot_step_response,
    plot_spectrogram,
)


class SpectroscopyCryoscopeEstimator(BaseEstimator):
    """Estimator for a LONG-TIME (spectroscopy) cryoscope.

    The complement of the Ramsey cryoscope: instead of reconstructing a phase, it
    measures the qubit's frequency DIRECTLY by spectroscopy at each wait-time into
    a parked flux pulse, so it reaches the microsecond tails a phase measurement
    cannot (the phase would wrap). Expects a 2-D map ``xarray.Dataset`` with:
        - Variable: ``'signal'`` (real spectroscopy response, e.g. the discriminated
          state population) **or** complex IQ (``'IQdata'``, or both ``'I'`` and
          ``'Q'``) — reduced to magnitude
        - Coordinates: ``'detuning'`` (drive detuning from the parked drive, Hz) and
          ``'wait_time'`` (delay into the flux pulse, **seconds**)
        - Optional scalar variable ``'center_offset_hz'``: the parked detuning the
          drive was centered on (arch-predicted ``quad * flux_amp**2``). The fitted
          peak center is RELATIVE to that drive, so the absolute detuning from the
          sweet spot is ``center_offset_hz + fitted_center``. Absent → 0.0.

    Analysis chain (composes existing tools; no new tool):

    1. **Per wait-time**, fit the spectroscopy line with a windowed Lorentzian
       (:func:`scqat.tools.peak_fit.fit_peaks`, ``max_peaks=1``) → the peak center
       ``center_detuning(t)``.
    2. **Absolute detuning** ``total(t) = center_offset_hz + center_detuning(t)``;
       **flux** ``∝ sqrt(|total(t)|)`` (quadratic arch), **normalized by the settled
       tail** (the longest waits) so the step response settles to ~1 and the taps
       are relative — no flux-frequency curvature calibration is needed.
    3. **Multi-exponential fit**
       (:func:`scqat.tools.step_response_fit.fit_step_response`) → the ``(amp,
       tau_s)`` predistortion tap components.

    SCQO's ``update`` writes the components as the flux channel's paired facts
    ``distortion_amp`` (= ``amp / a_dc``) and ``distortion_tau_s`` (seconds) — the
    same facts the Ramsey cryoscope writes (independent, whole-array REPLACE).
    """

    estimator_name = "spectroscopy_cryoscope"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Spectroscopy-cryoscope analysis requires a 'signal' variable, or "
                "complex 'IQdata', or both 'I' and 'Q'."
            )
        for coord in ("detuning", "wait_time"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"Spectroscopy-cryoscope analysis requires a '{coord}' coordinate."
                )

    def _signal_map(self, dataset: xr.Dataset) -> xr.DataArray:
        """The real (wait_time, detuning) spectroscopy map the peak fit consumes."""
        if "signal" in dataset.data_vars:
            da = dataset["signal"].astype(float)
        elif "IQdata" in dataset.data_vars:
            da = np.abs(dataset["IQdata"])
        else:
            da = np.abs(dataset["I"] + 1j * dataset["Q"])
        return da.transpose("wait_time", "detuning")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Reconstruct the long-time step response and fit its exponentials.

        Kwargs — flat and fully owned:
            start_fractions (sequence[float]): DESCENDING 0-1 fractions where each
                exponential component's fit starts (slowest first). Default
                ``(0.7, 0.3, 0.05)`` (the 14b long-time defaults). No MPM
                seeding here — the Matrix Pencil needs a UNIFORM axis and this
                experiment's wait grid is log-spaced (see
                ``tools.step_response_fit.mpm_tau_seeds``).
            stable_tail_frac (float): fraction of the record (longest waits) used to
                normalize the step response. Default ``0.15``.
            a_dc (float | None): pin the settled level of the exponential fit.
                Default ``1.0`` (the step response is tail-normalized); ``None``
                estimates it from the flattest tail.
            plus the peak-fit knobs (``prominence``, ``min_snr``, ``max_peaks``,
            ``fit_window_factor``, ...) forwarded to
            :func:`scqat.tools.peak_fit.fit_peaks`.

        Returns a dict with: success, component_amps, component_taus_s, a_dc,
        rms_residual, n_components, start_fractions, stable_tail_frac,
        center_offset_hz, n_peaks_found, wait_time_s, detuning_hz, signal (2-D),
        center_detuning, flux, step_response, best_fit.
        """
        start_fractions = list(kwargs.get("start_fractions", (0.7, 0.3, 0.05)))
        stable_tail_frac = float(kwargs.get("stable_tail_frac", 0.15))
        a_dc = kwargs.get("a_dc", 1.0)
        peak_knobs = {k: kwargs[k] for k in PEAK_KNOBS if k in kwargs}
        peak_knobs.pop("max_peaks", None)  # this estimator pins a single peak
        validate_peak_kwargs(peak_knobs)

        sig = self._signal_map(dataset)
        wait_s = np.asarray(sig.coords["wait_time"].values, dtype=float)
        detuning = np.asarray(sig.coords["detuning"].values, dtype=float)
        signal_map = np.asarray(sig.values, dtype=float)
        center_offset = (
            float(np.asarray(dataset["center_offset_hz"].values).item())
            if "center_offset_hz" in dataset else 0.0
        )

        # 1. per wait-time: the spectroscopy peak center relative to the parked drive
        n_wait = wait_s.size
        center = np.full(n_wait, np.nan)
        for k in range(n_wait):
            try:
                r = fit_peaks(detuning, signal_map[k], max_peaks=1, **peak_knobs)
            except Exception:
                continue
            if r["peaks"]:
                center[k] = float(r["peaks"][0]["detuning"])
        n_found = int(np.count_nonzero(np.isfinite(center)))

        # 2. absolute detuning -> flux -> tail-normalized step response
        total = center_offset + center
        flux = np.sqrt(np.abs(total))
        n_tail = max(1, int(round(n_wait * stable_tail_frac)))
        tail_vals = flux[-n_tail:][np.isfinite(flux[-n_tail:])]
        tail_mean = float(np.mean(tail_vals)) if tail_vals.size else float("nan")
        if tail_mean and np.isfinite(tail_mean):
            step_response = flux / tail_mean
        else:
            step_response = np.full_like(flux, np.nan)

        # 3. multi-exponential fit on the finite points (the wait axis may be
        # log-spaced; fit_step_response is fine on a non-uniform axis)
        good = np.isfinite(step_response)
        if good.sum() >= 2:
            fit = fit_step_response(
                wait_s[good], step_response[good], start_fractions, a_dc=a_dc,
            )
        else:
            # mirror fit_step_response's FULL failed-dict contract (incl. best_fit)
            fit = {"success": False, "components": [], "a_dc": float("nan"),
                   "rms": float("nan"), "best_fractions": list(start_fractions),
                   "best_fit": np.full(int(good.sum()), np.nan)}
        amps = [amp for amp, _ in fit["components"]]
        taus = [tau for _, tau in fit["components"]]

        # rebuild the fitted curve on the FULL wait axis for plotting
        fitted_dc = float(fit["a_dc"])
        best_fit = np.full(n_wait, fitted_dc, dtype=float)
        for amp, tau in fit["components"]:
            best_fit += amp * np.exp(-wait_s / tau)

        return {
            "success": bool(fit["success"]),
            "component_amps": amps,
            "component_taus_s": taus,
            "a_dc": fitted_dc,
            "rms_residual": float(fit["rms"]),
            "n_components": len(fit["components"]),
            "start_fractions": [float(f) for f in fit["best_fractions"]],
            "stable_tail_frac": stable_tail_frac,
            "center_offset_hz": center_offset,
            "n_peaks_found": n_found,
            "wait_time_s": wait_s,
            "detuning_hz": detuning,
            "signal": signal_map,
            "center_detuning": center,
            "flux": flux,
            "step_response": step_response,
            "best_fit": best_fit,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the tap components + fit scalars; drop the diagnostic arrays."""
        drop = {"wait_time_s", "detuning_hz", "signal", "center_detuning",
                "flux", "step_response", "best_fit"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """The raw 2-D spectroscopy map (wait_time x detuning), the fitted peak
        track and the step-response fit (over wait_time), with the taps in
        netCDF-safe ``.attrs`` — both figures redraw with no recomputation."""
        wait_s = np.asarray(results["wait_time_s"], dtype=float)
        detuning = np.asarray(results["detuning_hz"], dtype=float)
        attrs = {
            "success": int(bool(results["success"])),
            "a_dc": float(results["a_dc"]),
            "rms_residual": float(results["rms_residual"]),
            "n_components": int(results["n_components"]),
            "fit_component_amps": np.asarray(results["component_amps"], dtype=float),
            "fit_component_taus_s": np.asarray(results["component_taus_s"], dtype=float),
            "stable_tail_frac": float(results["stable_tail_frac"]),
            "center_offset_hz": float(results["center_offset_hz"]),
            "n_peaks_found": int(results["n_peaks_found"]),
        }
        return xr.Dataset(
            {
                "signal": (("wait_time", "detuning"), np.asarray(results["signal"], dtype=float)),
                "center_detuning": ("wait_time", np.asarray(results["center_detuning"], dtype=float)),
                "step_response": ("wait_time", np.asarray(results["step_response"], dtype=float)),
                "best_fit": ("wait_time", np.asarray(results["best_fit"], dtype=float)),
            },
            coords={"wait_time": wait_s, "detuning": detuning},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Draw the step-response fit and the spectroscopy-map diagnostic strictly
        from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures({
            self.estimator_name: lambda: plot_step_response(plot_data),
            "spectrogram": lambda: plot_spectrogram(plot_data),
        }, label=self.estimator_name)
