from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import (
    POS_ATTRS,
    BaseEstimator,
    stored_positions,
    with_iqdata,
)
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators.parity_switch.visualization import plot_psd, plot_trace
from scqat.tools.discriminate import discriminate_states
from scqat.tools.telegraph_psd import (
    fit_telegraph_psd,
    validate_telegraph_psd_kwargs,
)

#: shots kept for the IQ-plane panel — the full 1e5-shot cloud draws slowly and
#: bloats plotdata.nc; an even subsample shows the same two blobs.
_IQ_PANEL_MAX = 4096


class ParitySwitchEstimator(BaseEstimator):
    """Estimator for the parity-switch monitor: a fixed-sequence single-shot
    time trace -> the charge-parity switching rate.

    Pipeline: resolve the per-shot 0/1 trace (a ``state`` variable verbatim,
    or nearest-centre discrimination of per-shot I/Q against the stored blob
    centres), then fit the trace's Welch PSD with a Lorentzian knee
    (:func:`scqat.tools.telegraph_psd.fit_telegraph_psd`) — the rate is
    ``pi * corner`` (convention pinned in the tool's docstring).

    Dataset contract:
        - Coordinate ``shot_idx`` (uniform shot cadence).
        - Variables: per-shot ``state`` (0/1), OR complex ``IQdata``, OR both
          ``I`` and ``Q``.
        - ``shot_period_s`` — the shot cadence in seconds, as a scalar
          variable or dataset attr (the acquisition layer attaches it);
          required unless the ``dt_s`` kwarg is passed.
        - ``ref_pos_g_i``/``ref_pos_g_q``/``ref_pos_e_i``/``ref_pos_e_q`` —
          stored |0>/|1> centres; required in the I/Q mode unless
          ``user_mean`` is passed.
    """

    estimator_name = "parity_switch"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "shot_idx" not in dataset.coords:
            raise ValueError(
                "Parity-switch analysis requires a 'shot_idx' coordinate."
            )
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if "state" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Parity-switch analysis requires a per-shot 'state' variable, "
                "or complex 'IQdata', or both 'I' and 'Q'."
            )

    @staticmethod
    def _resolve_dt(dataset: xr.Dataset, dt_s: Optional[float]) -> float:
        if dt_s is None:
            if "shot_period_s" in dataset:
                dt_s = float(np.asarray(dataset["shot_period_s"].values).item())
            elif "shot_period_s" in dataset.attrs:
                dt_s = float(dataset.attrs["shot_period_s"])
            else:
                raise ValueError(
                    "parity_switch needs the shot cadence: attach a scalar "
                    "'shot_period_s' variable/attr to the dataset (the "
                    "acquisition layer's job) or pass dt_s= in seconds."
                )
        if not (np.isfinite(dt_s) and dt_s > 0):
            raise ValueError(
                f"shot period must be positive and finite, got {dt_s!r}"
            )
        return float(dt_s)

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Discriminate the shot trace and fit its PSD knee.

        Kwargs — flat and fully owned; unknown names raise:
            dt_s (float): Shot period in seconds; overrides the dataset's
                ``shot_period_s``.
            user_mean, user_std, outlier_sigma:
                Discrimination knobs (see
                :func:`scqat.tools.discriminate.discriminate_states`);
                ``user_mean`` overrides the stored ``ref_pos_*`` centres.
                Ignored when the dataset already carries a ``state`` variable.
            nperseg, window, detrend:
                PSD knobs (see :func:`scqat.tools.telegraph_psd.fit_telegraph_psd`).

        Returns the tool's two-tier result plus ``trace``, ``dt_s``,
        ``state_source`` and, when discriminated, the pinned centres
        (``pos_*``) and ``outlier_probability``.
        """
        dt_s = kwargs.pop("dt_s", None)
        user_mean = kwargs.pop("user_mean", None)
        user_std = kwargs.pop("user_std", None)
        outlier_sigma = kwargs.pop("outlier_sigma", 3)
        validate_telegraph_psd_kwargs(kwargs)  # everything left is a PSD knob

        dt = self._resolve_dt(dataset, dt_s)

        results: Dict[str, Any] = {}
        if "state" in dataset.data_vars:
            trace = np.rint(
                np.asarray(dataset["state"].values, dtype=float)
            ).astype(np.int8).ravel()
            results["state_source"] = "state_var"
        else:
            iq = with_iqdata(dataset)["IQdata"].squeeze()
            I = np.real(iq.values).astype(float)
            Q = np.imag(iq.values).astype(float)
            centres = user_mean
            if centres is None:
                pos = stored_positions(dataset)
                if pos is None:
                    raise ValueError(
                        "parity_switch needs the |0>/|1> centres to "
                        "discriminate the shot trace: attach the stored "
                        "ref_pos_g_i/ref_pos_g_q/ref_pos_e_i/ref_pos_e_q "
                        "variables (single_shot_readout's accepted pos_* "
                        "monitors) or pass user_mean=[[g_i, g_q], [e_i, e_q]]."
                    )
                centres = [[pos[0].real, pos[0].imag],
                           [pos[1].real, pos[1].imag]]
            disc = discriminate_states(
                I[None, :], Q[None, :], user_mean=centres,
                user_std=user_std, outlier_sigma=outlier_sigma,
            )
            # nearest-centre assignment against the pinned g/e reference: row 0
            # is the one (unprepared) block, its labels ARE the 0/1 trace
            trace = disc["state_label"][0].astype(np.int8)
            centres = np.asarray(centres, dtype=float)
            results.update(
                state_source="discriminated",
                outlier_probability=float(disc["outlier_probability"][0]),
                pos_g_i=float(centres[0][0]), pos_g_q=float(centres[0][1]),
                pos_e_i=float(centres[1][0]), pos_e_q=float(centres[1][1]),
            )

        results.update(fit_telegraph_psd(trace, dt, **kwargs))
        results["trace"] = trace
        results["dt_s"] = dt
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the rate + fit scalars; drop the per-shot/spectral arrays."""
        drop = {"trace", "psd_freq_hz", "psd", "psd_fit"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 0/1 trace (over ``shot_idx``/``time_s``), the PSD + fit
        curve (over ``psd_freq_hz``) and an IQ subsample when the input carried
        quadratures; every fit scalar lives in ``.attrs``."""
        trace = np.asarray(results["trace"], dtype=np.int8)
        dt = float(results["dt_s"])
        n = trace.size

        data_vars: Dict[str, Any] = {
            "state": ("shot_idx", trace),
            "psd": ("psd_freq_hz",
                    np.asarray(results["psd"], dtype=float)),
            "psd_fit": ("psd_freq_hz",
                        np.asarray(results["psd_fit"], dtype=float)),
        }
        coords: Dict[str, Any] = {
            "shot_idx": np.arange(n),
            "time_s": ("shot_idx", np.arange(n) * dt),
            "psd_freq_hz": np.asarray(results["psd_freq_hz"], dtype=float),
        }
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if has_iq:
            iq = with_iqdata(dataset)["IQdata"].squeeze()
            step = max(1, -(-iq.size // _IQ_PANEL_MAX))  # ceil: stay <= cap
            sub = iq.values[::step]
            data_vars["iq_i"] = ("iq_idx", np.real(sub).astype(float))
            data_vars["iq_q"] = ("iq_idx", np.imag(sub).astype(float))

        attrs: Dict[str, Any] = {
            k: results[k]
            for k in ("parity_rate_hz", "psd_corner_hz", "psd_amplitude",
                      "psd_white_floor", "n_transitions", "p_excited",
                      "dt_s", "state_source", "method")
            if k in results
        }
        attrs["success"] = int(bool(results.get("success")))
        for key in (*POS_ATTRS, "outlier_probability"):
            if key in results:
                attrs[key] = float(results[key])
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Telegraph-trace snippet + log-log PSD with the knee fit (+ the
        shared IQ-plane panel when the input carried quadratures). Draws only
        from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        figs = {"trace": plot_trace(plot_data), "psd": plot_psd(plot_data)}
        if has_iq_plane(plot_data):
            figs["iq_plane"] = plot_iq_plane(plot_data)
        return figs
