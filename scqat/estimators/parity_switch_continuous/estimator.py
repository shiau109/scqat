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
from scqat.core.figures import render_figures
from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane
from scqat.estimators.parity_switch_continuous.visualization import (
    plot_psd,
    plot_state_psd,
    plot_timetrace,
)
from scqat.tools.discriminate import discriminate_states
from scqat.tools.telegraph_psd import (
    fit_telegraph_psd,
    telegraph_spectrum,
    validate_telegraph_psd_kwargs,
)

#: shots kept for the IQ-plane panel — the full 1e5-shot cloud draws slowly and
#: bloats plotdata.nc; an even subsample shows the same two blobs.
_IQ_PANEL_MAX = 4096


class ParitySwitchContinuousEstimator(BaseEstimator):
    """Estimator for the CONTINUOUS parity-switch monitor: a fixed-sequence
    single-shot time trace -> the charge-parity switching rate.
    (The two-measurement-per-cycle sibling lives in
    ``estimators/parity_switch_discrete`` and fits the within-cycle
    ``m1 XOR m2`` instead of the consecutive-pair difference.)

    Pipeline: resolve the per-shot 0/1 readout (a ``state`` variable verbatim,
    or nearest-centre discrimination of per-shot I/Q against the stored blob
    centres), derive the PARITY as the consecutive-pair difference, then fit
    the PARITY's Welch PSD with a Lorentzian knee
    (:func:`scqat.tools.telegraph_psd.fit_telegraph_psd`) — the rate is
    ``pi * corner`` (convention pinned in the tool's docstring).

    WHICH SERIES IS FITTED, AND WHY IT IS NOT THE READOUT. The sequence carries
    no qubit reset and is a unitary, so it maps antipodal Bloch vectors to
    antipodal ones and each outcome inverts with the pole the previous shot
    left behind: ``s[i] = s[i-1] XOR parity[i]``. The readout trace is the
    running XOR of the parity; the pair series IS the parity telegraph. The
    readout's own spectrum is still computed and plotted, unfitted, as a
    diagnostic.

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

    estimator_name = "parity_switch_continuous"

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
                    "parity_switch_continuous needs the shot cadence: attach "
                    "a scalar 'shot_period_s' variable/attr to the dataset "
                    "(the acquisition layer's job) or pass dt_s= in seconds."
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
            model ({"constrained", "independent"}):
                Which PSD model to fit (see
                :data:`scqat.tools.telegraph_psd.TELEGRAPH_MODELS`). Default
                ``"constrained"`` — the reference single-F model. Only the parity
                fit reads it; the raw-state spectrum is always unfitted.
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
        # the model axis is not a welch knob — pop it before validating the rest
        model = kwargs.pop("model", "constrained")
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
                        "parity_switch_continuous needs the |0>/|1> centres to "
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

        # THE PARITY, not the readout, is what gets fitted. The sequence runs
        # without a qubit reset and is a unitary, so each shot's outcome
        # INVERTS with the pole the previous shot left behind:
        #     s[i] = s[i-1] XOR parity[i]
        # i.e. the readout trace is the running XOR of the parity, and the
        # consecutive-pair series IS the parity telegraph. Fitting the readout
        # instead fits an integrated telegraph and returns a meaningless rate
        # (module docstring of scqat.tools.telegraph_psd).
        if trace.size < 2:
            raise ValueError(
                f"parity_switch_continuous needs at least 2 shots to form a "
                f"parity (the parity is the difference between consecutive "
                f"shots), got {trace.size}."
            )
        parity = (trace[:-1] != trace[1:]).astype(np.int8)

        results.update(fit_telegraph_psd(parity, dt, model=model, **kwargs))
        # the odd-parity level, distinct from p_switch (the parity's own
        # switching fraction, which is what the guard reads). ~0.5 is HEALTHY:
        # it just says the chip sits in each parity about half the time.
        results["p_parity_odd"] = float(np.mean(parity)) if parity.size else float("nan")
        # the RAW READOUT's mean — a third fraction, and a different one again:
        # the readout is the running XOR, so this is not the parity level and
        # not a population. Reported because it is the readable check on
        # readout balance.
        results["p_state_high"] = float(np.mean(trace))

        # the readout's own spectrum, kept for the diagnostic figure ONLY and
        # deliberately never fitted — it is the integrated telegraph, so a
        # Lorentzian on it means nothing.
        state_freq, state_psd = telegraph_spectrum(trace, dt, **kwargs)
        results["state_psd_freq_hz"] = state_freq
        results["state_psd"] = state_psd

        results["trace"] = trace
        results["parity"] = parity
        results["dt_s"] = dt
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the rate + fit scalars; drop the per-shot/spectral arrays."""
        drop = {"trace", "parity", "psd_freq_hz", "psd", "psd_fit",
                "state_psd_freq_hz", "state_psd"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the measured 0/1 readout (over ``shot_idx``/``time_s``), the
        derived PARITY (over ``pair_idx``), the parity's PSD + Lorentzian fit
        (over ``psd_freq_hz``), the readout's own UNFITTED spectrum (over
        ``state_psd_freq_hz``) and an IQ subsample when the input carried
        quadratures; every fit scalar lives in ``.attrs``.

        The parity is the measurement: even (0) = the two shots agree, odd (1)
        = they differ, one entry shorter than the readout trace and each timed
        BETWEEN its two shots. Under this no-reset sequence that pair series IS
        the parity telegraph (class docstring), which is why it — and not the
        readout — carries the fit.
        """
        trace = np.asarray(results["trace"], dtype=np.int8)
        parity = np.asarray(results["parity"], dtype=np.int8)
        dt = float(results["dt_s"])
        n = trace.size

        data_vars: Dict[str, Any] = {
            "state": ("shot_idx", trace),
            "parity": ("pair_idx", parity),
            "psd": ("psd_freq_hz",
                    np.asarray(results["psd"], dtype=float)),
            "psd_fit": ("psd_freq_hz",
                        np.asarray(results["psd_fit"], dtype=float)),
            "state_psd": ("state_psd_freq_hz",
                          np.asarray(results["state_psd"], dtype=float)),
        }
        coords: Dict[str, Any] = {
            "shot_idx": np.arange(n),
            "time_s": ("shot_idx", np.arange(n) * dt),
            "pair_idx": np.arange(parity.size),
            # the pair sits BETWEEN its two shots, hence the half-step
            "pair_time_s": ("pair_idx", (np.arange(parity.size) + 0.5) * dt),
            "psd_freq_hz": np.asarray(results["psd_freq_hz"], dtype=float),
            "state_psd_freq_hz": np.asarray(results["state_psd_freq_hz"],
                                            dtype=float),
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
                      "psd_white_floor", "n_transitions", "p_switch",
                      "p_high", "p_parity_odd", "p_state_high", "psd_freq_min_hz",
                      "psd_freq_max_hz", "psd_contrast", "corner_margin_low",
                      "mapping_fidelity", "mapping_fidelity_floor",
                      "mapping_fidelity_ratio", "psd_model", "psd_fit_residual",
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
        """One shared-axis DEBUG panel holding the readout snippet over the
        derived parity snippet, the parity's log-log PSD with the knee fit, and
        the readout's own UNFITTED spectrum (+ the shared IQ-plane panel when
        the input carried quadratures). Draws only from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        pd = plot_data
        builders = {"timetrace": lambda: plot_timetrace(pd),
                    "psd": lambda: plot_psd(pd),
                    "state_psd": lambda: plot_state_psd(pd)}
        if has_iq_plane(pd):
            builders["iq_plane"] = lambda: plot_iq_plane(pd)
        return render_figures(builders, label=self.estimator_name)
