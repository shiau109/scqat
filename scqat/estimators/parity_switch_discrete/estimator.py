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
from scqat.estimators.parity_switch_discrete.visualization import (
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

#: shots kept for the IQ-plane panel — the full cloud (two measurements per
#: cycle) draws slowly and bloats plotdata.nc; an even subsample shows the same
#: two blobs.
_IQ_PANEL_MAX = 4096


class ParitySwitchDiscreteEstimator(BaseEstimator):
    """Estimator for the DISCRETE parity-switch monitor: two measurements per
    cycle -> the charge-parity switching rate.

    The sequence per cycle is M1 - depletion - x90 - idle - y90 - M2 - wait,
    repeated at a fixed cycle period. M1 projects (re-initializes) the qubit,
    the mapping block flips the pole iff the parity is odd, and M2 reads the
    result — so the parity of cycle ``i`` is simply the WITHIN-CYCLE difference
    ``m1[i] XOR m2[i]``, self-contained per cycle. (Contrast the continuous
    sibling in ``estimators/parity_switch_continuous``, where a single
    measurement per shot makes the readout the running XOR of the parity and
    the CONSECUTIVE-pair difference is fitted.)

    Pipeline: resolve the per-cycle (m1, m2) 0/1 pair (a ``state`` variable
    verbatim, or nearest-centre discrimination of per-measurement I/Q against
    the stored blob centres), reduce to ``parity = m1 XOR m2``, then fit the
    parity's Welch PSD with a Lorentzian knee
    (:func:`scqat.tools.telegraph_psd.fit_telegraph_psd`) — the rate is
    ``pi * corner``, sampled at the CYCLE period (``shot_period_s``).

    Because M1 re-projects every cycle, decay or error between M2 and the next
    M1 never corrupts a parity sample — it only shows up as a mismatch between
    ``m1[i+1]`` and ``m2[i]``, reported as the health diagnostic
    ``p_intercycle_flip`` (0 on a clean QND chain). A single bad measurement
    flips exactly ONE parity sample (the continuous variant's bad shot flips
    two).

    Dataset contract:
        - Coordinates ``shot_idx`` (cycle index, uniform cadence) and
          ``meas_idx`` (size 2; index 0 = M1, 1 = M2).
        - Variables over ``(shot_idx, meas_idx)``: per-measurement ``state``
          (0/1), OR complex ``IQdata``, OR both ``I`` and ``Q``.
        - ``shot_period_s`` — the CYCLE period in seconds (the telegraph
          timebase), as a scalar variable or dataset attr; required unless the
          ``dt_s`` kwarg is passed.
        - ``ref_pos_g_i``/``ref_pos_g_q``/``ref_pos_e_i``/``ref_pos_e_q`` —
          stored |0>/|1> centres; required in the I/Q mode unless
          ``user_mean`` is passed.
    """

    estimator_name = "parity_switch_discrete"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "shot_idx" not in dataset.coords:
            raise ValueError(
                "Discrete parity-switch analysis requires a 'shot_idx' "
                "(cycle index) coordinate."
            )
        if "meas_idx" not in dataset.coords:
            raise ValueError(
                "Discrete parity-switch analysis requires a 'meas_idx' "
                "coordinate (the two measurements of each cycle)."
            )
        if dataset.sizes.get("meas_idx") != 2:
            raise ValueError(
                f"'meas_idx' must have exactly 2 entries (M1, M2), got "
                f"{dataset.sizes.get('meas_idx')}."
            )
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if "state" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "Discrete parity-switch analysis requires a per-measurement "
                "'state' variable, or complex 'IQdata', or both 'I' and 'Q'."
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
                    "parity_switch_discrete needs the cycle period: attach a "
                    "scalar 'shot_period_s' variable/attr to the dataset (the "
                    "acquisition layer's job) or pass dt_s= in seconds."
                )
        if not (np.isfinite(dt_s) and dt_s > 0):
            raise ValueError(
                f"cycle period must be positive and finite, got {dt_s!r}"
            )
        return float(dt_s)

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Resolve the (m1, m2) pairs, reduce to the parity, fit its PSD knee.

        Kwargs — flat and fully owned; unknown names raise:
            dt_s (float): Cycle period in seconds; overrides the dataset's
                ``shot_period_s``.
            user_mean, user_std, outlier_sigma:
                Discrimination knobs (see
                :func:`scqat.tools.discriminate.discriminate_states`);
                ``user_mean`` overrides the stored ``ref_pos_*`` centres.
                Ignored when the dataset already carries a ``state`` variable.
            model ({"constrained", "independent"}):
                Which PSD model to fit (see
                :data:`scqat.tools.telegraph_psd.TELEGRAPH_MODELS`). Default
                ``"constrained"``. Only the parity fit reads it; the raw-M1
                spectrum is always unfitted.
            nperseg, window, detrend:
                PSD knobs (see :func:`scqat.tools.telegraph_psd.fit_telegraph_psd`).

        Returns the tool's two-tier result plus ``m1``/``m2``/``parity``,
        ``dt_s``, ``state_source``, the diagnostics ``p_parity_odd``,
        ``p_intercycle_flip``, ``p_m1_high``, ``p_m2_high`` and, when
        discriminated, the pinned centres (``pos_*``) and
        ``outlier_probability``.
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
            arr = np.rint(np.asarray(
                dataset["state"].transpose("shot_idx", "meas_idx").values,
                dtype=float,
            )).astype(np.int8)
            results["state_source"] = "state_var"
        else:
            iq = with_iqdata(dataset)["IQdata"].transpose("shot_idx", "meas_idx")
            # flatten C-order: (m1[0], m2[0], m1[1], m2[1], ...) — the physical
            # time order — discriminate all 2N points in one call, then fold
            # back to (cycle, measurement)
            flat = iq.values.ravel()
            I = np.real(flat).astype(float)
            Q = np.imag(flat).astype(float)
            centres = user_mean
            if centres is None:
                pos = stored_positions(dataset)
                if pos is None:
                    raise ValueError(
                        "parity_switch_discrete needs the |0>/|1> centres to "
                        "discriminate the measurements: attach the stored "
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
            # is the one (unprepared) block; fold its labels back per cycle
            arr = disc["state_label"][0].astype(np.int8).reshape(-1, 2)
            centres = np.asarray(centres, dtype=float)
            results.update(
                state_source="discriminated",
                outlier_probability=float(disc["outlier_probability"][0]),
                pos_g_i=float(centres[0][0]), pos_g_q=float(centres[0][1]),
                pos_e_i=float(centres[1][0]), pos_e_q=float(centres[1][1]),
            )

        m1 = arr[:, 0]
        m2 = arr[:, 1]

        # THE WITHIN-CYCLE DIFFERENCE IS THE PARITY. M1 projects the qubit,
        # the mapping block flips the pole iff the parity is odd, M2 reads it:
        #     m2[i] = m1[i] XOR parity[i]
        # so each cycle carries its own parity sample — no chain across cycles
        # (that is the continuous sibling's reduction, not this one's).
        parity = (m1 != m2).astype(np.int8)

        results.update(fit_telegraph_psd(parity, dt, model=model, **kwargs))
        # the odd-parity level, distinct from p_switch (the parity's own
        # switching fraction, which is what the guard reads). ~0.5 is HEALTHY.
        results["p_parity_odd"] = float(np.mean(parity))
        # the discrete variant's health check: absent decay/readout error
        # between M2 and the next M1, the next cycle starts where this one
        # ended (QND chain), so m1[i+1] == m2[i] exactly. Every mismatch is a
        # T1 decay or a readout error during the inter-cycle wait — it does NOT
        # corrupt any parity sample (M1 re-projects), it is purely diagnostic.
        results["p_intercycle_flip"] = (
            float(np.mean(m1[1:] != m2[:-1])) if m1.size >= 2 else float("nan"))
        # readout balance per measurement slot — two more level-fractions,
        # neither of which is the parity level.
        results["p_m1_high"] = float(np.mean(m1))
        results["p_m2_high"] = float(np.mean(m2))

        # the raw M1 trace's own spectrum, kept for the diagnostic figure ONLY
        # and deliberately never fitted — absent errors m1 is the running XOR
        # of the parity (the QND chain integrates it), so a Lorentzian on it
        # means nothing.
        state_freq, state_psd = telegraph_spectrum(m1, dt, **kwargs)
        results["state_psd_freq_hz"] = state_freq
        results["state_psd"] = state_psd

        results["m1"] = m1
        results["m2"] = m2
        results["parity"] = parity
        results["dt_s"] = dt
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the rate + fit scalars; drop the per-cycle/spectral arrays."""
        drop = {"m1", "m2", "parity", "psd_freq_hz", "psd", "psd_fit",
                "state_psd_freq_hz", "state_psd"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the two measured 0/1 traces ``m1``/``m2`` and the derived
        PARITY (all over ``shot_idx``/``time_s`` — the parity is per-cycle
        here, no pair offset), the parity's PSD + Lorentzian fit (over
        ``psd_freq_hz``), the raw M1 trace's UNFITTED spectrum (over
        ``state_psd_freq_hz``) and an IQ subsample when the input carried
        quadratures; every fit scalar lives in ``.attrs``.
        """
        m1 = np.asarray(results["m1"], dtype=np.int8)
        m2 = np.asarray(results["m2"], dtype=np.int8)
        parity = np.asarray(results["parity"], dtype=np.int8)
        dt = float(results["dt_s"])
        n = m1.size

        data_vars: Dict[str, Any] = {
            "m1": ("shot_idx", m1),
            "m2": ("shot_idx", m2),
            "parity": ("shot_idx", parity),
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
            "psd_freq_hz": np.asarray(results["psd_freq_hz"], dtype=float),
            "state_psd_freq_hz": np.asarray(results["state_psd_freq_hz"],
                                            dtype=float),
        }
        has_iq = "IQdata" in dataset.data_vars or (
            "I" in dataset.data_vars and "Q" in dataset.data_vars
        )
        if has_iq:
            iq = with_iqdata(dataset)["IQdata"]
            flat = iq.values.ravel()
            step = max(1, -(-flat.size // _IQ_PANEL_MAX))  # ceil: stay <= cap
            sub = flat[::step]
            data_vars["iq_i"] = ("iq_idx", np.real(sub).astype(float))
            data_vars["iq_q"] = ("iq_idx", np.imag(sub).astype(float))

        attrs: Dict[str, Any] = {
            k: results[k]
            for k in ("parity_rate_hz", "psd_corner_hz", "psd_amplitude",
                      "psd_white_floor", "n_transitions", "p_switch",
                      "p_high", "p_parity_odd", "p_intercycle_flip",
                      "p_m1_high", "p_m2_high", "psd_freq_min_hz",
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
        """Three stacked snippet rows (M1, M2, parity), the parity's log-log
        PSD with the knee fit, and the raw M1 spectrum, unfitted (+ the shared
        IQ-plane panel when the input carried quadratures). Draws only from
        ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        pd = plot_data
        builders = {"timetrace": lambda: plot_timetrace(pd),
                    "psd": lambda: plot_psd(pd),
                    "state_psd": lambda: plot_state_psd(pd)}
        if has_iq_plane(pd):
            builders["iq_plane"] = lambda: plot_iq_plane(pd)
        return render_figures(builders, label=self.estimator_name)
