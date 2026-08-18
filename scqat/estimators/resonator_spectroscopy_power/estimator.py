"""
Resonator Spectroscopy Power Estimator
======================================
Reduce a 2-D resonator-spectroscopy-vs-readout-power map to a 1-D
``center_frequency(power)`` curve by fitting the resonator dip **power-by-power**,
then pick the optimal readout power from where that centre stops shifting.

For every readout-power slice the estimator calls the family-shared per-trace
dip fit :func:`scqat.tools.dip_fit.fit_dip` — the same reduction used by 1-D
resonator spectroscopy and the vs-flux map, with ``dip_method`` selecting
``lorentzian`` (default) or ``circle`` — and records the dip centre and
linewidth. Stacking those per-slice centres yields the resonator centre
frequency as a function of readout power (and its FWHM), i.e. the 2-D
``(power, detuning)`` map is collapsed onto a 1-D trace.

On top of that reduction (which mirrors the dip-tracking stage of
:mod:`~scqat.estimators.resonator_spectroscopy_flux` one-for-one, with ``power``
in place of ``flux_bias``), a second stage picks the
**optimal readout power**: the low-power dispersive regime is where
``center_detuning(power)`` stops shifting, so it is found where the smoothed
``d(center)/d(power)`` first crosses below a (negative) threshold — the same
derivative-crossing heuristic the official ``02b`` node uses, but run on the
robust fitted centre trace instead of a raw ``idxmin`` proxy.

A third stage reads the TWO BRANCHES the same crossing separates. Below it the
qubit stays in |0> and dresses the resonator, so the dip sits at ``f_dress0``;
above it the qubit saturates and the dip walks to the bare resonator ``f_bare``.
Their difference is the Lamb shift ``g^2/Delta`` (reported as ``lamb_shift``).
This is what makes a punchout the independent source of a BARE resonator
frequency — a dispersive flux fit can only trade ``f_r0`` off against ``g``.

.. warning::
   ``frequency_shift`` is NOT a branch difference. It is the dip *detuning from
   the LO* at the chosen optimal power — i.e. the dressed frequency expressed as
   a detuning. The branch difference is ``lamb_shift``. The names are kept apart
   deliberately; renaming ``frequency_shift`` would break its consumers for no
   physics gain.

Expected xarray.Dataset contract
---------------------------------
The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).

Coordinates:
    - power     : 1-D float array – readout power in dB (relative to the current
                  readout amplitude, or absolute dBm — any log-scale power axis).
    - detuning  : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute readout frequency (Hz). Optional; when
                  present the centre trace is also reported in absolute frequency
                  and the resonator frequency at the optimal power is reported.
    - digital_amp   : (power,), optional – per-point digital amplitude of the
                  sweep; drawn as an amp/chain subplot under the map.
    - chain_setting : (power,), optional – per-point chain value (QM
                  full_scale_power_dbm / Qblox output_att; constant for a
                  fixed-chain amplitude sweep), labeled by ``chain_name``.
    - chain_name : scalar str, optional – axis label for ``chain_setting``.
    - power_axis_kind : scalar str, optional – x-axis label suffix
                  (e.g. "absolute dBm").
    - mode_label : scalar str, optional – mechanism tag added to the title
                  (e.g. "amplitude sweep (fast)" / "chain-stepped (slow)").
Data variables:
    - IQdata : (power, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (power, detuning) – the two quadratures, combined into IQdata.

Rows may carry a power-dependent overall scale (the measured |IQ| grows with the
readout drive amplitude) — per-slice dip fits are scale-invariant, and the
cross-power amplitude outlier test normalizes by each row's baseline scale, so
both raw-instrument and pre-normalized maps are handled.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.dip_fit import fit_dip, validate_dip_kwargs
from scqat.tools.robust import mad_outliers
from scqat.estimators.resonator_spectroscopy_power.visualization import plot_power_map


def _rolling_mean_nan(x: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean over ``window`` points, ignoring NaNs in each window.

    A dependency-free stand-in for xarray's ``rolling(...).mean()`` used by the
    official 02b analysis. Windows with no finite points stay NaN.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.full(n, np.nan)
    window = max(int(window), 1)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, lo + window)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(seg.mean())
    return out


def _pick_optimal_power(
    power: np.ndarray,
    center_detuning: np.ndarray,
    *,
    threshold_hz_per_dbm: float,
    smoothing_window: int,
    init_filter_window: int,
    buffer_dbm: float,
) -> "Tuple[float, float, float]":
    """``(optimal_power, crossing_power, settled_power)`` in dBm; NaNs when no
    crossing is found.

    Ports the official 02b heuristic: differentiate the resonator-centre trace
    with respect to power, smooth it, scale down the noisy leading edge, and take
    the first power whose smoothed ``d(center)/d(power)`` drops below the
    (negative) ``threshold_hz_per_dbm``; then step ``buffer_dbm`` below it. Runs on
    the robust fitted ``center_detuning(power)`` rather than a raw ``idxmin``.

    The three returns answer different questions and all are needed. The CROSSING
    is where the centre STARTS moving; the SETTLED power is the last point where
    it is still moving, i.e. where the transition ENDS. Together they bracket the
    punchout transition, which is what separates the dressed (low-power) branch
    from the bare (high-power) one — and the bracket matters: the derivative
    crosses the threshold several dB before the knee, so a bare plateau measured
    from the crossing would swallow the whole transition and understate the Lamb
    shift. The OPTIMAL power is the crossing stepped ``buffer_dbm`` lower, i.e. a
    safe operating point inside the dispersive regime.
    """
    power = np.asarray(power, dtype=float)
    center = np.asarray(center_detuning, dtype=float)
    finite = np.isfinite(center)
    if power.size < 3 or finite.sum() < 2:
        return float("nan"), float("nan"), float("nan")

    # Fill fit gaps so the derivative is defined on the full power grid.
    center_filled = np.interp(power, power[finite], center[finite])
    diff = np.gradient(center_filled, power)  # Hz/dBm
    # Drop implausibly large jumps (fit glitches), as 02b does.
    diff = np.where(np.abs(diff) < 1e6, diff, np.nan)

    avg = _rolling_mean_nan(diff, smoothing_window)
    # Scale down the leading (edge-effect) points so they cannot trip the
    # threshold prematurely (denominators window..1, matching 02b).
    m = min(int(init_filter_window), avg.size)
    for j in range(m):
        denom = init_filter_window - j
        if denom > 0 and np.isfinite(avg[j]):
            avg[j] = avg[j] / denom

    below = np.isfinite(avg) & (avg < threshold_hz_per_dbm)
    if not below.any():
        return float("nan"), float("nan"), float("nan")
    idx = int(np.argmax(below))  # first power below the threshold
    end = int(len(below) - 1 - np.argmax(below[::-1]))  # last one still moving
    crossing = float(power[idx])
    return crossing - float(buffer_dbm), crossing, float(power[end])


def _grow_plateau(center: np.ndarray, tol: float) -> int:
    """How many leading points of ``center`` stay flat within ``tol`` of the
    running median. Always at least 1 for a non-empty input."""
    if center.size == 0:
        return 0
    kept = 1
    for i in range(1, center.size):
        if abs(center[i] - float(np.median(center[:kept]))) > tol:
            break
        kept = i + 1
    return kept


def _branch_frequencies(
    power: np.ndarray,
    center_full_freq: np.ndarray,
    good: np.ndarray,
    *,
    tol_frac: float,
    min_points: int,
) -> "Tuple[float, float, int, int]":
    """``(f_dress0, f_bare, n_low, n_high)`` — the two punchout branches, in Hz.

    THE PHYSICS: at LOW power the resonator is dressed by a qubit that stays in
    |0>, so the dip sits at ``f_dress0``. Driven hard enough the qubit saturates
    and stops dressing it, so the dip walks to the BARE resonator ``f_bare``. The
    gap between the two plateaus is the Lamb shift ``g^2/Delta`` — which is why a
    punchout measures the bare frequency that a flux-map dispersive fit can only
    trade off against ``g``.

    Each plateau is GROWN inward from its end of the power axis for as long as
    the centre stays within ``tol_frac`` of the full centre span, and the moving
    middle simply never gets reached. Deliberately NOT derived from the
    optimal-power derivative: that derivative is smoothed over ~10 points so the
    threshold trips several dB early and clears several dB late, which would put
    the whole transition inside the bare plateau and halve the reported Lamb
    shift. The trace's own flatness is the honest test of where a plateau is.

    MEDIAN of each plateau, not mean, for the same reason the acceptance gates
    are MAD-based: one TLS jump in one slice must not move the answer. A side
    with fewer than ``min_points`` good points yields NaN for that branch alone;
    a punchout whose window only reached the dispersive regime still reports
    ``f_dress0``.
    """
    power = np.asarray(power, dtype=float)
    center = np.asarray(center_full_freq, dtype=float)
    keep = np.asarray(good, dtype=bool) & np.isfinite(center)
    if keep.sum() < 2:
        return float("nan"), float("nan"), 0, 0

    order = np.argsort(power)
    p_sorted = power[order][keep[order]]
    c_sorted = center[order][keep[order]]
    span = float(np.max(c_sorted) - np.min(c_sorted))
    tol = max(tol_frac * span, 1.0)  # a flat trace has no branches to separate

    n_low = _grow_plateau(c_sorted, tol)             # from the lowest power up
    n_high = _grow_plateau(c_sorted[::-1], tol)      # from the highest power down
    # A trace that never left one plateau (no punchout in the window) must not be
    # cut into two "branches" of itself.
    if n_low + n_high > p_sorted.size:
        return float(np.median(c_sorted)), float("nan"), int(p_sorted.size), 0

    f_dress0 = (float(np.median(c_sorted[:n_low]))
                if n_low >= min_points else float("nan"))
    f_bare = (float(np.median(c_sorted[-n_high:]))
              if n_high >= min_points else float("nan"))
    return f_dress0, f_bare, int(n_low), int(n_high)


class ResonatorSpectroscopyPowerEstimator(BaseEstimator):
    """
    Fit the resonator dip at every readout power, report the resonator centre
    frequency as a function of power, and pick the optimal readout power.

    The result dict reports, per power point, the dip ``center_detuning`` (and
    absolute ``center_full_freq`` when available), the ``fwhm`` and a per-point
    ``success`` flag, alongside the 2-D ``amplitude`` map kept for plotting; plus
    the scalar deliverables ``optimal_power`` / ``frequency_shift`` /
    ``resonator_frequency`` and an overall ``optimal_success`` flag; plus the two
    punchout branches ``f_dress0`` / ``f_bare`` with their ``lamb_shift`` and
    ``branch_success``.
    """

    estimator_name = "resonator_spectroscopy_power"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in ("power", "detuning"):
            if coord not in dataset.coords:
                raise ValueError(
                    f"ResonatorSpectroscopyPowerEstimator requires a '{coord}' coordinate."
                )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "ResonatorSpectroscopyPowerEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit the resonator dip in every ``power`` slice, stack the centres, and
        pick the optimal readout power from the centre trace.

        Each slice is handed to the family-shared per-trace dip fit
        :func:`scqat.tools.dip_fit.fit_dip` — ``dip_method`` selects
        ``"lorentzian"`` (default) or ``"circle"``, and the remaining kwargs
        (``baseline_order`` / ``delay``) are that method's knobs, validated
        BEFORE the slice loop (unknown names raise ValueError).

        Acceptance per power point happens in two stages: (1) the fitted dip centre
        must lie strictly **inside** the swept detuning window, and (2) the dip
        ``fwhm`` and the **baseline-normalized** dip amplitude (``|dip_amplitude|``
        divided by the row's 90th-percentile ``|IQ|`` squared — rows scale with the
        readout drive on real instruments) must not be robust (median/MAD) outliers
        across power. ``dip_amplitude_median``/``dip_amplitude_mad`` report the
        normalized statistic; ``dip_amplitude`` itself stays in raw ``|IQ|^2`` units.

        Keyword arguments
        -----------------
        n_sigma : float, optional
            Robust-sigma threshold for the width / amplitude outlier test
            (default 3.0). Not forwarded to the per-slice estimator.
        derivative_crossing_threshold_in_hz_per_dbm : float, optional
            Smoothed ``d(center)/d(power)`` threshold (negative) that marks the
            optimal-power crossing (default -50_000).
        derivative_smoothing_window_num_points : int, optional
            Rolling-mean window (points) for the derivative (default 10).
        moving_average_filter_window_num_points : int, optional
            Number of leading derivative points scaled down before thresholding
            (default 10).
        buffer_from_crossing_threshold_in_dbm : float, optional
            dBm stepped below the crossing to set the optimal power (default 1).
        branch_tol_frac : float, optional
            Flatness tolerance for growing each branch plateau inward from its end
            of the power axis, as a fraction of the full centre span (default
            0.15). Deliberately NOT derived from the optimal-power derivative,
            which is smoothed over ~10 points and so brackets the transition far
            too generously.
        branch_min_points : int, optional
            Minimum good points a plateau needs before its branch frequency is
            reported; a shorter side yields NaN for that branch alone (default 3).

        Returns
        -------
        dict
            ``{power, detuning, full_freq?, center_detuning, center_full_freq?,
            fwhm, dip_amplitude, success, in_window, outlier, good,
            fwhm_median, fwhm_mad, dip_amplitude_median, dip_amplitude_mad,
            amplitude_map, n_power, n_success, n_good, n_outlier,
            optimal_power, crossing_power, frequency_shift, resonator_frequency,
            optimal_success, f_dress0, f_bare, lamb_shift, n_low_plateau,
            n_high_plateau, branch_success}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))
        threshold = float(kwargs.pop("derivative_crossing_threshold_in_hz_per_dbm", -50_000.0))
        smoothing_window = int(kwargs.pop("derivative_smoothing_window_num_points", 10))
        init_filter_window = int(kwargs.pop("moving_average_filter_window_num_points", 10))
        buffer_dbm = float(kwargs.pop("buffer_from_crossing_threshold_in_dbm", 1.0))
        branch_tol_frac = float(kwargs.pop("branch_tol_frac", 0.15))
        branch_min_points = int(kwargs.pop("branch_min_points", 3))
        dip_method = str(kwargs.pop("dip_method", "lorentzian"))
        # Fail loudly BEFORE the per-slice loop — a typo'd knob must never be
        # swallowed by the per-slice fallback.
        validate_dip_kwargs(dip_method, kwargs)

        ds = with_iqdata(dataset)
        power = ds.coords["power"].values.astype(float)
        detuning = ds.coords["detuning"].values.astype(float)
        has_full_freq = "full_freq" in ds.coords
        full_freq = (
            ds.coords["full_freq"].values.ravel().astype(float) if has_full_freq else None
        )
        if dip_method == "circle" and full_freq is None:
            raise ValueError(
                "dip_method='circle' requires the 'full_freq' coordinate "
                "(absolute readout frequency in Hz)."
            )
        iq_map = ds["IQdata"].transpose("power", "detuning").values

        n_power = len(power)
        center_detuning = np.full(n_power, np.nan)
        center_full_freq = np.full(n_power, np.nan)
        fwhm = np.full(n_power, np.nan)
        dip_amplitude = np.full(n_power, np.nan)
        success = np.zeros(n_power, dtype=bool)

        for k in range(n_power):
            try:
                r = fit_dip(detuning, iq_map[k], full_freq=full_freq,
                            method=dip_method, **kwargs)
            except Exception:
                # Leave NaN / False for this power point and carry on
                # (fit-domain failure only: kwargs were validated up front).
                continue
            center_detuning[k] = r["detuning"]
            fwhm[k] = r["fwhm"]
            # Method-owned extra — only lorentzian reports a dip amplitude.
            dip_amplitude[k] = r.get("amplitude", np.nan)
            success[k] = bool(r["success"])
            if has_full_freq and "full_freq" in r:
                center_full_freq[k] = r["full_freq"]

        # (1) Strict window enforcement: the fitted dip centre must lie inside the
        # swept detuning window (a centre at an edge means the fit was pinned).
        det_lo, det_hi = float(detuning.min()), float(detuning.max())
        in_window = np.isfinite(center_detuning) & (center_detuning > det_lo) & (center_detuning < det_hi)
        valid = success & in_window

        # 2-D |IQ| amplitude map, oriented (power, detuning) — kept for plotting,
        # and its per-row median doubles as the row's baseline scale below.
        amplitude_map = np.abs(iq_map)

        # (2) Robust outlier rejection on the dip width and amplitude. The measured
        # |IQ| grows with the readout drive, so rows carry a power-dependent overall
        # scale and raw dip amplitudes are NOT comparable across power. Divide out
        # each row's baseline scale — a HIGH quantile of |IQ| over detuning (the top
        # decile sits on the off-resonant baseline even when the dip covers a sizable
        # fraction of the span; the median does not, and its dip-depth bias can flip
        # borderline flags) — squared to match the |IQ|^2 units of the fitted dip
        # amplitude. For pre-normalized data the row scale is ~constant across rows,
        # leaving the flags as before.
        row_scale = np.quantile(amplitude_map, 0.9, axis=1) ** 2
        rel_amp = np.abs(dip_amplitude) / np.maximum(row_scale, np.finfo(float).tiny)
        outlier_fwhm, fwhm_med, fwhm_mad = mad_outliers(fwhm, valid, n_sigma)
        outlier_amp, amp_med, amp_mad = mad_outliers(rel_amp, valid, n_sigma)
        outlier = valid & (outlier_fwhm | outlier_amp)
        good = valid & ~outlier

        # Optimal readout power from where the centre trace stops shifting, using
        # only the good (in-window, non-outlier) centres.
        center_for_pick = np.where(good, center_detuning, np.nan)
        optimal_power, crossing_power, settled_power = _pick_optimal_power(
            power,
            center_for_pick,
            threshold_hz_per_dbm=threshold,
            smoothing_window=smoothing_window,
            init_filter_window=init_filter_window,
            buffer_dbm=buffer_dbm,
        )

        frequency_shift = float("nan")
        resonator_frequency = float("nan")
        if np.isfinite(optimal_power) and good.any():
            idx = int(np.argmin(np.abs(power - optimal_power)))
            frequency_shift = float(center_detuning[idx])
            if has_full_freq:
                resonator_frequency = float(center_full_freq[idx])

        optimal_success = bool(
            np.isfinite(optimal_power)
            and np.isfinite(frequency_shift)
            and det_lo < frequency_shift < det_hi
        )

        # The two punchout BRANCHES. Absolute frequencies only: a bare/dressed
        # pair is meaningless as a detuning from a readout LO that may move, and
        # they are written to the device as absolute facts.
        f_dress0 = f_bare = lamb_shift = float("nan")
        n_low = n_high = 0
        if has_full_freq:
            f_dress0, f_bare, n_low, n_high = _branch_frequencies(
                power, center_full_freq, good,
                tol_frac=branch_tol_frac, min_points=branch_min_points,
            )
            lamb_shift = f_dress0 - f_bare
        branch_success = bool(np.isfinite(f_dress0) and np.isfinite(f_bare))

        results: Dict[str, Any] = {
            "power": power,
            "detuning": detuning,
            "center_detuning": center_detuning,
            "fwhm": fwhm,
            "dip_amplitude": dip_amplitude,
            "success": success,
            "in_window": in_window,
            "outlier": outlier,
            "good": good,
            "dip_method": dip_method,
            "fwhm_median": fwhm_med,
            "fwhm_mad": fwhm_mad,
            "dip_amplitude_median": amp_med,
            "dip_amplitude_mad": amp_mad,
            "amplitude_map": amplitude_map,
            "n_power": int(n_power),
            "n_success": int(success.sum()),
            "n_good": int(good.sum()),
            "n_outlier": int(outlier.sum()),
            "optimal_power": float(optimal_power),
            "crossing_power": float(crossing_power),
            "settled_power": float(settled_power),
            "frequency_shift": frequency_shift,
            "resonator_frequency": resonator_frequency,
            "optimal_success": optimal_success,
            "f_dress0": f_dress0,
            "f_bare": f_bare,
            "lamb_shift": lamb_shift,
            "n_low_plateau": int(n_low),
            "n_high_plateau": int(n_high),
            "branch_success": branch_success,
        }
        if has_full_freq:
            results["full_freq"] = ds.coords["full_freq"].values.ravel().astype(float)
            results["center_full_freq"] = center_full_freq

        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the 1-D power-indexed traces + scalar deliverables; drop the
        bulky 2-D map and the spectrum axes (those belong in the plot data)."""
        drop = {"amplitude_map", "detuning", "full_freq"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the 2-D amplitude map and the extracted centre trace into one
        self-sufficient Dataset so the figure redraws with no refitting.

        Alongside the raw linear ``amplitude``, stores the power-normalized
        ``amplitude_db`` (``20*log10|IQ| - power``) the figure colors by: with
        both response and drive on a log scale, subtracting the input power
        removes the swept-drive brightness gradient across rows."""
        power = np.asarray(results["power"], dtype=float)
        detuning = np.asarray(results["detuning"], dtype=float)
        amplitude = np.asarray(results["amplitude_map"], dtype=float)
        amplitude_db = (
            20.0 * np.log10(np.maximum(amplitude, np.finfo(float).tiny)) - power[:, None]
        )

        data_vars: Dict[str, Any] = {
            "amplitude": (("power", "detuning"), amplitude),
            "amplitude_db": (("power", "detuning"), amplitude_db),
            "center_detuning": ("power", np.asarray(results["center_detuning"], float)),
            "fwhm": ("power", np.asarray(results["fwhm"], float)),
            "dip_amplitude": ("power", np.asarray(results["dip_amplitude"], float)),
            "success": ("power", np.asarray(results["success"], bool)),
            "good": ("power", np.asarray(results["good"], bool)),
            "outlier": ("power", np.asarray(results["outlier"], bool)),
        }
        coords: Dict[str, Any] = {"power": power, "detuning": detuning}
        attrs: Dict[str, Any] = {
            "dip_method": str(results["dip_method"]),
            "n_power": int(results["n_power"]),
            "n_success": int(results["n_success"]),
            "n_good": int(results["n_good"]),
            "n_outlier": int(results["n_outlier"]),
            "optimal_power": float(results["optimal_power"]),
            "crossing_power": float(results["crossing_power"]),
            "frequency_shift": float(results["frequency_shift"]),
            "optimal_success": int(bool(results["optimal_success"])),
            # the two punchout branches, so the figure redraws them with no re-fit
            "f_dress0": float(results["f_dress0"]),
            "f_bare": float(results["f_bare"]),
            "lamb_shift": float(results["lamb_shift"]),
            "n_low_plateau": int(results["n_low_plateau"]),
            "n_high_plateau": int(results["n_high_plateau"]),
            "branch_success": int(bool(results["branch_success"])),
        }

        if "full_freq" in results:
            coords["full_freq"] = ("detuning", np.asarray(results["full_freq"], float))
            data_vars["center_full_freq"] = (
                "power", np.asarray(results["center_full_freq"], float)
            )
            attrs["has_full_freq"] = 1
            attrs["resonator_frequency"] = float(results["resonator_frequency"])
        else:
            attrs["has_full_freq"] = 0

        # Optional acquisition-chain provenance (attached by the caller as coords,
        # e.g. the scqo punchouts — both emit the SAME form). Pure pass-through —
        # provenance-blind; absent coords change nothing.
        # digital_amp(power) + chain_setting(power) + a chain_name label are drawn
        # as an amp/chain subplot under the map; power_axis_kind labels the x-axis
        # and mode_label tags the mechanism in the title.
        for key in ("digital_amp", "chain_setting"):
            if key in dataset.coords:
                arr = np.asarray(dataset.coords[key].values, dtype=float)
                if arr.shape == power.shape:
                    data_vars[key] = ("power", arr)
        if "chain_name" in dataset.coords:
            name = str(dataset.coords["chain_name"].values)
            if name:
                attrs["chain_name"] = name
        for key in ("power_axis_kind", "mode_label"):
            if key in dataset.coords:
                val = str(dataset.coords[key].values)
                if val:
                    attrs[key] = val

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Single figure: the power-normalized ``20*log10|IQ| - power`` map over
        (power, frequency) with the fitted resonator-centre trace, the
        optimal-power marker and the two branch frequencies overlaid, drawn from
        plot_data.

        Passed as a THUNK through ``render_figures`` so a plotter failure is
        skipped with a warning rather than dropping the run's only figure."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures(
            {"resonator_spectroscopy_power": lambda: plot_power_map(plot_data)},
            label=self.estimator_name,
        )
