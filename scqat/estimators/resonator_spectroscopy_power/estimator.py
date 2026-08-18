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

A third stage reads the TWO BRANCHES of the punchout. At low power the qubit
stays in |0> and dresses the resonator, so the dip sits at ``f_dress0``; driven
hard enough the qubit saturates and the dip walks to the bare resonator
``f_bare``. Their difference is the Lamb shift ``g^2/Delta`` (reported as
``lamb_shift``). This is what makes a punchout the independent source of a BARE
resonator frequency — a dispersive flux fit can only trade ``f_r0`` off against
``g``. The branches are found by ANCHORED TWO-BAND CLASSIFICATION
(:func:`_branch_frequencies`): each plateau is anchored at its own end of the
power axis, every point is classified against the two FIXED anchors, and the
contiguous in-band run from each end is the plateau — the points between the
two runs are the transition and belong to NEITHER branch. The run edges are
reported as ``dress_max_power`` / ``bare_min_power``: the highest power that is
still dispersive and the lowest that is fully punched out.

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
readout drive amplitude) — the per-slice dip fits are scale-invariant, so both
raw-instrument and pre-normalized maps are handled.

.. warning::
   Per-slice acceptance is in-window + FITTABLE-width only — deliberately NOT a
   cross-power population test. The dip's width and depth legitimately CHANGE
   across a punchout (the dressed dip is qubit-broadened, the bare one is not;
   measured 8 -> 3 MHz on 5Q4C q2), so a global median/MAD gate systematically
   flags the smaller population — on run 20260818-204626 it deleted the entire
   bare plateau before branch extraction ever saw it. The linewidth drop is
   reported (``fwhm``/``fwhm_median``) as the honest "did we truly saturate?"
   diagnostic, but nothing gates on it.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.core.figures import render_figures
from scqat.tools.dip_fit import fit_dip, validate_dip_kwargs
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
) -> "Tuple[float, float]":
    """``(optimal_power, crossing_power)`` in dBm; NaNs when no crossing is found.

    Ports the official 02b heuristic: differentiate the resonator-centre trace
    with respect to power, smooth it, scale down the noisy leading edge, and take
    the first power whose smoothed ``d(center)/d(power)`` drops below the
    (negative) ``threshold_hz_per_dbm``; then step ``buffer_dbm`` below it. Runs on
    the robust fitted ``center_detuning(power)`` rather than a raw ``idxmin``.

    The CROSSING is where the centre STARTS moving; the OPTIMAL power is the
    crossing stepped ``buffer_dbm`` lower, i.e. a safe operating point inside the
    dispersive regime. Deliberately NOT used for the branch plateaus: the
    derivative is smoothed over ~10 points, so it trips several dB early and
    clears several dB late — the plateaus are classified against their own
    anchors instead (:func:`_branch_frequencies`).
    """
    power = np.asarray(power, dtype=float)
    center = np.asarray(center_detuning, dtype=float)
    finite = np.isfinite(center)
    if power.size < 3 or finite.sum() < 2:
        return float("nan"), float("nan")

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
        return float("nan"), float("nan")
    idx = int(np.argmax(below))  # first power below the threshold
    crossing = float(power[idx])
    return crossing - float(buffer_dbm), crossing


def _plateau_members(in_band: np.ndarray) -> np.ndarray:
    """Member mask of the LEADING plateau run, tolerating isolated glitches.

    The run extends while points stay in-band, bridging a SINGLE out-of-band
    point whenever the very next point is back in-band — one TLS-jumped or
    glitched slice must not amputate an otherwise consistent plateau (run
    20260818-205631: one 2.9 MHz glitch at the 3rd slice cost the whole dressed
    branch under strict contiguity). Never two consecutive out-of-band points:
    a real transition departs the FIXED band monotonically and cannot re-enter
    after exactly one point, so this keeps the anti-creep guarantee. Bridged
    points are NOT members — they feed neither the median nor the boundary.
    """
    members = np.zeros(in_band.size, dtype=bool)
    i = 0
    while i < in_band.size:
        if in_band[i]:
            members[i] = True
            i += 1
        elif i + 1 < in_band.size and in_band[i + 1]:
            i += 1  # isolated glitch, bridged (and excluded)
        else:
            break
    return members


def _branch_frequencies(
    power: np.ndarray,
    center_full_freq: np.ndarray,
    good: np.ndarray,
    *,
    band_frac: float,
    min_points: int,
    anchor_points: int,
) -> "Tuple[float, float, float, float, int, int, np.ndarray]":
    """``(f_dress0, f_bare, dress_max_power, bare_min_power, n_low, n_high,
    branch_class)`` — the two punchout branches (Hz), their plateau boundary
    powers (dBm), and the per-point class over the INPUT power order
    (1 = dressed member, 2 = bare member, 0 = neither).

    THE PHYSICS: at LOW power the resonator is dressed by a qubit that stays in
    |0>, so the dip sits at ``f_dress0``. Driven hard enough the qubit saturates
    and stops dressing it, so the dip walks to the BARE resonator ``f_bare``. The
    gap between the two plateaus is the Lamb shift ``g^2/Delta`` — which is why a
    punchout measures the bare frequency that a flux-map dispersive fit can only
    trade off against ``g``.

    ANCHORED TWO-BAND CLASSIFICATION. Each plateau is ANCHORED at its own end of
    the power axis (median of the ``anchor_points`` extreme slices — a median of
    3 survives one bad end slice), every point is classified against the two
    FIXED anchors with a per-side band tolerance, and each plateau is the
    leading in-band run from its window end (isolated one-point glitches
    bridged — :func:`_plateau_members`). Two anti-creep devices, both learned
    from real data (run 20260818-204626, 5Q4C q2): the anchor never moves — a
    previous implementation grew each plateau against a LAGGING median and
    crept point-by-point into a gradual transition — and the run-from-the-end
    structure stops a transition point that wanders back into a band (the
    bifurcation-regime centre OVERSHOOTS below f_bare, then recovers toward it)
    from joining a plateau it is not connected to. Points in neither band are
    the transition and belong to NO branch.

    The band tolerance is ``band_frac x anchor separation`` — derived from the
    branch separation ONLY, never from the full centre span (which the
    transition overshoot inflates) and never from the anchor's own scatter (a
    glitched anchor slice would then WIDEN its own band and readmit exactly the
    transition creep this function exists to reject). The anchor scatter is
    used the other way round, as a CREDIBILITY gate: a side whose 2 robust
    sigmas exceed the band cannot certify a plateau at all and is refused —
    that is what a window truncated MID-TRANSITION looks like from its end.
    Plateaus noisier than the band lose points to the band test pseudo-randomly
    and, past isolated-glitch bridging, terminate early — for data that noisy
    relative to the Lamb shift, widening ``band_frac`` is the explicit,
    documented lever.

    MEDIAN of each plateau, not mean: one TLS jump in one slice must not move
    the answer. A side with fewer than ``min_points`` member points yields NaN
    for that branch (and its boundary power) alone; a punchout whose window only
    reached the dispersive regime still reports ``f_dress0``. When the anchors
    are closer than the plateau noise — or NEITHER side is credible — the trace
    resolved no punchout and is reported wholly as the dressed branch: this
    layer cannot tell an all-dispersive window from an all-punched-out one
    (a window that STARTED above the knee), unchanged from the previous
    behaviour.
    """
    power = np.asarray(power, dtype=float)
    center = np.asarray(center_full_freq, dtype=float)
    keep = np.asarray(good, dtype=bool) & np.isfinite(center)
    nan = float("nan")
    branch_class = np.zeros(power.size, dtype=int)
    if keep.sum() < 2:
        return nan, nan, nan, nan, 0, 0, branch_class

    order = np.argsort(power)
    kept_idx = order[keep[order]]          # original indices, ascending power
    p_sorted = power[kept_idx]
    c_sorted = center[kept_idx]
    n = p_sorted.size

    k = max(1, min(int(anchor_points), n // 2))
    low_anchor_pts = c_sorted[:k]
    high_anchor_pts = c_sorted[-k:]
    anchor_d = float(np.median(low_anchor_pts))
    anchor_b = float(np.median(high_anchor_pts))
    separation = abs(anchor_d - anchor_b)

    # Robust sigma of each anchor's own scatter -> the plateau noise floor.
    sigma_d = 1.4826 * float(np.median(np.abs(low_anchor_pts - anchor_d)))
    sigma_b = 1.4826 * float(np.median(np.abs(high_anchor_pts - anchor_b)))
    band = band_frac * separation
    credible_d = 2.0 * sigma_d <= band
    credible_b = 2.0 * sigma_b <= band

    def _one_plateau():
        branch_class[kept_idx] = 1
        return (float(np.median(c_sorted)), nan,
                float(p_sorted[-1]), nan, n, 0, branch_class)

    # No punchout resolved in the window: anchors indistinguishable, or no end
    # of the trace is settled enough to certify a plateau.
    if separation <= max(5.0 * sigma_d, 5.0 * sigma_b, 1.0):
        return _one_plateau()
    if not credible_d and not credible_b:
        return _one_plateau()

    # The tolerance is the band alone — see the docstring on why the anchor
    # scatter must never widen it.
    tol = max(band, 1.0)
    members_low = np.zeros(n, dtype=bool)
    members_high = np.zeros(n, dtype=bool)
    if credible_d:
        members_low = _plateau_members(np.abs(c_sorted - anchor_d) <= tol)
    if credible_b:
        members_high = _plateau_members(
            np.abs(c_sorted[::-1] - anchor_b) <= tol)[::-1]
    # Overlapping spans: no transition resolved between the bands, so there is
    # only one plateau to report.
    if members_low.any() and members_high.any():
        if int(np.flatnonzero(members_low)[-1]) >= int(np.flatnonzero(members_high)[0]):
            return _one_plateau()

    n_low = int(members_low.sum())
    n_high = int(members_high.sum())
    if n_low >= min_points:
        f_dress0 = float(np.median(c_sorted[members_low]))
        dress_max_power = float(p_sorted[np.flatnonzero(members_low)[-1]])
        branch_class[kept_idx[members_low]] = 1
    else:
        f_dress0, dress_max_power = nan, nan
    if n_high >= min_points:
        f_bare = float(np.median(c_sorted[members_high]))
        bare_min_power = float(p_sorted[np.flatnonzero(members_high)[0]])
        branch_class[kept_idx[members_high]] = 2
    else:
        f_bare, bare_min_power = nan, nan
    return (f_dress0, f_bare, dress_max_power, bare_min_power,
            n_low, n_high, branch_class)


class ResonatorSpectroscopyPowerEstimator(BaseEstimator):
    """
    Fit the resonator dip at every readout power, report the resonator centre
    frequency as a function of power, and pick the optimal readout power.

    The result dict reports, per power point, the dip ``center_detuning`` (and
    absolute ``center_full_freq`` when available), the ``fwhm`` and a per-point
    ``success`` flag, alongside the 2-D ``amplitude`` map kept for plotting; plus
    the scalar deliverables ``optimal_power`` / ``frequency_shift`` /
    ``resonator_frequency`` and an overall ``optimal_success`` flag; plus the two
    punchout branches ``f_dress0`` / ``f_bare`` with their ``lamb_shift``, the
    plateau boundary powers ``dress_max_power`` / ``bare_min_power``, the
    per-power ``branch_class`` (1 = dressed run, 2 = bare run, 0 = neither) and
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

        Acceptance per power point: (1) the fitted dip centre must lie strictly
        **inside** the swept detuning window, and (2) the fitted ``fwhm`` must be
        FITTABLE — wider than two frequency-grid steps and narrower than the
        swept window. Deliberately NO cross-power population test: a punchout's
        dip width and depth legitimately change between the dressed and bare
        branches (the dressed dip is qubit-broadened), so a global median/MAD
        gate systematically deletes the smaller plateau — see the module
        warning. ``fwhm_median``/``fwhm_mad`` are reported as diagnostics only.

        Keyword arguments
        -----------------
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
        branch_band_frac : float, optional
            Half-width of each plateau's classification band as a fraction of the
            anchor separation (default 0.08). Never span-derived: the transition
            overshoot inflates the full centre span.
        branch_anchor_points : int, optional
            Extreme-power slices whose median anchors each plateau (default 3).
        branch_min_points : int, optional
            Minimum in-band points a plateau needs before its branch frequency
            and boundary power are reported; a shorter side yields NaN for that
            branch alone (default 3).

        Returns
        -------
        dict
            ``{power, detuning, full_freq?, center_detuning, center_full_freq?,
            fwhm, dip_amplitude, success, in_window, outlier, good,
            fwhm_median, fwhm_mad,
            amplitude_map, n_power, n_success, n_good, n_outlier,
            optimal_power, crossing_power, frequency_shift, resonator_frequency,
            optimal_success, f_dress0, f_bare, lamb_shift, dress_max_power,
            bare_min_power, branch_class, n_low_plateau, n_high_plateau,
            branch_success}``
        """
        threshold = float(kwargs.pop("derivative_crossing_threshold_in_hz_per_dbm", -50_000.0))
        smoothing_window = int(kwargs.pop("derivative_smoothing_window_num_points", 10))
        init_filter_window = int(kwargs.pop("moving_average_filter_window_num_points", 10))
        buffer_dbm = float(kwargs.pop("buffer_from_crossing_threshold_in_dbm", 1.0))
        branch_band_frac = float(kwargs.pop("branch_band_frac", 0.08))
        branch_anchor_points = int(kwargs.pop("branch_anchor_points", 3))
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

        # 2-D |IQ| amplitude map, oriented (power, detuning) — kept for plotting.
        amplitude_map = np.abs(iq_map)

        # (2) Fittable-width gate ONLY: a width narrower than two grid steps or
        # wider than the swept window is unfittable garbage; anything between is
        # legal physics. Never a cross-power population test — the dressed and
        # bare branches have genuinely different widths, and a global median/MAD
        # gate deletes the smaller plateau (the run-20260818-204626 failure).
        grid = float(np.median(np.abs(np.diff(detuning)))) if detuning.size > 1 else 0.0
        fittable = (fwhm > 2.0 * grid) & (fwhm < (det_hi - det_lo))
        outlier = valid & ~fittable
        good = valid & ~outlier
        # Diagnostics only, nothing gates on them: the fwhm DROP across the
        # transition is the honest "did we truly saturate?" witness.
        fwhm_good = fwhm[good]
        fwhm_med = float(np.median(fwhm_good)) if fwhm_good.size else float("nan")
        fwhm_mad = (float(np.median(np.abs(fwhm_good - fwhm_med)))
                    if fwhm_good.size else float("nan"))

        # Optimal readout power from where the centre trace stops shifting, using
        # only the good (in-window, non-outlier) centres.
        center_for_pick = np.where(good, center_detuning, np.nan)
        optimal_power, crossing_power = _pick_optimal_power(
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
        dress_max_power = bare_min_power = float("nan")
        n_low = n_high = 0
        # Per-power class for diagnosis/plotting: 1 = dressed member, 2 = bare
        # member, 0 = neither (transition, bridged glitch, rejected, failed) —
        # exactly the points each branch median used.
        branch_class = np.zeros(n_power, dtype=int)
        if has_full_freq:
            (f_dress0, f_bare, dress_max_power, bare_min_power,
             n_low, n_high, branch_class) = _branch_frequencies(
                power, center_full_freq, good,
                band_frac=branch_band_frac, min_points=branch_min_points,
                anchor_points=branch_anchor_points,
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
            "amplitude_map": amplitude_map,
            "n_power": int(n_power),
            "n_success": int(success.sum()),
            "n_good": int(good.sum()),
            "n_outlier": int(outlier.sum()),
            "optimal_power": float(optimal_power),
            "crossing_power": float(crossing_power),
            "frequency_shift": frequency_shift,
            "resonator_frequency": resonator_frequency,
            "optimal_success": optimal_success,
            "f_dress0": f_dress0,
            "f_bare": f_bare,
            "lamb_shift": lamb_shift,
            "dress_max_power": float(dress_max_power),
            "bare_min_power": float(bare_min_power),
            "branch_class": branch_class,
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
            "branch_class": ("power", np.asarray(results["branch_class"], np.int8)),
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
            "dress_max_power": float(results["dress_max_power"]),
            "bare_min_power": float(results["bare_min_power"]),
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
