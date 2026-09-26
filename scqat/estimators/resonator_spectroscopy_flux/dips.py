"""Stage 1 of the resonator-vs-flux analysis: per-flux dip tracking.

Reduce a 2-D resonator-spectroscopy-vs-flux map to a 1-D
``center_frequency(flux)`` trace by locating and fitting the resonator dip
**flux-by-flux**. Per slice the dip candidate is first found robustly as the
minimum of the median-smoothed ``|IQ|`` **after subtracting a smooth background**
(so a sloped baseline can't send the argmin to a dark window edge), with a
minimum-depth gate; the family-shared per-trace fit
:func:`scqat.tools.dip_fit.fit_dip` (``dip_method`` selects ``lorentzian`` or
``circle``) then runs on a LOCAL window around that candidate — a full-span fit
chases off-resonance background structure and pins to window edges on real maps,
wasting perfectly good dips. When even the local refinement fails, the
smoothed-argmin centre itself is used (grid-accurate; flagged ``refined=False``).

These are plain functions: inside scqat they are the first stage of
:class:`.estimator.ResonatorSpectroscopyFluxEstimator`, and control repos (e.g.
the LCHQMDriver qualibrate node) call :func:`track_dips` directly when they need
the dip trace WITHOUT the flux-model fit — e.g. a coupler-driven sweep, where
the transmon flux-dependence model does not apply.

Dataset contract (the ``qubit`` dimension already removed):

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute readout frequency (Hz). Optional; when
                  present the centre trace is also reported in absolute frequency
                  (and REQUIRED by ``dip_method="circle"``).
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata.
"""

from typing import Any, Dict, Tuple

import numpy as np
import xarray as xr

from scqat.core.base_estimator import with_iqdata
from scqat.tools.dip_fit import DIP_KNOBS, fit_dip, validate_dip_kwargs
from scqat.tools.robust import mad_outliers
from scqat.tools.sweep_order import ascending


def check_dataset(dataset: xr.Dataset) -> None:
    """Validate the 2-D map contract (see module docstring); raise ValueError."""
    for coord in ("flux_bias", "detuning"):
        if coord not in dataset.coords:
            raise ValueError(
                f"resonator_spectroscopy_flux requires a '{coord}' coordinate."
            )
    if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
        raise ValueError(
            "resonator_spectroscopy_flux requires an 'IQdata' variable, or both 'I' and 'Q'."
        )


def _median_smooth(v: np.ndarray, k: int) -> np.ndarray:
    """Length-preserving running median (odd window ``k``)."""
    if len(v) < k or k < 2:
        return np.asarray(v, dtype=float)
    pad = k // 2
    vp = np.pad(np.asarray(v, dtype=float), pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(vp, k), axis=1)


def _robust_baseline(y: np.ndarray, deg: int = 2, iters: int = 3) -> np.ndarray:
    """Smooth background of a dip trace: a low-degree polynomial tracking the
    UPPER envelope (the resonator dip sits below it), fit with a few iterations
    that drop points falling well below the current fit. A flat 75th-percentile
    baseline is defeated by a sloped |IQ| background (bright off-resonance at one
    window edge, dark at the other) — then the global argmin lands on the dark
    window edge, not the dip. Detrending against this baseline removes the slope
    so the dip becomes the true minimum of the residual.
    """
    n = len(y)
    deg = min(deg, max(1, n - 1))
    x = np.linspace(-1.0, 1.0, n)  # conditioned abscissa for polyfit
    mask = np.ones(n, dtype=bool)
    base = np.polyval(np.polyfit(x, y, deg), x)
    for _ in range(iters):
        resid = y - base
        sel = resid[mask]
        sigma = float(np.std(sel)) if sel.size > deg + 1 else float(np.std(resid))
        if sigma <= 0:
            break
        new_mask = resid > -1.5 * sigma  # keep background; drop the dip (below)
        if new_mask.sum() < deg + 2 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
        base = np.polyval(np.polyfit(x[mask], y[mask], deg), x)
    return base


def _dip_candidate(amp: np.ndarray) -> Tuple[int, float, float]:
    """Robust dip candidate on one slice's ``|IQ|``.

    Returns ``(index, depth, coarse_fwhm_pts)``: the argmin of the smoothed trace
    AFTER subtracting a smooth background (:func:`_robust_baseline`), the
    fractional dip depth ``(baseline - min) / baseline`` at that point relative
    to the LOCAL background, and the full width (in points) of the residual dip
    at half depth (coarse linewidth for the fallback path). Detrending is what
    makes the argmin find the resonator dip instead of a sloped background's
    dark window edge.
    """
    n = len(amp)
    k = max(5, n // 25)
    if k % 2 == 0:
        k += 1
    sm = _median_smooth(amp, k)
    base = _robust_baseline(sm)
    resid = sm - base
    j = int(np.argmin(resid))
    base_j = float(base[j])
    if base_j <= 0:
        return j, 0.0, float("nan")
    depth = float((base_j - sm[j]) / base_j)
    # Full width at half depth of the residual dip (coarse fallback linewidth).
    half = 0.5 * float(resid[j])  # resid[j] < 0; half-way back up to the baseline
    left = j
    while left > 0 and resid[left] < half:
        left -= 1
    right = j
    while right < n - 1 and resid[right] < half:
        right += 1
    return j, depth, float(max(right - left, 1))


def track_dips(
    dataset: xr.Dataset,
    *,
    n_sigma: float = 3.0,
    edge_margin_frac: float = 0.06,
    min_dip_depth: float = 0.10,
    local_window_frac: float = 0.30,
    dip_method: str = "lorentzian",
    **dip_knobs,
) -> Dict[str, Any]:
    """Locate + fit the resonator dip in every ``flux_bias`` slice and stack the
    centres.

    Per slice: (a) the dip candidate is the minimum of the background-detrended
    median-smoothed ``|IQ|``, gated on a minimum relative depth; (b) the
    family-shared :func:`scqat.tools.dip_fit.fit_dip` (``dip_method`` +
    ``dip_knobs``) runs on a local window around the candidate; (c) if the
    refinement fails or wanders out of its window, the candidate itself is the
    centre (``refined=False``).

    Acceptance across flux then happens in two stages: (1) the centre must lie
    inside the swept detuning window, kept ``edge_margin_frac`` away from either
    edge, and (2) among the REFINED points, the dip ``fwhm`` and — when the dip
    method reports one — ``|dip_amplitude|`` must not be LOW-side robust
    (median/MAD) outliers: spurious noise-minimum "dips" are too narrow / too
    weak versus the real population, while genuinely broader/deeper points are
    physics. The surviving ``good`` points are what downstream
    frequency-vs-flux fits should use.

    Keyword arguments (all keyword-only)
    ------------------------------------
    n_sigma : float
        Robust-sigma threshold for the width / amplitude outlier test (default 3.0).
    edge_margin_frac : float
        Fraction of the swept detuning window treated as the rejection margin at
        each edge (default 0.06; 0 disables).
    min_dip_depth : float
        Minimum relative dip depth for a slice to count at all (default 0.10;
        protects flux points where the resonator dip vanishes).
    local_window_frac : float
        Width of the per-slice fit window as a fraction of the full detuning
        span, centred on the dip candidate (default 0.30; 0 fits the full span).
    dip_method : str
        Per-slice dip fit: ``"lorentzian"`` (default) or ``"circle"`` (needs the
        ``full_freq`` coordinate and meaningful phase data).
    **dip_knobs
        Knobs of the selected dip method (``baseline_order`` for lorentzian,
        ``delay`` for circle) — validated BEFORE the slice loop; unknown names
        raise ValueError here, never silently vanish.

    Returns
    -------
    dict
        ``{flux_bias, detuning, full_freq?, center_detuning, center_full_freq?,
        fwhm, dip_amplitude, dip_depth, success, refined, in_window, outlier,
        good, dip_method, fwhm_median, fwhm_mad, dip_amplitude_median,
        dip_amplitude_mad, amplitude_map, n_flux, n_success, n_refined, n_good,
        n_outlier}`` - every array in ASCENDING ``flux_bias`` / ``detuning``
        order whatever order the map was swept in; pair them with the returned
        axes, never with the caller's own.
    """
    check_dataset(dataset)
    # the sweep direction is provenance, never input (tools.sweep_order)
    dataset = ascending(dataset, "flux_bias", "detuning")
    # Fail loudly BEFORE any per-slice loop — a typo'd knob must never be
    # swallowed by the per-slice fallback.
    try:
        validate_dip_kwargs(dip_method, dip_knobs)
    except ValueError as err:
        raise ValueError(
            f"track_dips: {err} (own keyword-only tunables: n_sigma, "
            f"edge_margin_frac, min_dip_depth, local_window_frac, dip_method)"
        ) from None

    ds = with_iqdata(dataset)
    flux = ds.coords["flux_bias"].values.astype(float)
    detuning = ds.coords["detuning"].values.astype(float)
    has_full_freq = "full_freq" in ds.coords
    full_freq = (
        ds.coords["full_freq"].values.ravel().astype(float) if has_full_freq else None
    )
    if dip_method == "circle" and full_freq is None:
        raise ValueError(
            "track_dips: dip_method='circle' requires the 'full_freq' coordinate "
            "(absolute readout frequency in Hz)."
        )
    n_det = len(detuning)
    det_step = float(np.median(np.abs(np.diff(detuning)))) or 1.0

    # Complex map (flux, detuning): rows feed the per-slice fits; |rows| feed the
    # dip candidates and the 2-D plot.
    iq_map = ds["IQdata"].transpose("flux_bias", "detuning").values
    amplitude_map = np.abs(iq_map)

    n_flux = len(flux)
    center_detuning = np.full(n_flux, np.nan)
    center_full_freq = np.full(n_flux, np.nan)
    fwhm = np.full(n_flux, np.nan)
    dip_amplitude = np.full(n_flux, np.nan)
    dip_depth = np.full(n_flux, np.nan)
    success = np.zeros(n_flux, dtype=bool)
    refined = np.zeros(n_flux, dtype=bool)

    half_pts = max(int(round(local_window_frac * n_det / 2.0)), 12)

    for k in range(n_flux):
        j, depth, coarse_w_pts = _dip_candidate(amplitude_map[k])
        dip_depth[k] = depth
        if depth < min_dip_depth:
            continue  # no usable dip on this slice — leave NaN / False

        # Local fit window around the candidate (full slice when frac == 0).
        if local_window_frac > 0:
            lo, hi = max(j - half_pts, 0), min(j + half_pts + 1, n_det)
        else:
            lo, hi = 0, n_det
        det_local = detuning[lo:hi]
        ff_local = full_freq[lo:hi] if has_full_freq else None

        r = None
        try:
            r = fit_dip(det_local, iq_map[k, lo:hi], full_freq=ff_local,
                        method=dip_method, **dip_knobs)
        except Exception:
            r = None  # fit-domain failure only: kwargs were validated up front
        ok = (
            r is not None
            and bool(r["success"])
            and det_local.min() < r["detuning"] < det_local.max()
        )
        if ok:
            center_detuning[k] = r["detuning"]
            fwhm[k] = r["fwhm"]
            # Method-owned extra — only lorentzian reports a dip amplitude;
            # consumers of fit_dip may rely on detuning/fwhm/success alone.
            dip_amplitude[k] = r.get("amplitude", np.nan)
            refined[k] = True
            if has_full_freq and "full_freq" in r:
                center_full_freq[k] = r["full_freq"]
        else:
            # Coarse fallback: the smoothed-argmin candidate itself. Accurate to
            # the frequency grid — ample for the MHz-scale flux arch. fwhm from
            # the half-depth width; amplitude left NaN (a fitted dip amplitude
            # and a raw |IQ| depth are not comparable scales).
            center_detuning[k] = detuning[j]
            fwhm[k] = coarse_w_pts * det_step
            if has_full_freq:
                center_full_freq[k] = float(full_freq[j])
        success[k] = True

    # (1) Interior enforcement: the centre must lie inside the swept detuning
    # window, kept an `edge_margin_frac` away from either edge. A centre pinned
    # to (or hugging) an edge is unreliable regardless of how it was obtained.
    det_lo, det_hi = float(detuning.min()), float(detuning.max())
    margin = edge_margin_frac * (det_hi - det_lo)
    in_window = (
        np.isfinite(center_detuning)
        & (center_detuning > det_lo + margin)
        & (center_detuning < det_hi - margin)
    )
    valid = success & in_window

    # (2) Robust outlier rejection on the dip width and amplitude — judged among
    # the REFINED points only (coarse fallbacks carry incomparable scales and
    # already passed the depth gate), and LOW-SIDE only: a spurious noise-minimum
    # "dip" is too narrow / too weak versus the real population, while genuinely
    # broader/deeper points are physics (the dip broadens and deepens near the
    # arch bottom — exactly the points that pin the flux period). Slices whose
    # dip method reports no amplitude (circle) are NaN and exempt from the
    # amplitude gate automatically.
    judge = valid & refined
    outlier_fwhm, fwhm_med, fwhm_mad = mad_outliers(fwhm, judge, n_sigma, side="low")
    outlier_amp, amp_med, amp_mad = mad_outliers(
        np.abs(dip_amplitude), judge, n_sigma, side="low")
    outlier = judge & (outlier_fwhm | outlier_amp)
    good = valid & ~outlier

    results: Dict[str, Any] = {
        "flux_bias": flux,
        "detuning": detuning,
        "center_detuning": center_detuning,
        "fwhm": fwhm,
        "dip_amplitude": dip_amplitude,
        "dip_depth": dip_depth,
        "success": success,
        "refined": refined,
        "in_window": in_window,
        "outlier": outlier,
        "good": good,
        "dip_method": dip_method,
        "fwhm_median": fwhm_med,
        "fwhm_mad": fwhm_mad,
        "dip_amplitude_median": amp_med,
        "dip_amplitude_mad": amp_mad,
        "amplitude_map": amplitude_map,
        "n_flux": int(n_flux),
        "n_success": int(success.sum()),
        "n_refined": int(refined.sum()),
        "n_good": int(good.sum()),
        "n_outlier": int(outlier.sum()),
    }
    if has_full_freq:
        results["full_freq"] = full_freq
        results["center_full_freq"] = center_full_freq

    return results


def dips_plot_data(results: Dict[str, Any]) -> xr.Dataset:
    """Bundle the 2-D amplitude map and the extracted centre trace into one
    self-sufficient Dataset so the figure redraws with no refitting."""
    flux = np.asarray(results["flux_bias"], dtype=float)
    detuning = np.asarray(results["detuning"], dtype=float)
    amplitude = np.asarray(results["amplitude_map"], dtype=float)

    data_vars: Dict[str, Any] = {
        "amplitude": (("flux_bias", "detuning"), amplitude),
        "center_detuning": ("flux_bias", np.asarray(results["center_detuning"], float)),
        "fwhm": ("flux_bias", np.asarray(results["fwhm"], float)),
        "dip_amplitude": ("flux_bias", np.asarray(results["dip_amplitude"], float)),
        "dip_depth": ("flux_bias", np.asarray(results["dip_depth"], float)),
        "success": ("flux_bias", np.asarray(results["success"], bool)),
        "refined": ("flux_bias", np.asarray(results["refined"], bool)),
        "good": ("flux_bias", np.asarray(results["good"], bool)),
        "outlier": ("flux_bias", np.asarray(results["outlier"], bool)),
    }
    coords: Dict[str, Any] = {"flux_bias": flux, "detuning": detuning}
    attrs: Dict[str, Any] = {
        "dip_method": str(results["dip_method"]),
        "n_flux": int(results["n_flux"]),
        "n_success": int(results["n_success"]),
        "n_refined": int(results["n_refined"]),
        "n_good": int(results["n_good"]),
        "n_outlier": int(results["n_outlier"]),
    }

    if "full_freq" in results:
        coords["full_freq"] = ("detuning", np.asarray(results["full_freq"], float))
        data_vars["center_full_freq"] = (
            "flux_bias", np.asarray(results["center_full_freq"], float)
        )
        attrs["has_full_freq"] = 1
    else:
        attrs["has_full_freq"] = 0

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)
