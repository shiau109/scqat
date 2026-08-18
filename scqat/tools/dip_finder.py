"""Wideband resonator dip finder and ranking tool.

Locates transmission dips across wide frequency sweeps, estimates baselines,
ranks candidate dips by prominence and depth, and performs localized Lorentzian fits.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks, peak_widths

from .fit_lorentzian import FitLorentzian


DIP_FINDER_KNOBS = frozenset({
    "num_dips",
    "min_prominence_db",
    "min_snr",
    "baseline_window_points",
    "fit_window_points",
})


def validate_dip_finder_kwargs(knobs: Dict[str, Any]) -> None:
    """Raise ValueError if unknown knobs are passed."""
    unknown = set(knobs) - DIP_FINDER_KNOBS
    if unknown:
        raise ValueError(
            f"unknown dip finder knobs: {sorted(unknown)}; "
            f"allowed: {sorted(DIP_FINDER_KNOBS)}"
        )


def find_resonator_dips(
    freq: np.ndarray,
    iq_data: np.ndarray,
    num_dips: Optional[int] = None,
    min_prominence_db: float = 0.5,
    min_snr: float = 2.5,
    baseline_window_points: Optional[int] = None,
    fit_window_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Find and fit candidate resonator dips across a wideband frequency sweep.

    Parameters
    ----------
    freq : np.ndarray
        1-D array of frequencies in Hz.
    iq_data : np.ndarray
        1-D complex transmission array (I + 1j*Q) or real magnitude array.
    num_dips : int, optional
        Maximum number of candidate dips to select (e.g. from components.toml).
        If None or <= 0, returns all dips meeting the thresholds.
    min_prominence_db : float, default 0.5
        Minimum dip prominence in dB below local background.
    min_snr : float, default 2.5
        Minimum peak height in robust noise standard deviations (MAD).
    baseline_window_points : int, optional
        Window size for background baseline estimation. Defaults to max(15, n_points // 30).
    fit_window_points : int, optional
        Window size for localized Lorentzian fitting around each candidate dip.

    Returns
    -------
    dict
        Dictionary containing 'dips', 'baseline_db', 'mag_db', 'freq_hz', and 'num_dips_found'.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    n_points = freq.size
    if n_points < 5:
        return {
            "dips": [],
            "baseline_db": np.zeros_like(freq),
            "mag_db": np.zeros_like(freq),
            "freq_hz": freq,
            "num_dips_found": 0,
        }

    # 1. Compute magnitude in dB
    mag = np.abs(iq_data).ravel()
    mag_safe = np.maximum(mag, 1e-15)
    mag_db = 20.0 * np.log10(mag_safe)

    # 2. Compute background baseline (moving median filter to ignore sharp dips)
    if baseline_window_points is None:
        window_size = max(15, min(n_points // 25, 201))
        if window_size % 2 == 0:
            window_size += 1
    else:
        window_size = max(5, int(baseline_window_points))
        if window_size % 2 == 0:
            window_size += 1

    baseline_db = median_filter(mag_db, size=window_size, mode="reflect")
    # Smooth baseline slightly with uniform filter to avoid staircases
    baseline_db = uniform_filter1d(baseline_db, size=max(3, window_size // 3), mode="reflect")

    # Inverted signal: dips become positive peaks
    inverted_signal = baseline_db - mag_db

    # 3. Robust noise estimation using Median Absolute Deviation (MAD)
    med = float(np.median(inverted_signal))
    mad = float(np.median(np.abs(inverted_signal - med)))
    noise_std = max(1.4826 * mad, 1e-6)

    min_height = max(min_snr * noise_std, min_prominence_db * 0.5)

    # 4. Detect peaks in inverted signal
    peak_indices, properties = find_peaks(
        inverted_signal,
        prominence=min_prominence_db,
        height=min_height,
        distance=max(3, n_points // 500),
    )

    if len(peak_indices) == 0:
        return {
            "dips": [],
            "baseline_db": baseline_db,
            "mag_db": mag_db,
            "freq_hz": freq,
            "num_dips_found": 0,
        }

    prominences = properties.get("prominences", np.zeros(len(peak_indices)))
    heights = properties.get("peak_heights", np.zeros(len(peak_indices)))

    # Estimate widths for local fit window
    try:
        widths_res = peak_widths(inverted_signal, peak_indices, rel_height=0.5)
        half_widths = np.maximum(widths_res[0] / 2.0, 2.0)
    except Exception:
        half_widths = np.full(len(peak_indices), 5.0)

    # 5. Build candidate list and score
    candidates = []
    for idx, p_idx in enumerate(peak_indices):
        raw_f = freq[p_idx]
        prom = float(prominences[idx])
        depth = float(heights[idx])
        hw_pts = float(half_widths[idx])

        # Score balances prominence and peak depth
        score = prom * depth

        candidates.append({
            "peak_index": p_idx,
            "raw_freq": raw_f,
            "prominence": prom,
            "depth": depth,
            "half_width_pts": hw_pts,
            "score": score,
        })

    # Sort descending by score
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Cap to num_dips if requested
    if num_dips is not None and num_dips > 0:
        candidates = candidates[:num_dips]

    # 6. Fit local Lorentzian around each selected dip
    fit_half_span = (
        fit_window_points // 2
        if fit_window_points is not None
        else max(10, min(n_points // 50, 60))
    )

    dips = []
    for rank_idx, cand in enumerate(candidates, start=1):
        p_idx = cand["peak_index"]
        hw = max(int(np.ceil(cand["half_width_pts"] * 3)), 8)
        win = min(max(hw, fit_half_span), n_points // 4)

        start_idx = max(0, p_idx - win)
        stop_idx = min(n_points, p_idx + win + 1)

        sub_f = freq[start_idx:stop_idx]
        sub_mag = mag[start_idx:stop_idx]
        sub_mag_db = mag_db[start_idx:stop_idx]

        f_fit = cand["raw_freq"]
        kappa_fit = abs(float(sub_f[-1] - sub_f[0])) / 10.0
        success = False
        fit_x = sub_f
        fit_y = sub_mag

        try:
            # Fit Lorentzian dip on magnitude
            fitter = FitLorentzian(data=None, inverted=True, x=sub_f)
            fitter.x = sub_f
            fitter.y = sub_mag
            fitter.guess()
            result = fitter.fit()
            if result is not None and result.success:
                f_fit = float(result.best_values.get("x0", cand["raw_freq"]))
                gamma = abs(float(result.best_values.get("gamma", kappa_fit / 2.0)))
                kappa_fit = 2.0 * gamma
                success = True
                fit_x = np.linspace(sub_f[0], sub_f[-1], 200)
                fit_y = fitter.model_function(
                    fit_x,
                    f_fit,
                    result.best_values.get("amplitude", 0.0),
                    gamma,
                    result.best_values.get("offset", float(np.median(sub_mag))),
                )
        except Exception:
            success = False

        ql = float(f_fit / kappa_fit) if kappa_fit > 0 else 0.0

        dips.append({
            "rank": rank_idx,
            "frequency_hz": f_fit,
            "fwhm_hz": kappa_fit,
            "ql": ql,
            "depth_db": cand["depth"],
            "prominence_db": cand["prominence"],
            "raw_min_freq_hz": cand["raw_freq"],
            "success": success,
        })

    # Sort final dips by ascending frequency
    dips.sort(key=lambda d: d["frequency_hz"])

    # Reassign ranks 1..N based on prominence/depth
    sorted_by_prom = sorted(dips, key=lambda d: d["prominence_db"], reverse=True)
    for r_idx, d in enumerate(sorted_by_prom, start=1):
        d["rank"] = r_idx

    return {
        "dips": dips,
        "baseline_db": baseline_db,
        "mag_db": mag_db,
        "freq_hz": freq,
        "num_dips_found": len(dips),
    }
