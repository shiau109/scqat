"""
Resonator Spectroscopy Power Estimator
======================================
Reduce a 2-D resonator-spectroscopy-vs-readout-power map to a 1-D
``center_frequency(power)`` curve by fitting the resonator dip **power-by-power**,
then pick the optimal readout power from where that centre stops shifting.

For every readout-power slice the estimator delegates to
:class:`~scqat.estimators.resonator_spectroscopy.ResonatorSpectroscopyEstimator`
— the same single inverted-Lorentzian fit of the readout power ``|IQ|^2`` used for
1-D resonator spectroscopy — and records the dip centre and linewidth. Stacking
those per-slice centres yields the resonator centre frequency as a function of
readout power (and its FWHM), i.e. the 2-D ``(power, detuning)`` map is collapsed
onto a 1-D trace.

On top of that reduction (which mirrors
:class:`~scqat.estimators.resonator_spectroscopy_vs_flux.ResonatorSpectroscopyVsFluxEstimator`
one-for-one, with ``power`` in place of ``flux_bias``), a second stage picks the
**optimal readout power**: the low-power dispersive regime is where
``center_detuning(power)`` stops shifting, so it is found where the smoothed
``d(center)/d(power)`` first crosses below a (negative) threshold — the same
derivative-crossing heuristic the official ``02b`` node uses, but run on the
robust fitted centre trace instead of a raw ``idxmin`` proxy.

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
Data variables:
    - IQdata : (power, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (power, detuning) – the two quadratures, combined into IQdata.

Rows may carry a power-dependent overall scale (the measured |IQ| grows with the
readout drive amplitude) — per-slice dip fits are scale-invariant, and the
cross-power amplitude outlier test normalizes by each row's baseline scale, so
both raw-instrument and pre-normalized maps are handled.
"""

from typing import Any, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.estimators.resonator_spectroscopy import ResonatorSpectroscopyEstimator
from scqat.estimators.resonator_spectroscopy_vs_flux.estimator import _mad_outliers
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
) -> float:
    """Optimal readout power (dBm), or NaN when no crossing is found.

    Ports the official 02b heuristic: differentiate the resonator-centre trace
    with respect to power, smooth it, scale down the noisy leading edge, and take
    the first power whose smoothed ``d(center)/d(power)`` drops below the
    (negative) ``threshold_hz_per_dbm``; then step ``buffer_dbm`` below it. Runs on
    the robust fitted ``center_detuning(power)`` rather than a raw ``idxmin``.
    """
    power = np.asarray(power, dtype=float)
    center = np.asarray(center_detuning, dtype=float)
    finite = np.isfinite(center)
    if power.size < 3 or finite.sum() < 2:
        return float("nan")

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
        return float("nan")
    idx = int(np.argmax(below))  # first power below the threshold
    return float(power[idx]) - float(buffer_dbm)


class ResonatorSpectroscopyPowerEstimator(BaseEstimator):
    """
    Fit the resonator dip at every readout power, report the resonator centre
    frequency as a function of power, and pick the optimal readout power.

    The result dict reports, per power point, the dip ``center_detuning`` (and
    absolute ``center_full_freq`` when available), the ``fwhm`` and a per-point
    ``success`` flag, alongside the 2-D ``amplitude`` map kept for plotting; plus
    the scalar deliverables ``optimal_power`` / ``frequency_shift`` /
    ``resonator_frequency`` and an overall ``optimal_success`` flag.
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

        Each slice is handed to ``ResonatorSpectroscopyEstimator.extract_parameters``
        (single inverted Lorentzian on the readout power ``|IQ|^2``); ``kwargs``
        such as ``baseline_order`` / ``baseline_quantile`` are forwarded to it.

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

        Returns
        -------
        dict
            ``{power, detuning, full_freq?, center_detuning, center_full_freq?,
            fwhm, dip_amplitude, success, in_window, outlier, good,
            fwhm_median, fwhm_mad, dip_amplitude_median, dip_amplitude_mad,
            amplitude_map, n_power, n_success, n_good, n_outlier,
            optimal_power, frequency_shift, resonator_frequency, optimal_success}``
        """
        n_sigma = float(kwargs.pop("n_sigma", 3.0))
        threshold = float(kwargs.pop("derivative_crossing_threshold_in_hz_per_dbm", -50_000.0))
        smoothing_window = int(kwargs.pop("derivative_smoothing_window_num_points", 10))
        init_filter_window = int(kwargs.pop("moving_average_filter_window_num_points", 10))
        buffer_dbm = float(kwargs.pop("buffer_from_crossing_threshold_in_dbm", 1.0))

        ds = ResonatorSpectroscopyEstimator._with_iqdata(dataset)
        power = ds.coords["power"].values.astype(float)
        detuning = ds.coords["detuning"].values.astype(float)
        has_full_freq = "full_freq" in ds.coords

        single = ResonatorSpectroscopyEstimator()
        n_power = len(power)
        center_detuning = np.full(n_power, np.nan)
        center_full_freq = np.full(n_power, np.nan)
        fwhm = np.full(n_power, np.nan)
        dip_amplitude = np.full(n_power, np.nan)
        success = np.zeros(n_power, dtype=bool)

        for k in range(n_power):
            sl = ds.isel(power=k)
            try:
                r = single.extract_parameters(sl, **kwargs)
            except Exception:
                # Leave NaN / False for this power point and carry on.
                continue
            center_detuning[k] = r["detuning"]
            fwhm[k] = r["fwhm"]
            dip_amplitude[k] = r["amplitude"]
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
        amplitude_map = np.abs(ds["IQdata"].transpose("power", "detuning").values)

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
        outlier_fwhm, fwhm_med, fwhm_mad = _mad_outliers(fwhm, valid, n_sigma)
        outlier_amp, amp_med, amp_mad = _mad_outliers(rel_amp, valid, n_sigma)
        outlier = valid & (outlier_fwhm | outlier_amp)
        good = valid & ~outlier

        # Optimal readout power from where the centre trace stops shifting, using
        # only the good (in-window, non-outlier) centres.
        center_for_pick = np.where(good, center_detuning, np.nan)
        optimal_power = _pick_optimal_power(
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
            "frequency_shift": frequency_shift,
            "resonator_frequency": resonator_frequency,
            "optimal_success": optimal_success,
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
            "n_power": int(results["n_power"]),
            "n_success": int(results["n_success"]),
            "n_good": int(results["n_good"]),
            "n_outlier": int(results["n_outlier"]),
            "optimal_power": float(results["optimal_power"]),
            "frequency_shift": float(results["frequency_shift"]),
            "optimal_success": int(bool(results["optimal_success"])),
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
        (power, frequency) with the fitted resonator-centre trace and the
        optimal-power marker overlaid, drawn from plot_data."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"resonator_spectroscopy_power": plot_power_map(plot_data)}
