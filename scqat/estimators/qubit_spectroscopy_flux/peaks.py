"""Stage 1 of the qubit-vs-flux analysis: per-flux peak tracking.

Reduce a 2-D qubit-spectroscopy-vs-flux map to a peak **point-cloud**
``(flux, frequency, fwhm, amplitude)`` by fitting **flux-by-flux** with the
family-shared per-trace reduction :func:`scqat.tools.peak_fit.fit_peaks`,
pooled and gated by the generic map tracker
:func:`scqat.tools.peak_map.track_peaks`. All peaks found at each flux are
kept (two or more transitions can coexist — e.g. the 0-1 line and the
two-photon 0-2/2 line); assigning points to individual transition branches
belongs to the downstream flux-dependence fit.

These are plain functions: inside scqat they are stage 1 of both
:class:`.estimator.QubitSpectroscopyFluxEstimator` (point-cloud only) and
:class:`~scqat.estimators.qubit_flux_arch.QubitFluxArchEstimator` (cloud +
transmon arch fit), and control repos may call :func:`track_flux_peaks`
directly when they need the cloud without an estimator's artifacts.

Dataset contract (the ``qubit`` dimension already removed):

Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
    - detuning  : 1-D float array – drive-frequency detuning from the LO (Hz).
    - full_freq : (detuning,) absolute drive frequency (Hz). Optional; when
                  present peak centres are also reported in absolute frequency.
Data variables:
    - IQdata : (flux_bias, detuning) – complex demodulated signal (I + iQ), **or**
    - I, Q   : (flux_bias, detuning) – the two quadratures, combined into IQdata, **or**
    - a real signal named by the ``signal_var`` kwarg (e.g. ``"state"``).
"""

from typing import Any, Dict, Optional

import numpy as np
import xarray as xr

from scqat.core.base_estimator import with_iqdata
from scqat.tools.peak_fit import validate_peak_kwargs
from scqat.tools.peak_map import track_peaks


def check_flux_dataset(dataset: xr.Dataset) -> None:
    """Validate the 2-D map contract (see module docstring); raise ValueError."""
    for coord in ("flux_bias", "detuning"):
        if coord not in dataset.coords:
            raise ValueError(
                f"qubit_spectroscopy_flux requires a '{coord}' coordinate."
            )


def track_flux_peaks(
    dataset: xr.Dataset,
    *,
    n_sigma: float = 3.0,
    signal_var: Optional[str] = None,
    **peak_knobs,
) -> Dict[str, Any]:
    """Fit and collect every qubit peak in every ``flux_bias`` slice.

    Each flux row is handed to :func:`scqat.tools.peak_fit.fit_peaks` and every
    detected peak is pooled into a flat point-cloud with ``in_window`` /
    ``outlier`` / ``good`` acceptance masks (see
    :func:`scqat.tools.peak_map.track_peaks`). By default each flux slice is
    capped to its 4 most prominent peaks (``max_peaks=4``); pass
    ``max_peaks=None`` to keep every peak above the prominence threshold.

    Keyword arguments (all keyword-only)
    ------------------------------------
    n_sigma : float
        Robust-sigma threshold for the width / amplitude outlier test
        (default 3.0).
    signal_var : str, optional
        Real data variable to fit instead of ``|IQdata - ref|``
        (e.g. ``"state"``).
    **peak_knobs
        Knobs of :func:`~scqat.tools.peak_fit.fit_peaks` (``prominence``,
        ``max_peaks``, ...) — validated BEFORE the slice loop; unknown names
        raise ValueError here, never silently vanish.

    Returns
    -------
    dict
        ``{flux_bias, detuning, full_freq?, peak_flux, peak_flux_index,
        peak_detuning, peak_full_freq?, peak_fwhm, peak_amplitude,
        in_window, outlier, good, fwhm_median, fwhm_mad,
        peak_amplitude_median, peak_amplitude_mad, amplitude_map,
        n_flux, n_peaks, n_in_window, n_good, n_outlier}``
    """
    check_flux_dataset(dataset)
    # Fail loudly BEFORE any per-slice fit — a typo'd knob must never be
    # swallowed by the per-slice fallback.
    try:
        validate_peak_kwargs(peak_knobs)
    except ValueError as err:
        raise ValueError(
            f"track_flux_peaks: {err} (own keyword-only tunables: n_sigma, "
            f"signal_var)"
        ) from None
    # Default cap: keep each flux slice's 4 most-prominent peaks (headroom over
    # the typical 1-2 real transitions). setdefault preserves an explicit
    # max_peaks=None opt-out.
    peak_knobs.setdefault("max_peaks", 4)

    if signal_var is not None:
        signal_map = dataset[signal_var].transpose("flux_bias", "detuning").values
        ds = dataset
    else:
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "qubit_spectroscopy_flux needs an 'IQdata' variable, both 'I' "
                "and 'Q', or a real 'signal_var'."
            )
        ds = with_iqdata(dataset)
        signal_map = ds["IQdata"].transpose("flux_bias", "detuning").values

    flux = ds.coords["flux_bias"].values.astype(float)
    detuning = ds.coords["detuning"].values.astype(float)
    full_freq = (
        ds.coords["full_freq"].values.ravel().astype(float)
        if "full_freq" in ds.coords else None
    )

    cloud = track_peaks(
        flux, detuning, signal_map,
        full_freq=full_freq, n_sigma=n_sigma, **peak_knobs,
    )

    # Relabel the generic tracker keys into the flux vocabulary.
    results: Dict[str, Any] = {
        "flux_bias": cloud["x"],
        "detuning": cloud["y"],
        "peak_flux": cloud["peak_x"],
        "peak_flux_index": cloud["peak_x_index"],
        "peak_detuning": cloud["peak_y"],
        "peak_fwhm": cloud["peak_fwhm"],
        "peak_amplitude": cloud["peak_amplitude"],
        "in_window": cloud["in_window"],
        "outlier": cloud["outlier"],
        "good": cloud["good"],
        "fwhm_median": cloud["fwhm_median"],
        "fwhm_mad": cloud["fwhm_mad"],
        "peak_amplitude_median": cloud["peak_amplitude_median"],
        "peak_amplitude_mad": cloud["peak_amplitude_mad"],
        "amplitude_map": np.abs(signal_map),
        "n_flux": cloud["n_x"],
        "n_peaks": cloud["n_peaks"],
        "n_in_window": cloud["n_in_window"],
        "n_good": cloud["n_good"],
        "n_outlier": cloud["n_outlier"],
    }
    if full_freq is not None:
        results["full_freq"] = cloud["full_freq"]
        results["peak_full_freq"] = cloud["peak_full_freq"]

    return results


def flux_cloud_plotdata(results: Dict[str, Any]) -> xr.Dataset:
    """Bundle the 2-D signal map and the peak point-cloud (with good/outlier
    masks) into one self-sufficient Dataset so the figure redraws with no
    refitting."""
    flux = np.asarray(results["flux_bias"], dtype=float)
    detuning = np.asarray(results["detuning"], dtype=float)
    amplitude = np.asarray(results["amplitude_map"], dtype=float)
    n_peaks = int(results["n_peaks"])

    data_vars: Dict[str, Any] = {
        "amplitude": (("flux_bias", "detuning"), amplitude),
        "peak_flux": ("peak", np.asarray(results["peak_flux"], float)),
        "peak_detuning": ("peak", np.asarray(results["peak_detuning"], float)),
        "peak_fwhm": ("peak", np.asarray(results["peak_fwhm"], float)),
        "peak_amplitude": ("peak", np.asarray(results["peak_amplitude"], float)),
        "good": ("peak", np.asarray(results["good"], bool)),
        "outlier": ("peak", np.asarray(results["outlier"], bool)),
    }
    coords: Dict[str, Any] = {
        "flux_bias": flux, "detuning": detuning, "peak": np.arange(n_peaks),
    }
    attrs: Dict[str, Any] = {
        "n_flux": int(results["n_flux"]),
        "n_peaks": n_peaks,
        "n_good": int(results["n_good"]),
        "n_outlier": int(results["n_outlier"]),
    }

    if "full_freq" in results:
        coords["full_freq"] = ("detuning", np.asarray(results["full_freq"], float))
        data_vars["peak_full_freq"] = ("peak", np.asarray(results["peak_full_freq"], float))
        attrs["has_full_freq"] = 1
    else:
        attrs["has_full_freq"] = 0

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)
