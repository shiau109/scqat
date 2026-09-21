"""
AC-Stark Shift Estimator
========================
The qubit line against the amplitude of a resonator Stark tone: qubit
spectroscopy repeated at every amplitude of a tone parked on the readout
resonator, read out AFTER the tone's photons have left (SCQO's
``qubit_resonator_stark``).

Reading
-------
Each amplitude row is one qubit-spectroscopy trace, reduced by the
family-shared per-trace fit :func:`scqat.tools.peak_fit.fit_peaks` with
``max_peaks=1``. The second stage is this estimator's own physics: the
dispersive AC-Stark shift is linear in the resonator photon number, and the
photon number is linear in the tone POWER, i.e. in ``amp_prefactor ** 2``::

    f(a)    = f0 + s * a**2          (line centre)
    fwhm(a) = w0 + b * a**2          (measurement-induced dephasing broadening)

``s`` is therefore the shift at ``amp_prefactor = 1`` — the reference
amplitude the prefactor multiplies (SCQO: the calibrated ``readout_amp``).

Photon number
-------------
With ``chi_hz = (f_dress0 - f_dress1) / 2`` — the resonator pull between the
qubit's two states, SCQO's catalog convention — the qubit moves by
``-2 * chi_hz`` per photon, so ``n(a) = (f(a) - f0) / (-2 * chi_hz)``. A
transmon below its resonator has ``chi_hz > 0`` and shifts DOWN. A negative
photon number means the supplied chi disagrees in sign with the measured shift;
it is reported, never clipped.

Row acceptance — why there is no global width gate
--------------------------------------------------
A row enters the line fit when its peak centre sits at least one sweep step
inside the window, its fit converged (finite ``detuning_err`` — ``fit_peaks``
falls back to the initial guess with NaN errors when lmfit raises), and its
FWHM lies in ``[one step, window width)``. There is deliberately NO population
(median/MAD) gate on width or amplitude, unlike the vs-flux map tracker
``track_peaks``: the linewidth GROWS with the tone
power by physics, so a population gate always flags the high-power rows —
exactly the ones that carry the shift. Outliers are removed on the RESIDUALS
from a robust (Theil–Sen) line instead, which follow the trend; the reported
numbers are then an ordinary least-squares fit of the rows that remain.

The measured per-row curve is always reported next to the model values: at
high photon number the shift saturates, the rows bend away from the line, and
the clip marks them (``accepted`` but not ``in_fit``) — so the slope is the
low-power coefficient and the figures show where the line stopped holding.

Expected ``xarray.Dataset`` contract
------------------------------------
The ``target``/``qubit`` dimension is already removed (``repetition_data``).

Coordinates:
    - amp_prefactor : 1-D float — the tone amplitude as a factor of ``amp_ref``.
    - detuning      : 1-D float — qubit-drive detuning (Hz).
    - full_freq     : (detuning,) absolute drive frequency (Hz). Optional.
    - <twin_coord>  : (amp_prefactor,) the same points in absolute amplitude.
                      Optional; drawn as a secondary axis only.
Data variables:
    - IQdata (complex) or I and Q : (amp_prefactor, detuning).
    - ref_pos_g_i/_g_q/_e_i/_e_q : optional stored blob centres; the ground one
      is the radial reference for every row (the readout is the same in every
      row, so one reference serves them all).
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_ground, with_iqdata
from scqat.core.figures import render_figures
from scqat.estimators._twin_axis import twin_values
from scqat.estimators.ac_stark_shift.visualization import (
    plot_broadening,
    plot_shift,
    plot_stark_map,
)
from scqat.tools.iq_reduce import ground_ref, radial
from scqat.tools.peak_fit import fit_peaks, validate_peak_kwargs

AMP = "amp_prefactor"
DET = "detuning"

#: fewest rows the straight-line fit is reported from — two would always fit.
MIN_FIT_ROWS = 3

#: the estimator's own knobs, beside the fit_peaks ones (max_peaks excluded).
OWN_KNOBS = ("amp_ref", "chi_hz", "resid_sigma", "twin_coord", "twin_label")

#: the fit curves are drawn on this many points across the amplitude window.
_DENSE_POINTS = 101


def _line_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Ordinary least squares ``y = intercept + slope * x`` with standard errors.

    Closed form, not ``np.polyfit(cov=True)``: the covariance there needs more
    points than the three this estimator accepts. With exactly two degrees of
    freedom used up the errors are NaN, never a division by zero.
    """
    nan = float("nan")
    out = {"slope": nan, "intercept": nan, "slope_err": nan,
           "intercept_err": nan, "rms_residual": nan}
    n = x.size
    if n < 2:
        return out
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.sum((x - x_mean) ** 2))
    if sxx <= 0.0:
        return out
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / sxx)
    intercept = y_mean - slope * x_mean
    resid = y - (intercept + slope * x)
    out.update(slope=slope, intercept=intercept,
               rms_residual=float(np.sqrt(np.mean(resid ** 2))))
    if n > 2:
        s2 = float(np.sum(resid ** 2)) / (n - 2)
        out["slope_err"] = float(np.sqrt(s2 / sxx))
        out["intercept_err"] = float(np.sqrt(s2 * (1.0 / n + x_mean ** 2 / sxx)))
    return out


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple:
    """Median-of-pairwise-slopes line — the robust reference the residual clip
    measures against. A least-squares line would be dragged by the very row it
    is meant to judge, offsetting every other row's residual with it."""
    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    usable = dx != 0
    if not usable.any():
        return float("nan"), float("nan")
    slope = float(np.median((y[j] - y[i])[usable] / dx[usable]))
    return slope, float(np.median(y - slope * x))


class AcStarkShiftEstimator(BaseEstimator):
    """Fit the qubit line at every Stark-tone amplitude and the shift and
    broadening against ``amp_prefactor ** 2``."""

    estimator_name = "ac_stark_shift"

    #: class-level defaults for the companion amplitude axis; a caller passes
    #: ``twin_coord`` / ``twin_label`` per call (see ``_twin_axis``).
    twin_coord: Optional[str] = None
    twin_label: Optional[str] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        for coord in (AMP, DET):
            if coord not in dataset.coords:
                raise ValueError(f"AcStarkShiftEstimator requires a '{coord}' coordinate.")
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "AcStarkShiftEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
            )

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Locate the line in every amplitude row, then fit shift and width
        against ``amp_prefactor ** 2``.

        Keyword arguments — flat and fully owned; unknown names raise BEFORE any
        row is fitted
        ----------------------------------------------------------------------
        amp_ref : float, optional
            The absolute amplitude at ``amp_prefactor = 1``; turns the per-prefactor
            slopes into per-absolute-amplitude ones (``stark_hz_per_amp2``,
            ``photons_per_amp2``). Omitted -> those two are NaN.
        chi_hz : float, optional
            Dispersive shift, ``(f_dress0 - f_dress1) / 2`` in Hz. Omitted, None or
            zero -> no photon number.
        resid_sigma : float, optional
            Residual-clip threshold, in robust sigmas of the rows' scatter about
            a Theil–Sen line (default 4). The sigma is floored at the rows' median
            fit error so a near-perfect line cannot clip a row for a sub-error
            deviation.
        twin_coord, twin_label : optional
            A coordinate over ``amp_prefactor`` with the same points in absolute
            amplitude, drawn as a secondary axis (never used as a number here).
        ref, prominence, min_snr, merge_factor, min_fwhm_factor, fit_window_factor
            Knobs of :func:`scqat.tools.peak_fit.fit_peaks`. ``max_peaks`` is
            pinned to 1 and refused.

        Returns
        -------
        dict
            Scalars: ``success, stark_slope_hz, stark_slope_err_hz,
            intercept_detuning_hz, intercept_detuning_err_hz, intercept_freq_hz,
            broadening_slope_hz, fwhm_intercept_hz, stark_hz_per_amp2,
            photons_per_prefactor2, photons_per_amp2, chi_hz, amp_ref,
            has_photon, rms_residual_hz, n_rows, n_rows_accepted, n_rows_fit,
            ref_source``. Per-row arrays (over ``amp_prefactor``):
            ``peak_detuning, peak_detuning_err, peak_fwhm, peak_fwhm_err,
            peak_freq, accepted, in_fit, photon_number``. Plus ``detuning``,
            ``reduced_map`` (``|IQ - ref|``, never negated) and, when drawable,
            ``twin_values`` / ``twin_label``.
        """
        twin_coord = kwargs.pop("twin_coord", self.twin_coord)
        twin_label = kwargs.pop("twin_label", self.twin_label)
        amp_ref = kwargs.pop("amp_ref", None)
        chi_hz = kwargs.pop("chi_hz", None)
        resid_sigma = float(kwargs.pop("resid_sigma", 4.0))
        if "max_peaks" in kwargs:
            raise ValueError(
                "AcStarkShiftEstimator pins max_peaks=1 (one qubit line per row); "
                "do not pass it."
            )
        try:
            validate_peak_kwargs(kwargs)
        except ValueError as err:
            raise ValueError(
                f"AcStarkShiftEstimator: {err} (own keyword-only tunables: "
                f"{', '.join(OWN_KNOBS)}; max_peaks is pinned to 1)"
            ) from None

        ds = with_iqdata(dataset)
        amp = np.asarray(ds.coords[AMP].values, dtype=float)
        detuning = np.asarray(ds.coords[DET].values, dtype=float)
        iq_map = np.asarray(ds["IQdata"].transpose(AMP, DET).values)
        full_freq = (
            np.asarray(ds.coords["full_freq"].values, dtype=float).ravel()
            if "full_freq" in ds.coords else None
        )
        # An ascending frequency axis, always: fit_peaks builds its width bound as
        # detuning[-1] - detuning[0], which a descending axis inverts silently.
        order = np.argsort(detuning)
        detuning = detuning[order]
        iq_map = iq_map[:, order]
        if full_freq is not None:
            full_freq = full_freq[order]

        # the radial reference: supplied -> the stored ground blob -> per-row median
        ref = kwargs.pop("ref", None)
        ref_source = "supplied" if ref is not None else "median"
        if ref is None:
            stored = stored_ground(ds)
            if stored is not None:
                ref, ref_source = stored, "stored"

        n_amp, n_det = iq_map.shape
        step = float(np.median(np.diff(detuning))) if n_det > 1 else float("nan")
        low, high = float(detuning[0]), float(detuning[-1])
        window = high - low

        nan_row = np.full(n_amp, np.nan)
        peak_det, peak_err = nan_row.copy(), nan_row.copy()
        peak_fwhm, peak_fwhm_err, peak_freq = nan_row.copy(), nan_row.copy(), nan_row.copy()
        accepted = np.zeros(n_amp, dtype=bool)
        reduced_map = np.full((n_amp, n_det), np.nan)

        for row in range(n_amp):
            trace = iq_map[row]
            row_ref = complex(ref) if ref is not None else ground_ref(trace.real, trace.imag)
            # the raw view never depends on the fit succeeding
            reduced_map[row] = radial(trace.real, trace.imag, ref=row_ref)
            try:
                fit = fit_peaks(detuning, trace, full_freq=full_freq, ref=row_ref,
                                max_peaks=1, **kwargs)
            except Exception:  # noqa: BLE001 - fit-domain failure only; knobs validated above
                continue
            if not fit["peaks"]:
                continue
            peak = fit["peaks"][0]
            peak_det[row] = peak["detuning"]
            peak_err[row] = peak["detuning_err"]
            peak_fwhm[row] = peak["fwhm"]
            peak_fwhm_err[row] = peak["fwhm_err"]
            peak_freq[row] = peak.get("full_freq", np.nan)
            accepted[row] = bool(
                np.isfinite(peak_det[row]) and np.isfinite(peak_err[row])
                and low + step <= peak_det[row] <= high - step
                and step <= peak_fwhm[row] < window
            )

        x = amp ** 2
        in_fit = accepted.copy()
        # one residual pass against the TREND (see the module docstring), judged
        # from a robust line; only with enough rows that dropping some still
        # leaves a fit worth reporting
        if in_fit.sum() >= MIN_FIT_ROWS + 2:
            robust_slope, robust_intercept = _theil_sen(x[in_fit], peak_det[in_fit])
            if np.isfinite(robust_slope):
                resid = peak_det - (robust_intercept + robust_slope * x)
                centre = float(np.median(resid[in_fit]))
                mad_sigma = 1.4826 * float(np.median(np.abs(resid[in_fit] - centre)))
                sigma = max(mad_sigma, float(np.nanmedian(peak_err[in_fit])))
                if sigma > 0:
                    drop = in_fit & (np.abs(resid - centre) > resid_sigma * sigma)
                    if drop.any() and (in_fit & ~drop).sum() >= MIN_FIT_ROWS:
                        in_fit &= ~drop
        line = _line_fit(x[in_fit], peak_det[in_fit])
        width = _line_fit(x[in_fit], peak_fwhm[in_fit])

        n_fit = int(in_fit.sum())
        success = bool(n_fit >= MIN_FIT_ROWS and np.isfinite(line["slope"]))

        chi = float(chi_hz) if chi_hz is not None else float("nan")
        has_photon = bool(np.isfinite(chi) and chi != 0.0)
        per_photon_hz = -2.0 * chi if has_photon else float("nan")
        photon = (peak_det - line["intercept"]) / per_photon_hz if has_photon else nan_row.copy()
        photons_per_prefactor2 = line["slope"] / per_photon_hz if has_photon else float("nan")

        a_ref = float(amp_ref) if amp_ref is not None else float("nan")
        a_ref = a_ref if np.isfinite(a_ref) and a_ref > 0 else float("nan")

        intercept_freq = float("nan")
        if full_freq is not None and np.isfinite(line["intercept"]):
            # full_freq is the detuning axis shifted by the drive frequency
            intercept_freq = line["intercept"] + float(np.median(full_freq - detuning))

        results: Dict[str, Any] = {
            "success": success,
            "stark_slope_hz": line["slope"],
            "stark_slope_err_hz": line["slope_err"],
            "intercept_detuning_hz": line["intercept"],
            "intercept_detuning_err_hz": line["intercept_err"],
            "intercept_freq_hz": intercept_freq,
            "broadening_slope_hz": width["slope"],
            "fwhm_intercept_hz": width["intercept"],
            "stark_hz_per_amp2": line["slope"] / a_ref ** 2,
            "photons_per_prefactor2": photons_per_prefactor2,
            "photons_per_amp2": photons_per_prefactor2 / a_ref ** 2,
            "chi_hz": chi if has_photon else float("nan"),
            "amp_ref": a_ref,
            "has_photon": has_photon,
            "rms_residual_hz": line["rms_residual"],
            "n_rows": n_amp,
            "n_rows_accepted": int(accepted.sum()),
            "n_rows_fit": n_fit,
            "ref_source": ref_source,
            "amp_prefactor": amp,
            "peak_detuning": peak_det,
            "peak_detuning_err": peak_err,
            "peak_fwhm": peak_fwhm,
            "peak_fwhm_err": peak_fwhm_err,
            "peak_freq": peak_freq,
            "accepted": accepted,
            "in_fit": in_fit,
            "photon_number": photon,
            "detuning": detuning,
            "reduced_map": reduced_map,
        }
        twin = twin_values(dataset, AMP, twin_coord)
        if twin is not None:
            results["twin_values"] = twin
            results["twin_label"] = str(twin_label or twin_coord)
        return results

    # ------------------------------------------------------------------
    # Metadata + plot data
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the scalars and the per-row curve; drop the 2-D map, the
        frequency axis and the twin values."""
        drop = {"reduced_map", "detuning", "twin_values"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """The raw ``|IQ - ref|`` map, the per-row line positions and widths, and
        the two fitted curves on a dense amplitude axis — enough to redraw every
        figure. Never raises on a failed fit: fit-derived fields are NaN."""
        amp = np.asarray(results["amp_prefactor"], dtype=float)
        detuning = np.asarray(results["detuning"], dtype=float)
        finite_amp = amp[np.isfinite(amp)]
        dense = (np.linspace(float(finite_amp.min()), float(finite_amp.max()), _DENSE_POINTS)
                 if finite_amp.size else np.full(_DENSE_POINTS, np.nan))

        slope, intercept = results["stark_slope_hz"], results["intercept_detuning_hz"]
        b, w0 = results["broadening_slope_hz"], results["fwhm_intercept_hz"]

        data_vars: Dict[str, Any] = {
            "raw_signal": ((AMP, DET), np.asarray(results["reduced_map"], dtype=float)),
            "peak_detuning": (AMP, np.asarray(results["peak_detuning"], dtype=float)),
            "peak_detuning_err": (AMP, np.asarray(results["peak_detuning_err"], dtype=float)),
            "peak_fwhm": (AMP, np.asarray(results["peak_fwhm"], dtype=float)),
            "peak_fwhm_err": (AMP, np.asarray(results["peak_fwhm_err"], dtype=float)),
            "accepted": (AMP, np.asarray(results["accepted"], dtype=bool)),
            "in_fit": (AMP, np.asarray(results["in_fit"], dtype=bool)),
            "photon_number": (AMP, np.asarray(results["photon_number"], dtype=float)),
            "fit_detuning": ("amp_dense", intercept + slope * dense ** 2),
            "fit_fwhm": ("amp_dense", w0 + b * dense ** 2),
        }
        coords: Dict[str, Any] = {AMP: amp, DET: detuning, "amp_dense": dense}
        attrs: Dict[str, Any] = {
            "success": int(bool(results["success"])),
            "stark_slope_hz": float(slope),
            "intercept_detuning_hz": float(intercept),
            "broadening_slope_hz": float(b),
            "fwhm_intercept_hz": float(w0),
            "chi_hz": float(results["chi_hz"]),
            "has_photon": int(bool(results["has_photon"])),
            "photons_per_prefactor2": float(results["photons_per_prefactor2"]),
            "amp_ref": float(results["amp_ref"]),
            "n_rows_fit": int(results["n_rows_fit"]),
            "rms_residual_hz": float(results["rms_residual_hz"]),
            "ref_source": str(results["ref_source"]),
        }
        if results.get("twin_values") is not None:
            data_vars["twin"] = (AMP, np.asarray(results["twin_values"], dtype=float))
            attrs["twin_label"] = str(results.get("twin_label", ""))
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
        """Three isolated figures, drawn from ``plot_data`` alone: the raw map
        with the tracked line, the shift against ``amp_prefactor**2`` and the
        width against ``amp_prefactor**2``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures(
            {
                "map": lambda: plot_stark_map(plot_data),
                "shift": lambda: plot_shift(plot_data),
                "broadening": lambda: plot_broadening(plot_data),
            },
            label=self.estimator_name,
        )
