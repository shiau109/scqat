from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.tools.peak_fit import PEAK_KNOBS, fit_peaks
from scqat.estimators.ac_stark_shift.visualization import (
    plot_raw_2d_with_peaks,
    plot_shift_vs_power,
)


class AcStarkShiftEstimator(BaseEstimator):
    """
    AC-Stark shift: qubit spectroscopy vs readout amplitude.

    Expects an xarray.Dataset with:
        - Variable:    ``IQdata`` (complex), or ``I`` and ``Q`` to build it
        - Coordinate:  ``detuning`` — drive-frequency detuning (Hz)
        - Coordinate:  the readout-amplitude axis named by :attr:`power_coord`
          (default ``readout_amp_ratio``)

    For each amplitude slice the qubit line is located by the family-shared
    per-trace reduction :func:`scqat.tools.peak_fit.fit_peaks` on
    ``|IQdata - ref|`` with ``max_peaks=1`` (peak detuning).
    With ``chi_eff`` the detuning is converted to readout photon number
    ``n = detuning / chi_eff`` (where ``chi_eff = 2*chi``), and ``n`` is fit
    linearly against ``amp**2`` (slope = photons per amp²).

    This is the Estimator form of ``notebooks/ac_stark_spectroscopy.ipynb``.
    """

    estimator_name = "ac_stark_shift"
    power_coord: str = "readout_amp_ratio"

    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if self.power_coord not in dataset.coords:
            raise ValueError(f"AC-Stark shift requires a '{self.power_coord}' coordinate.")
        if "detuning" not in dataset.coords:
            raise ValueError("AC-Stark shift requires a 'detuning' coordinate.")
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError("AC-Stark shift requires an 'IQdata' variable, or both 'I' and 'Q'.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Locate f01 per amplitude slice and fit the Stark shift vs amp².

        Kwargs:
            chi_eff (float): Dispersive shift (2*chi, same units as detuning/1e6,
                i.e. MHz) used to convert detuning to photon number. If omitted,
                the analysis stays in detuning and reports no photon number.
            amp_ref (float): Scale ``power_coord`` by this to get a physical
                readout amplitude (default: use the ratio as-is).
            power_coord (str): Override the swept coordinate name.
            ref, prominence, fit_window_factor, ...: knobs of
                :func:`scqat.tools.peak_fit.fit_peaks` (``max_peaks`` excluded —
                this analysis pins ``max_peaks=1``). Unknown names raise before
                any per-slice fit.

        Returns a dict: power_coord, power_values, amp_squared, peak_detuning,
        peak_freq, chi_eff, photon_number, has_photon, fit_slope, fit_intercept,
        amp_ref.
        """
        power_coord = kwargs.pop("power_coord", self.power_coord)
        chi_eff = kwargs.pop("chi_eff", None)
        amp_ref = kwargs.pop("amp_ref", None)
        # Flat surface, validated up front: everything left must be a fit_peaks
        # knob, and max_peaks is pinned to 1 (exactly one qubit line expected).
        unknown = set(kwargs) - (PEAK_KNOBS - {"max_peaks"})
        if unknown:
            raise ValueError(
                f"Unknown keyword argument(s) {sorted(unknown)} for "
                f"AcStarkShiftEstimator; valid: "
                f"{sorted((PEAK_KNOBS - {'max_peaks'}) | {'power_coord', 'chi_eff', 'amp_ref'})} "
                f"(max_peaks is pinned to 1 here)"
            )

        ds = with_iqdata(dataset)
        amp = np.asarray(ds.coords[power_coord].values, dtype=float)
        detuning = np.asarray(ds.coords["detuning"].values, dtype=float)
        full_freq = (
            ds.coords["full_freq"].values.ravel().astype(float)
            if "full_freq" in ds.coords else None
        )
        iq_map = ds["IQdata"].transpose(power_coord, "detuning").values

        peak_det = np.full(amp.shape, np.nan)
        peak_freq = np.full(amp.shape, np.nan)
        for i in range(len(amp)):
            try:
                res = fit_peaks(detuning, iq_map[i], full_freq=full_freq,
                                max_peaks=1, **kwargs)
            except Exception:
                continue  # fit-domain failure only: kwargs were validated up front
            peaks = res["peaks"]
            if peaks:
                peak_det[i] = float(peaks[0]["detuning"])
                peak_freq[i] = float(peaks[0].get("full_freq", np.nan))

        amp_scaled = amp * amp_ref if amp_ref is not None else amp
        amp_squared = amp_scaled ** 2

        has_photon = chi_eff is not None
        if has_photon:
            photon = (peak_det / 1e6) / float(chi_eff)
            y = photon
        else:
            photon = np.full(amp.shape, np.nan)
            y = peak_det

        mask = np.isfinite(y) & np.isfinite(amp_squared)
        if np.count_nonzero(mask) >= 2:
            slope, intercept = np.polyfit(amp_squared[mask], y[mask], 1)
        else:
            slope = intercept = np.nan

        return {
            "power_coord": power_coord,
            "power_values": amp_scaled,
            "amp_squared": amp_squared,
            "peak_detuning": peak_det,
            "peak_freq": peak_freq,
            "chi_eff": (float(chi_eff) if has_photon else None),
            "has_photon": has_photon,
            "photon_number": photon,
            "fit_slope": float(slope),
            "fit_intercept": float(intercept),
            "amp_ref": (float(amp_ref) if amp_ref is not None else None),
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Bundle the raw 2D map, per-amp peak / photon number, and the linear fit
        curve into one self-sufficient Dataset so the figures redraw from the saved
        ``*_plotdata.nc`` alone."""
        power_coord = results["power_coord"]
        ds = with_iqdata(dataset)
        detuning = np.asarray(ds.coords["detuning"].values, dtype=float)
        raw = np.abs(ds["IQdata"].transpose(power_coord, "detuning").values)

        amp = np.asarray(results["power_values"], dtype=float)
        amp_sq = np.asarray(results["amp_squared"], dtype=float)
        has_photon = bool(results["has_photon"])
        y = np.asarray(results["photon_number" if has_photon else "peak_detuning"], dtype=float)

        data_vars: Dict[str, Any] = {
            "raw_signal": ([power_coord, "detuning"], np.asarray(raw, dtype=float)),
            "peak_detuning": (power_coord, np.asarray(results["peak_detuning"], dtype=float)),
            "amp_squared": (power_coord, amp_sq),
            "y_value": (power_coord, y),
        }
        coords: Dict[str, Any] = {power_coord: amp, "detuning": detuning}
        attrs: Dict[str, Any] = {
            "power_coord": power_coord,
            "has_photon": int(has_photon),
            "chi_eff": float(results["chi_eff"]) if has_photon else np.nan,
            "amp_ref": float(results["amp_ref"]) if results["amp_ref"] is not None else np.nan,
            "fit_slope": float(results["fit_slope"]),
            "fit_intercept": float(results["fit_intercept"]),
        }

        if np.isfinite(results["fit_slope"]) and amp_sq.size:
            P_fine = np.linspace(float(np.nanmin(amp_sq)), float(np.nanmax(amp_sq)), 200)
            coords["P_fine"] = P_fine
            data_vars["fit_y"] = ("P_fine", results["fit_slope"] * P_fine + results["fit_intercept"])

        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the AC-Stark figures, drawing only from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return {
            "raw_2d": plot_raw_2d_with_peaks(plot_data),
            "shift_vs_power": plot_shift_vs_power(plot_data),
        }
