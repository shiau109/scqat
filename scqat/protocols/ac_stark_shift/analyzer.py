from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.qubit_spectroscopy import QubitSpectroscopyAnalyzer
from scqat.protocols.ac_stark_shift.visualization import (
    plot_raw_2d_with_peaks,
    plot_shift_vs_power,
)


class AcStarkShiftAnalyzer(BaseAnalyzer):
    """
    AC-Stark shift: qubit spectroscopy vs readout amplitude.

    Expects an xarray.Dataset with:
        - Variable:    ``IQdata`` (complex), or ``I`` and ``Q`` to build it
        - Coordinate:  ``detuning`` — drive-frequency detuning (Hz)
        - Coordinate:  the readout-amplitude axis named by :attr:`power_coord`
          (default ``readout_amp_ratio``)

    For each amplitude slice the qubit line is located by reusing
    :class:`QubitSpectroscopyAnalyzer` on ``|IQdata - ref|`` (peak detuning).
    With ``chi_eff`` the detuning is converted to readout photon number
    ``n = detuning / chi_eff`` (where ``chi_eff = 2*chi``), and ``n`` is fit
    linearly against ``amp**2`` (slope = photons per amp²).

    This is the Analyzer form of ``notebooks/ac_stark_spectroscopy.ipynb``.
    """

    protocol_name = "ac_stark_shift"
    power_coord: str = "readout_amp_ratio"

    # ------------------------------------------------------------------
    @staticmethod
    def _with_iqdata(dataset: xr.Dataset) -> xr.Dataset:
        if "IQdata" in dataset:
            return dataset
        if "I" in dataset and "Q" in dataset:
            return dataset.assign(IQdata=dataset["I"] + 1j * dataset["Q"])
        raise ValueError("AC-Stark shift requires an 'IQdata' variable, or both 'I' and 'Q'.")

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
            ref, prominence, fit_window_factor: forwarded to QubitSpectroscopyAnalyzer.

        Returns a dict: power_coord, power_values, amp_squared, peak_detuning,
        peak_freq, chi_eff, photon_number, has_photon, fit_slope, fit_intercept,
        amp_ref.
        """
        power_coord = kwargs.get("power_coord", self.power_coord)
        chi_eff = kwargs.get("chi_eff", None)
        amp_ref = kwargs.get("amp_ref", None)
        qs_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("ref", "prominence", "fit_window_factor")}

        ds = self._with_iqdata(dataset)
        amp = np.asarray(ds.coords[power_coord].values, dtype=float)

        qs = QubitSpectroscopyAnalyzer()
        peak_det = np.full(amp.shape, np.nan)
        peak_freq = np.full(amp.shape, np.nan)
        for i, val in enumerate(amp):
            sub = ds.sel({power_coord: val})
            try:
                res = qs.extract_parameters(sub, max_peaks=1, **qs_kwargs)
                peaks = res.get("peaks", [])
                if peaks:
                    peak_det[i] = float(peaks[0]["detuning"])
                    peak_freq[i] = float(peaks[0].get("full_freq", np.nan))
            except Exception:
                pass

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
        ds = self._with_iqdata(dataset)
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
