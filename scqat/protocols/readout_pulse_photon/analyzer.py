from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.protocols.qubit_spectroscopy import QubitSpectroscopyAnalyzer
from scqat.protocols.readout_pulse_photon.visualization import (
    plot_raw_2d_with_peaks,
    plot_photon_vs_delay,
)


class ReadoutPulsePhotonAnalyzer(BaseAnalyzer):
    """
    Time-dependent readout-resonator photon number: qubit spectroscopy vs delay.

    Expects an xarray.Dataset with:
        - Variable:    ``IQdata`` (complex), or ``I`` and ``Q`` to build it
        - Coordinate:  ``detuning`` — drive-frequency detuning (Hz)
        - Coordinate:  the delay axis named by :attr:`time_coord`
          (default ``delay_time``)

    For each delay slice the qubit line is located by reusing
    :class:`QubitSpectroscopyAnalyzer` on ``|IQdata - ref|`` (peak detuning).
    With ``chi_eff`` the detuning is converted to an intra-resonator photon
    number ``n = detuning / chi_eff`` (``chi_eff = 2*chi``); the photon-number
    trace vs delay shows the resonator filling/ring-down, and a steady-state
    value can be averaged over a chosen delay window.

    This is the Analyzer form of ``notebooks/ac_stark_readout.ipynb``.
    """

    protocol_name = "readout_pulse_photon"
    time_coord: str = "delay_time"

    # ------------------------------------------------------------------
    @staticmethod
    def _with_iqdata(dataset: xr.Dataset) -> xr.Dataset:
        if "IQdata" in dataset:
            return dataset
        if "I" in dataset and "Q" in dataset:
            return dataset.assign(IQdata=dataset["I"] + 1j * dataset["Q"])
        raise ValueError("Readout-pulse photon requires an 'IQdata' variable, or both 'I' and 'Q'.")

    def _check_data(self, dataset: xr.Dataset) -> None:
        if self.time_coord not in dataset.coords:
            raise ValueError(f"Readout-pulse photon requires a '{self.time_coord}' coordinate.")
        if "detuning" not in dataset.coords:
            raise ValueError("Readout-pulse photon requires a 'detuning' coordinate.")
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError("Readout-pulse photon requires an 'IQdata' variable, or both 'I' and 'Q'.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Locate the qubit line per delay and build the photon-number trace.

        Kwargs:
            chi_eff (float): Dispersive shift (2*chi, same units as detuning/1e6,
                i.e. MHz) for detuning -> photon number. If omitted, stays in detuning.
            steady_state_window (tuple[float, float]): (t_lo, t_hi) delay range to
                average for a steady-state value (default: none).
            time_coord (str): Override the delay coordinate name.
            ref, prominence, fit_window_factor: forwarded to QubitSpectroscopyAnalyzer.

        Returns a dict: time_coord, delay_times, peak_detuning, photon_number,
        has_photon, chi_eff, steady_state_window, steady_state_value.
        """
        time_coord = kwargs.get("time_coord", self.time_coord)
        chi_eff = kwargs.get("chi_eff", None)
        window = kwargs.get("steady_state_window", None)
        qs_kwargs = {k: v for k, v in kwargs.items()
                     if k in ("ref", "prominence", "fit_window_factor")}

        ds = self._with_iqdata(dataset)
        delays = np.asarray(ds.coords[time_coord].values, dtype=float)

        qs = QubitSpectroscopyAnalyzer()
        peak_det = np.full(delays.shape, np.nan)
        for i, t in enumerate(ds.coords[time_coord].values):
            sub = ds.sel({time_coord: t})
            try:
                res = qs.extract_parameters(sub, max_peaks=1, **qs_kwargs)
                peaks = res.get("peaks", [])
                if peaks:
                    peak_det[i] = float(peaks[0]["detuning"])
            except Exception:
                pass

        has_photon = chi_eff is not None
        photon = (peak_det / 1e6) / float(chi_eff) if has_photon else np.full(delays.shape, np.nan)
        y = photon if has_photon else peak_det

        steady_value = np.nan
        if window is not None:
            lo, hi = window
            wmask = (delays >= lo) & (delays <= hi) & np.isfinite(y)
            if wmask.any():
                steady_value = float(np.nanmean(y[wmask]))

        return {
            "time_coord": time_coord,
            "delay_times": delays,
            "peak_detuning": peak_det,
            "photon_number": photon,
            "has_photon": has_photon,
            "chi_eff": (float(chi_eff) if has_photon else None),
            "steady_state_window": (tuple(window) if window is not None else None),
            "steady_state_value": steady_value,
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """Bundle the raw 2D map and the per-delay peak / photon-number trace into
        one self-sufficient Dataset so the figures redraw from the saved
        ``*_plotdata.nc`` alone."""
        time_coord = results["time_coord"]
        ds = self._with_iqdata(dataset)
        detuning = np.asarray(ds.coords["detuning"].values, dtype=float)
        raw = np.abs(ds["IQdata"].transpose(time_coord, "detuning").values)

        delays = np.asarray(results["delay_times"], dtype=float)
        has_photon = bool(results["has_photon"])
        y = np.asarray(results["photon_number" if has_photon else "peak_detuning"], dtype=float)

        attrs: Dict[str, Any] = {
            "time_coord": time_coord,
            "has_photon": int(has_photon),
            "chi_eff": float(results["chi_eff"]) if has_photon else np.nan,
            "steady_state_value": float(results["steady_state_value"]),
        }
        window = results["steady_state_window"]
        if window is not None:
            attrs["steady_state_lo"] = float(window[0])
            attrs["steady_state_hi"] = float(window[1])

        return xr.Dataset(
            {
                "raw_signal": ([time_coord, "detuning"], np.asarray(raw, dtype=float)),
                "peak_detuning": (time_coord, np.asarray(results["peak_detuning"], dtype=float)),
                "y_value": (time_coord, y),
            },
            coords={time_coord: delays, "detuning": detuning},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the photon-trace figures, drawing only from ``plot_data``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return {
            "raw_2d": plot_raw_2d_with_peaks(plot_data),
            "photon_vs_delay": plot_photon_vs_delay(plot_data),
        }
