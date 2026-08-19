"""Broadband Resonator Spectroscopy Estimator.

Processes continuous wideband transmission sweeps (I, Q), normalizes baselines,
detects candidate resonator transmission dips, and fits local Lorentzian parameters.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.tools.dip_finder import find_resonator_dips, validate_dip_finder_kwargs
from .visualization import plot_broadband_spectrum


class BroadbandResonatorSpectroscopyEstimator(BaseEstimator):
    """Estimate candidate resonator frequencies from wideband spectroscopy data."""

    estimator_name = "broadband_resonator_spectroscopy"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_freq = "frequency" in dataset.coords or "frequency_hz" in dataset.coords
        if not has_freq:
            raise ValueError(
                "BroadbandResonatorSpectroscopyEstimator requires a 'frequency' "
                "or 'frequency_hz' coordinate."
            )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "BroadbandResonatorSpectroscopyEstimator requires an 'IQdata' "
                "variable, or both 'I' and 'Q'."
            )

    @classmethod
    def _arrays(cls, dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        ds = with_iqdata(dataset)
        freq_key = "frequency" if "frequency" in ds.coords else "frequency_hz"
        freq = ds.coords[freq_key].values.astype(float).ravel()
        iq = ds["IQdata"].values.ravel()
        return freq, iq

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Extract candidate resonator dips and fit their resonance properties."""
        validate_dip_finder_kwargs(kwargs)
        self._check_data(dataset)
        freq, iq = self._arrays(dataset)

        num_dips = kwargs.get("num_dips")
        min_prominence_db = kwargs.get("min_prominence_db", 0.5)
        min_snr = kwargs.get("min_snr", 2.5)
        baseline_window_points = kwargs.get("baseline_window_points")
        fit_window_points = kwargs.get("fit_window_points")

        finder_out = find_resonator_dips(
            freq,
            iq,
            num_dips=num_dips,
            min_prominence_db=min_prominence_db,
            min_snr=min_snr,
            baseline_window_points=baseline_window_points,
            fit_window_points=fit_window_points,
        )

        dips = finder_out["dips"]
        resonator_freqs = [float(d["frequency_hz"]) for d in dips]

        success = len(dips) > 0
        if num_dips is not None and num_dips > 0:
            success = len(dips) >= num_dips

        return {
            "dips": dips,
            "resonator_frequencies_hz": resonator_freqs,
            "num_dips_found": len(dips),
            "num_dips_requested": num_dips,
            "baseline_db": finder_out["baseline_db"],
            "mag_db": finder_out["mag_db"],
            "freq_hz": finder_out["freq_hz"],
            "success": success,
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Produce small, JSON-serializable metadata summary."""
        clean_dips = []
        for d in results.get("dips", []):
            clean_dips.append({
                "rank": int(d.get("rank", 0)),
                "frequency_hz": float(d.get("frequency_hz", 0.0)),
                "fwhm_hz": float(d.get("fwhm_hz", 0.0)),
                "ql": float(d.get("ql", 0.0)),
                "depth_db": float(d.get("depth_db", 0.0)),
                "prominence_db": float(d.get("prominence_db", 0.0)),
                "success": bool(d.get("success", False)),
            })
        return {
            "estimator_name": self.estimator_name,
            "resonator_frequencies_hz": results.get("resonator_frequencies_hz", []),
            "num_dips_found": results.get("num_dips_found", 0),
            "num_dips_requested": results.get("num_dips_requested"),
            "dips": clean_dips,
            "success": bool(results.get("success", False)),
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Construct plot_data Dataset for standalone offline figure rendering."""
        freq, iq = self._arrays(dataset)
        mag_db = results.get("mag_db", 20.0 * np.log10(np.maximum(np.abs(iq), 1e-15)))
        baseline_db = results.get("baseline_db", np.zeros_like(freq))
        unwrapped_phase = np.unwrap(np.angle(iq))
        delay_s = 0.0
        if len(freq) > 1:
            f_c = freq - np.mean(freq)
            slope, intercept = np.polyfit(f_c, unwrapped_phase, deg=1)
            delay_s = float(-slope / (2.0 * np.pi))
            phase_rad = unwrapped_phase - (slope * f_c + intercept)
        else:
            phase_rad = unwrapped_phase

        clean_dips = self.extract_metadata(results)["dips"]

        ds = xr.Dataset(
            data_vars={
                "mag_db": ("frequency", mag_db),
                "baseline_db": ("frequency", baseline_db),
                "phase_rad": ("frequency", phase_rad),
            },
            coords={"frequency": freq},
            attrs={
                "estimator_name": self.estimator_name,
                "success": int(results.get("success", False)),
                "num_dips_found": results.get("num_dips_found", len(clean_dips)),
                "dips_json": json.dumps(clean_dips),
                "cable_delay_s": delay_s,
            },
        )
        return ds

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Render the broadband spectrum figure."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        if plot_data is None:
            return {}
        return {"broadband_resonator_spectroscopy": plot_broadband_spectrum(plot_data)}

    def plot(self, plot_data: xr.Dataset) -> Dict[str, plt.Figure]:
        """Render the broadband spectrum figure directly from plot_data."""
        return {"broadband_resonator_spectroscopy": plot_broadband_spectrum(plot_data)}

