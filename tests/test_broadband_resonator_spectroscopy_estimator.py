"""Tests for BroadbandResonatorSpectroscopyEstimator and find_resonator_dips tool."""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import BroadbandResonatorSpectroscopyEstimator
from scqat.tools.dip_finder import find_resonator_dips


def _generate_synthetic_broadband_data(
    freqs: np.ndarray,
    dip_freqs: list[float],
    dip_kappas: list[float],
    dip_depths: list[float],
    noise_level: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic complex transmission with multiple Lorentzian dips."""
    # Baseline with mild ripple
    baseline_mag = 0.8 + 0.1 * np.sin(2 * np.pi * freqs / 1e9)
    baseline_phase = -2 * np.pi * freqs * 50e-9  # Cable delay
    s21 = baseline_mag * np.exp(1j * baseline_phase)

    # Multiply by transmission dips
    for f0, kappa, depth in zip(dip_freqs, dip_kappas, dip_depths):
        lorentz_dip = 1.0 - depth / (1.0 + 2j * (freqs - f0) / kappa)
        s21 *= lorentz_dip

    # Add Gaussian noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, noise_level, freqs.shape) + 1j * rng.normal(0, noise_level, freqs.shape)
    s21 += noise

    return np.real(s21), np.imag(s21)


def test_find_resonator_dips():
    freqs = np.linspace(4.0e9, 8.0e9, 4001)
    true_dips = [5.2e9, 6.1e9, 7.3e9]
    kappas = [3.0e6, 4.0e6, 2.5e6]
    depths = [0.8, 0.7, 0.9]

    i_data, q_data = _generate_synthetic_broadband_data(freqs, true_dips, kappas, depths)
    iq = i_data + 1j * q_data

    res = find_resonator_dips(freqs, iq, num_dips=3, min_prominence_db=1.0)
    dips = res["dips"]

    assert len(dips) == 3
    found_freqs = [d["frequency_hz"] for d in dips]

    # Verify each true dip is found within 10 MHz
    for true_f in true_dips:
        min_dist = min(abs(ff - true_f) for ff in found_freqs)
        assert min_dist < 10.0e6


def test_broadband_estimator_analyze(tmp_path):
    freqs = np.linspace(4.0e9, 8.0e9, 2001)
    true_dips = [5.5e9, 6.8e9]
    kappas = [5.0e6, 4.0e6]
    depths = [0.85, 0.75]

    i_data, q_data = _generate_synthetic_broadband_data(freqs, true_dips, kappas, depths)

    ds = xr.Dataset(
        data_vars={"I": ("frequency", i_data), "Q": ("frequency", q_data)},
        coords={"frequency": freqs},
    )

    estimator = BroadbandResonatorSpectroscopyEstimator()
    results, figures = estimator.analyze(ds, output_dir=str(tmp_path), num_dips=2)

    assert results["success"] is True
    assert len(results["resonator_frequencies_hz"]) == 2
    assert "broadband_resonator_spectroscopy" in figures

    # Check that artifact files were created
    assert (tmp_path / "broadband_resonator_spectroscopy_metadata.json").is_file()
    assert (tmp_path / "broadband_resonator_spectroscopy_plotdata.nc").is_file()
    assert (tmp_path / "broadband_resonator_spectroscopy.png").is_file()
