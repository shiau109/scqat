"""Tests for the ParametricDriveResonanceEstimator.

The fixed-time parametric-drive node produces a 2-D ``amplitude_ratio`` x
``driving_frequency`` map of P(|1>); a parametric resonance shows up as a peak in
frequency whose centre drifts with the drive amplitude. The estimator fits each
amplitude slice (delegating to QubitSpectroscopyEstimator) and returns a cleaned
point-cloud of peaks. These tests synthesise a drifting Lorentzian ridge and check
the recovered peak positions, the validation contract, and the analyze round-trip.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import ParametricDriveResonanceEstimator
from scqat.estimators.parametric_drive_resonance import (
    ParametricDriveResonanceEstimator as SubpkgEstimator,
)


def _make_map(n_amp=7, n_freq=121, noise=4e-3, seed=0):
    """A Lorentzian peak per amplitude row whose centre drifts linearly."""
    amp = np.linspace(1.4, 1.8, n_amp)
    freq = np.linspace(330e6, 350e6, n_freq)
    rng = np.random.default_rng(seed)
    hwhm = (freq[-1] - freq[0]) / 25.0
    f0 = np.linspace(336e6, 344e6, n_amp)  # ridge centre vs amplitude

    state = np.empty((n_amp, n_freq))
    for k in range(n_amp):
        lor = 0.6 / (1.0 + ((freq - f0[k]) / hwhm) ** 2)
        state[k] = 0.1 + lor + noise * rng.standard_normal(n_freq)

    ds = xr.Dataset(
        {"state": (("amplitude_ratio", "driving_frequency"), state)},
        coords={"amplitude_ratio": amp, "driving_frequency": freq},
    )
    return ds, amp, f0


class TestParametricDriveResonanceEstimator:

    def test_imports_match(self):
        assert ParametricDriveResonanceEstimator is SubpkgEstimator
        assert ParametricDriveResonanceEstimator.estimator_name == "parametric_drive_resonance"

    def test_check_data_requires_coords(self):
        est = ParametricDriveResonanceEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"state": ("driving_frequency", [0.0, 1.0])},
                                       coords={"driving_frequency": [0.0, 1.0]}))

    def test_finds_drifting_ridge(self):
        ds, amp, f0 = _make_map()
        res = ParametricDriveResonanceEstimator().extract_parameters(ds)
        assert res["n_amp"] == len(amp)
        # At least one good peak per amplitude row on a clean ridge.
        assert res["n_good"] >= len(amp) - 1
        # Kept peaks should track the planted centres.
        good = res["good"]
        assert good.any()
        kept_amp = res["peak_amp_ratio"][good]
        kept_freq = res["peak_frequency"][good]
        for a, f in zip(kept_amp, kept_freq):
            expected = np.interp(a, amp, f0)
            assert f == pytest.approx(expected, abs=1e6)

    def test_metadata_drops_map(self):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res = est.extract_parameters(ds)
        meta = est.extract_metadata(res)
        assert "amplitude_map" not in meta
        assert "driving_frequency" not in meta
        assert {"peak_amp_ratio", "peak_frequency", "good", "n_good"} <= set(meta)

    def test_plot_data_layout(self):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        assert pd["amplitude"].dims == ("amplitude_ratio", "driving_frequency")
        assert pd["peak_frequency"].dims == ("peak",)
        assert "good" in pd and "outlier" in pd

    def test_analyze_roundtrip(self, tmp_path):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "parametric_drive_resonance_metadata.json").exists()
        assert (tmp_path / "parametric_drive_resonance_plotdata.nc").exists()
        assert set(figs) == {"parametric_drive_resonance"}
        assert isinstance(figs["parametric_drive_resonance"], plt.Figure)
        plt.close("all")
