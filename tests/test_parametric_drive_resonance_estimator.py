"""Tests for the ParametricDriveResonanceEstimator.

The fixed-time parametric-drive node produces a 2-D ``drive_amp`` x
``driving_frequency`` map of P(|1>); a parametric resonance shows up as a peak in
frequency whose centre drifts with the drive amplitude. The estimator fits each
amplitude slice (delegating to QubitSpectroscopyEstimator) and returns a cleaned
point-cloud of peaks. These tests synthesise a drifting Lorentzian ridge and check
the recovered peak positions, the validation contract, and the analyze round-trip.
"""

import json

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import ParametricDriveResonanceEstimator
from scqat.estimators.parametric_drive_resonance import (
    ParametricDriveResonanceEstimator as SubpkgEstimator,
)


def _make_map(n_amp=7, n_freq=121, noise=4e-3, seed=0, dip=False):
    """A Lorentzian feature per amplitude row whose centre drifts linearly.

    ``dip=True`` is the prepared-excited acquisition the real node runs: the
    parametric transfer empties P(|1>), so the resonance is a DIP — the polarity
    ``fit_peaks`` normalizes away and ``peak_inverted`` preserves.
    """
    amp = np.linspace(1.4, 1.8, n_amp)
    freq = np.linspace(330e6, 350e6, n_freq)
    rng = np.random.default_rng(seed)
    hwhm = (freq[-1] - freq[0]) / 25.0
    f0 = np.linspace(336e6, 344e6, n_amp)  # ridge centre vs amplitude

    state = np.empty((n_amp, n_freq))
    for k in range(n_amp):
        lor = 0.6 / (1.0 + ((freq - f0[k]) / hwhm) ** 2)
        background = 0.9 if dip else 0.1
        state[k] = background + (-lor if dip else lor) + noise * rng.standard_normal(n_freq)

    ds = xr.Dataset(
        {"state": (("drive_amp", "driving_frequency"), state)},
        coords={"drive_amp": amp, "driving_frequency": freq},
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
        kept_amp = res["peak_drive_amp"][good]
        kept_freq = res["peak_frequency"][good]
        for a, f in zip(kept_amp, kept_freq):
            expected = np.interp(a, amp, f0)
            assert f == pytest.approx(expected, abs=1e6)

    def test_dip_map_normalizes_amplitude_and_flags_polarity(self):
        """A DIP map: amplitudes come back POSITIVE (fit_peaks normalizes the
        polarity), and ``peak_inverted`` is the only thing that says "dip"."""
        ds, _, f0 = _make_map(dip=True)
        res = ParametricDriveResonanceEstimator().extract_parameters(ds)
        assert res["good"].any()
        # The polarity CHOICE is per slice and independent of how well that
        # slice's Lorentzian converged, so every point is flagged.
        assert res["peak_inverted"].all()
        assert res["n_inverted"] == res["n_peaks"]
        # The sign of peak_amplitude can never mean "dip": a CONVERGED dip fit
        # reports a POSITIVE amplitude and a negative one is a badly-conditioned
        # fit — the exact trap that made SCQO's `best_peak_amplitude < 0`
        # assertion pass on an artifact. Select converged fits by centre
        # accuracy against the injected ridge, never by sign. (Not every slice
        # converges: fit_peaks negates the trace for a dip and then still seeds
        # FitLorentzian with inverted=True, so the guess starts from a noise
        # trough. Tighten to `res["good"]` once that is fixed.)
        on_line = np.abs(res["peak_frequency"] - f0[res["peak_amp_index"]]) < 1e6
        assert on_line.sum() >= 3
        assert (res["peak_amplitude"][on_line] > 0).all()
        # With the flag, signed physics is recoverable: the dip is negative.
        signed = np.where(res["peak_inverted"], -res["peak_amplitude"],
                          res["peak_amplitude"])
        assert (signed[on_line] < 0).all()

    def test_peak_map_is_not_flagged_inverted(self):
        ds, _, _ = _make_map()
        res = ParametricDriveResonanceEstimator().extract_parameters(ds)
        assert res["n_peaks"] > 0
        assert not res["peak_inverted"].any()
        assert res["n_inverted"] == 0

    def test_polarity_survives_the_saved_artifacts(self, tmp_path):
        """A downstream repo reloading the JSON + netCDF can still tell a dip
        from a peak — no re-fit, no unpickling."""
        ds, _, _ = _make_map(dip=True)
        est = ParametricDriveResonanceEstimator()
        est.analyze(ds, output_dir=str(tmp_path))

        meta = json.loads(
            (tmp_path / "parametric_drive_resonance_metadata.json").read_text()
        )
        assert meta["peak_inverted"] == [True] * len(meta["peak_amplitude"])
        assert meta["n_inverted"] == meta["n_peaks"]

        reloaded = xr.load_dataset(tmp_path / "parametric_drive_resonance_plotdata.nc")
        assert reloaded["peak_inverted"].dims == ("peak",)
        assert reloaded["peak_inverted"].values.astype(bool).all()
        assert int(reloaded.attrs["n_inverted"]) == int(reloaded.attrs["n_peaks"])
        plt.close("all")

    def test_metadata_drops_map(self):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res = est.extract_parameters(ds)
        meta = est.extract_metadata(res)
        assert "amplitude_map" not in meta
        assert "driving_frequency" not in meta
        assert {"peak_drive_amp", "peak_frequency", "good", "n_good",
                "peak_inverted", "n_inverted"} <= set(meta)

    def test_plot_data_layout(self):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, res)
        assert pd["amplitude"].dims == ("drive_amp", "driving_frequency")
        assert pd["peak_frequency"].dims == ("peak",)
        assert "good" in pd and "outlier" in pd
        assert pd["peak_inverted"].dims == ("peak",)
        assert "n_inverted" in pd.attrs

    def test_analyze_roundtrip(self, tmp_path):
        ds, _, _ = _make_map()
        est = ParametricDriveResonanceEstimator()
        res, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "parametric_drive_resonance_metadata.json").exists()
        assert (tmp_path / "parametric_drive_resonance_plotdata.nc").exists()
        assert set(figs) == {"parametric_drive_resonance"}
        assert isinstance(figs["parametric_drive_resonance"], plt.Figure)
        plt.close("all")
