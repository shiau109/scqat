"""Tests for the QubitSpectroscopyEstimator peak detection / fitting.

Synthesises qubit-spectroscopy IQ sweeps with one or more Lorentzian lines and checks
the estimator returns the right peak count, recovers each line, exposes the
``amplitude``/``fwhm``/``full_freq`` fields the LCHQMDriver node relies on, and that
selecting the primary line by Lorentzian **area** (``|amplitude|·fwhm``) genuinely differs
from selecting by peak **height**.

The area-selection rule lives in the node (LCHQMDriver, which can't be imported here — it
pulls in ``qm``), so these tests replicate the node's selection key inline and assert the
estimator's output is accurate enough for it to pick the intended peak.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import QubitSpectroscopyEstimator
from scqat.estimators.qubit_spectroscopy import (
    QubitSpectroscopyEstimator as SubpkgEstimator,
)
from scqat.estimators.qubit_spectroscopy.estimator import _merge_overlapping_peaks
from scqat.tools.fit_lorentzian import lorentzian


# Node's primary-peak rule: largest Lorentzian area (∝ |amplitude|·fwhm).
def _area(p):
    return abs(p["amplitude"]) * p["fwhm"]


def _height(p):
    return abs(p["amplitude"])


def _make_ds(peaks, n=801, lo=5.0e9, span=100e6, seed=0):
    """Qubit-spectroscopy IQ sweep with emission Lorentzians.

    ``peaks`` is a list of ``(x0, amplitude, gamma)``. The estimator builds
    ``signal = |IQdata - median|`` from ``I``/``Q``; with ``Q ≈ 0`` and a ~0 baseline the
    median is ~0, so the signal reproduces the Lorentzian bumps.

    NOTE: keep multi-peak fixtures well separated (≥ ~80 MHz). A peak's fit window is
    ``5 × find_peaks width``, so a broad line adjacent to a tall narrow one makes the broad
    fit lock onto the narrow peak (a pre-existing estimator limitation). Wide separation
    keeps each fit clean — do not narrow it when editing these fixtures.
    """
    detuning = np.linspace(-span, span, n)
    I = np.zeros_like(detuning)
    for x0, amp, gamma in peaks:
        I = I + lorentzian(detuning, x0, amp, gamma, 0.0)
    rng = np.random.default_rng(seed)
    I = I + 1e-3 * rng.standard_normal(n)
    Q = 1e-3 * rng.standard_normal(n)
    return xr.Dataset(
        {"I": ("detuning", I), "Q": ("detuning", Q)},
        coords={"detuning": detuning, "full_freq": ("detuning", detuning + lo)},
    )


class TestQubitSpectroscopyEstimator:

    def test_imports_match(self):
        assert QubitSpectroscopyEstimator is SubpkgEstimator
        assert QubitSpectroscopyEstimator.estimator_name == "qubit_spectroscopy"

    def test_single_peak(self):
        lo = 5.0e9
        x0, amp, gamma = 10e6, 0.8, 3e6
        ds = _make_ds([(x0, amp, gamma)], lo=lo)
        results = QubitSpectroscopyEstimator().extract_parameters(ds)
        peaks = results["peaks"]
        assert len(peaks) == 1
        pk = peaks[0]
        assert pk["detuning"] == pytest.approx(x0, abs=0.5e6)
        assert pk["fwhm"] == pytest.approx(2 * gamma, rel=0.15)
        assert pk["full_freq"] == pytest.approx(x0 + lo, abs=0.5e6)

    def test_two_peaks_detected(self):
        ds = _make_ds([(-60e6, 1.0, 1e6), (20e6, 0.8, 2e6)])
        results = QubitSpectroscopyEstimator().extract_parameters(ds)
        assert len(results["peaks"]) == 2

    def test_area_vs_height_selection_disagree(self):
        # Tall-narrow at -60 MHz (height 1.0, area-metric ~2.0e6) vs short-broad at +20 MHz
        # (height 0.5, area-metric ~4.0e6): area and height selection must pick different lines.
        ds = _make_ds([(-60e6, 1.0, 1e6), (20e6, 0.5, 4e6)])
        peaks = QubitSpectroscopyEstimator().extract_parameters(ds)["peaks"]
        assert len(peaks) == 2

        by_area = max(peaks, key=_area)
        by_height = max(peaks, key=_height)
        assert by_area is not by_height
        assert by_area["detuning"] == pytest.approx(20e6, abs=1e6)   # broad / larger area
        assert by_height["detuning"] == pytest.approx(-60e6, abs=1e6)  # tall / larger height

    def test_max_peaks_caps_count(self):
        three = [(-70e6, 1.0, 1e6), (0e6, 0.7, 1e6), (70e6, 0.4, 1e6)]
        est = QubitSpectroscopyEstimator()
        capped = est.extract_parameters(_make_ds(three), max_peaks=2)["peaks"]
        assert len(capped) == 2
        all_peaks = est.extract_parameters(_make_ds(three), max_peaks=None)["peaks"]
        assert len(all_peaks) == 3

    def test_metadata_drops_fit_curves(self):
        ds = _make_ds([(0e6, 0.9, 2e6)])
        est = QubitSpectroscopyEstimator()
        results = est.extract_parameters(ds)
        meta = est.extract_metadata(results)
        assert meta["peaks"], "expected at least one peak in metadata"
        for pk in meta["peaks"]:
            assert "fit_x" not in pk and "fit_y" not in pk
            assert {"detuning", "amplitude", "fwhm"} <= set(pk)

    def test_plot_data_per_peak_arrays(self):
        ds = _make_ds([(-60e6, 1.0, 1e6), (20e6, 0.5, 4e6)])
        est = QubitSpectroscopyEstimator()
        results = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, results)

        n = len(results["peaks"])
        assert pd.attrs["n_peaks"] == n
        assert pd["peak_fit"].dims == ("peak", "detuning")
        assert pd.sizes["peak"] == n
        for var in ("peak_fit", "peak_detuning", "peak_fwhm"):
            assert var in pd

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_ds([(0e6, 0.9, 2e6)])
        est = QubitSpectroscopyEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path))

        assert (tmp_path / "qubit_spectroscopy_metadata.json").exists()
        assert (tmp_path / "qubit_spectroscopy_plotdata.nc").exists()
        assert set(figs) == {"spectrum"}
        assert isinstance(figs["spectrum"], plt.Figure)
        plt.close("all")


class TestPeakMerging:
    """De-duplication of near-coincident Lorentzian fits (the duplicate-peak fix)."""

    @staticmethod
    def _pk(detuning, amplitude, fwhm):
        return {"detuning": float(detuning), "amplitude": float(amplitude), "fwhm": float(fwhm)}

    def test_merge_helper_collapses_overlapping_keeps_larger_area(self):
        # Mimics run #66 q5: a broad central line (large area) and a narrow
        # shoulder bump 4.4 MHz away (centres within the summed half-widths).
        broad = self._pk(-0.5e6, 1.0, 8.56e6)   # area ~8.6e6
        narrow = self._pk(3.87e6, 1.0, 2.43e6)  # area ~2.4e6
        merged = _merge_overlapping_peaks([broad, narrow], merge_factor=1.0)
        assert len(merged) == 1
        assert merged[0]["detuning"] == pytest.approx(-0.5e6)  # the larger-area line kept

    def test_merge_helper_keeps_separated_lines(self):
        # Mimics run #66 q4: ~91 MHz apart -> genuinely two transitions, never merged.
        far_a = self._pk(-91e6, 1.0, 1.48e6)
        far_b = self._pk(0.5e6, 1.0, 22.72e6)
        merged = _merge_overlapping_peaks([far_a, far_b], merge_factor=1.0)
        assert len(merged) == 2

    def test_merge_factor_zero_disables(self):
        broad = self._pk(-0.5e6, 1.0, 8.56e6)
        narrow = self._pk(3.87e6, 1.0, 2.43e6)
        assert len(_merge_overlapping_peaks([broad, narrow], merge_factor=0)) == 2

    def test_overlapping_peaks_merged_end_to_end(self):
        # A tall narrow bump riding on a broad line: clean synthetic version of the
        # q5 duplicate. With merging on -> one peak; with it off -> the duplicate returns.
        ds = _make_ds([(0e6, 1.0, 4e6), (4e6, 1.2, 0.8e6)])
        est = QubitSpectroscopyEstimator()
        merged = est.extract_parameters(ds, merge_factor=1.0)["peaks"]
        unmerged = est.extract_parameters(ds, merge_factor=0)["peaks"]
        assert len(merged) == 1
        assert len(unmerged) >= len(merged)

    def test_two_peaks_survive_default_merge(self):
        # The genuine two-transition sweep must stay two peaks under the default
        # merge_factor=1.0 (guards against over-merging in the full pipeline).
        ds = _make_ds([(-60e6, 1.0, 1e6), (20e6, 0.8, 2e6)])
        peaks = QubitSpectroscopyEstimator().extract_parameters(ds)["peaks"]
        assert len(peaks) == 2

    def test_subresolution_spike_dropped(self):
        # A single-sample spike fits as a delta-like Lorentzian (fwhm << step) and
        # must be dropped by the min-width guard, leaving only the real line.
        ds = _make_ds([(0e6, 1.0, 4e6)])
        ds["I"].values[5] += 0.5  # isolated one-sample spike far from the real peak
        est = QubitSpectroscopyEstimator()
        step = abs(float(np.median(np.diff(ds.detuning.values))))
        default = est.extract_parameters(ds)["peaks"]
        assert all(pk["fwhm"] >= 0.5 * step for pk in default)
        assert any(abs(pk["detuning"]) < 5e6 for pk in default)  # real line recovered
