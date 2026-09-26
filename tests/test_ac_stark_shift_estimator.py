"""Tests for the AcStarkShiftEstimator — the qubit line against a Stark tone.

SCQO's ``qubit_resonator_stark`` repeats qubit spectroscopy at every amplitude of
a tone parked on the readout resonator. The line moves as ``f0 + s * a**2`` and
broadens as ``w0 + b * a**2`` (``a`` = the amplitude prefactor, so ``a**2`` is the
tone power). These tests plant both and check they come back, together with the
photon-number conversion's SIGN (the catalog's ``chi = (f_dress0 - f_dress1) / 2``),
the residual clip, the knob validation, and that a failed fit still draws.
"""

import json

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import AcStarkShiftEstimator
from scqat.estimators.ac_stark_shift import AcStarkShiftEstimator as SubpkgEstimator

F0 = 2.0e6          # the zero-photon line, Hz from the drive
SLOPE = -14.0e6     # Hz per amp_prefactor**2 — a transmon below its resonator
W0 = 1.5e6          # zero-photon FWHM
B = 1.2e6           # FWHM growth per amp_prefactor**2
AMPS = np.linspace(0.0, 1.5, 9)
DETUNING = np.linspace(-40e6, 10e6, 201)
FIGURES = {"map", "shift", "broadening"}


def _lorentzian(detuning, centre, fwhm):
    return 1.0 / (1.0 + ((detuning - centre) / (fwhm / 2.0)) ** 2)


def _make_map(*, noise=4e-3, seed=0, outlier_row=None, centres=None):
    """I/Q over (amp_prefactor, detuning): a Lorentzian bump on I per row."""
    rng = np.random.default_rng(seed)
    centres = F0 + SLOPE * AMPS ** 2 if centres is None else centres
    fwhms = W0 + B * AMPS ** 2
    i_data = np.empty((AMPS.size, DETUNING.size))
    for row, (centre, fwhm) in enumerate(zip(centres, fwhms)):
        if row == outlier_row:
            centre = centre + 15e6  # far off the trend, still inside the window
        i_data[row] = 0.1 + 0.5 * _lorentzian(DETUNING, centre, fwhm)
    i_data += noise * rng.standard_normal(i_data.shape)
    q_data = 0.05 + noise * rng.standard_normal(i_data.shape)
    return xr.Dataset(
        {"I": (("amp_prefactor", "detuning"), i_data),
         "Q": (("amp_prefactor", "detuning"), q_data)},
        coords={"amp_prefactor": AMPS, "detuning": DETUNING},
    )


def _fit(ds, **kwargs):
    return AcStarkShiftEstimator().extract_parameters(ds, **kwargs)


def test_imports_match():
    assert AcStarkShiftEstimator is SubpkgEstimator
    assert AcStarkShiftEstimator.estimator_name == "ac_stark_shift"


@pytest.mark.parametrize("missing", ["amp_prefactor", "detuning"])
def test_check_data_requires_both_axes(missing):
    ds = _make_map().rename({missing: "other"})
    with pytest.raises(ValueError, match=missing):
        AcStarkShiftEstimator()._check_data(ds)


def test_recovers_the_planted_shift_intercept_and_broadening():
    res = _fit(_make_map())
    assert res["success"] is True
    assert res["n_rows_fit"] == AMPS.size
    assert res["stark_slope_hz"] == pytest.approx(SLOPE, rel=0.02)
    assert res["intercept_detuning_hz"] == pytest.approx(F0, abs=0.2e6)
    assert res["broadening_slope_hz"] == pytest.approx(B, rel=0.15)
    assert res["fwhm_intercept_hz"] == pytest.approx(W0, abs=0.25e6)
    assert np.isfinite(res["stark_slope_err_hz"]) and res["stark_slope_err_hz"] > 0
    # the measured per-row curve is reported, not just the model
    assert res["peak_detuning"] == pytest.approx(F0 + SLOPE * AMPS ** 2, abs=0.3e6)


def test_photon_number_follows_the_catalog_chi_sign():
    """chi = (f_dress0 - f_dress1)/2 > 0 for a transmon below its resonator, which
    shifts DOWN by 2*chi per photon: a negative slope must give POSITIVE photons."""
    chi = 0.5e6
    res = _fit(_make_map(), chi_hz=chi)
    assert res["has_photon"] is True
    assert res["photons_per_prefactor2"] == pytest.approx(SLOPE / (-2 * chi), rel=0.02)
    assert res["photons_per_prefactor2"] > 0
    assert res["photon_number"][-1] == pytest.approx(
        SLOPE * AMPS[-1] ** 2 / (-2 * chi), rel=0.05)


@pytest.mark.parametrize("chi", [None, 0.0, float("nan")])
def test_no_usable_chi_means_no_photon_number(chi):
    res = _fit(_make_map(), chi_hz=chi)
    assert res["has_photon"] is False
    assert np.isnan(res["photons_per_prefactor2"])
    assert np.isnan(res["photon_number"]).all()


def test_amp_ref_turns_the_slopes_absolute():
    res = _fit(_make_map(), chi_hz=0.5e6, amp_ref=0.08)
    assert res["stark_hz_per_amp2"] == pytest.approx(res["stark_slope_hz"] / 0.08 ** 2)
    assert res["photons_per_amp2"] == pytest.approx(res["photons_per_prefactor2"] / 0.08 ** 2)
    assert np.isnan(_fit(_make_map())["stark_hz_per_amp2"])


def test_an_off_trend_row_is_clipped_on_the_residuals():
    """The row is a clean, in-window line — every row gate passes it — so only
    the residual clip against the TREND can keep it out of the slope."""
    res = _fit(_make_map(outlier_row=5))
    assert res["accepted"][5] and not res["in_fit"][5]
    assert res["n_rows_fit"] == AMPS.size - 1
    assert res["stark_slope_hz"] == pytest.approx(SLOPE, rel=0.02)


def test_a_broad_high_power_row_is_not_gated_away():
    """No population width gate: the widest (highest-power) rows stay in."""
    res = _fit(_make_map())
    assert res["in_fit"][-1] and res["in_fit"][-2]


def test_max_peaks_is_refused():
    with pytest.raises(ValueError, match="max_peaks"):
        _fit(_make_map(), max_peaks=2)


def test_an_unknown_knob_is_refused_before_any_row_is_fitted():
    with pytest.raises(ValueError, match="prominance"):
        _fit(_make_map(), prominance=0.2)


def test_figures_render_on_a_failed_fit(tmp_path):
    rng = np.random.default_rng(1)
    ds = _make_map()
    ds["I"].values[:] = 0.1 + 4e-3 * rng.standard_normal(ds["I"].shape)
    res, figs = AcStarkShiftEstimator().analyze(ds, output_dir=str(tmp_path))
    assert res["success"] is False
    assert res["n_rows_fit"] == 0
    assert set(figs) == FIGURES
    written = {p.name for p in tmp_path.iterdir()}
    assert {f"ac_stark_shift_{name}.png" for name in FIGURES} <= written
    assert {"ac_stark_shift_metadata.json", "ac_stark_shift_plotdata.nc"} <= written


def test_artifacts_round_trip_and_redraw_from_plot_data(tmp_path):
    est = AcStarkShiftEstimator()
    res, _ = est.analyze(_make_map(), output_dir=str(tmp_path), chi_hz=0.5e6)
    meta = json.loads((tmp_path / "ac_stark_shift_metadata.json").read_text())
    assert meta["estimator_name"] == "ac_stark_shift"
    assert meta["stark_slope_hz"] == pytest.approx(res["stark_slope_hz"])
    assert "reduced_map" not in meta and len(meta["peak_detuning"]) == AMPS.size
    plot_data = est.load_plot_data(str(tmp_path))
    assert plot_data["raw_signal"].dims == ("amp_prefactor", "detuning")
    assert plot_data.attrs["has_photon"] == 1
    figs = est.generate_figures(None, None, plot_data=plot_data)
    assert set(figs) == FIGURES


def test_axis_order_is_irrelevant():
    ds = _make_map()
    a = _fit(ds)["stark_slope_hz"]
    b = _fit(ds.transpose("detuning", "amp_prefactor"))["stark_slope_hz"]
    assert a == pytest.approx(b)


@pytest.mark.parametrize("dims", [("detuning",), ("amp_prefactor",),
                                  ("amp_prefactor", "detuning")])
def test_sweep_direction_cannot_change_the_answer(order_free, dims):
    """Either axis (or both) walked high -> low gives the identical fit - the
    per-row curve, the absolute line and the twin axis included - and the right
    one."""
    ds = _make_map().assign_coords(
        digital_amp=("amp_prefactor", 0.08 * AMPS),
        full_freq=("detuning", DETUNING + 5.1e9))
    res = order_free(AcStarkShiftEstimator(), ds, dims, chi_hz=-1.0e6, amp_ref=0.08,
                     twin_coord="digital_amp", twin_label="absolute")
    assert res["stark_slope_hz"] == pytest.approx(SLOPE, rel=0.02)


def test_the_twin_amplitude_rides_the_plot_data():
    ds = _make_map().assign_coords(digital_amp=("amp_prefactor", 0.08 * AMPS))
    est = AcStarkShiftEstimator()
    res = est.extract_parameters(ds, twin_coord="digital_amp", twin_label="absolute")
    plot_data = est.build_plot_data(ds, res)
    assert plot_data["twin"].values == pytest.approx(0.08 * AMPS)
    assert plot_data.attrs["twin_label"] == "absolute"
    assert set(est.generate_figures(None, None, plot_data=plot_data)) == FIGURES


def test_the_stored_ground_blob_is_the_reference():
    ds = _make_map().assign(ref_pos_g_i=0.1, ref_pos_g_q=0.05,
                            ref_pos_e_i=0.6, ref_pos_e_q=0.05)
    res = _fit(ds)
    assert res["ref_source"] == "stored"
    assert res["stark_slope_hz"] == pytest.approx(SLOPE, rel=0.02)


def test_full_freq_gives_the_absolute_zero_photon_line():
    drive = 5.1e9
    ds = _make_map().assign_coords(full_freq=("detuning", DETUNING + drive))
    res = _fit(ds)
    assert res["intercept_freq_hz"] == pytest.approx(drive + res["intercept_detuning_hz"])
