"""Tests for the QubitSpectroscopyFluxEstimator's artifact contract, focused on
the per-peak POLARITY flag.

``fit_peaks`` normalizes polarity per slice — when the negative branch wins it
negates the trace and fits a POSITIVE Lorentzian — so ``peak_amplitude`` alone
cannot tell an absorption dip from an emission peak, and a negative value there
means a badly-conditioned fit rather than a dip. ``peak_inverted`` is what
carries the distinction through ``track_peaks``' pooling; these tests pin it
across the stage function, the estimator's metadata projection, the plot_data
Dataset, and the saved JSON + netCDF a downstream repo reloads.

``ref_scope`` (the other axis of this estimator's kwarg surface) is covered in
``test_flux_ref_scope.py``.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.estimators import QubitSpectroscopyFluxEstimator
from scqat.estimators.qubit_spectroscopy_flux import (
    QubitSpectroscopyFluxEstimator as SubpkgEstimator,
    flux_cloud_plotdata,
    track_flux_peaks,
)


def _flux_map_ds(n_flux=7, n_det=161, dip=False, seed=1):
    """A qubit line sweeping the window with flux, as a REAL (discriminated)
    signal — the only input whose polarity can go either way, since the complex
    path reduces radially to ``|IQ - ref|`` (always a positive bump)."""
    flux = np.linspace(-0.2, 0.2, n_flux)
    det = np.linspace(-50e6, 50e6, n_det)
    rng = np.random.default_rng(seed)
    centers = 30e6 * np.sin(np.pi * flux / 0.4)
    state = np.empty((n_flux, n_det))
    for k in range(n_flux):
        line = 0.6 / (1.0 + ((det - centers[k]) / 3e6) ** 2)
        background = 0.9 if dip else 0.1
        state[k] = background + (-line if dip else line) + 3e-3 * rng.standard_normal(n_det)
    return xr.Dataset(
        {"state": (("flux_bias", "detuning"), state)},
        coords={"flux_bias": flux, "detuning": det,
                "full_freq": ("detuning", det + 5e9)},
    )


def test_imports_match():
    assert QubitSpectroscopyFluxEstimator is SubpkgEstimator
    assert QubitSpectroscopyFluxEstimator.estimator_name == "qubit_spectroscopy_flux"


def test_dip_slices_normalize_amplitude_and_flag_polarity():
    ds = _flux_map_ds(dip=True)
    res = track_flux_peaks(ds, signal_var="state")
    assert res["good"].any()
    # The polarity CHOICE is per slice and independent of how well that slice's
    # Lorentzian converged, so every point is flagged.
    assert res["peak_inverted"].all()
    assert res["n_inverted"] == res["n_peaks"]
    # A CONVERGED dip fit reports a POSITIVE amplitude — the sign carries no
    # physics. Select converged fits by centre accuracy against the injected
    # line, never by sign. (Not every slice converges: fit_peaks negates the
    # trace for a dip and then still seeds FitLorentzian with inverted=True, so
    # the guess starts from a noise trough. Tighten to `res["good"]` once that
    # is fixed.)
    flux = ds.coords["flux_bias"].values
    centers = 30e6 * np.sin(np.pi * flux / 0.4)
    on_line = np.abs(res["peak_detuning"] - centers[res["peak_flux_index"]]) < 2e6
    assert on_line.sum() >= 3
    assert (res["peak_amplitude"][on_line] > 0).all()
    # With the flag the signed physics is recoverable.
    signed = np.where(res["peak_inverted"], -res["peak_amplitude"], res["peak_amplitude"])
    assert (signed[on_line] < 0).all()


def test_peak_slices_are_not_flagged_inverted():
    res = track_flux_peaks(_flux_map_ds(), signal_var="state")
    assert res["n_peaks"] > 0
    assert not res["peak_inverted"].any()
    assert res["n_inverted"] == 0


def test_plotdata_carries_polarity_per_peak():
    res = track_flux_peaks(_flux_map_ds(dip=True), signal_var="state")
    pd = flux_cloud_plotdata(res)
    assert pd["peak_inverted"].dims == ("peak",)
    assert pd["peak_inverted"].values.astype(bool).all()
    assert int(pd.attrs["n_inverted"]) == int(pd.attrs["n_peaks"])


def test_metadata_keeps_polarity_and_still_drops_the_maps():
    est = QubitSpectroscopyFluxEstimator()
    res = est.extract_parameters(_flux_map_ds(dip=True), signal_var="state")
    meta = est.extract_metadata(res)
    assert {"amplitude_map", "reduced_map", "detuning", "full_freq"} & set(meta) == set()
    assert {"peak_flux", "peak_detuning", "peak_inverted", "n_inverted"} <= set(meta)


def test_polarity_survives_the_saved_artifacts(tmp_path):
    """A downstream repo reloading the JSON + netCDF can still tell a dip from a
    peak — no re-fit, no unpickling."""
    est = QubitSpectroscopyFluxEstimator()
    _, figs = est.analyze(_flux_map_ds(dip=True), signal_var="state",
                          output_dir=str(tmp_path))
    assert "qubit_spectroscopy_flux" in figs

    meta = json.loads((tmp_path / "qubit_spectroscopy_flux_metadata.json").read_text())
    assert meta["peak_inverted"] == [True] * len(meta["peak_amplitude"])
    assert meta["n_inverted"] == meta["n_peaks"]

    reloaded = xr.load_dataset(tmp_path / "qubit_spectroscopy_flux_plotdata.nc")
    assert reloaded["peak_inverted"].values.astype(bool).all()
    assert int(reloaded.attrs["n_inverted"]) == int(reloaded.attrs["n_peaks"])
    plt.close("all")
