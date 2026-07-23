"""``ref_scope`` on the qubit-vs-flux stage-1 reduction (``track_flux_peaks``):
``per_slice`` (default — safe when the readout condition moves with flux,
DC-held bias) vs ``global`` (pulsed flux: every slice reads out at idle, so ONE
median of the whole map is valid and more stable)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators._iq_plane import plot_iq_plane
from scqat.estimators.qubit_spectroscopy_flux import flux_cloud_plotdata, track_flux_peaks


def _flux_map_ds(n_flux=9, n_det=161, seed=0):
    """Flux map with a moving Lorentzian line over a FIXED ground point (the
    pulsed-flux acquisition: readout at idle every slice)."""
    flux = np.linspace(-0.2, 0.2, n_flux)
    det = np.linspace(-50e6, 50e6, n_det)
    rng = np.random.default_rng(seed)
    centers = 40e6 * np.sin(np.pi * flux / 0.4)  # the line sweeps the window
    z = np.empty((n_flux, n_det), dtype=complex)
    for k in range(n_flux):
        peak = 0.8 / (1 + ((det - centers[k]) / 3e6) ** 2)
        z[k] = (0.3 - 0.1j) + peak * np.exp(1j * 0.9)
        z[k] += 2e-3 * (rng.standard_normal(n_det) + 1j * rng.standard_normal(n_det))
    return xr.Dataset(
        {"I": (("flux_bias", "detuning"), z.real), "Q": (("flux_bias", "detuning"), z.imag)},
        coords={"flux_bias": flux, "detuning": det, "full_freq": ("detuning", det + 5e9)},
    )


def test_default_is_per_slice():
    res = track_flux_peaks(_flux_map_ds())
    assert res["ref_scope"] == "per_slice"
    assert flux_cloud_plotdata(res).attrs["ref_scope"] == "per_slice"


def test_global_scope_uses_one_map_median():
    ds = _flux_map_ds()
    res = track_flux_peaks(ds, ref_scope="global")
    assert res["ref_scope"] == "global"
    ri, rq = np.asarray(res["ref_i"]), np.asarray(res["ref_q"])
    ok = np.isfinite(ri) & np.isfinite(rq)
    assert ok.any()
    # one constant reference echoed by every row...
    assert np.allclose(ri[ok], ri[ok][0]) and np.allclose(rq[ok], rq[ok][0])
    # ...equal to the complex median of the WHOLE 2-D map
    assert ri[ok][0] == pytest.approx(float(np.median(ds["I"].values)))
    assert rq[ok][0] == pytest.approx(float(np.median(ds["Q"].values)))
    # the line is still found across the map with the global reference
    assert res["n_good"] >= 5
    # the shared panel renders ONE star and titles the scope
    pd = flux_cloud_plotdata(res)
    assert pd.attrs["ref_scope"] == "global"
    fig = plot_iq_plane(pd)
    assert "global" in fig.axes[0].get_title()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "global radial ref (median)" in labels
    plt.close(fig)


def test_ref_scope_guards():
    ds = _flux_map_ds()
    with pytest.raises(ValueError, match="ref_scope"):
        track_flux_peaks(ds, ref_scope="bogus")
    # a real (already-reduced) signal_var has no IQ plane to take a median in
    ds2 = ds.assign(state=(("flux_bias", "detuning"), np.abs(ds["I"].values)))
    with pytest.raises(ValueError, match="complex IQ"):
        track_flux_peaks(ds2, ref_scope="global", signal_var="state")
