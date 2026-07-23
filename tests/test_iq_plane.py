"""Smoke tests for the shared IQ-plane diagnostic panel (estimators/_iq_plane)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scqat.estimators._iq_plane import has_iq_plane, plot_iq_plane


def _radial_pd(n=101):
    det = np.linspace(-50e6, 50e6, n)
    z = (2.0 - 1.0j) + 0.5 * np.exp(1j * 0.8) / (1 + (det / 5e6) ** 2)
    return xr.Dataset(
        {"iq_i": ("detuning", np.real(z)), "iq_q": ("detuning", np.imag(z))},
        coords={"detuning": det},
        attrs={"ref_iq_real": 2.0, "ref_iq_imag": -1.0, "ref_source": "median"},
    )


def _axial_pd(n=101):
    amp = np.linspace(0, 2, n)
    P = 0.5 - 0.5 * np.cos(np.pi * amp)
    z = (0.4 + 0.1j) + P * 3.0 * np.exp(1j * 1.1)
    return xr.Dataset(
        {"iq_i": ("amp_prefactor", np.real(z)), "iq_q": ("amp_prefactor", np.imag(z))},
        coords={"amp_prefactor": amp},
        attrs={"reduction_method": "pca", "reduction_angle": -1.1},
    )


def _map_pd(n_flux=5, n_det=41):
    flux = np.linspace(-0.1, 0.1, n_flux)
    det = np.linspace(-50e6, 50e6, n_det)
    i = np.random.default_rng(0).standard_normal((n_flux, n_det))
    q = np.random.default_rng(1).standard_normal((n_flux, n_det))
    return xr.Dataset(
        {
            "iq_i": (("flux_bias", "detuning"), i),
            "iq_q": (("flux_bias", "detuning"), q),
            "ref_i": ("flux_bias", i.mean(axis=1)),
            "ref_q": ("flux_bias", q.mean(axis=1)),
        },
        coords={"flux_bias": flux, "detuning": det},
    )


@pytest.mark.parametrize("maker,kind", [
    (_radial_pd, "RADIAL"),
    (_axial_pd, "AXIAL"),
    (_map_pd, "RADIAL"),
])
def test_panel_draws_and_states_its_kind(maker, kind):
    pd = maker()
    assert has_iq_plane(pd)
    fig = plot_iq_plane(pd)
    assert isinstance(fig, plt.Figure)
    # the user must be able to READ whether the reference is radial or axial
    assert kind in fig.axes[0].get_title()
    plt.close(fig)


def test_missing_iq_vars():
    pd = xr.Dataset({"signal": ("x", [1.0, 2.0])}, coords={"x": [0, 1]})
    assert not has_iq_plane(pd)
    with pytest.raises(ValueError):
        plot_iq_plane(pd)


def test_axial_positions_draw_two_stored_blobs():
    """When the axis came from the stored |0>/|1> centroids (pos_* attrs), the
    panel draws both blob positions + the g->e axis."""
    g = 0.4 + 0.1j
    e = g + 3.0 * np.exp(1j * 1.1)
    pd = _axial_pd()
    pd.attrs.update(
        reduction_method="positions", reduction_angle=float(-np.angle(e - g)),
        pos_g_i=float(g.real), pos_g_q=float(g.imag),
        pos_e_i=float(e.real), pos_e_q=float(e.imag),
    )
    fig = plot_iq_plane(pd)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "|0> position (stored)" in labels
    assert "|1> position (stored)" in labels
    assert "axial axis (positions)" in labels
    plt.close(fig)


def test_global_map_ref_draws_single_point_and_titles_scope():
    """ref_scope='global' renders ONE reference star (not a degenerate per-slice
    trajectory) and says so in the title; absent attr keeps per-slice rendering."""
    pd = _map_pd()
    pd.attrs["ref_scope"] = "global"
    pd["ref_i"].values[:] = 0.25  # a global ref is one constant echoed per slice
    pd["ref_q"].values[:] = -0.5
    fig = plot_iq_plane(pd)
    assert "global" in fig.axes[0].get_title()
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert "global radial ref (median)" in labels
    plt.close(fig)
