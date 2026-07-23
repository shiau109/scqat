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
