"""The stored-reference path: ``ref_pos_*`` dataset variables -> reduction reference.

``single_shot_readout`` measures the |0>/|1> blob centers; the acquisition layer
(SCQO) attaches them to later datasets as per-target ``ref_pos_g_i/g_q/e_i/e_q``
variables. These tests cover the base helpers (``stored_positions`` /
``stored_ground``) and ``reduced_signal``'s priority chain:
explicit kwarg -> stored positions -> PCA.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.core.base_estimator import (
    reduced_signal,
    stored_ground,
    stored_positions,
)

POS_G = 0.5 - 0.2j
POS_E = POS_G + 3.0 * np.exp(1j * 0.7)


def _iq_ds(n=100, with_positions=True, nan_e=False):
    """Rabi-like sweep running the cloud from POS_G (P=0) toward POS_E (P=1)."""
    amp = np.linspace(0.0, 2.0, n)
    P = 0.5 - 0.5 * np.cos(np.pi * amp / 0.8)
    z = POS_G + P * (POS_E - POS_G)
    ds = xr.Dataset(
        {"I": ("amp_prefactor", np.real(z)), "Q": ("amp_prefactor", np.imag(z))},
        coords={"amp_prefactor": amp},
    )
    if with_positions:
        ds["ref_pos_g_i"] = float(POS_G.real)
        ds["ref_pos_g_q"] = float(POS_G.imag)
        ds["ref_pos_e_i"] = float("nan") if nan_e else float(POS_E.real)
        ds["ref_pos_e_q"] = float(POS_E.imag)
    return ds


def test_stored_positions_reads_all_four():
    pos = stored_positions(_iq_ds())
    assert pos is not None
    assert pos[0] == pytest.approx(POS_G)
    assert pos[1] == pytest.approx(POS_E)
    assert stored_ground(_iq_ds()) == pytest.approx(POS_G)


def test_stored_positions_absent_or_nonfinite_is_none():
    assert stored_positions(_iq_ds(with_positions=False)) is None
    assert stored_ground(_iq_ds(with_positions=False)) is None
    # all four must be finite — a single NaN center invalidates the set
    assert stored_positions(_iq_ds(nan_e=True)) is None
    assert stored_ground(_iq_ds(nan_e=True)) is None


def test_reduced_signal_prefers_stored_positions():
    sig = reduced_signal(_iq_ds())
    assert sig.attrs["reduction_method"] == "positions"
    assert sig.attrs["reduction_angle"] == pytest.approx(-np.angle(POS_E - POS_G))
    # the used centers are stamped for the IQ-plane panel
    assert sig.attrs["pos_g_i"] == pytest.approx(POS_G.real)
    assert sig.attrs["pos_e_q"] == pytest.approx(POS_E.imag)
    # deterministic direction: ground (first sample, P=0) low, excited high
    # (the sampled grid does not hit P=1 exactly -> tolerance ~1e-3)
    vals = np.asarray(sig.values, dtype=float)
    assert vals.max() - vals[0] == pytest.approx(abs(POS_E - POS_G), rel=1e-3)


def test_explicit_kwargs_win_over_stored():
    ds = _iq_ds()
    sig = reduced_signal(ds, angle=0.0)
    assert sig.attrs["reduction_method"] == "angle"
    assert "pos_g_i" not in sig.attrs
    explicit = [0.0 + 0.0j, 1.0 + 0.0j]
    sig2 = reduced_signal(ds, positions=explicit)
    assert sig2.attrs["reduction_method"] == "positions"
    assert sig2.attrs["pos_e_i"] == pytest.approx(1.0)  # the kwarg, not the vars


def test_pre_reduced_signal_ignores_positions():
    ds = _iq_ds()
    ds["signal"] = ("amp_prefactor", np.linspace(0.0, 1.0, ds.sizes["amp_prefactor"]))
    sig = reduced_signal(ds)
    assert sig.attrs["reduction_method"] == "signal"


def test_no_positions_falls_back_to_pca():
    sig = reduced_signal(_iq_ds(with_positions=False))
    assert sig.attrs["reduction_method"] == "pca"
    assert "pos_g_i" not in sig.attrs
