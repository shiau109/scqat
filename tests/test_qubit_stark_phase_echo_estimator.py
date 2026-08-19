"""Tests for the QubitStarkPhaseEchoEstimator.

The probe measures a Hahn echo with an off-resonant Stark tone in the second
arm, read out in two bases: x90 -> <Z> = sin(phi), -y90 -> <Z> = cos(phi), with
the AC-Stark phase phi ~ k * stark_amp**2. These tests synthesise that data
(as discriminated population and as raw I/Q at an arbitrary readout rotation)
and check that the estimator recovers the Stark coefficient k, plus the data
guards and the analyze() artifact round-trip.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import QubitStarkPhaseEchoEstimator
from scqat.estimators.qubit_stark_phase_echo import (
    QubitStarkPhaseEchoEstimator as SubpkgEstimator,
)


def _populations(amp, k, phi0=0.0):
    """(n_amp, 2) populations: basis 0 = x90 (sin), basis 1 = -y90 (cos).

    ``phi0`` is a constant phase offset at amp=0 (a residual echo/readout phase)."""
    phi = phi0 + k * amp ** 2
    p_sin = 0.5 * (1.0 - np.sin(phi))  # x90  close -> <Z> = sin(phi)
    p_cos = 0.5 * (1.0 - np.cos(phi))  # -y90 close -> <Z> = cos(phi)
    return np.stack([p_sin, p_cos], axis=1)


def _make_signal(k=5.0, n=21, noise=2e-3, seed=0, amp_min=0.0, amp_max=1.0, phi0=0.0):
    """Discriminated averaged population dataset (variable ``signal``)."""
    amp = np.linspace(amp_min, amp_max, n)
    rng = np.random.default_rng(seed)
    P = _populations(amp, k, phi0) + noise * rng.standard_normal((n, 2))
    return xr.Dataset(
        {"signal": (("stark_amp", "meas_basis"), P)},
        coords={"stark_amp": amp, "meas_basis": [0, 1]},
    )


def _make_iq(k=5.0, n=21, theta=0.6, sep=3.0, noise=2e-3, seed=0, amp_min=0.0, amp_max=1.0, phi0=0.0):
    """Raw I/Q dataset: both bases share one ground center + g->e vector."""
    amp = np.linspace(amp_min, amp_max, n)
    rng = np.random.default_rng(seed)
    P = _populations(amp, k, phi0)
    pos0 = 0.3 - 0.7j
    d = sep * np.exp(1j * theta)
    z = pos0 + P * d + noise * (rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2)))
    return xr.Dataset(
        {"I": (("stark_amp", "meas_basis"), np.real(z)),
         "Q": (("stark_amp", "meas_basis"), np.imag(z))},
        coords={"stark_amp": amp, "meas_basis": [0, 1]},
    )


class TestQubitStarkPhaseEchoEstimator:

    def test_imports_match(self):
        assert QubitStarkPhaseEchoEstimator is SubpkgEstimator
        assert QubitStarkPhaseEchoEstimator.estimator_name == "qubit_stark_phase_echo"

    def test_recovers_coefficient_from_signal(self):
        res = QubitStarkPhaseEchoEstimator().extract_parameters(_make_signal(k=5.0))
        assert res["success"] is True
        assert res["stark_coeff"] == pytest.approx(5.0, rel=0.05)
        # phi anchored to 0 at the smallest amplitude.
        assert res["phase"][0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("theta", [0.0, 0.8, -1.7])
    def test_recovers_coefficient_from_iq(self, theta):
        # Any readout rotation: the pooled axial reduction + phasor is rotation-robust.
        res = QubitStarkPhaseEchoEstimator().extract_parameters(_make_iq(k=5.0, theta=theta))
        assert res["success"] is True
        assert res["stark_coeff"] == pytest.approx(5.0, rel=0.07)
        assert res["reduction_method"] == "pca"

    def test_sign_of_coefficient_follows_physics(self):
        # A negative Stark coefficient (phi winds the other way) is reported negative.
        res = QubitStarkPhaseEchoEstimator().extract_parameters(_make_signal(k=-4.0))
        assert res["stark_coeff"] == pytest.approx(-4.0, rel=0.05)

    def test_recovers_coefficient_symmetric_sweep(self):
        # phi ~ k*a^2 is minimized at a=0 (the MIDDLE of a -1..1 sweep), not at the
        # first point -- the anchor must find the smallest |amp|, not index 0.
        ds = _make_signal(k=5.0, n=41, amp_min=-1.0, amp_max=1.0)
        res = QubitStarkPhaseEchoEstimator().extract_parameters(ds)
        assert res["success"] is True
        assert res["stark_coeff"] == pytest.approx(5.0, rel=0.05)
        i0 = int(np.argmin(np.abs(ds["stark_amp"].values)))
        assert res["phase"][i0] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("phi0", [np.pi / 2, 2.0, -1.3])
    def test_recovers_coefficient_with_phase_offset(self, phi0):
        # A residual echo/readout phase at amp=0 puts the anchor anywhere on the
        # circle (not at sin=0). The circle fit measures phase about the true
        # center, so the coefficient is still recovered (this broke the old
        # single-anchor offset estimate -> distorted, folded phase).
        ds = _make_signal(k=5.0, n=41, amp_min=-1.0, amp_max=1.0, phi0=phi0)
        res = QubitStarkPhaseEchoEstimator().extract_parameters(ds)
        assert res["success"] is True
        assert res["stark_coeff"] == pytest.approx(5.0, rel=0.05)

    def test_stored_positions_resolve_axis(self):
        theta, sep = 0.6, 3.0
        ds = _make_iq(k=5.0, theta=theta, sep=sep)
        pos0 = 0.3 - 0.7j
        pos1 = pos0 + sep * np.exp(1j * theta)
        ds["ref_pos_g_i"], ds["ref_pos_g_q"] = float(pos0.real), float(pos0.imag)
        ds["ref_pos_e_i"], ds["ref_pos_e_q"] = float(pos1.real), float(pos1.imag)
        res = QubitStarkPhaseEchoEstimator().extract_parameters(ds)
        assert res["reduction_method"] == "positions"
        assert res["stark_coeff"] == pytest.approx(5.0, rel=0.07)

    def test_check_data_requires_coords_and_two_bases(self):
        est = QubitStarkPhaseEchoEstimator()
        with pytest.raises(ValueError):  # no stark_amp
            est._check_data(xr.Dataset({"signal": ("meas_basis", [0, 1])},
                                       coords={"meas_basis": [0, 1]}))
        with pytest.raises(ValueError):  # no meas_basis
            est._check_data(xr.Dataset({"signal": ("stark_amp", [0, 1])},
                                       coords={"stark_amp": [0, 1]}))
        with pytest.raises(ValueError):  # wrong number of bases
            est._check_data(xr.Dataset(
                {"signal": (("stark_amp", "meas_basis"), np.zeros((2, 3)))},
                coords={"stark_amp": [0, 1], "meas_basis": [0, 1, 2]}))

    def test_metadata_drops_arrays(self):
        est = QubitStarkPhaseEchoEstimator()
        res = est.extract_parameters(_make_signal())
        meta = est.extract_metadata(res)
        for k in ("s_sin", "s_cos", "best_fit"):
            assert k not in meta
        assert {"stark_coeff", "intercept", "phase", "success"} <= set(meta)

    def test_plot_data_layout(self):
        est = QubitStarkPhaseEchoEstimator()
        res = est.extract_parameters(_make_signal())
        pd = est.build_plot_data(_make_signal(), res)
        assert {"s_sin", "s_cos", "phase", "best_fit"} <= set(pd.data_vars)
        assert pd["phase"].dims == ("stark_amp",)
        assert pd.attrs["success"] == 1

    def test_analyze_roundtrip(self, tmp_path):
        est = QubitStarkPhaseEchoEstimator()
        res, figs = est.analyze(_make_iq(), output_dir=str(tmp_path))
        assert (tmp_path / "qubit_stark_phase_echo_metadata.json").exists()
        assert (tmp_path / "qubit_stark_phase_echo_plotdata.nc").exists()
        assert set(figs) == {"qubit_stark_phase_echo", "quadratures", "phasor"}
        assert isinstance(figs["qubit_stark_phase_echo"], plt.Figure)
        assert isinstance(figs["phasor"], plt.Figure)
        plt.close("all")
