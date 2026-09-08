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


#: dressed-state Stark phase params: quadratic while W << D, near-linear beyond.
#: Tuned so the full turn lands mid-window (~a = 0.49) with the sweep running well
#: into saturation — the regime where the global quadratic and the measured curve
#: give genuinely different answers, as they do on 5Q4C.
SAT_GAIN, SAT_RABI, SAT_DETUNING = 3.0, 6.0, 1.0


def _saturating_phase(amp):
    """``phi(a)`` in the SATURATING regime — the real one at 50 MHz detuning.

    ``gain * (sqrt(D**2 + W**2) - D)`` with ``W = rabi * a``: the dressed-state
    shift, whose small-drive limit is the ``k*a**2`` the estimator fits. A global
    quadratic over a window that leaves that limit is materially wrong, which is
    exactly what ``amp_2pi`` must not be taken from."""
    w = SAT_RABI * np.asarray(amp, dtype=float)
    return SAT_GAIN * (np.hypot(SAT_DETUNING, w) - SAT_DETUNING)


def _make_saturating_signal(n=41, noise=2e-3, seed=0, amp_max=1.0, sign=1.0):
    """Discriminated dataset whose phase follows :func:`_saturating_phase`."""
    amp = np.linspace(0.0, amp_max, n)
    phi = sign * _saturating_phase(amp)
    rng = np.random.default_rng(seed)
    P = np.stack([0.5 * (1.0 - np.sin(phi)), 0.5 * (1.0 - np.cos(phi))], axis=1)
    P = P + noise * rng.standard_normal((n, 2))
    return xr.Dataset(
        {"signal": (("stark_amp", "meas_basis"), P)},
        coords={"stark_amp": amp, "meas_basis": [0, 1]},
    )


def _amp_at_full_turn():
    """The analytic amplitude where :func:`_saturating_phase` reaches 2*pi."""
    ratio = 2.0 * np.pi / SAT_GAIN + SAT_DETUNING
    return float(np.sqrt(ratio ** 2 - SAT_DETUNING ** 2) / SAT_RABI)


def _with_absolute_amp(ds, baked=0.25):
    """Attach the absolute-amplitude companion coordinate SCQO's probe supplies."""
    return ds.assign_coords(
        digital_amp=("stark_amp", baked * ds["stark_amp"].values))


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

    def test_amp_2pi_comes_from_the_measured_curve_not_the_fit(self):
        # THE point of amp_2pi: on a saturating curve the k*a^2 fit lands between
        # the small-drive and dressed-state regimes, so sqrt(2pi/k) is materially
        # wrong (~18% in amplitude on real 5Q4C data). The reported value must
        # track the measured crossing, not the fit.
        res = QubitStarkPhaseEchoEstimator().extract_parameters(_make_saturating_signal())
        expected = _amp_at_full_turn()
        assert res["amp_2pi"] == pytest.approx(expected, abs=0.01)
        from_fit = np.sqrt(2 * np.pi / res["stark_coeff"])
        assert abs(from_fit - expected) > 0.05  # the two really do disagree here

    def test_amp_2pi_is_a_magnitude_for_a_negative_stark_shift(self):
        # 5Q4C winds NEGATIVE (k = -6.73). The full turn is |phi| = 2pi, so the
        # answer is the same amplitude with either sign of the coefficient.
        res = QubitStarkPhaseEchoEstimator().extract_parameters(
            _make_saturating_signal(sign=-1.0))
        assert res["stark_coeff"] < 0
        assert res["amp_2pi"] == pytest.approx(_amp_at_full_turn(), abs=0.01)

    def test_amp_2pi_nan_when_the_sweep_never_reaches_a_full_turn(self):
        # Extrapolating a SATURATING curve would invent the answer low; refuse.
        res = QubitStarkPhaseEchoEstimator().extract_parameters(_make_signal(k=3.0))
        assert max(abs(res["phase"])) < 2 * np.pi
        assert np.isnan(res["amp_2pi"])

    def test_absolute_amp_companion_reported_in_both_frames(self):
        est = QubitStarkPhaseEchoEstimator()
        ds = _with_absolute_amp(_make_saturating_signal(), baked=0.25)
        res = est.extract_parameters(ds, twin_coord="digital_amp",
                                     twin_label="absolute amplitude (normalized)")
        assert res["twin_label"] == "absolute amplitude (normalized)"
        assert res["twin_values"] == pytest.approx(0.25 * ds["stark_amp"].values)
        # the same crossing, expressed in the absolute frame
        assert res["amp_2pi_twin"] == pytest.approx(0.25 * res["amp_2pi"], rel=1e-6)
        # and it rides plot_data, so a saved plotdata.nc redraws the second axis
        pd = est.build_plot_data(ds, res)
        assert pd["twin"].dims == ("stark_amp",)
        assert pd.attrs["amp_2pi_twin"] == pytest.approx(res["amp_2pi_twin"])
        assert pd.attrs["amp_2pi"] == pytest.approx(res["amp_2pi"])

    def test_absent_or_undrawable_companion_is_simply_not_carried(self):
        est = QubitStarkPhaseEchoEstimator()
        ds = _make_saturating_signal()
        res = est.extract_parameters(ds, twin_coord="digital_amp")  # not present
        assert "twin_values" not in res and "amp_2pi_twin" not in res
        assert "twin" not in est.build_plot_data(ds, res)

    def test_figures_render_without_a_companion_or_a_full_turn(self):
        # Decorations must never cost the raw figure: no twin axis, no crossing.
        est = QubitStarkPhaseEchoEstimator()
        ds = _make_signal(k=3.0)
        res = est.extract_parameters(ds)
        figs = est.generate_figures(ds, res, plot_data=est.build_plot_data(ds, res))
        assert set(figs) == {"qubit_stark_phase_echo", "quadratures", "phasor"}
        plt.close("all")

    def test_figures_render_with_the_absolute_axis(self):
        est = QubitStarkPhaseEchoEstimator()
        ds = _with_absolute_amp(_make_saturating_signal())
        res = est.extract_parameters(ds, twin_coord="digital_amp",
                                     twin_label="absolute amplitude (normalized)")
        figs = est.generate_figures(ds, res, plot_data=est.build_plot_data(ds, res))
        for name in ("qubit_stark_phase_echo", "quadratures"):
            ax = figs[name].axes[0]
            assert ax.get_xlabel().startswith("stark amplitude")  # primary unchanged
            # the secondary top axis (a CHILD of the primary) carries the absolute frame
            labels = [child.get_xlabel() for child in ax.child_axes]
            assert "absolute amplitude (normalized)" in labels
        plt.close("all")

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
