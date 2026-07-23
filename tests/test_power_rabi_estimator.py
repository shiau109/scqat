"""Tests for the PowerRabiEstimator cosine fit / pi-amplitude extraction.

Synthesises a single-pulse power-Rabi amplitude sweep and checks the estimator
recovers the pi-pulse amplitude prefactor, exposes the ``opt_amp_prefactor`` /
``success`` fields the LCHQMDriver node relies on, and that ``analyze`` round-trips
its metadata + plot-data artifacts.

Convention: ``signal`` is high at zero amplitude (ground-state readout) and dips at
the pi pulse — i.e. ``0.5 + 0.5*cos(pi * x / factor_pi)`` — matching the reused
qualibration_libs derivation ``opt = (pi - phase) / (2*pi*f)`` (first minimum of the
fitted cosine). The pi pulse then sits at ``x = factor_pi``.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest

from scqat.estimators import PowerRabiEstimator
from scqat.estimators.power_rabi import PowerRabiEstimator as SubpkgEstimator


def _make_ds(factor_pi=0.8, n=200, amp_min=0.0, amp_max=1.99, noise=2e-3, seed=0, rising=False):
    """Power-Rabi sweep with a pi pulse at ``x = factor_pi``.

    ``rising=False`` is the ground-readout convention (signal high at zero amplitude,
    dipping at the pi pulse); ``rising=True`` is the opposite sign (signal low at zero,
    peaking at the pi pulse) — the case that traps a fixed phi=0 / a>=0 fit at a~0.
    """
    amp_prefactor = np.linspace(amp_min, amp_max, n)
    sign = -1.0 if rising else 1.0
    signal = 0.5 + sign * 0.5 * np.cos(np.pi * amp_prefactor / factor_pi)
    rng = np.random.default_rng(seed)
    signal = signal + noise * rng.standard_normal(n)
    return xr.Dataset(
        {"signal": ("amp_prefactor", signal)},
        coords={"amp_prefactor": amp_prefactor},
    )


def _make_ds_iq(factor_pi=0.8, theta=0.7, n=200, amp_max=1.99, sep=3.0, noise=2e-3, seed=0):
    """Power-Rabi placed in the IQ plane at readout rotation ``theta``: the excited
    fraction ``P = 0.5 - 0.5*cos(pi*x/factor_pi)`` runs the cloud along the g->e axis.
    The reduction (axial/PCA) must recover the same cosine for any theta."""
    amp = np.linspace(0.0, amp_max, n)
    P = 0.5 - 0.5 * np.cos(np.pi * amp / factor_pi)
    d = sep * np.exp(1j * theta)
    pos0 = 0.5 - 0.2j
    rng = np.random.default_rng(seed)
    z = pos0 + P * d + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return xr.Dataset(
        {"I": ("amp_prefactor", np.real(z)), "Q": ("amp_prefactor", np.imag(z))},
        coords={"amp_prefactor": amp},
    )


class TestPowerRabiEstimator:

    def test_imports_match(self):
        assert PowerRabiEstimator is SubpkgEstimator
        assert PowerRabiEstimator.estimator_name == "power_rabi"

    @pytest.mark.parametrize("rising", [False, True])
    def test_recovers_pi_prefactor(self, rising):
        # Both readout signs must recover the pi prefactor: 'rising' is the case that
        # traps a fixed phi=0 / a>=0 cosine fit at a flat (a~0) line.
        factor_pi = 0.8
        ds = _make_ds(factor_pi=factor_pi, rising=rising)
        results = PowerRabiEstimator().extract_parameters(ds)
        assert results["success"] is True
        assert results["opt_amp_prefactor"] == pytest.approx(factor_pi, rel=0.05)
        # f corresponds to a half-period at the pi pulse: f = 1 / (2 * factor_pi).
        assert results["f"] == pytest.approx(1.0 / (2 * factor_pi), rel=0.05)
        # The fit must actually track the oscillation (no degenerate a~0 line).
        assert abs(results["a"]) > 0.2

    @pytest.mark.parametrize("theta", [0.0, 0.7, 1.9, -2.5])
    def test_recovers_pi_prefactor_from_iq(self, theta):
        # The axial/PCA reduction must recover the pi prefactor from raw I/Q at any
        # readout rotation — the case the old rename(I->signal) got wrong.
        factor_pi = 0.8
        res = PowerRabiEstimator().extract_parameters(_make_ds_iq(factor_pi=factor_pi, theta=theta))
        assert res["success"] is True
        assert res["opt_amp_prefactor"] == pytest.approx(factor_pi, rel=0.07)
        assert res["reduction_method"] == "pca"

    def test_check_data_requires_signal_and_coord(self):
        est = PowerRabiEstimator()
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"I": ("amp_prefactor", [0, 1])},
                                       coords={"amp_prefactor": [0, 1]}))
        with pytest.raises(ValueError):
            est._check_data(xr.Dataset({"signal": ("x", [0, 1])}, coords={"x": [0, 1]}))

    def test_metadata_drops_arrays(self):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results = est.extract_parameters(ds)
        meta = est.extract_metadata(results)
        assert "best_fit" not in meta and "fit_report" not in meta
        assert {"a", "f", "phi", "c", "opt_amp_prefactor", "success"} <= set(meta)

    def test_plot_data_layout(self):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results = est.extract_parameters(ds)
        pd = est.build_plot_data(ds, results)
        assert "signal" in pd and "best_fit" in pd
        assert pd["signal"].dims == ("amp_prefactor",)
        assert pd.attrs["opt_amp_prefactor"] == pytest.approx(results["opt_amp_prefactor"])

    def test_analyze_roundtrip(self, tmp_path):
        ds = _make_ds()
        est = PowerRabiEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path))
        assert (tmp_path / "power_rabi_metadata.json").exists()
        assert (tmp_path / "power_rabi_plotdata.nc").exists()
        # pre-reduced 'signal' input -> no IQ cloud -> no iq_plane panel
        assert set(figs) == {"amplitude"}
        assert isinstance(figs["amplitude"], plt.Figure)
        plt.close("all")

    def test_analyze_roundtrip_iq_adds_iq_plane(self, tmp_path):
        est = PowerRabiEstimator()
        results, figs = est.analyze(_make_ds_iq(), output_dir=str(tmp_path))
        assert set(figs) == {"amplitude", "iq_plane"}
        assert (tmp_path / "power_rabi_iq_plane.png").exists()
        pd = est.load_plot_data(str(tmp_path))
        assert "iq_i" in pd and "iq_q" in pd
        assert pd.attrs["reduction_method"] == "pca"
        plt.close("all")

    def test_angle_zero_is_preserved_in_plot_data(self):
        """angle=0 (readout axis already on I) is a real value — it must reach the
        plotdata attrs as 0.0, never be coerced to NaN (falsy-zero regression)."""
        est = PowerRabiEstimator()
        ds = _make_ds_iq(theta=0.0)
        results = est.extract_parameters(ds, angle=0.0)
        assert results["reduction_method"] == "angle"
        assert results["reduction_angle"] == 0.0
        pd = est.build_plot_data(ds, results)
        assert pd.attrs["reduction_angle"] == 0.0
