"""Multi-method ResonatorSpectroscopyEstimator tests.

Covers the strategy-dispatch contract: COMMON_KEYS enforcement, per-method
recovery on synthetic data, cross-method agreement (method changes robustness,
not physics), metadata bulk-dropping, and the netCDF plot-data roundtrip that
proves offline replot dispatch works from the saved file alone.
"""

import json
import os

import numpy as np
import pytest
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scqat.estimators.resonator_spectroscopy import ResonatorSpectroscopyEstimator
from scqat.estimators.resonator_spectroscopy.estimator import COMMON_KEYS
from scqat.estimators.resonator_spectroscopy.methods import METHODS
from scqat.tools.fit_notch_circle import notch_s21

FR, QL, ABSQC = 7.2e9, 3000.0, 4200.0
FWHM_TRUE = FR / QL  # 2.4 MHz


def _notch_dataset(fr=FR, Ql=QL, absQc=ABSQC, phi0=0.02, a=0.9, alpha=0.5,
                   delay=30e-9, lo_offset=-1.5e6, span=12e6, n=241,
                   noise=0.006, seed=11, with_full_freq=True):
    """Synthetic notch S21 as an estimator-contract dataset (I/Q + detuning).

    The LO sits at ``fr + lo_offset``, so the dip appears at detuning
    ``-lo_offset``. ``phi0`` defaults small so the magnitude dip is nearly
    symmetric (fair ground for both methods)."""
    lo = fr + lo_offset
    det = np.linspace(-span / 2, span / 2, n)
    f = lo + det
    rng = np.random.default_rng(seed)
    z = notch_s21(f, fr, Ql, absQc, phi0, a=a, alpha=alpha, delay=delay)
    z = z + rng.normal(0, noise, n) + 1j * rng.normal(0, noise, n)
    coords = {"detuning": det}
    if with_full_freq:
        coords["full_freq"] = ("detuning", f)
    return xr.Dataset(
        {"I": ("detuning", z.real), "Q": ("detuning", z.imag)}, coords=coords
    )


class TestDispatch:

    def test_unknown_method_raises(self):
        est = ResonatorSpectroscopyEstimator()
        with pytest.raises(ValueError, match="lorentzian"):
            est.extract_parameters(_notch_dataset(), method="nope")

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_common_keys_and_provenance(self, method):
        est = ResonatorSpectroscopyEstimator()
        results = est.extract_parameters(_notch_dataset(), method=method)
        assert COMMON_KEYS <= results.keys()
        assert results["method"] == method
        assert "full_freq" in results  # coord present -> absolute freq reported

    def test_circle_requires_full_freq(self):
        est = ResonatorSpectroscopyEstimator()
        with pytest.raises(ValueError, match="full_freq"):
            est.extract_parameters(
                _notch_dataset(with_full_freq=False), method="circle"
            )

    def test_lorentzian_works_without_full_freq(self):
        est = ResonatorSpectroscopyEstimator()
        results = est.extract_parameters(
            _notch_dataset(with_full_freq=False), method="lorentzian"
        )
        assert results["success"]
        assert "full_freq" not in results


class TestLorentzianMethod:

    def test_recovery_on_sloped_background(self):
        """The failure mode that motivated the joint fit: the environment
        (a, delay) tilts/curves the power background; centre and width must
        come out unbiased anyway (symmetric line: phi0=0)."""
        ds = _notch_dataset(phi0=0.0, seed=3)
        est = ResonatorSpectroscopyEstimator()
        r = est.extract_parameters(ds, method="lorentzian", baseline_order=2)
        assert r["success"]
        assert r["detuning"] == pytest.approx(1.5e6, abs=0.1 * FWHM_TRUE)
        assert r["fwhm"] == pytest.approx(FWHM_TRUE, rel=0.15)
        assert r["full_freq"] == pytest.approx(FR, abs=0.1 * FWHM_TRUE)
        # extras present
        assert "amplitude" in r and "bg_c1" in r and "baseline_order" in r

    def test_noise_only_fails_gracefully(self):
        rng = np.random.default_rng(0)
        n = 201
        det = np.linspace(-6e6, 6e6, n)
        ds = xr.Dataset(
            {"I": ("detuning", 1.0 + rng.normal(0, 0.01, n)),
             "Q": ("detuning", rng.normal(0, 0.01, n))},
            coords={"detuning": det},
        )
        est = ResonatorSpectroscopyEstimator()
        r = est.extract_parameters(ds, method="lorentzian")
        # no dip: either the fit fails or the "dip" is a noise wiggle; the
        # result must stay a valid contract dict either way
        assert COMMON_KEYS <= r.keys()
        assert isinstance(r["success"], bool)


class TestCircleMethod:

    def test_recovery_and_extras(self):
        ds = _notch_dataset(phi0=0.2, seed=5)
        est = ResonatorSpectroscopyEstimator()
        r = est.extract_parameters(ds, method="circle")
        assert r["success"]
        assert r["detuning"] == pytest.approx(1.5e6, abs=0.05 * FWHM_TRUE)
        assert r["fwhm"] == pytest.approx(FWHM_TRUE, rel=0.1)
        assert r["full_freq"] == pytest.approx(FR, abs=0.05 * FWHM_TRUE)
        assert r["Ql"] == pytest.approx(QL, rel=0.1)
        qi_true = 1.0 / (1.0 / QL - np.cos(0.2) / ABSQC)
        assert r["Qi_dia_corr"] == pytest.approx(qi_true, rel=0.15)
        assert r["delay"] == pytest.approx(30e-9, rel=0.1)

    def test_phaseless_data_fails_gracefully(self):
        """Simulated-backend style data (Q = pure noise) has no meaningful
        phase; the degenerate circle fit must be gated out by the linewidth
        resolution floor, not reported as success."""
        rng = np.random.default_rng(1)
        n = 201
        det = np.linspace(-5e6, 5e6, n)
        mag = 1.0 - 0.8 / (1.0 + (det / 0.7e6) ** 2)
        ds = xr.Dataset(
            {"I": ("detuning", mag + rng.normal(0, 0.01, n)),
             "Q": ("detuning", rng.normal(0, 0.01, n))},
            coords={"detuning": det, "full_freq": ("detuning", det + 6.0e9)},
        )
        est = ResonatorSpectroscopyEstimator()
        r = est.extract_parameters(ds, method="circle")
        assert COMMON_KEYS <= r.keys()
        assert r["success"] is False


class TestCrossMethodAgreement:

    def test_same_physics_both_methods(self):
        """Method choice changes robustness, not physics: on a (nearly)
        symmetric line both must report the same resonance and linewidth."""
        ds = _notch_dataset(phi0=0.02, seed=7)
        est = ResonatorSpectroscopyEstimator()
        r_lor = est.extract_parameters(ds, method="lorentzian")
        r_cir = est.extract_parameters(ds, method="circle")
        assert r_lor["success"] and r_cir["success"]
        assert r_lor["detuning"] == pytest.approx(r_cir["detuning"], abs=0.1 * FWHM_TRUE)
        assert r_lor["fwhm"] == pytest.approx(r_cir["fwhm"], rel=0.15)


class TestArtifacts:

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_metadata_drops_bulk_and_is_json_safe(self, method, tmp_path):
        est = ResonatorSpectroscopyEstimator()
        results = est.extract_parameters(_notch_dataset(), method=method)
        metadata = est.extract_metadata(results)
        for bulky in METHODS[method].bulky_keys:
            assert bulky not in metadata
        est.save_metadata(metadata, str(tmp_path))
        with open(os.path.join(str(tmp_path), "resonator_spectroscopy_metadata.json"),
                  encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["method"] == method
        assert loaded["estimator_name"] == "resonator_spectroscopy"

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_plotdata_netcdf_roundtrip_and_replot(self, method, tmp_path):
        """plot_data must survive to_netcdf (no complex vars) and the figure
        must redraw from the loaded file alone (attrs-based dispatch)."""
        ds = _notch_dataset()
        est = ResonatorSpectroscopyEstimator()
        results, figs = est.analyze(ds, output_dir=str(tmp_path), method=method)
        for fig in figs.values():
            plt.close(fig)

        loaded = est.load_plot_data(str(tmp_path))
        assert loaded.attrs["method"] == method
        for name, var in loaded.data_vars.items():
            assert not np.iscomplexobj(var.values), f"complex var {name!r} in plot_data"

        figs2 = est.generate_figures(None, None, plot_data=loaded)
        assert set(figs2) == {"resonator_spectroscopy"}
        for fig in figs2.values():
            plt.close(fig)


@pytest.mark.parametrize("method", ["lorentzian", "circle"])
def test_sweep_direction_cannot_change_the_answer(order_free, method):
    """The readout window walked high -> low gives the identical fit in both
    methods (the circle one used to RAISE on it) - and the right one."""
    results = order_free(ResonatorSpectroscopyEstimator(), _notch_dataset(),
                         "detuning", method=method)
    assert results["success"]
    assert results["full_freq"] == pytest.approx(FR, abs=0.1 * FR / QL)
