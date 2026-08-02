"""Tests for the XY-Z delay estimator.

Synthetic prepared_state (|g>, |e>) x relative_time data whose |e> - |g>
contrast is a triangle peaked at the true delay. The estimator must recover the
peak on both the discriminated ``signal`` path and the raw I/Q path (via the
shared axial projection), require its ``xy_duration_s`` kwarg, and produce a
plot_data Dataset the figure can be drawn from.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.xyz_delay import XyzDelayEstimator
from scqat.tools.fit_triangle import triangle_peak

T0_NS = 12.0
XY_DURATION_S = 40e-9


def _relative_time_s():
    return np.arange(-60.0, 60.0) * 1e-9


def _contrast(t_s):
    """|e> - |g> triangle peaked at T0_NS, half-width = pi length."""
    return triangle_peak(t_s, amp=0.9, t0=T0_NS * 1e-9, half_width=XY_DURATION_S, offset=0.0)


def _state_dataset():
    t = _relative_time_s()
    contrast = _contrast(t)
    rng = np.random.default_rng(0)
    ground = 0.05 + rng.normal(0.0, 0.01, t.size)
    excited = 0.05 + contrast + rng.normal(0.0, 0.01, t.size)
    signal = np.stack([ground, excited])  # (prepared_state, relative_time)
    return xr.Dataset(
        {"signal": (("prepared_state", "relative_time"), signal)},
        coords={"prepared_state": [0, 1], "relative_time": t},
    )


def _iq_dataset():
    t = _relative_time_s()
    contrast = _contrast(t)
    rng = np.random.default_rng(1)
    # Populations placed on a rotated g->e axis; the estimator's axial reduction
    # must recover the contrast regardless of the readout rotation.
    theta = 0.7
    pos0 = 0.3 + 0.2j
    sep = 3.0
    ground_pop = 0.05 + rng.normal(0.0, 0.01, t.size)
    excited_pop = 0.05 + contrast + rng.normal(0.0, 0.01, t.size)
    zg = pos0 + ground_pop * sep * np.exp(1j * theta)
    ze = pos0 + excited_pop * sep * np.exp(1j * theta)
    i_data = np.stack([np.real(zg), np.real(ze)])
    q_data = np.stack([np.imag(zg), np.imag(ze)])
    return xr.Dataset(
        {"I": (("prepared_state", "relative_time"), i_data),
         "Q": (("prepared_state", "relative_time"), q_data)},
        coords={"prepared_state": [0, 1], "relative_time": t},
    )


class TestXyzDelayEstimator:

    def test_state_path_recovers_delay(self):
        est = XyzDelayEstimator()
        results, _ = est.analyze(_state_dataset(), skip_figures=True, xy_duration_s=XY_DURATION_S)
        assert results["success"] is True
        assert results["t0_s"] == pytest.approx(T0_NS * 1e-9, abs=2e-9)
        assert results["reduction_method"] == "state"

    def test_iq_path_recovers_delay(self):
        est = XyzDelayEstimator()
        results, _ = est.analyze(_iq_dataset(), skip_figures=True, xy_duration_s=XY_DURATION_S)
        assert results["success"] is True
        assert results["t0_s"] == pytest.approx(T0_NS * 1e-9, abs=2e-9)
        assert results["reduction_method"] == "axial"

    def test_requires_xy_duration(self):
        est = XyzDelayEstimator()
        with pytest.raises(ValueError, match="xy_duration_s"):
            est.analyze(_state_dataset(), skip_figures=True)

    def test_unknown_kwarg_rejected(self):
        est = XyzDelayEstimator()
        with pytest.raises(ValueError, match="Unknown"):
            est.analyze(_state_dataset(), skip_figures=True, xy_duration_s=XY_DURATION_S, bogus=1)

    def test_missing_coord_raises(self):
        est = XyzDelayEstimator()
        bad = _state_dataset().drop_vars("prepared_state")
        with pytest.raises(ValueError, match="prepared_state"):
            est.analyze(bad, skip_figures=True, xy_duration_s=XY_DURATION_S)

    def test_plot_data_is_self_sufficient(self):
        est = XyzDelayEstimator()
        results = est.extract_parameters(_state_dataset(), xy_duration_s=XY_DURATION_S)
        plot_data = est.build_plot_data(_state_dataset(), results)
        assert "difference" in plot_data and "best_fit" in plot_data
        assert "relative_time" in plot_data.coords
        assert plot_data.attrs["success"] == 1
        figs = est.generate_figures(None, None, plot_data=plot_data)
        assert "xyz_delay" in figs

    def test_metadata_drops_arrays(self):
        est = XyzDelayEstimator()
        results = est.extract_parameters(_state_dataset(), xy_duration_s=XY_DURATION_S)
        meta = est.extract_metadata(results)
        assert "difference" not in meta and "best_fit" not in meta
        assert "t0_s" in meta
