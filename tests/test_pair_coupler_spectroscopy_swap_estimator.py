"""Synthetic tests for the coupler swap-spectroscopy estimator.

Two arms per frequency: the ramp arm, where the coupler excitation is swapped into
the probe (a Lorentzian at f_c on the probe's population), and the reference arm,
where it is not. A DECOY - a feature exciting the probe directly - sits in both arms;
the estimator must see through it when it cancels, and flag it when it does not.
Numbers are 5Q4C q1_q2's: a 500 MHz window around the 6.80 GHz prediction, 2 MHz
steps, 300 shots.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_coupler_spectroscopy_swap import (
    PairCouplerSpectroscopySwapEstimator,
)

LABELS = ["00", "01", "10", "11"]
FREQ = np.linspace(6.55e9, 7.05e9, 251)


def _line(f, center, height, fwhm):
    return height / (1 + ((f - center) / (fwhm / 2)) ** 2)


def _dataset(*, f_c=6.8123e9, height=0.35, fwhm=6e6, probe="high", decoy=None,
             second=None, freq=FREQ, shots=300, seed=3):
    """``decoy = (center, ramp_height, reference_height)``; ``second = (center,
    height)`` adds a second ramp-only line (a two-photon transition)."""
    rng = np.random.default_rng(seed)
    other = "low" if probe == "high" else "high"
    arms = {}
    for arm in (1, 0):
        p = {probe: np.full(freq.size, 0.02), other: np.full(freq.size, 0.02)}
        if arm and f_c is not None:
            p[probe] = p[probe] + _line(freq, f_c, height, fwhm)
        if arm and second is not None:
            p[probe] = p[probe] + _line(freq, second[0], second[1], fwhm)
        if decoy is not None:
            p[probe] = p[probe] + _line(freq, decoy[0], decoy[1] if arm else decoy[2], 4e6)
        ph, pl = np.clip(p["high"], 0, 1), np.clip(p["low"], 0, 1)
        probs = np.stack([(1 - ph) * (1 - pl), (1 - ph) * pl, ph * (1 - pl), ph * pl])
        arms[arm] = np.stack([rng.multinomial(shots, probs[:, j]) / shots
                              for j in range(freq.size)], axis=1)
    joint = np.stack([arms[1], arms[0]], axis=1)   # (joint_state, ramp_played, freq)
    return xr.Dataset(
        {"joint_population": (("joint_state", "ramp_played", "tone_freq_hz"), joint)},
        coords={"joint_state": LABELS, "ramp_played": [1, 0], "tone_freq_hz": freq})


def _fit(ds, **kwargs):
    return PairCouplerSpectroscopySwapEstimator().extract_parameters(ds, **kwargs)


def test_the_coupler_line_is_found_through_a_cancelling_decoy():
    r = _fit(_dataset(decoy=(6.70e9, 0.3, 0.3)))
    assert r["success"]
    assert r["f_c_hz"] == pytest.approx(6.8123e9, abs=1e6)
    assert r["fwhm_hz"] == pytest.approx(6e6, rel=0.3)
    assert r["peak_height"] == pytest.approx(0.35, rel=0.2)
    assert r["snr"] > 10
    assert not r["reference_feature"] and not r["peak_at_edge"] and not r["no_peak"]
    # the decoy shows in the reference arm and is reported there, far from f_c
    assert any(abs(f - 6.70e9) < 2e6 for f in r["reference_peaks_hz"])


def test_a_line_the_reference_arm_shares_is_not_the_coupler():
    """No coupler line at all, and a decoy the ramp arm happens to show more of:
    the difference peaks at the decoy, and the reference arm gives it away."""
    r = _fit(_dataset(f_c=None, decoy=(6.70e9, 0.45, 0.25)))
    assert r["reference_feature"] == 1 and not r["success"]
    assert r["f_c_hz"] == pytest.approx(6.70e9, abs=2e6)


def test_the_order_is_not_a_result(order_free):
    ascending = _dataset(decoy=(6.70e9, 0.3, 0.3)).sortby("ramp_played")  # fixture: [0, 1]
    up = order_free(PairCouplerSpectroscopySwapEstimator(), ascending,
                    ("tone_freq_hz", "ramp_played"))
    assert up["success"]


def test_a_second_ramp_only_line_is_listed():
    """At high power the coupler's two-photon 0-2 line sits ~100 MHz below."""
    r = _fit(_dataset(second=(6.71e9, 0.12)))
    assert r["success"] and r["multiple_peaks"] == 1
    assert r["f_c_hz"] == pytest.approx(6.8123e9, abs=1e6)       # the stronger one
    assert any(abs(f - 6.71e9) < 2e6 for f in r["other_peaks_hz"])


def test_the_low_member_can_be_the_probe():
    r = _fit(_dataset(probe="low"), probe="low")
    assert r["success"] and r["f_c_hz"] == pytest.approx(6.8123e9, abs=1e6)
    # read as the wrong member there is nothing to find
    assert _fit(_dataset(probe="low"), probe="high")["no_peak"] == 1


def test_a_line_at_the_window_edge_fails():
    r = _fit(_dataset(f_c=6.553e9))
    assert r["peak_at_edge"] == 1 and not r["success"]


def test_unknown_kwargs_and_a_missing_arm_raise():
    ds = _dataset()
    with pytest.raises(ValueError, match="unknown kwargs"):
        _fit(ds, min_snrr=3)
    est = PairCouplerSpectroscopySwapEstimator()
    with pytest.raises(ValueError, match="both arms"):
        est.analyze(ds.sel(ramp_played=[1]))


def test_figures_render_without_a_peak(tmp_path):
    est = PairCouplerSpectroscopySwapEstimator()
    results, figures = est.analyze(_dataset(f_c=None), output_dir=str(tmp_path),
                                   lo_hz=6.8e9, ramp_duration_ns=400.0,
                                   high_name="q1", low_name="q2")
    assert results["no_peak"] == 1 and not results["success"]
    assert set(figures) == {"spectrum"}
    plot = xr.open_dataset(tmp_path / "pair_coupler_spectroscopy_swap_plotdata.nc")
    assert plot.attrs["high_name"] == "q1" and plot.attrs["lo_hz"] == 6.8e9
    plot.close()
    assert (tmp_path / "pair_coupler_spectroscopy_swap_metadata.json").exists()
