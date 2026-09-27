"""Synthetic tests for the coupler ZZ-spectroscopy estimator.

Two arms per frequency: the pi arm, where the pi member gets a pi after the tone
(spoiled by the ZZ while the coupler is excited, so its population DIPS on the
coupler's lines), and the reference arm, where it waits instead. The pi member's
readout may also see the coupler directly, in both arms. Numbers are 5Q4C q1_q2's:
f01 = 7.058 GHz, alpha = -135 MHz, a 500 MHz window at 1 MHz steps, 300 shots. The pi
member sits near 0.9, where the shot noise (~0.02 on the difference) is larger than
experiment 3's, so the planted occupations are 0.45 and 0.3.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_coupler_spectroscopy_zz import PairCouplerSpectroscopyZZEstimator

LABELS = ["00", "01", "10", "11"]
FREQ = np.linspace(6.85e9, 7.35e9, 501)
F01, ALPHA = 7.058e9, -135e6
LADDER = [(F01, 5e6, 0.45), (F01 + ALPHA / 2, 4e6, 0.3)]   # coupler occupation


def _line(f, center, height, fwhm):
    return height / (1 + ((f - center) / (fwhm / 2)) ** 2)


def _dataset(*, coupler=LADDER, contrast=0.88, spoil=0.9, seen=0.0, pi_member="high",
             freq=FREQ, shots=300, seed=3):
    """``coupler`` = ``(center, fwhm, occupation)`` lines of the coupler population the
    tone leaves; the pi arm loses ``spoil`` of its ``contrast`` wherever the coupler
    is excited. ``seen`` of the coupler population shows on the pi member's readout
    in BOTH arms."""
    rng = np.random.default_rng(seed)
    p_c = np.zeros(freq.size)
    for center, fwhm, height in coupler:
        p_c = p_c + _line(freq, center, height, fwhm)
    arms = {}
    for arm in (1, 0):
        mine = 0.02 + seen * p_c + (contrast * (1 - spoil * p_c) if arm else 0.0)
        other = np.full(freq.size, 0.02)
        ph, pl = (mine, other) if pi_member == "high" else (other, mine)
        ph, pl = np.clip(ph, 0, 1), np.clip(pl, 0, 1)
        probs = np.stack([(1 - ph) * (1 - pl), (1 - ph) * pl, ph * (1 - pl), ph * pl])
        arms[arm] = np.stack([rng.multinomial(shots, probs[:, j]) / shots
                              for j in range(freq.size)], axis=1)
    joint = np.stack([arms[1], arms[0]], axis=1)   # (joint_state, pi_played, freq)
    return xr.Dataset(
        {"joint_population": (("joint_state", "pi_played", "tone_freq_hz"), joint)},
        coords={"joint_state": LABELS, "pi_played": [1, 0], "tone_freq_hz": freq})


def _fit(ds, **kwargs):
    return PairCouplerSpectroscopyZZEstimator().extract_parameters(ds, **kwargs)


def test_the_dip_is_f01_and_alpha_comes_from_f02_half():
    r = _fit(_dataset())
    assert r["success"]
    assert r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert r["fwhm_hz"] == pytest.approx(5e6, rel=0.3)
    assert r["dip_depth"] == pytest.approx(0.88 * 0.9 * 0.45, rel=0.2)
    assert r["snr"] > 10
    assert r["alpha_hz"] == pytest.approx(ALPHA, abs=2e6)
    assert r["pi_contrast"] == pytest.approx(0.88, abs=0.03)
    assert (r["n_lines"], r["n_ladder_lines"]) == (2, 2)
    assert not (r["no_line"] or r["unexplained_lines"] or r["peak_at_edge"])


def test_a_single_dip_is_f01_without_alpha():
    r = _fit(_dataset(coupler=LADDER[:1]))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert np.isnan(r["alpha_hz"]) and r["n_ladder_lines"] == 1


def test_the_readout_seeing_the_coupler_cancels_between_the_arms():
    """5Q4C q1's readout sees q1_q2_c: a bump in both arms, gone from the difference."""
    r = _fit(_dataset(seen=0.4))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert r["alpha_hz"] == pytest.approx(ALPHA, abs=2e6)


def test_the_low_member_can_get_the_pi():
    r = _fit(_dataset(pi_member="low"), pi_member="low")
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    # read as the other member there is no pi and nothing to find
    wrong = _fit(_dataset(pi_member="low"), pi_member="high")
    assert wrong["no_line"] == 1 and abs(wrong["pi_contrast"]) < 0.02


def test_a_dip_off_the_ladder_fails():
    r = _fit(_dataset(coupler=LADDER + [(6.88e9, 4e6, 0.3)]))
    assert r["unexplained_lines"] == 1 and not r["success"]
    assert r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    [left_over] = r["unexplained_lines_hz"]
    assert min(abs(left_over - 6.88e9), abs(left_over - (F01 + ALPHA / 2))) < 1e6


def test_a_dip_one_point_wide_is_ignored():
    """Above f01, so kept it would have been called f01."""
    r = _fit(_dataset(coupler=LADDER + [(7.2e9, 0.5e6, 0.6)]))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert not any(abs(f - 7.2e9) < 5e6 for f in r["lines_hz"])


def test_a_poor_pi_makes_the_dip_shallower_not_displaced():
    r = _fit(_dataset(contrast=0.5, shots=1000))       # a poor pi also costs SNR
    assert r["pi_contrast"] == pytest.approx(0.5, abs=0.03)
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=1e6)


def test_a_pi_too_wide_to_be_spoiled_shows_no_line():
    """The expected outcome with a 16 ns x180 against a sub-MHz ZZ."""
    r = _fit(_dataset(spoil=0.001))
    assert r["no_line"] == 1 and not r["success"]
    assert r["pi_contrast"] == pytest.approx(0.88, abs=0.03)


def test_a_line_at_the_window_edge_fails():
    """3 MHz inside the window: flagged at the edge, or - half a dip, half the
    prominence - not found at all; never a SUCCESSFUL f01 (30 of 30 seeds)."""
    r = _fit(_dataset(coupler=[(6.853e9, 5e6, 0.45)]))
    assert not r["success"] and (r["peak_at_edge"] or r["no_line"])


def test_the_order_is_not_a_result(order_free):
    ascending = _dataset().sortby("pi_played")      # the fixture wants [0, 1]
    up = order_free(PairCouplerSpectroscopyZZEstimator(), ascending,
                    ("tone_freq_hz", "pi_played"))
    assert up["success"]


def test_unknown_kwargs_and_a_missing_arm_raise():
    ds = _dataset()
    with pytest.raises(ValueError, match="unknown kwargs"):
        _fit(ds, tone_on="low")
    with pytest.raises(ValueError, match="pi_member must be"):
        _fit(ds, pi_member="both")
    with pytest.raises(ValueError, match="both arms"):
        PairCouplerSpectroscopyZZEstimator().analyze(ds.sel(pi_played=[1]))


def test_no_line_fails_and_the_figure_still_renders(tmp_path):
    est = PairCouplerSpectroscopyZZEstimator()
    results, figures = est.analyze(_dataset(coupler=[]), output_dir=str(tmp_path),
                                   lo_hz=7.1e9, high_name="q1", low_name="q2")
    assert results["no_line"] == 1 and not results["success"]
    assert set(figures) == {"spectrum"}
    plot = xr.open_dataset(tmp_path / "pair_coupler_spectroscopy_zz_plotdata.nc")
    assert plot.attrs["high_name"] == "q1" and plot.attrs["lo_hz"] == 7.1e9
    assert plot["difference"].dims == ("tone_freq_hz",)
    plot.close()
    assert (tmp_path / "pair_coupler_spectroscopy_zz_metadata.json").exists()


def test_figures_render_with_every_marker(tmp_path):
    est = PairCouplerSpectroscopyZZEstimator()
    results, figures = est.analyze(_dataset(coupler=LADDER + [(6.88e9, 4e6, 0.3)]),
                                   output_dir=str(tmp_path))
    assert results["unexplained_lines"] and not np.isnan(results["alpha_hz"])
    assert set(figures) == {"spectrum"}
