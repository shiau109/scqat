"""Synthetic tests for the coupler swap-spectroscopy estimator.

Two arms per frequency: the ramp arm, where the coupler excitation is swapped into a
member, and the reference arm, where it is not. The coupler shows its multi-photon
ladder (f01, f02/2 = f01 + alpha/2, f03/3 ~ f01 + alpha); a member's own feature
sits in both arms. Numbers are 5Q4C q1_q2's at idle 0.16 V: f01 = 7.058 GHz, alpha
= -135 MHz, a 500 MHz window at 1 MHz steps, 300 shots.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.pair_coupler_spectroscopy_swap import (
    PairCouplerSpectroscopySwapEstimator,
)

LABELS = ["00", "01", "10", "11"]
FREQ = np.linspace(6.85e9, 7.35e9, 501)
F01, ALPHA = 7.058e9, -135e6
LADDER = [(F01, 5e6, 0.35), (F01 + ALPHA / 2, 4e6, 0.2)]


def _line(f, center, height, fwhm):
    return height / (1 + ((f - center) / (fwhm / 2)) ** 2)


def _dataset(*, coupler=LADDER, to_high=0.6, reference=0.0, member_lines=(),
             freq=FREQ, shots=300, seed=3):
    """``coupler`` = ``(center, fwhm, height)`` lines of the ramp arm, split
    ``to_high`` : ``1 - to_high`` between the members; the reference arm carries
    ``reference`` of each on the high member (a neighbour's readout seeing the
    coupler). ``member_lines`` = ``(center, fwhm, height)`` features of the high
    member alone, identical in both arms."""
    rng = np.random.default_rng(seed)
    arms = {}
    for arm in (1, 0):
        ph, pl = np.full(freq.size, 0.02), np.full(freq.size, 0.02)
        for center, fwhm, height in coupler:
            shape = _line(freq, center, height, fwhm)
            if arm:
                ph, pl = ph + to_high * shape, pl + (1 - to_high) * shape
            else:
                ph = ph + reference * shape
        for center, fwhm, height in member_lines:
            ph = ph + _line(freq, center, height, fwhm)
        ph, pl = np.clip(ph, 0, 1), np.clip(pl, 0, 1)
        probs = np.stack([(1 - ph) * (1 - pl), (1 - ph) * pl, ph * (1 - pl), ph * pl])
        arms[arm] = np.stack([rng.multinomial(shots, probs[:, j]) / shots
                              for j in range(freq.size)], axis=1)
    joint = np.stack([arms[1], arms[0]], axis=1)   # (joint_state, ramp_played, freq)
    return xr.Dataset(
        {"joint_population": (("joint_state", "ramp_played", "tone_freq_hz"), joint)},
        coords={"joint_state": LABELS, "ramp_played": [1, 0], "tone_freq_hz": freq})


def _fit(ds, **kwargs):
    return PairCouplerSpectroscopySwapEstimator().extract_parameters(ds, **kwargs)


def test_f01_is_the_top_of_the_ladder_and_alpha_comes_from_f02_half():
    r = _fit(_dataset())
    assert r["success"]
    assert r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert r["fwhm_hz"] == pytest.approx(5e6, rel=0.3)
    assert r["peak_height"] == pytest.approx(0.35, rel=0.2)
    assert r["snr"] > 10
    assert r["alpha_hz"] == pytest.approx(ALPHA, abs=2e6)
    assert 0 < r["alpha_stderr_hz"] < 2e6
    assert r["f02_half_hz"] == pytest.approx(F01 + ALPHA / 2, abs=1e6)
    assert (r["n_coupler_lines"], r["n_ladder_lines"]) == (2, 2)
    assert not (r["no_line"] or r["unexplained_lines"] or r["peak_at_edge"])


def test_a_broadened_f01_under_a_stronger_f02_half_is_still_f01():
    """5Q4C q2_q3 through q3's line: f01 ~30 MHz wide and lower than f02/2."""
    f01, alpha = 7.156e9, -148e6
    r = _fit(_dataset(coupler=[(f01, 30e6, 0.15), (f01 + alpha / 2, 4e6, 0.45)]))
    assert r["success"]
    assert r["f_c_hz"] == pytest.approx(f01, abs=10e6)      # a third of its FWHM
    assert r["alpha_hz"] == pytest.approx(alpha, abs=20e6)


def test_the_f03_third_rung_is_placed_on_the_same_ladder():
    r = _fit(_dataset(coupler=LADDER + [(F01 + ALPHA + 2e6, 4e6, 0.15)]))
    assert r["success"] and (r["n_coupler_lines"], r["n_ladder_lines"]) == (3, 3)
    assert r["f03_third_hz"] == pytest.approx(F01 + ALPHA + 2e6, abs=1e6)
    assert r["alpha_hz"] == pytest.approx(ALPHA, abs=2e6)


def test_a_single_coupler_line_is_f01_without_alpha():
    r = _fit(_dataset(coupler=LADDER[:1]))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert np.isnan(r["alpha_hz"]) and r["n_ladder_lines"] == 1


def test_the_excitation_landing_on_either_member_counts():
    for to_high in (0.0, 1.0):
        r = _fit(_dataset(to_high=to_high), tone_on="high")
        assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
        landed, missed = ((r["landing_high"], r["landing_low"]) if to_high
                          else (r["landing_low"], r["landing_high"]))
        assert landed == pytest.approx(0.35, rel=0.25) and abs(missed) < 0.05


def test_a_reference_arm_seeing_the_coupler_does_not_veto_it():
    """5Q4C q1's readout sees q1_q2_c directly, so the reference shows f01 too."""
    r = _fit(_dataset(reference=0.4))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert r["alpha_hz"] == pytest.approx(ALPHA, abs=2e6)


def test_a_member_line_in_both_arms_is_not_the_coupler():
    """Equal in both arms and ABOVE f01: taken for a coupler line it would be f01."""
    r = _fit(_dataset(member_lines=[(7.25e9, 6e6, 0.3)]))
    assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert any(abs(f - 7.25e9) < 1e6 for f in r["member_lines_hz"])
    assert r["n_coupler_lines"] == 2


def test_a_coupler_line_off_the_ladder_fails():
    """6.88 GHz fits no rung of alpha = -135 MHz, and f02/2 fits no rung of the
    alpha 6.88 would imply: one of the two is left over - which one depends on the
    noise, the verdict does not."""
    r = _fit(_dataset(coupler=LADDER + [(6.88e9, 4e6, 0.15)]))
    assert r["unexplained_lines"] == 1 and not r["success"]
    assert r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
    assert r["n_coupler_lines"] == 3 and r["n_ladder_lines"] == 2
    [left_over] = r["unexplained_lines_hz"]
    assert min(abs(left_over - 6.88e9), abs(left_over - (F01 + ALPHA / 2))) < 1e6


def test_a_line_one_or_two_points_wide_is_ignored():
    """A sub-resolution spike above f01 would be called f01 if it were kept. One
    point wide it always drops out. Two points wide its fitted FWHM sits at the
    two-step gate, so now and then one survives - it then fits no rung and the run
    FAILS; it never passes as f01."""
    listed = lambda r: r["coupler_lines_hz"] + r["member_lines_hz"] + r["unexplained_lines_hz"]
    for seed in range(4):
        r = _fit(_dataset(coupler=LADDER + [(7.2e9, 0.5e6, 0.5)], seed=seed))
        assert r["success"] and r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
        assert not any(abs(f - 7.2e9) < 5e6 for f in listed(r))
        r = _fit(_dataset(coupler=LADDER + [(7.2005e9, 1e6, 0.5)], seed=seed))
        if r["success"]:
            assert r["f_c_hz"] == pytest.approx(F01, abs=0.5e6)
            assert not any(abs(f - 7.2e9) < 5e6 for f in listed(r))
        else:
            assert r["unexplained_lines"] == 1


def test_a_line_at_the_window_edge_fails():
    r = _fit(_dataset(coupler=[(6.853e9, 5e6, 0.35)]))
    assert r["peak_at_edge"] == 1 and not r["success"]


def test_the_order_is_not_a_result(order_free):
    ascending = _dataset().sortby("ramp_played")      # the fixture wants [0, 1]
    up = order_free(PairCouplerSpectroscopySwapEstimator(), ascending,
                    ("tone_freq_hz", "ramp_played"))
    assert up["success"]


def test_unknown_kwargs_and_a_missing_arm_raise():
    ds = _dataset()
    with pytest.raises(ValueError, match="unknown kwargs"):
        _fit(ds, min_snrr=3)
    with pytest.raises(ValueError, match="tone_on must be"):
        _fit(ds, tone_on="middle")
    est = PairCouplerSpectroscopySwapEstimator()
    with pytest.raises(ValueError, match="both arms"):
        est.analyze(ds.sel(ramp_played=[1]))


def test_no_line_fails_and_the_figure_still_renders(tmp_path):
    est = PairCouplerSpectroscopySwapEstimator()
    results, figures = est.analyze(_dataset(coupler=[]), output_dir=str(tmp_path),
                                   tone_on="low", lo_hz=7.1e9, ramp_duration_ns=936.0,
                                   high_name="q1", low_name="q2")
    assert results["no_line"] == 1 and not results["success"]
    assert set(figures) == {"spectrum"}
    plot = xr.open_dataset(tmp_path / "pair_coupler_spectroscopy_swap_plotdata.nc")
    assert plot.attrs["high_name"] == "q1" and plot.attrs["lo_hz"] == 7.1e9
    assert plot["total"].dims == ("ramp_played", "tone_freq_hz")
    plot.close()
    assert (tmp_path / "pair_coupler_spectroscopy_swap_metadata.json").exists()


def test_figures_render_with_every_marker(tmp_path):
    est = PairCouplerSpectroscopySwapEstimator()
    results, figures = est.analyze(
        _dataset(coupler=LADDER + [(6.88e9, 4e6, 0.15)], member_lines=[(7.25e9, 6e6, 0.3)]),
        output_dir=str(tmp_path), tone_on="low")
    assert results["unexplained_lines"] and results["member_lines_hz"]
    assert set(figures) == {"spectrum"}
