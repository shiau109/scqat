"""Tests for ``tools.coupler_ladder``: coupler lines and their multi-photon ladder.

Numbers are 5Q4C's: q1_q2_c at 7.058 GHz with alpha = -135 MHz, 1 MHz steps.
"""

import numpy as np
import pytest

from scqat.tools.coupler_ladder import find_lines, lines_curve, read_ladder

FREQ = np.linspace(6.85e9, 7.35e9, 501)
F01, ALPHA = 7.058e9, -135e6


def _trace(lines, noise=0.012, seed=0):
    rng = np.random.default_rng(seed)
    y = np.full(FREQ.size, 0.04)
    for center, fwhm, height in lines:
        y = y + height / (1 + ((FREQ - center) / (fwhm / 2)) ** 2)
    return y + noise * rng.standard_normal(FREQ.size)


def _line(f, fwhm=5e6, amplitude=0.3, err=0.2e6):
    return {"full_freq": f, "fwhm": fwhm, "amplitude": amplitude, "detuning_err": err}


def test_two_equal_lines_are_both_found():
    """The tool's automatic polarity takes two equal lines for a dip; fixed to peaks
    here, both come back (seed 8 is one where auto would have lost them)."""
    lines, fit = find_lines(FREQ, _trace([(7.0e9, 5e6, 0.3), (7.2e9, 5e6, 0.3)], seed=8))
    assert [p["full_freq"] for p in lines] == [pytest.approx(7.0e9, abs=0.5e6),
                                               pytest.approx(7.2e9, abs=0.5e6)]
    curve = lines_curve(FREQ, lines, fit)
    assert curve.max() == pytest.approx(0.34, abs=0.03)


def test_a_broad_line_is_not_pulled_onto_a_stronger_narrow_neighbour():
    lines, _ = find_lines(FREQ, _trace([(7.156e9, 30e6, 0.15), (7.082e9, 4e6, 0.45)], seed=3))
    assert [p["full_freq"] for p in lines] == [pytest.approx(7.082e9, abs=0.5e6),
                                               pytest.approx(7.156e9, abs=5e6)]


def test_a_one_point_spike_and_noise_give_no_line():
    assert find_lines(FREQ, _trace([(7.1e9, 0.5e6, 0.5)]))[0] == []
    assert find_lines(FREQ, _trace([]))[0] == []
    assert np.isnan(lines_curve(FREQ, [], {})).all()


def test_f01_is_the_top_and_alpha_comes_from_f02_half():
    lad = read_ladder([_line(F01 + ALPHA / 2, amplitude=0.4), _line(F01, amplitude=0.2),
                       _line(F01 + ALPHA + 3e6, amplitude=0.1)])
    assert lad["f01"]["full_freq"] == F01
    assert lad["alpha_hz"] == pytest.approx(ALPHA)
    assert lad["alpha_stderr_hz"] == pytest.approx(2 * np.hypot(0.2e6, 0.2e6))
    assert lad["f03_third"]["full_freq"] == F01 + ALPHA + 3e6
    assert lad["n_ladder_lines"] == 3 and lad["unexplained"] == []


def test_a_single_line_is_f01_without_alpha_and_none_is_none():
    lad = read_ladder([_line(F01)])
    assert lad["n_ladder_lines"] == 1 and np.isnan(lad["alpha_hz"])
    assert lad["f02_half"] is None and lad["f03_third"] is None
    assert read_ladder([]) is None


def test_a_line_off_the_ladder_is_left_over():
    """6.88 GHz fits no rung of alpha = -135 MHz; f02/2 is the larger area, so it
    keeps the ladder and 6.88 is the one left over."""
    lad = read_ladder([_line(F01), _line(F01 + ALPHA / 2, amplitude=0.2),
                       _line(6.88e9, amplitude=0.1)])
    assert [p["full_freq"] for p in lad["unexplained"]] == [6.88e9]
    assert lad["alpha_hz"] == pytest.approx(ALPHA)


def test_an_alpha_outside_the_range_places_nothing():
    lad = read_ladder([_line(F01), _line(F01 - 20e6)])          # alpha -40 / -20 MHz
    assert len(lad["unexplained"]) == 1 and np.isnan(lad["alpha_hz"])
    assert read_ladder([_line(F01), _line(F01 - 20e6)],
                       alpha_range_hz=(-100e6, -30e6))["alpha_hz"] == pytest.approx(-40e6)
