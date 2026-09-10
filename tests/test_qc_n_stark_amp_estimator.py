"""Synthetic-grid tests for the N-swap AC-Stark amplitude estimator.

The AC-Stark sibling of ``qc_n_swap_amp``: same ``joint_population`` form, drawn
over ``(stark_amp, swap_count)`` with an INTEGER swap-count axis. The shared
summary / plot-data / figure code is exercised in depth by
``test_pair_swap_chevron_estimator``; here we pin this estimator's own axis
names (including the integer count axis), the compensating-amplitude read-off
and the artifact writeout.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_n_stark_amp import QcNStarkAmpEstimator
from scqat.estimators.qc_n_stark_amp.estimator import trace_contrast

SA0, N0 = 0.0, 7  # where the (synthetic) transfer peaks: ideal stark amplitude, swap count
LABELS = ["00", "01", "10", "11"]

FIGURES = {"qc_n_stark_amp", "compensation"}


def _n_stark_amp_ds() -> xr.Dataset:
    sa = np.linspace(-0.1, 0.1, 11)          # AC-Stark amplitude (prefactor)
    n = np.arange(0, 11)                      # swap count N (integer)
    spot = np.exp(-((sa[:, None] - SA0) ** 2) / 0.002) * np.exp(-((n[None, :] - N0) ** 2) / 8.0)
    p11 = np.full((sa.size, n.size), 0.01)
    p10 = 0.84 * spot                        # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - spot)                # driven (low) member depletes at the peak
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])
    return xr.Dataset(
        {"joint_population": (("joint_state", "stark_amp", "swap_count"), jp)},
        coords={"joint_state": LABELS, "stark_amp": sa, "swap_count": n},
    )


#: the compensating stark amplitude the oscillating grid below is built around,
#: and the FASTEST period the grid is allowed to reach at the edge of the sweep.
#: Two counts per cycle is pi/2 a swap, the limit the reading assumes no row
#: crosses, so a test grid stays clear of it exactly as a real sweep must.
A_COMP = 0.35
EDGE_PERIOD = 2.5


def _oscillating_ds(a_comp: float = A_COMP, period: float = 6.0,
                    edge_period: float = EDGE_PERIOD, n_max: int = 13,
                    seed: int = 3) -> xr.Dataset:
    """A grid whose transfer really oscillates in N, detuned by the stark tone.

    The same model the SCQO experiment simulates: a fixed resonant exchange
    ``2J`` detuned by ``delta(a) = slope * (a - a_comp)``, so after ``N`` swaps
    the transfer is ``(2J)^2/omega^2 * sin^2(pi*omega*N*t_sw)`` with
    ``omega = sqrt(delta^2 + (2J)^2)``. On ``a_comp`` the oscillation is at its
    strongest (prefactor 1) and its slowest (``omega`` at its minimum); off it
    the contrast falls and the period shortens -- which is exactly the pair of
    criteria the estimator picks on. ``t_sw`` is set from ``period`` so the
    compensated row's period in N is known independently of the fit, and the
    detuning slope is DERIVED so the farthest row lands on ``edge_period``:
    ``omega/2J = period/edge_period`` there, which is what keeps the whole grid
    inside the pi/2-per-swap assumption instead of aliasing at its edges.
    """
    rng = np.random.default_rng(seed)
    amps = np.linspace(0.0, 1.0, 21)
    n = np.arange(0, n_max, dtype=float)
    two_j = 1.0
    t_sw = 1.0 / (two_j * period)
    reach = float(np.max(np.abs(amps - a_comp)))
    slope = two_j * np.sqrt(max((period / edge_period) ** 2 - 1.0, 0.0)) / reach
    omega = np.sqrt((slope * (amps - a_comp)) ** 2 + two_j ** 2)
    swap = ((two_j ** 2 / omega ** 2)[:, None]
            * np.sin(np.pi * omega[:, None] * n[None, :] * t_sw) ** 2)
    p10 = np.clip(0.96 * swap, 0.0, 1.0)              # transfer onto the high member
    p01 = np.clip(0.96 * (1.0 - swap), 0.0, 1.0)      # the driven (low) member
    p11 = np.full_like(p10, 0.01)
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.clip(np.stack([p00, p01, p10, p11])
                 + rng.normal(0.0, 0.01, (4, amps.size, n.size)), 0.0, 1.0)
    return xr.Dataset(
        {"joint_population": (("joint_state", "stark_amp", "swap_count"), jp)},
        coords={"joint_state": LABELS, "stark_amp": amps, "swap_count": n},
    )


def test_summarizes_spot_on_amp_and_count_axes():
    ds = _n_stark_amp_ds()
    est = QcNStarkAmpEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"
    assert res["best_stark_amp"] == pytest.approx(SA0, abs=0.02)
    assert res["best_swap_count"] == pytest.approx(N0, abs=1)
    assert res["n_stark_amp"] == 11 and res["n_swap_count"] == 11


def test_plot_data_uses_amp_count_axes_and_joint_basis():
    ds = _n_stark_amp_ds()
    est = QcNStarkAmpEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert {"p00", "p01", "p10", "p11"} <= set(pd.data_vars)
    assert set(pd.coords) == {"stark_amp", "swap_count", "swap_count_dense"}
    assert pd.attrs["axis0"] == "stark_amp"
    assert pd.attrs["axis1"] == "swap_count"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    ds = _n_stark_amp_ds().rename({"swap_count": "rounds"})
    with pytest.raises(ValueError, match="swap_count"):
        QcNStarkAmpEstimator()._check_data(ds)


# --- the compensating-amplitude read-off -----------------------------------


def test_picks_the_amplitude_with_the_strongest_slowest_oscillation():
    res = QcNStarkAmpEstimator().extract_parameters(
        _oscillating_ds(), drive_side="low")
    # every row of this grid oscillates, so every row fits
    assert res["n_osc_ok"] == 21
    # both criteria, and hence the combined pick, land on the compensation point
    assert res["compensating_stark_amp"] == pytest.approx(A_COMP, abs=0.05)
    assert res["max_osc_contrast_stark_amp"] == pytest.approx(A_COMP, abs=0.05)
    assert res["max_osc_period_stark_amp"] == pytest.approx(A_COMP, abs=0.05)
    assert res["osc_criteria_agree"] == 1
    # the period there is the one the grid was built with, and theta = pi/period
    assert res["compensating_osc_period"] == pytest.approx(6.0, rel=0.1)
    assert res["compensating_theta_rad"] == pytest.approx(np.pi / 6.0, rel=0.1)


def test_the_pick_refines_between_swept_amplitudes():
    res = QcNStarkAmpEstimator().extract_parameters(
        _oscillating_ds(), drive_side="low")
    assert res["compensating_is_refined"] == 1
    # the interpolated optimum stays inside the bin it was found in (the grid
    # step is 0.05) and does not run away from the grid pick
    assert res["compensating_stark_amp_refined"] == pytest.approx(A_COMP, abs=0.03)
    assert abs(res["compensating_stark_amp_refined"]
               - res["compensating_stark_amp"]) <= 0.025


def test_the_pick_is_the_largest_contrast_and_longest_period_measured():
    res = QcNStarkAmpEstimator().extract_parameters(
        _oscillating_ds(), drive_side="low")
    ok = np.asarray(res["osc_success"], dtype=bool)
    contrast = np.asarray(res["osc_contrast"])[ok]
    period = np.asarray(res["osc_period"])[ok]
    # the reported extrema ARE the extrema of the reported curves -- the pick is
    # read off the columns, not off a separate reduction
    assert res["max_osc_contrast"] == pytest.approx(contrast.max())
    assert res["max_osc_period"] == pytest.approx(period.max())
    # and the curves fall away on both sides of it, which is why one point wins
    i = int(np.argmax(contrast))
    assert contrast[0] < contrast[i] and contrast[-1] < contrast[i]
    assert period[0] < period[i] and period[-1] < period[i]


def test_the_nyquist_amplitude_is_degenerate_but_the_contrast_is_not():
    """Why the amplitude criterion is read off the trace and not off the fit.

    At the Nyquist period the samples are ``a*cos(phi)*(-1)**N + c``, so ONLY the
    product is determined: the two parameter sets below produce byte-identical
    data. A fitted ``a`` is therefore not a measurement there -- on 5Q4C q1_q2
    (run 20260908-173919) the row at stark_amp 0.05 fitted ``a = 0.355`` while
    swinging 0.178, and won a criterion it had no business winning."""
    n = np.arange(15.0)
    quiet = 0.3 + 0.178 * np.cos(np.pi * n + 0.0)
    loud = 0.3 + 0.356 * np.cos(np.pi * n + np.pi / 3)      # cos(pi/3) = 1/2
    assert loud == pytest.approx(quiet)                     # the same samples
    # the model cannot tell them apart; the measured contrast can only say what
    # the trace actually did, which is the same for both
    assert trace_contrast(quiet) == pytest.approx(0.178, rel=0.02)
    assert trace_contrast(loud) == pytest.approx(trace_contrast(quiet))


def test_the_fastest_row_reports_how_close_the_map_ran_to_the_limit():
    """``min_osc_period`` is the standing assumption's dashboard.

    The whole reading assumes no row swaps by more than pi/2 -- two counts per
    cycle -- because past that an oscillation aliases and reads SLOWER the faster
    it truly is. Nothing in the fitted periods can reveal a violation, so the map
    reports its fastest row and the operator keeps the assumption. Measured on
    5Q4C q1_q2 (run 20260908-173919): 2.02, inside by one percent."""
    inside = QcNStarkAmpEstimator().extract_parameters(
        _oscillating_ds(), drive_side="low")
    assert inside["min_osc_period"] == pytest.approx(EDGE_PERIOD, rel=0.15)
    assert inside["min_osc_period"] == pytest.approx(
        np.nanmin(inside["osc_period"]))

    # a map driven to a full swap sits ON the limit and says so
    at_limit = QcNStarkAmpEstimator().extract_parameters(
        _oscillating_ds(period=2.0, edge_period=2.0), drive_side="low")
    assert at_limit["min_osc_period"] == pytest.approx(2.0, abs=0.1)

def test_rejected_rows_cannot_win_the_pick():
    ds = _oscillating_ds()
    # blank the compensated row: it must lose the pick to a fitted neighbour
    # rather than winning it with a NaN period
    blanked = int(np.argmin(np.abs(ds["stark_amp"].values - A_COMP)))
    ds["joint_population"].values[:, blanked, :] = np.nan
    res = QcNStarkAmpEstimator().extract_parameters(ds, drive_side="low")
    assert res["osc_success"][blanked] == 0
    assert np.isnan(res["osc_period"][blanked])
    assert res["n_osc_ok"] == 20
    assert res["compensating_stark_amp"] != pytest.approx(
        ds["stark_amp"].values[blanked])


def test_no_fittable_row_leaves_the_pick_nan():
    ds = _oscillating_ds()
    ds["joint_population"].values[:] = np.nan
    res = QcNStarkAmpEstimator().extract_parameters(ds, drive_side="low")
    assert res["n_osc_ok"] == 0
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["max_osc_period"])
    assert res["osc_criteria_agree"] == 0


def test_plot_data_carries_the_fit_columns_and_the_pick():
    ds = _oscillating_ds()
    est = QcNStarkAmpEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds, drive_side="low"),
                             drive_side="low")
    assert {"transfer", "transfer_fit", "osc_contrast", "osc_amplitude",
            "osc_period", "osc_r_squared", "osc_success"} <= set(pd.data_vars)
    assert pd["transfer"].dims == ("stark_amp", "swap_count")
    assert pd["osc_period"].dims == ("stark_amp",)
    # the fitted curve rides its own dense axis, 10 points per measured count
    assert pd["transfer_fit"].dims == ("stark_amp", "swap_count_dense")
    assert pd.sizes["swap_count_dense"] == 10 * (pd.sizes["swap_count"] - 1) + 1
    assert pd.attrs["compensating_stark_amp"] == pytest.approx(A_COMP, abs=0.05)
    assert pd.attrs["compensating_stark_amp_refined"] == pytest.approx(A_COMP, abs=0.03)
    assert pd.attrs["n_osc_ok"] == 21


def test_plot_data_degrades_without_a_fit():
    """The replot path passes a results dict that never held the fit."""
    ds = _oscillating_ds()
    pd = QcNStarkAmpEstimator().build_plot_data(ds, {}, drive_side="low")
    assert np.isfinite(pd["transfer"].values).all()          # raw data survives
    assert np.isnan(pd["transfer_fit"].values).all()
    assert pd.sizes["swap_count_dense"] == 10 * (pd.sizes["swap_count"] - 1) + 1
    assert np.isnan(pd["osc_period"].values).all()
    assert np.isnan(pd["osc_contrast"].values).all()
    assert np.isnan(pd.attrs["compensating_stark_amp"])
    assert pd.attrs["n_osc_ok"] == 0


def test_metadata_drops_the_bulky_maps_and_keeps_the_curves():
    ds = _oscillating_ds()
    est = QcNStarkAmpEstimator()
    meta = est.extract_metadata(est.extract_parameters(ds, drive_side="low"))
    assert not [k for k in meta if k.startswith("_")]
    assert len(meta["osc_period"]) == 21
    assert len(meta["osc_contrast"]) == 21
    assert "compensating_stark_amp" in meta


# --- artifacts --------------------------------------------------------------


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _n_stark_amp_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = QcNStarkAmpEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == FIGURES
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_n_stark_amp.png" in written
    assert "qc_n_stark_amp_compensation.png" in written


def test_analyze_writes_artifacts(tmp_path):
    est = QcNStarkAmpEstimator()
    _res, figs = est.analyze(_oscillating_ds(), output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert set(figs) == FIGURES
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_n_stark_amp.png" in written
    assert "qc_n_stark_amp_compensation.png" in written
    assert "qc_n_stark_amp_plotdata.nc" in written
    assert "qc_n_stark_amp_metadata.json" in written
