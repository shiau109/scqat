"""Synthetic-grid tests for the fixed-N flux x stark estimator.

The two-amplitude member of the pair-swap map family: same ``joint_population``
form as ``qc_n_swap_amp``, but BOTH axes are amplitudes (the swap count is frozen
to a scalar and never reaches the dataset). The shared summary / plot-data /
figure code is exercised in depth by ``test_pair_swap_chevron_estimator``; here we
pin this estimator's own axis names, its compensation-ridge fit and the artifact
writeout.

The ridge fixture injects the physics the estimator claims to invert: a detuned
exchange whose per-round phase is the SUM of a flux-induced term and a
stark-induced one, so ``phi = 0`` is a straight line across the plane and the
compensating amplitude at resonance is known exactly. The three bands that
matter are separated by ``N * theta``:

  ``N*theta <= pi/2``   both gates open; the refined angle is readable
  ``pi/2 < N*theta <= pi``  only the ridge is valid — the arcsin has folded
  ``N*theta > pi``      the per-row argmax has jumped to a side lobe; nothing
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators.qc_swap_flux_stark import QcSwapFluxStarkEstimator

VA0, A0 = 0.15, 0.4  # where the (synthetic) transfer peaks: resonance V, compensating factor
LABELS = ["00", "01", "10", "11"]

#: ridge fixture: resonance flux (V), compensating stark factor at resonance.
RIDGE_V0, RIDGE_A0 = -0.1500, 0.5
#: detuning at the window edge, in units of 2J — sets how fast the envelope falls.
EDGE_DETUNING = 1.5


def _flux_stark_ds() -> xr.Dataset:
    va = np.linspace(0.10, 0.20, 11)          # control flux amplitude (V)
    a = np.linspace(0.0, 1.0, 9)              # stark amplitude factor
    spot = (np.exp(-((va[:, None] - VA0) ** 2) / 5e-5)
            * np.exp(-((a[None, :] - A0) ** 2) / 0.05))
    p11 = np.full((va.size, a.size), 0.01)
    p10 = 0.84 * spot                        # transfer onto the undriven (high) member
    p01 = 0.88 * (1.0 - spot)                # driven (low) member depletes at the peak
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    jp = np.stack([p00, p01, p10, p11])
    return xr.Dataset(
        {"joint_population": (("joint_state", "flux_amp_v", "stark_amp"), jp)},
        coords={"joint_state": LABELS, "flux_amp_v": va, "stark_amp": a},
    )


#: the two phase-vs-amplitude laws the fixture can inject. ``linear`` is the
#: convenient one; ``quadratic`` is what a real Stark tone does near zero
#: amplitude, and it is the one that punishes a global straight-line fit.
PHASE_LAWS = {
    "linear": (lambda a, k: k * a, lambda p, k: p / k),
    "quadratic": (lambda a, k: k * a ** 2,
                  lambda p, k: np.sqrt(np.clip(p, 0.0, None) / k)),
}


def _ridge_ds(n_swaps: int, theta0: float, *, nv: int = 41, na: int = 21,
              noise: float = 0.004, seed: int = 7, phase_law: str = "linear",
              k_phi: float = 4.0, half_span: float = 0.005,
              edge_detuning: float = EDGE_DETUNING, gap_factor: float = 1.0):
    """A repeated detuned exchange whose ``phi = 0`` locus is known exactly.

    Returns ``(dataset, truth)`` where ``truth["ridge"]`` is the compensating
    stark amplitude of every flux row and ``truth["theta"]`` the resonant
    exchange angle. The detuning does two things: it caps the per-round exchange
    through the envelope (that is what localizes the feature in flux) and it
    winds its own between-round phase, which the stark tone subtracts.

    ``phase_law`` picks how the tone's phase depends on its amplitude.
    ``"quadratic"`` is the real one -- 5Q4C's echo measured a curve that is
    quadratic near zero and only straightens past ~0.6 -- and under it the
    ``phi = 0`` locus is a CURVE in amplitude while staying a straight line in
    phase. That is what makes a local read the right one.

    The composite is the EXACT closed form the estimator inverts,
    ``T = sin^2(theta) * [sin(N*theta_eff)/sin(theta_eff)]^2`` — not SCQO's
    ``1 - tilt^2`` shorthand, whose axis component is un-normalized and would
    move the row optimum off ``phi = 0`` by a bias the estimator is not claiming
    to model.
    """
    forward, inverse = PHASE_LAWS[phase_law]
    v = np.linspace(RIDGE_V0 - half_span, RIDGE_V0 + half_span, nv)
    a = np.linspace(0.0, 1.0, na)
    j_hz = 5e6
    t_sw_s = theta0 / (2.0 * np.pi * j_hz)
    delta = (2 * j_hz) * edge_detuning * (v - RIDGE_V0) / half_span
    env = ((2 * j_hz) ** 2 / (delta ** 2 + (2 * j_hz) ** 2))[:, None]
    theta = np.sqrt(env) * theta0
    # The phase the flux pulse itself winds, offset so phi = 0 lands on
    # RIDGE_A0 at the resonance. ``gap_factor`` is how much longer the phase
    # accrues than the exchange does -- on 5Q4C a 40 ns pulse sits inside a
    # ~120 ns round -- and it is what lets the ridge wrap before the detuning
    # envelope has killed the rows.
    phi_flux = (2 * np.pi * delta * t_sw_s * gap_factor)[:, None] \
        + forward(RIDGE_A0, k_phi)
    phi = phi_flux - forward(a, k_phi)[None, :]
    theta_eff = np.arccos(np.clip(np.cos(phi / 2) * np.cos(theta), -1.0, 1.0))
    ratio = np.sin(n_swaps * theta_eff) / np.sin(theta_eff)
    swap = np.clip(np.sin(theta) ** 2 * ratio ** 2, 0.0, 1.0)

    rng = np.random.default_rng(seed)
    p01 = np.clip(0.96 * swap + rng.normal(0, noise, swap.shape), 0.0, 1.0)
    p10 = np.clip(0.96 * (1.0 - swap) + rng.normal(0, noise, swap.shape), 0.0, 1.0)
    p11 = np.full_like(p01, 0.01)
    p00 = np.clip(1.0 - (p01 + p10 + p11), 0.0, 1.0)
    ds = xr.Dataset(
        {"joint_population": (("joint_state", "flux_amp_v", "stark_amp"),
                              np.stack([p00, p01, p10, p11]))},
        coords={"joint_state": LABELS, "flux_amp_v": v, "stark_amp": a},
    )
    truth = {"ridge": inverse(phi_flux[:, 0], k_phi), "theta": theta0,
             "flux": v, "stark": a,
             "amp_2pi": float(inverse(2 * np.pi, k_phi))}
    return ds, truth


def _analyze(ds, **kwargs):
    est = QcSwapFluxStarkEstimator()
    res = est.extract_parameters(ds, drive_side="high", **kwargs)
    return est, res


def test_summarizes_spot_on_both_amplitude_axes():
    ds = _flux_stark_ds()
    est = QcSwapFluxStarkEstimator()
    est._check_data(ds)
    res = est.extract_parameters(ds, drive_side="low")
    assert res["success"] is True
    assert res["partner"] == "p_high"
    assert res["best_flux_amp_v"] == pytest.approx(VA0, abs=0.01)
    assert res["best_stark_amp"] == pytest.approx(A0, abs=0.13)
    assert res["n_flux_amp_v"] == 11 and res["n_stark_amp"] == 9


def test_plot_data_uses_flux_stark_axes_and_joint_basis():
    ds = _flux_stark_ds()
    est = QcSwapFluxStarkEstimator()
    pd = est.build_plot_data(ds, est.extract_parameters(ds), drive_side="low")
    assert set(pd.data_vars) == {
        "p00", "p01", "p10", "p11", "transfer",
        "row_stark_amp", "row_contrast", "ridge_stark_amp", "ridge_transfer",
        "ridge_theta_rad", "row_ok",
    }
    assert set(pd.coords) == {"flux_amp_v", "stark_amp"}
    assert pd.attrs["axis0"] == "flux_amp_v"
    assert pd.attrs["axis1"] == "stark_amp"
    assert pd.attrs["transfer_state"] == "p10"


def test_rejects_missing_coordinate():
    """The stark axis is this estimator's own; a dataset carrying the sibling's
    count axis instead must be refused by name, not silently analysed."""
    ds = _flux_stark_ds().rename({"stark_amp": "swap_count"})
    with pytest.raises(ValueError, match="stark_amp"):
        QcSwapFluxStarkEstimator()._check_data(ds)


# --- the compensation ridge -------------------------------------------------

@pytest.mark.parametrize("phase_law", ["linear", "quadratic"])
def test_ridge_recovers_the_injected_compensation_and_angle(phase_law):
    """Both gates open: the read returns the injected compensation AND the angle.

    Parametrized over the phase law on purpose. A QUADRATIC phase makes the
    ``phi = 0`` locus a curve in amplitude, and a local read has to survive that
    -- it is the reason there is no global fit here any more.
    """
    theta0 = np.pi / 5                      # N*theta = 1.257 < pi/2
    ds, truth = _ridge_ds(2, theta0, phase_law=phase_law)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    assert res["ridge_ok"] == 1 and res["branch_ok"] == 1
    assert res["n_ridge_rows"] >= 10
    assert res["resonance_in_gap"] == 0
    step = float(truth["stark"][1] - truth["stark"][0])
    assert res["compensating_stark_amp"] == pytest.approx(RIDGE_A0, abs=step)
    assert res["resonance_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    assert res["swap_angle_rad_refined"] == pytest.approx(theta0, rel=0.1)
    assert res["swap_angle_consistent"] == 1
    assert res["n_fold_rows"] == 0          # below pi/2 there is nothing to unfold
    # the per-row optima must track the injected locus, not merely bracket it
    used = np.asarray(res["row_ok"], dtype=bool)
    got = np.asarray(res["ridge_stark_amp"], dtype=float)
    deviation = np.abs(got - truth["ridge"])[used]
    assert float(np.median(deviation)) < step


def test_a_curved_ridge_defeats_a_global_straight_line():
    """The local read beats a whole-axis line when the phase law is quadratic.

    Not a style preference: on 5Q4C's N=5 map a line across the axis missed the
    per-row optima by 0.042 stark units against 0.016 for a phase-aware read,
    and the compensation it implied moved by more than a grid step.
    """
    theta0 = np.pi / 5
    ds, truth = _ridge_ds(2, theta0, phase_law="quadratic")
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    flux = np.asarray(truth["flux"])
    used = np.asarray(res["row_ok"], dtype=bool)
    got = np.asarray(res["ridge_stark_amp"], dtype=float)
    line = np.polyval(np.polyfit(flux[used], got[used], 1), flux)
    local_err = abs(res["compensating_stark_amp"] - RIDGE_A0)
    line_err = abs(float(np.interp(RIDGE_V0, flux, line)) - RIDGE_A0)
    assert local_err < line_err
    assert np.abs(line - got)[used].max() > np.abs(got - truth["ridge"])[used].max()


def test_a_sloppy_prior_still_yields_a_precise_angle():
    """The prior is a BRANCH SELECTOR — 30% off must not move the answer."""
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, exact = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    _est, sloppy = _analyze(ds, swap_count=2, swap_angle_rad=0.7 * theta0)
    assert sloppy["branch_ok"] == 1
    assert sloppy["swap_angle_rad_refined"] == pytest.approx(
        exact["swap_angle_rad_refined"], rel=1e-12)


def test_mid_band_reports_the_ridge_but_not_the_angle():
    """``pi/2 < N*theta <= pi``: the argmax is still phi=0, the arcsin has folded.

    This is the band the two-gate split exists for — 5Q4C q1_q2's N=2 map of
    2026-09-20 sat here, and collapsing the gates into one would have thrown
    away a compensating amplitude the map really did carry.
    """
    theta0 = 0.95                            # N*theta = 1.90, between the gates
    ds, _truth = _ridge_ds(2, theta0)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    assert res["ridge_ok"] == 1 and res["branch_ok"] == 0
    step = float(ds["stark_amp"].values[1] - ds["stark_amp"].values[0])
    assert res["compensating_stark_amp"] == pytest.approx(RIDGE_A0, abs=step)
    assert res["resonance_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    # the resonance is only findable because the fold was UNDONE: on the
    # principal branch it would have read as a local minimum
    assert res["n_fold_rows"] > 0
    assert np.isnan(res["swap_angle_rad_refined"])
    assert res["swap_angle_consistent"] == 0


def test_past_the_ridge_gate_nothing_is_reported():
    """``N*theta > pi``: the per-row argmax has jumped to a side lobe."""
    ds, _truth = _ridge_ds(3, 1.2)           # N*theta = 3.6 > pi
    est, res = _analyze(ds, swap_count=3, swap_angle_rad=1.2)
    assert res["ridge_ok"] == 0 and res["branch_ok"] == 0
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["resonance_flux_amp_v"])
    assert np.isnan(res["swap_angle_rad_refined"])
    assert np.isnan(res["ridge_slope_per_v"])
    # the measured per-row optima are still there, for the operator to judge
    assert np.isfinite(np.asarray(res["row_stark_amp"], dtype=float)).any()


def test_without_a_prior_both_gates_stay_shut():
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, res = _analyze(ds, swap_count=2)
    assert res["ridge_ok"] == 0 and res["branch_ok"] == 0
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["swap_angle_rad_prior"])
    # the rows are measured whether or not a prior was supplied; only the
    # reading that needs a branch selector is withheld
    assert res["n_ridge_rows"] > 4
    assert np.isfinite(np.asarray(res["row_contrast"], dtype=float)).all()


def test_a_stark_inert_map_yields_no_ridge_rows():
    """N=1 leaves the stark axis inert BY CONSTRUCTION — T(1) = sin^2(theta).

    The regression for run ``20260920-192637-643``, whose ``best_stark_amp``
    was an argmax over 21 statistically identical cells. The contrast filter,
    not a gate, is what has to catch it.
    """
    ds, _truth = _ridge_ds(1, np.pi / 5)
    # one swap: strip the stark dependence the way the hardware does
    jp = ds["joint_population"].values
    ds["joint_population"].values = np.repeat(
        jp.mean(axis=2, keepdims=True), jp.shape[2], axis=2)
    est, res = _analyze(ds, swap_count=1, swap_angle_rad=np.pi / 5)
    assert res["n_ridge_rows"] == 0
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["ridge_slope_per_v"])


def test_low_contrast_rows_are_excluded_from_the_fit():
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    _est, loose = _analyze(ds, swap_count=2, swap_angle_rad=theta0,
                           min_row_contrast=0.05)
    _est, tight = _analyze(ds, swap_count=2, swap_angle_rad=theta0,
                           min_row_contrast=0.8)
    assert loose["n_ridge_rows"] > tight["n_ridge_rows"]
    assert loose["max_row_contrast"] == pytest.approx(tight["max_row_contrast"])


@pytest.mark.parametrize("kwargs, match", [
    ({"swap_count": 0}, "swap_count"),
    ({"swap_angle_rad": 0.0}, "swap_angle_rad"),
    ({"swap_angle_rad": 2.0}, "swap_angle_rad"),
    ({"min_row_contrast": 1.5}, "min_row_contrast"),
])
def test_knobs_are_validated_before_the_row_loop(kwargs, match):
    """A typo'd knob must raise, not become an all-NaN map that reads as bad data."""
    ds, _truth = _ridge_ds(2, np.pi / 5)
    with pytest.raises(ValueError, match=match):
        QcSwapFluxStarkEstimator().extract_parameters(
            ds, drive_side="high", **kwargs)


def test_plot_data_carries_the_ridge_columns_and_attrs():
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    pd = est.build_plot_data(ds, res, drive_side="high", swap_count=2,
                             swap_angle_rad=theta0)
    assert pd["transfer"].dims == ("flux_amp_v", "stark_amp")
    for key in ("row_stark_amp", "row_contrast", "ridge_stark_amp",
                "ridge_transfer", "ridge_theta_rad", "row_ok"):
        assert pd[key].dims == ("flux_amp_v",)
    assert pd.attrs["compensating_stark_amp"] == pytest.approx(
        res["compensating_stark_amp"])
    assert pd.attrs["branch_ok"] == 1
    assert pd.attrs["swap_count"] == 2


def test_replot_path_degrades_to_nan_columns():
    """``build_plot_data(ds, {})`` is the replot path — it must not raise."""
    ds, _truth = _ridge_ds(2, np.pi / 5)
    pd = QcSwapFluxStarkEstimator().build_plot_data(ds, {}, drive_side="high")
    assert np.isnan(pd["ridge_stark_amp"].values).all()
    assert (pd["row_ok"].values == 0).all()
    assert np.isfinite(pd["transfer"].values).any()   # raw is always there
    assert np.isnan(pd.attrs["compensating_stark_amp"])


def test_metadata_drops_the_bulky_intermediate_and_keeps_the_columns():
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    meta = est.extract_metadata(res)
    assert not any(k.startswith("_") for k in meta)
    assert len(meta["ridge_stark_amp"]) == ds.sizes["flux_amp_v"]
    assert meta["swap_count"] == 2


def test_figures_render_on_a_failed_fit(tmp_path):
    ds = _flux_stark_ds()
    ds["joint_population"].values[:] = np.nan
    res, figs = QcSwapFluxStarkEstimator().analyze(
        ds, output_dir=str(tmp_path), skip_figures=False, drive_side="low")
    assert res["success"] is False
    assert set(figs) == {"qc_swap_flux_stark", "ridge"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_swap_flux_stark.png" in written
    assert "qc_swap_flux_stark_ridge.png" in written


def test_analyze_writes_artifacts(tmp_path):
    est = QcSwapFluxStarkEstimator()
    _res, figs = est.analyze(_flux_stark_ds(), output_dir=str(tmp_path),
                             skip_figures=False, drive_side="low")
    assert set(figs) == {"qc_swap_flux_stark", "ridge"}
    written = {p.name for p in tmp_path.iterdir()}
    assert "qc_swap_flux_stark.png" in written
    assert "qc_swap_flux_stark_ridge.png" in written
    assert "qc_swap_flux_stark_plotdata.nc" in written
    assert "qc_swap_flux_stark_metadata.json" in written


def test_the_periodic_ridge_is_unwrapped_before_it_is_read():
    """``phi = 0`` is ``phi = 0 mod 2pi``, so a wide flux window wraps the optimum.

    The regression for run ``20260920-212001-377``, whose ridge ran down to
    stark 0.08 at -150.5 mV and re-entered at 0.95 one row later. Left wrapped,
    any interpolation through that jump is meaningless.
    """
    theta0 = np.pi / 5
    # a flux window wide enough to drive phi past a full turn
    ds, truth = _ridge_ds(2, theta0, k_phi=7.0, gap_factor=5.0, nv=45)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    star = np.asarray(res["row_stark_amp"], dtype=float)
    ridge = np.asarray(res["ridge_stark_amp"], dtype=float)
    used = np.asarray(res["row_ok"], dtype=bool)
    span = float(truth["stark"].max() - truth["stark"].min())
    assert np.nanmax(np.abs(np.diff(star[used]))) > 0.4 * span, "fixture must wrap"
    assert np.nanmax(np.abs(np.diff(ridge[used]))) < 0.4 * span, "unwrap failed"
    assert res["ridge_wrap_amp"] == pytest.approx(truth["amp_2pi"], rel=0.15)


def test_a_calibrated_period_is_used_and_cross_checked():
    """``stark_amp_2pi`` from the echo makes the wrap exact and checks itself."""
    theta0 = np.pi / 5
    ds, truth = _ridge_ds(2, theta0, k_phi=7.0, gap_factor=5.0, nv=45)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0,
                        stark_amp_2pi=truth["amp_2pi"])
    assert res["stark_amp_2pi_prior"] == pytest.approx(truth["amp_2pi"])
    assert res["wrap_consistent"] == 1
    step = float(truth["stark"][1] - truth["stark"][0])
    assert res["compensating_stark_amp"] == pytest.approx(RIDGE_A0, abs=step)
    # a period that disagrees with the map is reported, never silently trusted
    _est, wrong = _analyze(ds, swap_count=2, swap_angle_rad=theta0,
                           stark_amp_2pi=0.5 * truth["amp_2pi"])
    assert wrong["wrap_consistent"] == 0


def test_a_resonance_inside_a_dead_band_is_refused():
    """Near a full swap the resonance row carries no stark signal at all.

    There is then nothing local to interpolate, and reaching it would take a
    model of phase-vs-amplitude this estimator deliberately does not carry. The
    numbers are withheld and the flag says why -- the case of 5Q4C's N=2 map.
    """
    ds, _truth = _ridge_ds(2, 1.45)          # sin^2(2*theta) ~ 0.04 on resonance
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=1.45)
    assert res["ridge_ok"] == 1
    assert res["resonance_in_gap"] == 1
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["resonance_flux_amp_v"])
    # the rows that DID have signal are still reported
    assert res["n_ridge_rows"] > 4


def test_stark_amp_2pi_must_be_positive():
    ds, _truth = _ridge_ds(2, np.pi / 5)
    with pytest.raises(ValueError, match="stark_amp_2pi"):
        QcSwapFluxStarkEstimator().extract_parameters(
            ds, drive_side="high", swap_count=2, stark_amp_2pi=0.0)
