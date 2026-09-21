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
              edge_detuning: float = EDGE_DETUNING, gap_factor: float = 1.0,
              window_offset: float = 0.0, descending: bool = False,
              exact_arch: bool = False):
    """A repeated detuned exchange whose ``phi = 0`` locus is known exactly.

    Returns ``(dataset, truth)`` where ``truth["ridge"]`` is the compensating
    stark amplitude of every flux row and ``truth["theta"]`` the resonant
    exchange angle. The detuning does two things: it caps the per-round exchange
    through the envelope (that is what localizes the feature in flux) and it
    winds its own between-round phase, which the stark tone subtracts.

    ``window_offset`` moves the flux WINDOW (V) while the resonance stays at
    ``RIDGE_V0``, and ``descending`` sweeps it downward, the way every 5Q4C run
    so far was taken. ``edge_detuning`` is the detuning at ``half_span`` from
    the resonance, in units of ``2J``.

    ``exact_arch`` makes the per-round exchange the EXACT detuned one,
    ``sin(theta0*s)/s`` with ``s = sqrt(1 + u^2)`` -- Fourier-limited and
    flat-topped, the shape the estimator's arch fit models and the hardware
    shows. The default ``theta0/s`` is a narrower stand-in: symmetric about the
    same resonance with the same height, which is all the older ridge tests
    need. The detuned round's own diagonal phase is not modelled separately;
    it is part of the flux-linear phase either way.

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
    v = RIDGE_V0 + window_offset + np.linspace(-half_span, half_span, nv)
    if descending:
        v = v[::-1]
    a = np.linspace(0.0, 1.0, na)
    j_hz = 5e6
    t_sw_s = theta0 / (2.0 * np.pi * j_hz)
    delta = (2 * j_hz) * edge_detuning * (v - RIDGE_V0) / half_span
    s = np.sqrt(1.0 + (delta / (2 * j_hz)) ** 2)[:, None]
    theta = (np.arcsin(np.clip(np.sin(theta0 * s) / s, -1.0, 1.0)) if exact_arch
             else theta0 / s)
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
        "ridge_theta_rad", "row_ok", "arch_fit",
    }
    assert set(pd.coords) == {"flux_amp_v", "stark_amp", "flux_amp_v_dense"}
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

@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("phase_law", ["linear", "quadratic"])
def test_ridge_recovers_the_injected_compensation_and_angle(phase_law, descending):
    """Both gates open: the read returns the injected compensation AND the angle.

    Parametrized over the phase law on purpose. A QUADRATIC phase makes the
    ``phi = 0`` locus a curve in amplitude, and a local read has to survive that
    -- it is the reason there is no global fit here any more. And over the
    sweep direction, because every 5Q4C run so far swept the flux DOWNWARD.
    """
    theta0 = np.pi / 5                      # N*theta = 1.257 < pi/2
    ds, truth = _ridge_ds(2, theta0, phase_law=phase_law, descending=descending)
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    assert res["ridge_ok"] == 1 and res["branch_ok"] == 1
    assert res["n_ridge_rows"] >= 10
    assert res["resonance_at_edge"] == 0 and res["resonance_unresolved"] == 0
    assert res["compensation_in_gap"] == 0
    step = float(abs(truth["stark"][1] - truth["stark"][0]))
    assert res["compensating_stark_amp"] == pytest.approx(RIDGE_A0, abs=step)
    assert res["resonance_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    # the error is reported with the value, and it is small but not zero
    assert 0.0 < res["resonance_flux_err_v"] < 5e-4
    assert 0.0 < res["compensating_stark_err"] < step
    assert res["swap_angle_rad_refined"] == pytest.approx(theta0, rel=0.1)
    assert res["swap_angle_err_rad"] > 0.0
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
    """The prior is a BRANCH SELECTOR — 30% off must not move the answer.

    It picks the arch fit's angle band and adds one seed to its grid; the
    minimum the fit converges to is the same, to the optimizer's tolerance.
    """
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, exact = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    _est, sloppy = _analyze(ds, swap_count=2, swap_angle_rad=0.7 * theta0)
    assert sloppy["branch_ok"] == 1
    assert sloppy["swap_angle_rad_refined"] == pytest.approx(
        exact["swap_angle_rad_refined"], rel=1e-6)
    assert sloppy["resonance_flux_amp_v"] == pytest.approx(
        exact["resonance_flux_amp_v"], abs=1e-9)


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
    # the resonance is findable because the arch is fitted in the FOLDED band,
    # where its centre is a dip between two maxima -- the largest row is one of
    # those maxima, not the resonance
    assert res["resonance_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    # the per-row angle drawn beside it is unfolded to match
    assert res["n_fold_rows"] > 0
    assert np.isnan(res["swap_angle_rad_refined"])
    assert np.isnan(res["swap_angle_err_rad"])
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
    # a shut gate is the N's limitation, not the data's -- no refusal flag, and
    # no fit was attempted
    assert res["resonance_unresolved"] == 0 and res["resonance_at_edge"] == 0
    assert np.isnan(res["resonance_flux_err_v"])
    # the measured per-row optima are still there, for the operator to judge
    assert np.isfinite(np.asarray(res["row_stark_amp"], dtype=float)).any()


def test_without_a_prior_both_gates_stay_shut():
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    est, res = _analyze(ds, swap_count=2)
    assert res["ridge_ok"] == 0 and res["branch_ok"] == 0
    assert np.isnan(res["compensating_stark_amp"])
    assert np.isnan(res["swap_angle_rad_prior"])
    assert res["resonance_unresolved"] == 0
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
    # the gate was OPEN (one swap at pi/5 is well inside it), so the missing
    # answer is the data's and is flagged as such
    assert res["ridge_ok"] == 1
    assert res["resonance_unresolved"] == 1


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
    # the arch is drawn on a denser flux axis spanning the same window
    assert pd["arch_fit"].dims == ("flux_amp_v_dense",)
    dense = pd["flux_amp_v_dense"].values
    assert dense.size > pd.sizes["flux_amp_v"]
    assert dense.min() == pytest.approx(pd["flux_amp_v"].values.min())
    assert dense.max() == pytest.approx(pd["flux_amp_v"].values.max())
    assert np.isfinite(pd["arch_fit"].values).any()
    assert pd.attrs["compensating_stark_amp"] == pytest.approx(
        res["compensating_stark_amp"])
    assert pd.attrs["resonance_flux_err_v"] == pytest.approx(
        res["resonance_flux_err_v"])
    assert pd.attrs["branch_ok"] == 1
    assert pd.attrs["swap_count"] == 2


def test_replot_path_degrades_to_nan_columns():
    """``build_plot_data(ds, {})`` is the replot path — it must not raise."""
    ds, _truth = _ridge_ds(2, np.pi / 5)
    pd = QcSwapFluxStarkEstimator().build_plot_data(ds, {}, drive_side="high")
    assert np.isnan(pd["ridge_stark_amp"].values).all()
    assert (pd["row_ok"].values == 0).all()
    assert np.isnan(pd["arch_fit"].values).all()
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


def test_a_dead_band_on_resonance_withholds_only_the_compensation():
    """Near a full swap the resonance rows carry no stark signal at all.

    The arch fit still reaches the resonance from the two flanks -- on 5Q4C's
    N=2 map it read -149.164 +- 0.024 mV across a seven-row dead band, within
    0.1 mV of three independent reads. What it cannot reach is the
    COMPENSATION there: that would be interpolated across the band, and the
    stark axis is not a linear picture of phase. So only that number is
    withheld, and the flag says why.
    """
    ds, _truth = _ridge_ds(2, 1.45)          # sin^2(2*theta) ~ 0.06 on resonance
    est, res = _analyze(ds, swap_count=2, swap_angle_rad=1.45)
    assert res["ridge_ok"] == 1 and res["branch_ok"] == 0
    used = np.asarray(res["row_ok"], dtype=bool)
    flux = ds["flux_amp_v"].values
    centre = int(np.argmin(np.abs(flux - RIDGE_V0)))
    assert not used[centre - 1: centre + 2].any(), "fixture must bury the resonance"
    assert res["compensation_in_gap"] == 1
    assert np.isnan(res["compensating_stark_amp"])
    assert res["resonance_at_edge"] == 0 and res["resonance_unresolved"] == 0
    assert res["resonance_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    # the rows that DID have signal are still reported
    assert res["n_ridge_rows"] > 4


def test_stark_amp_2pi_must_be_positive():
    ds, _truth = _ridge_ds(2, np.pi / 5)
    with pytest.raises(ValueError, match="stark_amp_2pi"):
        QcSwapFluxStarkEstimator().extract_parameters(
            ds, drive_side="high", swap_count=2, stark_amp_2pi=0.0)


def test_the_best_row_is_reported_without_any_prior():
    """WHERE the compensated transfer peaks is a measurement, not a claim.

    An operator who ran without priors still needs a flux to act on, so
    `ridge_peak_flux_amp_v` is computed from the rows alone. On a clean,
    unfolded arch it lands where the fitted resonance does; the fit is what
    to prefer once there is noise on the top (see the flat-top test below).
    """
    theta0 = np.pi / 5
    ds, _truth = _ridge_ds(2, theta0)
    _est, bare = _analyze(ds, swap_count=2)
    assert bare["ridge_ok"] == 0
    assert np.isnan(bare["resonance_flux_amp_v"])
    assert bare["ridge_peak_flux_amp_v"] == pytest.approx(RIDGE_V0, abs=5e-4)
    assert 0.0 < bare["ridge_peak_transfer"] <= 1.05

    _est, gated = _analyze(ds, swap_count=2, swap_angle_rad=theta0)
    assert gated["branch_ok"] == 1
    # the prior adds a claim and a fit; it does not move the raw measurement
    assert gated["ridge_peak_flux_amp_v"] == bare["ridge_peak_flux_amp_v"]
    step = float(abs(np.diff(ds["flux_amp_v"].values)[0]))
    assert gated["ridge_peak_flux_amp_v"] == pytest.approx(
        gated["resonance_flux_amp_v"], abs=0.5 * step)


# --- the arch fit, and when its centre is refused ----------------------------

#: a 5Q4C-like N=5 map (2026-09-21): 29 rows across 7 mV swept downward, a
#: Fourier-limited arch falling to about half height at the window edges, and
#: shot noise of the order 200 averages leave. The stark window is one period.
FLAT_TOP = dict(exact_arch=True, edge_detuning=6.2, half_span=0.0035, nv=29,
                descending=True, k_phi=7.0)
FLAT_TOP_PERIOD = 2 * np.pi / 7.0


def _flat_top(**overrides):
    kwargs = {**FLAT_TOP, "noise": 0.025, **overrides}
    ds, truth = _ridge_ds(5, 0.21, **kwargs)
    _est, res = _analyze(ds, swap_count=5, swap_angle_rad=0.23,
                         stark_amp_2pi=FLAT_TOP_PERIOD)
    return ds, res


def test_the_arch_fit_beats_the_largest_row_on_a_flat_top():
    """The reason the resonance is fitted: the top of the arch is flat.

    The regression for run ``20260921-123240-776``, whose largest row sat
    1.3 mV from the arch's centre -- which, at that ridge's slope, moved the
    compensation by more than two stark steps. A shot-noise bootstrap put the
    largest row at +-0.79 mV and the fit at +-0.10 mV; the same ordering has to
    hold here across noise realizations.
    """
    fit_err, row_err = [], []
    for seed in range(8):
        _ds, res = _flat_top(seed=seed)
        assert res["resonance_unresolved"] == 0 and res["resonance_at_edge"] == 0
        fit_err.append(res["resonance_flux_amp_v"] - RIDGE_V0)
        row_err.append(res["ridge_peak_flux_amp_v"] - RIDGE_V0)
    fit_rms = float(np.sqrt(np.mean(np.square(fit_err))))
    row_rms = float(np.sqrt(np.mean(np.square(row_err))))
    assert fit_rms < 0.25 * row_rms
    assert fit_rms < 1e-4                    # well inside half a 0.25 mV step


def test_a_window_that_misses_the_resonance_is_refused_at_the_edge():
    """A centre the rows never reached is an extrapolation, not a measurement.

    The regression for run ``20260921-105115-861``: its 3 mV window started at
    -151.00 mV, the arch's centre fitted to -151.01 +- 1.98, and the old read
    reported -150.85 -- a row next to the window's end -- as the resonance
    without a word. Now it is refused and the error is left in place to show
    why, and the arch is still drawn.
    """
    ds, res = _flat_top(noise=0.01, window_offset=1.2 * FLAT_TOP["half_span"])
    assert res["resonance_at_edge"] == 1
    for key in ("resonance_flux_amp_v", "compensating_stark_amp",
                "swap_angle_rad_refined"):
        assert np.isnan(res[key]), key
    assert np.isfinite(res["resonance_flux_err_v"])
    # the raw peak is still a measurement, and it is at the window's near end
    flux = ds["flux_amp_v"].values
    step = float(abs(np.diff(flux)[0]))
    assert res["ridge_peak_flux_amp_v"] == pytest.approx(flux.min(), abs=2 * step)
    est = QcSwapFluxStarkEstimator()
    pd = est.build_plot_data(ds, res, drive_side="high")
    assert np.isfinite(pd["arch_fit"].values).any()


def test_a_flat_topped_window_is_refused_as_unresolved():
    """A window that only covers the flat top cannot place the centre.

    Every row is live and the centre is inside the window, so nothing but the
    fit's own error can tell this run from a good one.
    """
    _ds, res = _flat_top(edge_detuning=2.0, seed=2)
    assert res["n_ridge_rows"] == 29
    assert res["resonance_unresolved"] == 1
    assert res["resonance_at_edge"] == 0
    assert np.isnan(res["resonance_flux_amp_v"])
    assert np.isnan(res["compensating_stark_amp"])
    # the numbers that explain the refusal stay
    assert np.isfinite(res["resonance_flux_err_v"])
    assert res["arch_r_squared"] < 0.5
    assert np.isfinite(res["ridge_peak_flux_amp_v"])
