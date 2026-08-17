"""The shared secondary-axis helper + its two carrying estimators.

The twin is a DECORATION: a caller may hand an estimator a second scale over the
same swept points, and it must either be drawn faithfully or ignored silently.
These tests pin both halves — the faithful case (values, answer, figure,
netCDF round-trip) and every way it is allowed to be absent.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr

from scqat.estimators._twin_axis import add_twin_axis, twin_at, twin_values
from scqat.estimators.power_rabi import PowerRabiEstimator
from scqat.estimators.readout_fidelity import ReadoutPowerFidelityEstimator

BASE = 0.1  # a stand-in stored pulse amplitude


def _rabi_dataset(twin=None, n=101):
    amp = np.linspace(0.0, 2.0, n)
    signal = 0.5 - 0.5 * np.cos(np.pi * amp)
    coords = {"amp_prefactor": amp}
    if twin is not None:
        coords["digital_amp"] = ("amp_prefactor", np.asarray(twin, dtype=float))
    return xr.Dataset({"signal": ("amp_prefactor", signal)}, coords=coords)


# --------------------------------------------------------------- the helper
def test_twin_values_returns_the_companion_scale():
    amp = np.linspace(0.0, 2.0, 11)
    ds = _rabi_dataset(twin=amp * BASE, n=11)
    got = twin_values(ds, "amp_prefactor", "digital_amp")
    assert got is not None
    np.testing.assert_allclose(got, amp * BASE)


@pytest.mark.parametrize(
    "name, twin",
    [
        ("all_nan", np.full(11, np.nan)),
        ("one_nan", np.concatenate([[np.nan], np.linspace(0.1, 1.0, 10)])),
        ("non_monotone", np.sin(np.linspace(0, 6, 11))),
        ("constant", np.full(11, 0.3)),
    ],
)
def test_undrawable_twins_are_none_never_an_exception(name, twin):
    """A twin that cannot carry an invertible axis is dropped, not raised on —
    a decoration must never be able to fail an analysis."""
    ds = _rabi_dataset(twin=twin, n=11)
    assert twin_values(ds, "amp_prefactor", "digital_amp") is None, name


def test_absent_or_unnamed_twin_is_none():
    ds = _rabi_dataset(n=11)
    assert twin_values(ds, "amp_prefactor", None) is None
    assert twin_values(ds, "amp_prefactor", "not_a_coord") is None


def test_length_mismatch_is_rejected():
    """A companion of the wrong length would silently mislabel every tick."""
    ds = _rabi_dataset(n=11)
    ds = ds.assign_coords(other=("other", np.linspace(0, 1, 5)))
    assert twin_values(ds, "amp_prefactor", "other") is None


def test_twin_at_interpolates_and_survives_nan():
    sweep = np.array([0.0, 1.0, 2.0])
    twin = sweep * BASE
    assert twin_at(sweep, twin, 1.5) == pytest.approx(0.15)
    assert np.isnan(twin_at(sweep, twin, float("nan")))


def test_add_twin_axis_extrapolates_past_the_ends():
    """matplotlib asks for the transform over PADDED axis limits, so a clamping
    map (plain np.interp) would pile ticks up at the edges."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sweep = np.linspace(0.0, 2.0, 21)
    add_twin_axis(ax, sweep, sweep * BASE, "absolute")
    secondary = ax.child_axes[0]
    forward = secondary._functions[0]
    assert float(forward(np.array([1.0]))[0]) == pytest.approx(0.1)
    assert float(forward(np.array([2.5]))[0]) == pytest.approx(0.25)  # not clamped
    assert float(forward(np.array([-0.5]))[0]) == pytest.approx(-0.05)
    plt.close(fig)


def test_add_twin_axis_handles_a_descending_companion():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    sweep = np.linspace(0.0, 2.0, 21)
    add_twin_axis(ax, sweep, 1.0 - sweep * BASE, "descending")
    forward = ax.child_axes[0]._functions[0]
    assert float(forward(np.array([2.0]))[0]) == pytest.approx(0.8)
    plt.close(fig)


# ------------------------------------------------------------- power_rabi
def test_power_rabi_reports_the_optimum_in_both_frames():
    amp = np.linspace(0.0, 2.0, 101)
    ds = _rabi_dataset(twin=amp * BASE)
    results = PowerRabiEstimator().extract_parameters(
        ds, twin_coord="digital_amp", twin_label="pi_amp"
    )
    assert results["opt_twin_value"] == pytest.approx(
        results["opt_amp_prefactor"] * BASE
    )
    assert results["twin_label"] == "pi_amp"


def test_power_rabi_without_a_twin_carries_no_twin_keys():
    """Consumers test with `if key in results`, so absence must be real."""
    results = PowerRabiEstimator().extract_parameters(_rabi_dataset())
    for key in ("twin_values", "twin_label", "opt_twin_value"):
        assert key not in results


def test_power_rabi_metadata_keeps_the_scalar_and_drops_the_array():
    amp = np.linspace(0.0, 2.0, 101)
    estimator = PowerRabiEstimator()
    results = estimator.extract_parameters(
        _rabi_dataset(twin=amp * BASE), twin_coord="digital_amp"
    )
    metadata = estimator.extract_metadata(results)
    assert "opt_twin_value" in metadata
    assert "twin_values" not in metadata


def test_power_rabi_figure_draws_the_twin_from_plot_data_alone(tmp_path):
    """scqat's self-enforcing rule: a saved plotdata must redraw every figure.
    Round-tripping through netCDF is the honest form of that check."""
    amp = np.linspace(0.0, 2.0, 101)
    estimator = PowerRabiEstimator()
    ds = _rabi_dataset(twin=amp * BASE)
    results = estimator.extract_parameters(
        ds, twin_coord="digital_amp", twin_label="pi_amp"
    )
    plot_data = estimator.build_plot_data(ds, results)
    assert "twin" in plot_data.data_vars

    path = tmp_path / "power_rabi_plotdata.nc"
    plot_data.to_netcdf(path)
    figures = estimator.generate_figures(None, None, plot_data=xr.open_dataset(path))
    secondary = figures["amplitude"].axes[0].child_axes
    assert secondary and secondary[0].get_xlabel() == "pi_amp"


def test_power_rabi_twin_kwargs_do_not_widen_the_reduction_knobs():
    """The twin kwargs are this estimator's own; tools/iq_reduce.py is shared by
    five families and its whitelist must stay closed."""
    with pytest.raises(ValueError):
        PowerRabiEstimator().extract_parameters(_rabi_dataset(), nonsense=1)


# -------------------------------------------------------- readout_fidelity
def _fidelity_dataset(twin=True, n_shots=400):
    rng = np.random.default_rng(0)
    amps = np.linspace(0.4, 1.8, 8)
    I = np.empty((amps.size, 2, n_shots))
    Q = np.empty_like(I)
    for j, a in enumerate(amps):
        separation = 2.6 * a
        for state in (0, 1):
            flip = 0.02 if state == 0 else 0.05
            actual = np.where(rng.random(n_shots) < flip, 1 - state, state)
            I[j, state] = actual * separation + rng.normal(0, 1.0, n_shots)
            Q[j, state] = rng.normal(0, 1.0, n_shots)
    coords = {
        "amp_prefactor": amps,
        "prepared_state": np.array([0, 1]),
        "shot_idx": np.arange(n_shots),
    }
    if twin:
        coords["digital_amp"] = ("amp_prefactor", amps * 0.08)
    dims = ("amp_prefactor", "prepared_state", "shot_idx")
    return xr.Dataset({"I": (dims, I), "Q": (dims, Q)}, coords=coords)


def test_readout_fidelity_best_twin_is_a_lookup_not_an_interpolation():
    """The twin is indexed identically to sweep_values, so the answer is the
    companion value AT best_index — exact, not interpolated."""
    ds = _fidelity_dataset()
    results = ReadoutPowerFidelityEstimator().extract_parameters(
        ds, twin_coord="digital_amp", twin_label="readout_amp"
    )
    expected = ds.coords["digital_amp"].values[results["best_index"]]
    assert results["best_twin_value"] == pytest.approx(expected)
    assert results["best_twin_value"] == pytest.approx(
        results["best_sweep_value"] * 0.08
    )


def test_readout_fidelity_without_a_twin_carries_no_twin_keys():
    results = ReadoutPowerFidelityEstimator().extract_parameters(
        _fidelity_dataset(twin=False)
    )
    for key in ("twin_values", "twin_label", "best_twin_value"):
        assert key not in results


def test_readout_fidelity_sweep_figures_carry_the_twin_axis():
    estimator = ReadoutPowerFidelityEstimator()
    ds = _fidelity_dataset()
    results = estimator.extract_parameters(
        ds, twin_coord="digital_amp", twin_label="readout_amp"
    )
    plot_data = estimator.build_plot_data(ds, results)
    figures = estimator.generate_figures(None, None, plot_data=plot_data)
    carrying = {name for name, fig in figures.items() if fig.axes[0].child_axes}
    # the answer-carrying figure plus the two other 1-D sweep curves; the IQ-plane
    # panel deliberately stays out (its sweep is a colourbar, not an axis)
    assert {"fidelity", "snr", "separation"} <= carrying
    assert "means_on_IQ" in figures and "means_on_IQ" not in carrying


def test_readout_fidelity_rejects_a_typoed_twin_kwarg():
    """The whitelist is UNIONED with TWIN_KNOBS, never loosened — a typo must
    still raise before any per-slice fit swallows it."""
    with pytest.raises(ValueError, match="twin_coordinate"):
        ReadoutPowerFidelityEstimator().extract_parameters(
            _fidelity_dataset(), twin_coordinate="digital_amp"
        )
