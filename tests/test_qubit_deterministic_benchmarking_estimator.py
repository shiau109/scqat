"""Tests for the QubitDeterministicBenchmarkingEstimator.

Synthesises repeated-gate error accumulation: an amplitude factor ``a`` away from
the ideal one leaves a fixed per-gate over/under-rotation, so the population
oscillates in the REPETITION count at ``omega = k*(a - a_ideal)`` under a
depolarizing envelope. The estimator fits each repetition trace and reports
``opt_factor`` — the amplitude multiplier where the rotation error crosses zero.

Two properties carry the weight here, because both fail SILENTLY:

* **Orientation.** ``damped_cosine_zero_phase`` fixes ``phi = 0`` and the fit bounds
  ``A >= 0``, so the model can only describe a trace that starts HIGH. The sequence
  starts in |0>, so N=0 is an extremum by construction — but whether that extremum
  reads high or low depends on the readout sign. An un-oriented trace collapses onto
  the flat ``A ~ 0`` bound, and ``curve_fit`` is wrapped in ``try/except``, so the
  failure surfaces as ``omega = 0`` / ``opt_factor = 1.0`` rather than an error.
* **IQ reduction.** The readout blobs sit at an arbitrary rotation in the IQ plane,
  so reading raw ``I`` gives an arbitrarily-signed, arbitrarily-scaled trace. Only
  the axial projection onto the |0>-|1> axis means "population".
"""

import numpy as np
import xarray as xr
import pytest

from scqat.estimators import QubitDeterministicBenchmarkingEstimator
from scqat.estimators.qubit_deterministic_benchmarking import (
    QubitDeterministicBenchmarkingEstimator as SubpkgEstimator,
)

A_IDEAL = 1.012
K_RAD = np.pi  # rotation error per unit amplitude-factor error, per gate


def _pz(reps, amp_factors, *, a_ideal=A_IDEAL, decay=4e-3, rising=False):
    """Ground-population traces, one row per amplitude factor.

    ``rising=False`` starts HIGH at N=0 (P0 convention). ``rising=True`` is the
    inverted readout sign — the case the orientation step exists to absorb.
    """
    rows = []
    for a in amp_factors:
        omega = K_RAD * (float(a) - a_ideal)
        trace = 0.5 + 0.45 * np.exp(-decay * reps) * np.cos(omega * reps)
        rows.append(1.0 - trace if rising else trace)
    return np.asarray(rows)


def _signal_ds(*, n_amp=11, rising=False, a_ideal=A_IDEAL):
    reps = np.arange(0, 102, 2).astype(float)
    amp = np.linspace(0.9, 1.1, n_amp)
    return xr.Dataset(
        {"signal": (("amp_factor", "repetition"), _pz(reps, amp, a_ideal=a_ideal, rising=rising))},
        coords={"amp_factor": amp, "repetition": reps},
    )


def _iq_ds(*, n_amp=11, theta=0.7, seed=0):
    """The same physics placed in the IQ plane at an arbitrary blob rotation, the
    way an averaged readout actually returns it."""
    reps = np.arange(0, 102, 2).astype(float)
    amp = np.linspace(0.9, 1.1, n_amp)
    pz = _pz(reps, amp)
    rng = np.random.default_rng(seed)
    pos0 = complex(rng.normal(), rng.normal())
    pos1 = pos0 + 3.0 * np.exp(1j * theta)
    mixed = pos0 + (1.0 - pz) * (pos1 - pos0)  # P(excited) = 1 - P0
    noise = rng.normal(0, 2e-3, mixed.shape) + 1j * rng.normal(0, 2e-3, mixed.shape)
    iq = mixed + noise
    return xr.Dataset(
        {"I": (("amp_factor", "repetition"), iq.real),
         "Q": (("amp_factor", "repetition"), iq.imag)},
        coords={"amp_factor": amp, "repetition": reps},
    )


def test_exports_match():
    assert QubitDeterministicBenchmarkingEstimator is SubpkgEstimator


def test_recovers_the_optimal_amplitude_factor():
    results = QubitDeterministicBenchmarkingEstimator().extract_parameters(_signal_ds())
    # The sign heuristic pivots at a = 1.0, not at a_ideal, so points between the two
    # get the wrong sign and bias the zero crossing slightly. It CONVERGES (the next
    # run starts from an a_ideal nearer 1.0), which is why a small bias is tolerable.
    assert results["opt_factor"] == pytest.approx(A_IDEAL, abs=0.02)
    assert results["opt_factor"] != 1.0, "exactly 1.0 means the fit degraded silently"
    assert results["unit"] == "P0"
    assert len(results["omegas"]) == 11


def test_inverted_readout_sign_recovers_the_same_answer():
    """The orientation guarantee: flipping the readout sign must not change the
    physics. Un-oriented, the a>=0 bound traps this at omega~0 for every trace."""
    upright = QubitDeterministicBenchmarkingEstimator().extract_parameters(_signal_ds())
    flipped = QubitDeterministicBenchmarkingEstimator().extract_parameters(
        _signal_ds(rising=True))
    assert flipped["opt_factor"] == pytest.approx(upright["opt_factor"], abs=1e-9)
    assert np.allclose(flipped["omegas"], upright["omegas"], atol=1e-9)


def test_rotated_iq_reduces_axially():
    """Raw I at an arbitrary blob rotation is not the signal; the axial projection is."""
    results = QubitDeterministicBenchmarkingEstimator().extract_parameters(_iq_ds())
    assert results["opt_factor"] == pytest.approx(A_IDEAL, abs=0.02)


def test_single_amplitude_run_is_degenerate_but_clean():
    """One amplitude point cannot locate a zero crossing — opt_factor stays 1.0 and
    nothing raises (this is the default-parameter path)."""
    ds = _signal_ds(n_amp=1)
    results = QubitDeterministicBenchmarkingEstimator().extract_parameters(ds)
    assert results["opt_factor"] == 1.0
    assert len(results["omegas"]) == 1


def test_missing_repetition_coord_is_refused():
    ds = _signal_ds().rename({"repetition": "n"})
    with pytest.raises(ValueError, match="repetition"):
        QubitDeterministicBenchmarkingEstimator().analyze(ds, output_dir=None, skip_figures=True)


def test_missing_signal_source_is_refused():
    ds = _signal_ds().drop_vars("signal")
    with pytest.raises(ValueError, match="signal"):
        QubitDeterministicBenchmarkingEstimator().analyze(ds, output_dir=None, skip_figures=True)


def test_analyze_round_trips_artifacts(tmp_path):
    est = QubitDeterministicBenchmarkingEstimator()
    results, figures = est.analyze(_signal_ds(), output_dir=str(tmp_path))
    assert (tmp_path / "qubit_deterministic_benchmarking_metadata.json").exists()
    assert (tmp_path / "qubit_deterministic_benchmarking_plotdata.nc").exists()
    assert "qubit_deterministic_benchmarking" in figures
    # metadata is the small projection, not the bulky intermediates
    assert set(est.extract_metadata(results)) == {"opt_factor", "unit"}
    import matplotlib.pyplot as plt

    plt.close("all")
