"""Run ``ParametricDriveDecoherenceEstimator`` on SCQ.jl simulation data.

"Simulation = virtual experiment": a SCQ.jl parametric-drive frequency sweep
synthesizes rho_11(t) per driving frequency, the same observable the real node
measures. ``load_parametric_sim_h5`` puts it into the estimator's
``(driving_frequency, driving_time, state)`` contract, so the SAME estimator used
on real data (see ``run_decoherence_estimator.py``) analyses it unchanged -- the
only difference is the readout normalisation is the identity (rho11_offset=0,
rho11_scale=1), because a simulated population needs no readout correction.

Usage
-----
    python examples/parametric_drive/run_decoherence_on_sim.py                  # bundled example sweep
    python examples/parametric_drive/run_decoherence_on_sim.py path/to/sim.h5   # your SCQ.jl sweep

Artifacts land under ``examples/parametric_drive/output/<tag>/``.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt

from scqat.workflows.parametric_sim import load_parametric_sim_h5
from scqat.estimators import ParametricDriveDecoherenceEstimator

# A ready SCQ.jl parametric-drive omega_flux sweep (legacy projector layout).
EXAMPLE_SIM = r"D:/github/SCQ.jl/data/parametric_drive/omega_flux_sweep.h5"
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# The sim time grid can be very fine (tens of thousands of points); thin it so
# the per-frequency Hankel/decoherence fits stay fast.
TIME_STRIDE = 40


def run(h5_path: str, tag: str | None = None, *, time_stride: int = TIME_STRIDE,
        channel_convention: str = "projectors", show: bool = False):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    tag = tag or "sim_" + os.path.splitext(os.path.basename(h5_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, tag)

    ds = load_parametric_sim_h5(h5_path, time_stride=time_stride,
                                channel_convention=channel_convention)
    estimator = ParametricDriveDecoherenceEstimator()
    # rho11_offset=0 / rho11_scale=1: the simulated population is already rho_11.
    results, figs = estimator.analyze(
        ds, output_dir=out_dir, verbose=False, rho11_offset=0.0, rho11_scale=1.0
    )
    print(
        f"[{tag}] freqs={results['n_freq']}  times={ds.sizes['driving_time']}  "
        f"decoh_ok={results['n_decoh_ok']}  -> {out_dir}"
    )
    if show:
        plt.show()
    else:
        plt.close("all")
    return out_dir


def main(argv):
    path = argv[1] if len(argv) > 1 else EXAMPLE_SIM
    if not os.path.exists(path):
        print(f"[skip] sim file not found -> {path}")
        return
    run(path, show=len(argv) > 1)
    print(f"\nArtifacts (metadata + plot data + figures) under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main(sys.argv)
