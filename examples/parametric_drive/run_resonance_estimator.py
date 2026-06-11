"""Run ``ParametricDriveResonanceEstimator`` on a fixed-time parametric-drive map.

This is the estimator the renewed ``LCH_qubit_parametric_drive_fixed_time`` node
calls: on the 2-D ``amplitude_ratio`` x ``driving_frequency`` map (measured at a
fixed drive time) it fits a Lorentzian per amplitude slice and returns the cleaned
resonance-peak point-cloud, plus the map figure with peaks/outliers overlaid.

No fixed-time dataset was provided, so by default this runs on a **synthetic**
drifting-ridge map (so the example is runnable as-is). Pass a real ``ds_raw.h5``
to analyse measured data:

    python examples/parametric_drive/run_resonance_estimator.py                 # synthetic demo
    python examples/parametric_drive/run_resonance_estimator.py path/ds_raw.h5  # real data

Artifacts are written under ``examples/parametric_drive/output/<tag>/<qubit>/``.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.parsers.xarray_h5_parser import load_xarray_h5
from scqat.parsers import repetition_data
from scqat.estimators import ParametricDriveResonanceEstimator

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def synthetic_map(n_amp: int = 9, n_freq: int = 161, noise: float = 4e-3, seed: int = 0):
    """A Lorentzian resonance per amplitude row whose centre drifts with amplitude."""
    amp = np.linspace(1.4, 1.8, n_amp)
    freq = np.linspace(328e6, 332e6, n_freq)
    rng = np.random.default_rng(seed)
    hwhm = (freq[-1] - freq[0]) / 25.0
    f0 = np.linspace(329e6, 331e6, n_amp)
    state = np.empty((n_amp, n_freq))
    for k in range(n_amp):
        lor = 0.6 / (1.0 + ((freq - f0[k]) / hwhm) ** 2)
        state[k] = 0.1 + lor + noise * rng.standard_normal(n_freq)
    return xr.Dataset(
        {"state": (("amplitude_ratio", "driving_frequency"), state)},
        coords={"amplitude_ratio": amp, "driving_frequency": freq},
    )


def run(ds: xr.Dataset, tag: str, *, show: bool = False, **estimator_kwargs):
    out_dir = os.path.join(OUTPUT_ROOT, tag)
    per_qubit = repetition_data(ds, "qubit") if "qubit" in ds.dims else [ds]
    estimator = ParametricDriveResonanceEstimator()
    for sq in per_qubit:
        qname = str(sq["qubit"].values.item()) if "qubit" in sq.coords else "q"
        qdir = os.path.join(out_dir, qname)
        results, figs = estimator.analyze(sq, output_dir=qdir, **estimator_kwargs)
        print(
            f"[{tag}/{qname}] amplitudes={results['n_amp']}  "
            f"peaks={results['n_peaks']}  kept={results['n_good']}  -> {qdir}"
        )
        if show:
            plt.show()
        else:
            plt.close("all")
    return out_dir


def main(argv):
    if len(argv) > 1:
        path = argv[1]
        ds = load_xarray_h5(path)
        tag = os.path.basename(os.path.dirname(path))
        run(ds, tag=tag, show=True)
    else:
        print("[info] no ds_raw.h5 given -> running the synthetic drifting-ridge demo.")
        run(synthetic_map(), tag="synthetic_demo")
    print(f"\nArtifacts (metadata + plot data + figures) under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main(sys.argv)
