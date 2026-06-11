"""
Offline readout-fidelity analysis on saved raw data — no QM, no node.

Runs ``StateDiscriminationEstimator`` (2D-GMM single-shot discrimination at the current
readout settings) against a real LCH_readout_fidelity run's ``ds_raw.h5``. Run cell-by-cell
in VS Code (each ``# %%`` is a cell) or headless:
``python analysis/try_readout_fidelity_new.py``.

    (A) RE-FIT + REPLOT  — re-fit the saved raw data per qubit, regenerate the estimator's
                            own figures (raw / 2DHist / outliers / fit_residue), and print
                            the metadata.
    (B) SUMMARY TABLE    — readout fidelity per qubit (mean of the confusion-matrix
                            diagonal, the same quantity the node uses for node.outcomes).

This node has no sweep; per experiment you only set the data path. The estimator consumes
I/Q directly, so no ``prep`` is needed. Everything else is reused from ``_harness.py``.

No-re-fit alternative: a run saved with ``save_plot_data=True`` can be redrawn without
refitting via ``replot(EST, from_plotdata=DATA_DIR)``.
"""

# %% Setup — make scqat + _harness importable, load the dataset once, pick the estimator
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.dirname(_HERE)  # repo root, so `scqat` resolves when run headless
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _harness import load, slices, replot
from scqat.estimators.state_discrimination import StateDiscriminationEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-10\#162_LCH_readout_fidelity_225829\ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = StateDiscriminationEstimator()
DS = load(DATA)


# %% (A) RE-FIT + REPLOT — regenerate the estimator's figures from saved raw
# Figures land in <run>/replot_check/<qubit>__<figname>.png. print_metadata=False keeps
# the output tidy (the GMM metadata is bulky); the (B) cell prints the fidelity summary.
figures = replot(EST, slices(DS),
                 out_dir=os.path.join(DATA_DIR, "replot_check"),
                 print_metadata=False)


# %% (B) SUMMARY TABLE — readout fidelity = mean of the direct_counts diagonal
print(f"\n{'qubit':6} {'readout_fidelity':>16}")
print("-" * 24)
for name, sq in slices(DS):
    res = EST.analyze(sq, output_dir=None, skip_figures=True)[0]
    dc = np.asarray(res["direct_counts"])
    n = min(dc.shape[0], dc.shape[1])
    fidelity = float(np.mean([dc[k, k] for k in range(n)]))
    print(f"{name:6} {fidelity:>16.4f}")
