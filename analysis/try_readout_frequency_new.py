"""
Offline readout-frequency analysis on saved raw data — no QM, no node.

Runs ``ReadoutFreqFidelityEstimator`` (per-frequency state discrimination -> optimal
readout detuning) against a real LCH_readout_frequency run's ``ds_raw.h5``. Run
cell-by-cell in VS Code (each ``# %%`` is a cell) or headless:
``python analysis/try_readout_frequency_new.py``.

    (A) RE-FIT + REPLOT  — re-fit the saved raw data per qubit, regenerate the estimator's
                            own figures (std / outlier / norm_res / fidelity / mean_distance
                            / mean_I / mean_Q / means_on_IQ), and print the metadata.
    (B) SUMMARY TABLE    — best detuning / fidelity / success per qubit.

Per experiment you only set the data path; the readout estimator consumes I/Q directly,
so no ``prep`` is needed. Everything else is reused from ``_harness.py``.

No-re-fit alternative: a run saved with ``save_plot_data=True`` can be redrawn without
refitting via ``replot(EST, from_plotdata=DATA_DIR)``.
"""

# %% Setup — make scqat + _harness importable, load the dataset once, pick the estimator
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.dirname(_HERE)  # repo root, so `scqat` resolves when run headless
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _harness import load, slices, replot
from scqat.estimators.readout_fidelity import ReadoutFreqFidelityEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-10\#165_LCH_readout_frequency_232708\ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = ReadoutFreqFidelityEstimator()
DS = load(DATA)


# %% (A) RE-FIT + REPLOT — regenerate the estimator's figures from saved raw
# Figures land in <run>/replot_check/<qubit>__<figname>.png; metadata is printed per qubit.
# ds_raw already carries I/Q + shot_idx/frequency/prepared_state, so no prep is needed.
figures = replot(EST, slices(DS),
                 out_dir=os.path.join(DATA_DIR, "replot_check"))


# %% (B) SUMMARY TABLE — best detuning (relative to the readout IF) / fidelity / success
print(f"\n{'qubit':6} {'best_detuning/MHz':>18} {'best_fidelity':>14} {'success':>9}")
print("-" * 50)
for name, sq in slices(DS):
    res = EST.analyze(sq, output_dir=None, skip_figures=True)[0]
    det = res["best_sweep_value"]
    det_mhz = det / 1e6 if det is not None else float("nan")
    print(f"{name:6} {det_mhz:>18.3f} {res['best_fidelity']:>14.4f} {str(res['success']):>9}")
