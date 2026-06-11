"""
Offline readout-power analysis on saved raw data — no QM, no node.

Runs ``ReadoutPowerFidelityEstimator`` (per-amplitude state discrimination -> optimal
readout amp_prefactor, constrained by ``outliers_threshold``) against a real
LCH_readout_power run's ``ds_raw.h5``. Run cell-by-cell in VS Code (each ``# %%`` is a
cell) or headless: ``python analysis/try_readout_power_new.py``.

    (A) RE-FIT + REPLOT  — re-fit the saved raw data per qubit, regenerate the estimator's
                            own figures (std / outlier / norm_res / fidelity / mean_distance
                            / mean_I / mean_Q / means_on_IQ), and print the metadata.
    (B) SUMMARY TABLE    — best amp_prefactor / fidelity / success per qubit.

Per experiment you only set the data path and ``OUTLIERS_THRESHOLD`` (mirrors the node's
parameter); the readout estimator consumes I/Q directly, so no ``prep`` is needed.
Everything else is reused from ``_harness.py``.

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
from scqat.estimators.readout_fidelity import ReadoutPowerFidelityEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-10\#166_LCH_readout_power_232839\ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)
OUTLIERS_THRESHOLD = 0.98  # mirrors the node's parameter; gates which amplitudes are eligible

EST = ReadoutPowerFidelityEstimator()
DS = load(DATA)


# %% (A) RE-FIT + REPLOT — regenerate the estimator's figures from saved raw
# Figures land in <run>/replot_check/<qubit>__<figname>.png; metadata is printed per qubit.
# outliers_threshold is forwarded to analyze so the offline best-point matches the node.
figures = replot(EST, slices(DS),
                 out_dir=os.path.join(DATA_DIR, "replot_check"),
                 outliers_threshold=OUTLIERS_THRESHOLD)


# %% (B) SUMMARY TABLE — best amp_prefactor / fidelity / success per qubit
print(f"\n{'qubit':6} {'best_amp_prefactor':>18} {'best_fidelity':>14} {'success':>9}")
print("-" * 50)
for name, sq in slices(DS):
    res = EST.analyze(sq, output_dir=None, skip_figures=True,
                      outliers_threshold=OUTLIERS_THRESHOLD)[0]
    amp = res["best_sweep_value"]
    amp = amp if amp is not None else float("nan")
    print(f"{name:6} {amp:>18.4f} {res['best_fidelity']:>14.4f} {str(res['success']):>9}")
