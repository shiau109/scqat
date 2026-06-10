"""
Offline power-Rabi analysis on saved raw data — no QM, no node.

Runs ``PowerRabiEstimator`` (cosine fit -> optimal pi-amplitude prefactor) against a
real LCH_power_rabi run's ``ds_raw.h5``. Run cell-by-cell in VS Code (each ``# %%`` is a
cell) or headless: ``python analysis/try_power_rabi_new.py``.

    (A) RE-FIT + REPLOT  — re-fit the saved raw data per qubit, regenerate the estimator's
                            own amplitude figures, and print the extracted parameters.
    (B) SUMMARY TABLE    — opt_amp_prefactor / frequency / success per qubit.

Per experiment you only set: the data path and ``prep`` (raw -> estimator input).
Everything else is reused from ``_harness.py``.
"""

# %% Setup — make scqat + _harness importable, load the dataset once, pick the estimator
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.dirname(_HERE)  # repo root, so `scqat` resolves when run headless
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import xarray as xr

from _harness import load, slices, replot
from scqat.estimators.power_rabi import PowerRabiEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-10\#73_LCH_power_rabi_102035\ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = PowerRabiEstimator()
DS = load(DATA)


# %% Per-experiment glue — prep (raw -> estimator input)
def prep(sq: xr.Dataset) -> xr.Dataset:
    # PowerRabiEstimator fits the 'signal' variable over the 'amp_prefactor' coord.
    # The node stores raw I/Q; use I as the signal (matching the node's analyse_data).
    return sq.rename({"I": "signal"})


# %% (A) RE-FIT + REPLOT — regenerate the estimator's amplitude figures from saved raw
# Figures land in <run>/replot_check/<qubit>__amplitude.png; metadata is printed per qubit.
figures = replot(EST, slices(DS, prep=prep),
                 out_dir=os.path.join(DATA_DIR, "replot_check"))


# %% (B) SUMMARY TABLE — opt_amp_prefactor / frequency / success per qubit
print(f"\n{'qubit':6} {'opt_amp_prefactor':>18} {'f':>12} {'success':>9}")
print("-" * 48)
for name, sq in slices(DS, prep=prep):
    res = EST.analyze(sq, output_dir=None, skip_figures=True)[0]
    print(f"{name:6} {res['opt_amp_prefactor']:>18.4f} {res['f']:>12.4f} {str(res['success']):>9}")
