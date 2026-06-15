"""
Offline Ramsey analysis on saved raw data — no QM, no node.

Runs ``RamseyEstimator`` (auto single-vs-beat damped-oscillation fit -> drive detuning
``f_1`` + dephasing time ``tau_1``) against a real LCH_Ramsey run's ``ds_raw.h5``. Run
cell-by-cell in VS Code (each ``# %%`` is a cell) or headless:
``python analysis/try_ramsey_new.py``.

    (A) RE-FIT + REPLOT  — re-fit the saved raw data per qubit, regenerate the estimator's
                            own time_domain + fft_spectrum figures, and print the params.
    (B) SUMMARY TABLE    — model_type / f_1 / tau_1 per qubit.
    (C) Test force_model — re-fit forcing 'single' vs 'beat' vs auto-detect.
    (D) REPLOT no re-fit — reconstruct figures from a run saved with save_plot_data=True.

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
from scqat.estimators.ramsey import RamseyEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-10\#133_LCH_Ramsey_171821\ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = RamseyEstimator()
DS = load(DATA)


# %% Per-experiment glue — prep (raw -> estimator input)
def prep(sq: xr.Dataset) -> xr.Dataset:
    # RamseyEstimator fits the 'signal' variable over the 'idle_time' coord. The node
    # stores raw I/Q, or 'state' when state discrimination is on; use I as the signal
    # (or state), matching LCH_Ramsey.analyse_data.
    if "I" in sq:
        return sq.rename({"I": "signal"})
    if "state" in sq:
        return sq.rename({"state": "signal"})
    return sq


# %% (A) RE-FIT + REPLOT — regenerate the estimator's time_domain + fft_spectrum figures
# Figures land in <run>/replot_check/<qubit>__<figname>.png; metadata is printed per qubit.
figures = replot(EST, slices(DS, prep=prep),
                 out_dir=os.path.join(DATA_DIR, "replot_check"))


# %% (B) SUMMARY TABLE — model_type / f_1 / tau_1 per qubit (f_1, tau_1 in 1/idle_time units)
print(f"\n{'qubit':6} {'model':>8} {'f_1':>14} {'tau_1':>14}")
print("-" * 46)
for name, sq in slices(DS, prep=prep):
    res = EST.analyze(sq, output_dir=None, skip_figures=True)[0]
    print(f"{name:6} {res['model_type']:>8} {res['f_1']:>14.6g} {res['tau_1']:>14.6g}")


# %% (C) Test force_model — re-fit forcing 'single' vs 'beat' vs auto-detect (None)
# force_model is RamseyEstimator's only extract_parameters knob (the analog of the
# qubit_spectroscopy prominence grid). Figures per setting land in <run>/replot_<setting>/.
for fm in ("single", "beat", None):
    setting = fm or "auto"
    print(f"\n=== force_model={setting} ===")
    replot(EST, slices(DS, prep=prep),
           out_dir=os.path.join(DATA_DIR, f"replot_{setting}"),
           force_model=fm)


# %% (D) REPLOT with NO re-fit — from a run saved with save_plot_data=True
# Point at any LCH_Ramsey run folder that contains plotdata_*.h5; the time_domain + FFT
# figures are reconstructed straight from the saved plot-data (zero recomputation).
SAVED_RUN = r"<<< EDIT: an LCH_Ramsey run folder containing plotdata_*.h5 >>>"
if os.path.isdir(SAVED_RUN):
    replot(EST, from_plotdata=SAVED_RUN, out_dir=os.path.join(SAVED_RUN, "replot_check"))
else:
    print(f"(D) skipped — set SAVED_RUN to a real run folder with plotdata_*.h5 "
          f"(got {SAVED_RUN!r})")
