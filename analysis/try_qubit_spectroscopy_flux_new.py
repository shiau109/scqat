"""
Offline qubit-spectroscopy-vs-flux re-fit on saved raw data — no QM, no node.

Sibling of ``try_qubit_spectroscopy_new.py`` for the **2-D flux map**: the estimator
``QubitSpectroscopyFluxEstimator`` fits the qubit peak(s) **flux-by-flux** and returns a
point-cloud ``(flux, detuning, fwhm, amplitude)`` (not a single peak), so the single-slice
``compare``/``adapt`` flow doesn't apply here — the per-flux single-Lorentzian comparison
already lives in ``try_qubit_spectroscopy_new.py``. The natural offline uses for this
estimator are therefore:

    (C) REPLOT / RE-FIT  — re-fit the saved ``ds_raw`` and regenerate the flux-map figures.
    (B) TEST PARAMETERS  — re-fit across a grid of cleaning/detection kwargs (here: ``n_sigma``).

Run cell-by-cell in VS Code (each ``# %%`` is a cell) or headless:
``python analysis/try_qubit_spectroscopy_flux_new.py``. Reuses the ``_harness.py`` engine
(``load``, ``slices``, ``replot``); per experiment you only set the data path and ``prep``.
"""

# %% Setup — make _harness importable, load the dataset once, pick the estimator
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import xarray as xr

from _harness import load, slices, replot
from scqat.estimators.qubit_spectroscopy_flux import QubitSpectroscopyFluxEstimator

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-09\#71_LCH_qubit_spectroscopy_flux_234456/ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = QubitSpectroscopyFluxEstimator()
DS = load(DATA)


# %% Per-experiment glue — prep (raw -> estimator input)
def prep(sq: xr.Dataset) -> xr.Dataset:
    # This run's ds_raw is already processed: it carries the flux_bias + detuning coords
    # (and full_freq, so peaks are also reported in absolute frequency) plus the I/Q
    # quadratures, and QubitSpectroscopyFluxEstimator builds IQdata from I/Q itself.
    # So there is nothing to reshape — pass the per-qubit slice straight through.
    return sq


# %% (C) RE-FIT saved raw — regenerate the flux-map figures with the STATISTICAL OUTLIER
# REJECTION DISABLED (n_sigma=np.inf).
#
# Expected truth for #50: q4 has TWO transition branches with very different width/height —
# an always-on line near ~3.925 GHz and a fainter line near ~3.840 GHz that is present at only
# some flux. The estimator pools every peak's width/amplitude and flags MAD outliers, so with
# the default n_sigma=3.0 the minority ~3.840 GHz branch is wrongly rejected (q4: 8 dropped).
# Setting n_sigma=np.inf turns the test off (z > inf is never true -> good == in_window), which
# keeps BOTH branches. q5 has no real qubit line here (a few scattered detections only).
# The estimator now DEFAULTS to max_peaks=4 (each flux slice capped to its 4 most-prominent
# peaks — headroom over q4's 2 real branches), so a noisy slice can't spray spurious peaks
# once the outlier test is off; no need to pass it here. Use max_peaks=None to keep all.
replot(EST, slices(DS, prep=prep),
       out_dir=os.path.join(DATA_DIR, "replot"), n_sigma=np.inf, prominence=0.5)


# %% (B) TEST PARAMETERS — same estimator + data, swept n_sigma (default vs outlier test OFF)
# Compact per-qubit count table for each n_sigma (reuses slices + analyze; no replotting), then
# a full replot per setting into its own folder so the figures can be compared. For q4 expect
# n_good to jump (lower ~3.840 GHz branch recovered) and n_outlier -> 0 as n_sigma -> inf.
def summarize(slices_, **kw) -> None:
    print(f"{'qubit':6} {'n_peaks':>8} {'n_good':>7} {'n_outlier':>10} {'n_flux':>7}  ({kw})")
    print("-" * 60)
    for name, sq in slices_:
        r = EST.analyze(sq, output_dir=None, skip_figures=True, **kw)[0]
        print(f"{name:6} {r['n_peaks']:>8} {r['n_good']:>7} {r['n_outlier']:>10} {r['n_flux']:>7}")


# for ns in (3.0, np.inf):
summarize(slices(DS, prep=prep), prominence=0.1, n_sigma=np.inf)
replot(EST, slices(DS, prep=prep),
        out_dir=os.path.join(DATA_DIR, f"replot_nsigma_inf"),
        prominence=0.1, n_sigma=np.inf)
