"""
Offline qubit-spectroscopy analysis on saved raw data — no QM, no node.

Three use cases, one engine (``_harness.py``). Run cell-by-cell in VS Code (each
``# %%`` is a cell) or headless: ``python analysis/try_qubit_spectroscopy_new.py``.

    (A) Try a NEW approach   — edit ``m_new``; compare it against the estimator.
    (B) Test PARAMETERS      — run the estimator across a grid of kwargs.
    (C) REPLOT a skipped run — regenerate the estimator's figures from stored ds_raw.

Per experiment you only set: the data path, ``prep`` (raw -> estimator input) and
``adapt`` (results -> normalized plot fields). Everything else is reused.
"""

# %% Setup — make _harness importable, load the dataset once, pick the estimator
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import xarray as xr

from _harness import load, slices, compare, estimator_method, replot
from scqat.estimators.qubit_spectroscopy import QubitSpectroscopyEstimator
from scqat.tools.fit_lorentzian import FitLorentzian, lorentzian

DATA = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-09\#54_LCH_qubit_spectroscopy_090054/ds_raw.h5"
DATA_DIR = os.path.dirname(DATA)

EST = QubitSpectroscopyEstimator()
DS = load(DATA)


# %% Per-experiment glue — prep (raw -> estimator input) + adapt (results -> plot fields)
def prep(sq: xr.Dataset) -> xr.Dataset:
    # qubit_spectroscopy needs a complex IQdata variable
    return sq.assign(IQdata=sq.I + 1j * sq.Q)


def adapt(res: dict, sq: xr.Dataset):
    peaks = res.get("peaks", [])
    if not peaks:
        return None
    p = peaks[0]
    return {
        "detuning": float(p["detuning"]), "fwhm": float(p["fwhm"]),
        "x": sq.detuning.values.astype(float),
        "y": np.asarray(res["signal_corrected"], float),
        "fit_x": np.asarray(p["fit_x"], float),
        "fit_y": np.asarray(p["fit_y"], float),
    }


# the current estimator as a reusable baseline method
m_estimator = estimator_method(EST, adapt, label="estimator (baseline)", max_peaks=1)


# %% (A) Try a NEW approach — <<< EDIT m_new >>> (example: rotated-I + single Lorentzian)
def m_new(sq: xr.Dataset) -> dict:
    det = sq.detuning.values.astype(float)
    I = sq.I.values.astype(float)
    Q = sq.Q.values.astype(float)
    iq = I + 1j * Q

    k = int(np.argmax(np.abs(iq - iq.mean())))
    ang = np.arctan2(Q[k] - Q.mean(), I[k] - I.mean())
    i_rot = I * np.cos(ang) + Q * np.sin(ang)

    base = np.polyval(np.polyfit(det, i_rot, 1), det)
    y = i_rot - base
    inverted = abs(y.min()) > abs(y.max())
    da = xr.DataArray(y, coords={"x": det}, dims="x")
    fit = FitLorentzian(
        da, inverted=inverted,
        bounds={"x0": (det.min(), det.max()), "gamma": (0.0, det.max() - det.min())},
    ).fit()
    p = fit.params
    x0, amp = float(p["x0"].value), float(p["amplitude"].value)
    gamma, off = float(p["gamma"].value), float(p["offset"].value)
    return {
        "label": "new (rotated-I + Lorentzian)", "ok": True,
        "detuning": x0, "fwhm": 2 * abs(gamma),
        "x": det, "y": y, "fit_x": det, "fit_y": lorentzian(det, x0, amp, gamma, off),
    }


compare(slices(DS, prep=prep), [m_estimator, m_new],
        out_png=os.path.join(DATA_DIR, "compare_new_vs_estimator.png"))


# %% (B) Test PARAMETERS — same estimator + data, swept kwargs (here: peak prominence)
param_methods = [
    estimator_method(EST, adapt, label=f"prominence={p}", max_peaks=2, prominence=p)
    for p in (0.05, 0.1, 0.2, 0.4)
]
compare(slices(DS, prep=prep), param_methods,
        out_png=os.path.join(DATA_DIR, "compare_params.png"))


# %% (C) REPLOT by RE-FITTING saved raw — figures a plot-skipped run would have made
replot(EST, slices(DS, prep=prep),
       out_dir=os.path.join(DATA_DIR, "replot"), max_peaks=1)


# %% (D) REPLOT with NO re-fit — from a run saved with save_plot_data=True
# Point at any LCHQM run folder that contains plotdata_*.h5; figures are reconstructed
# straight from the saved plot-data (zero recomputation).
SAVED_RUN = r"D:\SynologyDrive\LiChiehHsiao\AS\SynologyDrive\data\raw_data\2026-06-09\#54_LCH_qubit_spectroscopy_090054"
replot(EST, from_plotdata=SAVED_RUN, out_dir=os.path.join(SAVED_RUN, "replot_check"))
