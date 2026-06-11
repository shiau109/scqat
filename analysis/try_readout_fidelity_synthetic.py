"""
Synthetic readout-fidelity sweep — generate fake IQ data with KNOWN distributions and
check that the readout-fidelity estimator recovers them. No QM, no saved data.

Edit the PARAMETERS cell, then run cell-by-cell (each ``# %%`` is a VS Code cell) or
headless: ``python analysis/try_readout_fidelity_synthetic.py``.

Model
-----
At each swept point the two readout states are 2-D Gaussian blobs:
  * prepared_state 0 -> blob centred at CENTER0,
  * prepared_state 1 -> blob CENTER0 + SEPARATION * û,   û = (cos ANGLE, sin ANGLE).
Each blob has width SIGMA, optionally elongated by SIGMA_RATIO along the separation axis
û and rotated by ANGLE (make ANGLE vary with the sweep to mimic a readout-frequency phase
roll — the case that used to alias the fitted σ). A fraction P_MISASSIGN of every state's
shots are placed in the OTHER blob to mimic readout/prep error; this is the bimodal
contamination the estimator's robust (MAD) σ is meant to ignore.

The generator returns the ground truth (separation, σ, SNR, ideal fidelity) per sweep
point so the last cell can print KNOWN vs RECOVERED and you can see how well — and under
what parameters — the estimator holds up.
"""

# %% Setup — make scqat importable, define the synthetic generator
import os
import sys

import numpy as np
import xarray as xr
from scipy.special import erf

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
_ROOT = os.path.dirname(_HERE)  # repo root, so `scqat` resolves when run headless
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scqat.estimators.readout_fidelity import (
    ReadoutFreqFidelityEstimator,
    ReadoutPowerFidelityEstimator,
    ReadoutFidelityEstimator,
)


def _ideal_fidelity(sep, sigma_proj, p_mis):
    """Mean-diagonal fidelity a perfect nearest-centre classifier would score: each
    correctly-prepared shot is mis-labelled with probability Q(sep / 2σ_proj) (the
    Gaussian tail across the midpoint boundary), and the P_MISASSIGN shots sitting in the
    other blob are mis-labelled the complementary amount. σ_proj is the blob width along
    the separation axis."""
    phi = 0.5 * (1 + erf((sep / (2 * sigma_proj)) / np.sqrt(2)))  # P(correct | own blob)
    return (1 - p_mis) * phi + p_mis * (1 - phi)


def make_readout_sweep(sweep_values, n_shots, separation, sigma, coord, *,
                       sigma_ratio=1.0, angle=0.0, center0=(0.0, 0.0),
                       p_misassign=0.0, seed=0):
    """Build a synthetic readout sweep Dataset (vars I/Q over coords
    ``coord``/``prepared_state``/``shot_idx``) plus a ``truth`` dict. Every blob/error
    parameter may be a scalar (constant across the sweep) or a length-S array."""
    rng = np.random.default_rng(seed)
    S = len(sweep_values)
    sep = np.broadcast_to(np.asarray(separation, float), (S,))
    sig = np.broadcast_to(np.asarray(sigma, float), (S,))
    ang = np.broadcast_to(np.asarray(angle, float), (S,))
    pmis = np.broadcast_to(np.asarray(p_misassign, float), (S,))
    c0 = np.asarray(center0, float)

    s_par = sig * np.sqrt(sigma_ratio)    # width along the separation axis
    s_perp = sig / np.sqrt(sigma_ratio)   # width perpendicular to it
    sigma_rms = np.sqrt((s_par**2 + s_perp**2) / 2)  # isotropic-equivalent σ the estimator reports

    I = np.empty((S, 2, n_shots))
    Q = np.empty((S, 2, n_shots))
    for i in range(S):
        u = np.array([np.cos(ang[i]), np.sin(ang[i])])   # separation direction
        w = np.array([-u[1], u[0]])                       # perpendicular direction
        centers = (c0, c0 + sep[i] * u)
        L = np.column_stack([s_par[i] * u, s_perp[i] * w])  # covariance = L @ L.T
        for s in range(2):
            own = rng.random(n_shots) >= pmis[i]            # True -> own blob, False -> other
            base = np.where(own[:, None], centers[s], centers[1 - s])
            pts = base + rng.standard_normal((n_shots, 2)) @ L.T
            I[i, s], Q[i, s] = pts[:, 0], pts[:, 1]

    truth = {
        "separation": np.array(sep),
        "sigma": sigma_rms,                       # matches the estimator's reported std
        "snr": np.array(sep) / sigma_rms,         # matches the estimator's snr (sep / σ)
        "ideal_fidelity": _ideal_fidelity(np.array(sep), s_par, np.array(pmis)),
    }
    ds = xr.Dataset(
        {"I": ([coord, "prepared_state", "shot_idx"], I),
         "Q": ([coord, "prepared_state", "shot_idx"], Q)},
        coords={coord: sweep_values, "prepared_state": [0, 1], "shot_idx": np.arange(n_shots)},
    )
    return ds, truth


# %% PARAMETERS — EDIT ME ----------------------------------------------------------------
SWEEP_COORD = "frequency"                          # "frequency" | "amp_prefactor" | any name
SWEEP_VALUES = np.linspace(-2e6, 2e6, 21)          # the swept axis (Hz here)
N_SHOTS = 2000                                     # shots per (sweep point, state)

SIGMA = 7e-5                                        # blob width (scalar or length-S array)
# Separation between the |0>/|1> blobs. Here: a Gaussian bump peaking at band centre, so
# fidelity/SNR have a clear interior optimum. Try a constant, or a monotonic ramp for power.
SEPARATION = 1.5e-4 + 2.0e-4 * np.exp(-0.5 * (SWEEP_VALUES / 8e5) ** 2)

SIGMA_RATIO = 1.0          # 1.0 = round blobs; >1 elongates ALONG the separation axis
ANGLE = 0.0                # blob orientation (rad). e.g. np.linspace(0, np.pi, SWEEP_VALUES.size)
                           # rotates the blobs across the sweep (mimics the readout-freq phase roll;
                           # combine with SIGMA_RATIO>1 to stress the rotation-robust σ).
CENTER0 = (0.0, 0.0)       # |0> blob centre
P_MISASSIGN = 0.00         # fraction of each state's shots placed in the OTHER blob (readout error)
SEED = 0

OUTLIERS_THRESHOLD = 0.98  # only used when SWEEP_COORD == "amp_prefactor"
OUT_DIR = os.path.join(_HERE, "synthetic_figs")    # where figures/metadata are written (None = don't save)
# ---------------------------------------------------------------------------------------


# %% Build the synthetic dataset with known truth
DS, TRUTH = make_readout_sweep(
    SWEEP_VALUES, N_SHOTS, SEPARATION, SIGMA, SWEEP_COORD,
    sigma_ratio=SIGMA_RATIO, angle=ANGLE, center0=CENTER0,
    p_misassign=P_MISASSIGN, seed=SEED,
)


# %% Run the estimator (figures + metadata saved to OUT_DIR)
if SWEEP_COORD == "frequency":
    EST, KW = ReadoutFreqFidelityEstimator(), {}
elif SWEEP_COORD == "amp_prefactor":
    EST, KW = ReadoutPowerFidelityEstimator(), {"outliers_threshold": OUTLIERS_THRESHOLD}
else:
    EST, KW = ReadoutFidelityEstimator(), {"sweep_coord": SWEEP_COORD}

RES, FIGS = EST.analyze(DS, output_dir=OUT_DIR, **KW)
print(f"estimator: {EST.estimator_name}")
print(f"figures:   {sorted(FIGS)}")
if OUT_DIR:
    print(f"saved ->   {OUT_DIR}")


# %% KNOWN vs RECOVERED — does the estimator reproduce the planted distributions?
mean = np.asarray(RES["mean"])                                   # (S, center, iq)
sep_fit = np.linalg.norm(mean[:, 0, :] - mean[:, 1, :], axis=1)  # recovered separation
sig_fit, snr_fit, fid_fit = (np.asarray(RES[k]) for k in ("std", "snr", "fidelity"))

hdr = (f"{'idx':>3} {'sweep':>11} "
       f"{'sep_true':>9} {'sep_fit':>9} {'sig_true':>9} {'sig_fit':>9} "
       f"{'snr_true':>9} {'snr_fit':>8} {'fid_true':>9} {'fid_fit':>8}")
print("\n" + hdr)
print("-" * len(hdr))
for i, v in enumerate(SWEEP_VALUES):
    print(f"{i:>3} {v:>11.4g} "
          f"{TRUTH['separation'][i]:>9.3e} {sep_fit[i]:>9.3e} "
          f"{TRUTH['sigma'][i]:>9.3e} {sig_fit[i]:>9.3e} "
          f"{TRUTH['snr'][i]:>9.3f} {snr_fit[i]:>8.3f} "
          f"{TRUTH['ideal_fidelity'][i]:>9.4f} {fid_fit[i]:>8.4f}")

# overall recovery error + where each curve peaks
def _rel_err(true, fit):
    true = np.asarray(true, float)
    return float(np.nanmean(np.abs(fit - true) / np.where(true == 0, np.nan, np.abs(true))))

print(f"\nmean |rel error|:  sigma {_rel_err(TRUTH['sigma'], sig_fit):.1%} | "
      f"separation {_rel_err(TRUTH['separation'], sep_fit):.1%} | "
      f"snr {_rel_err(TRUTH['snr'], snr_fit):.1%}")
print(f"fidelity peak:     true @ {SWEEP_VALUES[np.argmax(TRUTH['ideal_fidelity'])]:.4g} | "
      f"estimator best @ {RES['best_sweep_value']:.4g} "
      f"(fid {RES['best_fidelity']:.4f}, success={RES['success']})")
