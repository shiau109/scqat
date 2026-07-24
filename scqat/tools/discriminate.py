"""Per-slice IQ state discrimination — the reduction shared by a FAMILY.

One block of single-shot I/Q clouds (one row per prepared state) in, the
trained GMM + assignment summary out. This is the pure-math reduction behind
every discrimination experiment: single-shot readout uses it once, the
readout-fidelity sweeps call it once per sweep point, and qubit tomography
calls it once on its training shots. Per the repo rule ("anything used by more
than one estimator lives in tools/"), it lives here — estimators compose it,
they never call each other.

Pipeline (single method): robust (MAD) per-state blob width -> 2-D histogram
binning -> centres seeded from the per-state density maxima then refined onto
each blob's density PEAK by nearest-centre-confined mean-shift (:func:`_mode_refine`;
skipped when ``user_mean`` pins the centres) -> global 2-D multi-Gaussian fit
(:class:`scqat.tools.fit_gaussian2d.FitMultiGaussian2D`; centres pinned, one
shared width + per-Gaussian amplitude vary) -> per-state amplitude refit ->
nearest-centre shot assignment -> confusion fractions + n-sigma outlier tagging.

Result contract
---------------
``{trained_paras, fitted_paras, gaussian_norms, direct_counts, state_label,
outlier_mask, outlier_probability, norm_res, fit_residues, density, hist_x,
hist_y}``. ``trained_paras`` is ``{mean (n_center, 2), std, covariance,
amp}``. ``direct_counts`` has the FIXED shape ``(n_state, n_center)`` — row k
is the fraction of prepared-state-k shots assigned to each centre, including
centres that captured no shot at all (columns of zeros). Historical note: the
estimator-era reduction sized this axis by the highest label actually
assigned, so a centre with no shots silently shortened the matrix and
consumers indexing ``counts[1, 1]`` crashed on degenerate data.

Callers that loop over slices and collect tunables in a dict should call
:func:`validate_discriminate_kwargs` ONCE before the loop, so a typo'd knob
dies loudly instead of being swallowed by a per-slice ``try/except``.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .fit_gaussian2d import FitMultiGaussian2D

#: caller-selectable knobs of :func:`discriminate_states` — the single source
#: of truth dict-collecting callers validate against.
DISCRIMINATE_KNOBS = frozenset({"user_mean", "user_std", "outlier_sigma"})


def validate_discriminate_kwargs(knobs: Dict) -> None:
    """Raise ValueError for an unknown knob — call BEFORE slice loops."""
    unknown = set(knobs) - DISCRIMINATE_KNOBS
    if unknown:
        raise ValueError(
            f"Unknown knob(s) {sorted(unknown)} for state discrimination; "
            f"valid: {sorted(DISCRIMINATE_KNOBS)}"
        )


def _robust_sigma(a: np.ndarray) -> np.ndarray:
    """Per-state Gaussian width from the median absolute deviation
    (MAD / 0.6745, the consistency factor for a normal distribution); one value
    per row. MAD is used instead of the plain standard deviation because each
    prepared state's data is slightly bimodal — a few percent of shots are
    mis-assigned and sit in the OTHER blob — and a std of that bimodal data is
    inflated by the inter-blob separation. MAD ignores that minority tail and
    tracks the dominant blob's width."""
    med = np.median(a, axis=1, keepdims=True)
    return np.median(np.abs(a - med), axis=1) / 0.67448975


def _bin_histograms(I: np.ndarray, Q: np.ndarray, user_std: Optional[float]):
    """Bin the raw shots into per-state 2-D densities.

    Per-state single-blob width for bin sizing and as the GMM sigma seed: the
    rotation-invariant RMS of the two axes, sqrt((sigma_I**2 + sigma_Q**2) / 2),
    of the robust (MAD-based) per-axis widths.
     * RMS (not min(sigma_I, sigma_Q)) keeps the bin size from oscillating when
       an elongated blob rotates with a swept parameter (e.g. readout frequency,
       whose resonator phase rotates the IQ blobs) — min would alias into a
       period-2 zigzag in std/norm_res/outliers across the sweep.
     * MAD (not std) keeps the few-percent of mis-assigned shots in the other
       blob from inflating the width (a std of the bimodal per-state data
       overestimates sigma, blowing up the n-sigma circles and the SNR).
    """
    sig_I = _robust_sigma(I)
    sig_Q = _robust_sigma(Q)
    std_init = float(np.min(np.sqrt((sig_I ** 2 + sig_Q ** 2) / 2)))

    step = (user_std if user_std else std_init) / 3

    I_all, Q_all = I.ravel(), Q.ravel()
    xedges = np.arange(I_all.min(), I_all.max() + step, step)
    yedges = np.arange(Q_all.min(), Q_all.max() + step, step)
    if len(xedges) < 2:
        xedges = np.linspace(I_all.min(), I_all.max(), 2)
    if len(yedges) < 2:
        yedges = np.linspace(Q_all.min(), Q_all.max(), 2)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    n_state = I.shape[0]
    density = np.zeros((n_state, len(ycenters), len(xcenters)))
    mean_init = []
    for i in range(n_state):
        H, _, _ = np.histogram2d(I[i], Q[i], bins=[xedges, yedges], density=True)
        density[i] = H.T
        max_idx = np.unravel_index(np.argmax(H), H.shape)
        mean_init.append(np.array([xcenters[max_idx[0]], ycenters[max_idx[1]]]))

    return density, xcenters, ycenters, np.array(mean_init), std_init


def _mode_refine(pts: np.ndarray, center: np.ndarray, bandwidth: float,
                 n_iter: int = 50, tol: float = 1e-3) -> np.ndarray:
    """Gaussian mean-shift: climb from ``center`` to the local density PEAK of
    ``pts``.

    Each step moves to the Gaussian-kernel-weighted mean
    ``sum(w_i * pts_i) / sum(w_i)`` with ``w_i = exp(-|pts_i - c|**2 / 2h**2)``.
    Because the kernel falls off with distance, a low-density tail (readout T1
    smear, |2> leakage) is down-weighted and the fixed point is the MODE, not the
    centroid — the accurate density peak, robust to skew. ``pts`` is ``(N, 2)``;
    ``bandwidth`` is the blob width (the robust sigma). Returns ``center``
    unchanged for an empty ``pts`` (a centre that captured no shot)."""
    c = np.asarray(center, dtype=float)
    if pts.shape[0] == 0:
        return c
    h2 = 2.0 * bandwidth ** 2
    for _ in range(n_iter):
        w = np.exp(-((pts - c) ** 2).sum(axis=1) / h2)
        total = w.sum()
        if total == 0:
            break
        nxt = (w[:, None] * pts).sum(axis=0) / total
        if np.hypot(*(nxt - c)) < tol * bandwidth:
            c = nxt
            break
        c = nxt
    return c


def _refine_centres(I: np.ndarray, Q: np.ndarray, seed: np.ndarray,
                    bandwidth: float, n_pass: int = 2) -> np.ndarray:
    """Move each seed centre onto its blob's density peak (mean-shift), keeping
    the blobs separate via nearest-centre hard assignment.

    Mean-shift on the POOLED shots would let two close (~2 sigma) centres collapse
    onto one merged mode; assigning each shot to its nearest centre first confines
    each mean-shift to its own blob. Two passes (reassign after moving) remove the
    small assignment-boundary bias. Auto-seed path only — a pinned ``user_mean`` is
    never refined."""
    pts = np.column_stack([I.ravel(), Q.ravel()])
    centres = np.asarray(seed, dtype=float).copy()
    for _ in range(n_pass):
        dist = np.stack([np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) for c in centres])
        label = dist.argmin(axis=0)
        centres = np.array([
            _mode_refine(pts[label == k], centres[k], bandwidth)
            for k in range(len(centres))
        ])
    return centres


def _gmm_fit(density: np.ndarray, x: np.ndarray, y: np.ndarray,
             mean: Sequence, std: float):
    """One constrained multi-Gaussian fit: centres pinned, one shared width."""
    n_gauss = len(mean)
    fitter = FitMultiGaussian2D(density, x, y, n_gauss=n_gauss)
    fitter.params["offset"].set(value=0, vary=False)
    for i in range(n_gauss):
        fitter.params[f"g{i}_x0"].set(value=mean[i][0], vary=False)
        fitter.params[f"g{i}_y0"].set(value=mean[i][1], vary=False)
        if i == 0:
            fitter.params[f"g{i}_sigma_x"].set(value=std, vary=False)
        else:
            fitter.params[f"g{i}_sigma_x"].set(expr="g0_sigma_x")
        fitter.params[f"g{i}_sigma_y"].set(expr="g0_sigma_x")
    return fitter.fit()


def _gmm_params(fit_result, n_gauss: int) -> Dict[str, Any]:
    """Unpack lmfit results into a plain dict."""
    mean, amp = [], []
    for i in range(n_gauss):
        mean.append(np.array([fit_result.params[f"g{i}_x0"].value,
                              fit_result.params[f"g{i}_y0"].value]))
        amp.append(fit_result.params[f"g{i}_amp"].value)
    std = fit_result.params["g0_sigma_x"].value
    return {"mean": np.array(mean), "std": std, "covariance": std ** 2,
            "amp": np.array(amp)}


def discriminate_states(
    I: np.ndarray,
    Q: np.ndarray,
    *,
    user_mean: Optional[Sequence] = None,
    user_std: Optional[float] = None,
    outlier_sigma: float = 3,
) -> Dict[str, Any]:
    """Train the GMM on one block of prepared-state IQ clouds and assign shots.

    Parameters
    ----------
    I, Q : 2-D float arrays, shape ``(n_prepared_state, n_shot)``
        Single-shot quadratures, one row per prepared state (row order defines
        the confusion-matrix row order).
    user_mean : sequence of (I, Q) pairs, optional
        Pin the GMM centres instead of seeding them from the per-state density
        maxima. Together with ``user_std`` this skips the global fit entirely.
    user_std : float, optional
        Pin the shared Gaussian width (also sets the histogram bin size).
    outlier_sigma : float, optional
        A shot farther than ``outlier_sigma * std`` from every centre is
        tagged an outlier (default 3).
    """
    I = np.asarray(I, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n_state, n_shot = I.shape

    # 1. Bin into per-state 2-D histograms (+ centre/width seeds)
    density, hist_x, hist_y, mean_init, std_init = _bin_histograms(I, Q, user_std)

    # 1b. Refine the coarse-histogram-argmax seed onto each blob's density PEAK
    # (mean-shift). The seed is quantized to a ~sigma/3 bin and, with overlapping
    # or skewed blobs, lands a fraction of sigma off the true peak; the GMM fit
    # below pins the centres, so without this step that error is frozen into the
    # reported mean (and, downstream, the stored readout reference + discriminator
    # rotation). A user-pinned centre is taken verbatim (no refinement).
    if user_mean is None:
        mean_init = _refine_centres(I, Q, mean_init, user_std if user_std else std_init)

    # 2. Train the global GMM model
    if user_mean is not None and user_std is not None:
        trained_paras: Dict[str, Any] = {
            "mean": np.array(user_mean), "std": user_std,
            "covariance": user_std ** 2, "amp": np.ones(len(user_mean)),
        }
    else:
        fit_result = _gmm_fit(
            np.sum(density, axis=0), hist_x, hist_y,
            mean_init if user_mean is None else user_mean,
            user_std if user_std else std_init,
        )
        trained_paras = _gmm_params(fit_result, len(mean_init))

    # 3. Fit individual prepared states (amplitudes against the trained model)
    fitted_paras: List[Dict[str, Any]] = []
    fit_residues = np.zeros_like(density)
    norm_res = np.full(n_state, np.nan)
    for i in range(n_state):
        fit_result = _gmm_fit(density[i], hist_x, hist_y,
                              trained_paras["mean"], trained_paras["std"])
        fitted_paras.append(_gmm_params(fit_result, len(trained_paras["mean"])))
        residue = density[i] - fit_result.best_fit.reshape(density[i].shape)
        fit_residues[i] = residue
        total = np.nansum(density[i])
        norm_res[i] = np.nansum(residue) / total if total != 0 else np.nan

    # 4. Distances, labels, confusion fractions, outliers
    mean = np.asarray(trained_paras["mean"], dtype=float)
    n_center = len(mean)
    # (n_center, n_state, n_shot) Euclidean distances via broadcasting
    dist = np.sqrt((I[None, :, :] - mean[:, 0, None, None]) ** 2
                   + (Q[None, :, :] - mean[:, 1, None, None]) ** 2)
    state_label = np.argmin(dist, axis=0)

    # FIXED-shape confusion fractions (n_state, n_center): a centre that
    # captured no shot contributes a column of zeros instead of shrinking
    # the matrix (see module docstring).
    direct_counts = np.stack(
        [np.bincount(row, minlength=n_center) for row in state_label]
    ) / n_shot

    amps = np.array([res["amp"] for res in fitted_paras])
    gaussian_norms = amps / np.sum(amps, axis=1, keepdims=True)

    outlier_mask = dist.min(axis=0) > (outlier_sigma * np.mean(trained_paras["std"]))
    outlier_probability = np.count_nonzero(outlier_mask, axis=1) / n_shot

    return {
        "trained_paras": trained_paras,
        "fitted_paras": fitted_paras,
        "gaussian_norms": gaussian_norms,
        "direct_counts": direct_counts,
        "state_label": state_label,
        "outlier_mask": outlier_mask,
        "outlier_probability": outlier_probability,
        "norm_res": norm_res,
        "fit_residues": fit_residues,
        "density": density,
        "hist_x": hist_x,
        "hist_y": hist_y,
    }
