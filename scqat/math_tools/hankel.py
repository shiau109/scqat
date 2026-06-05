"""
Hankel / Matrix-Pencil signal decomposition.

A pure, reusable algorithm (no I/O, no plotting) that decomposes a 1-D
time-series into a sum of complex exponentials

    y(t) ≈ Σ_i  r_i · exp(s_i · t),   s_i = decay_rate_i + 2πi·freq_i

via a Hankel matrix SVD followed by the Matrix Pencil Method (MPM) or HSVD.
Returns each mode's frequency, decay rate, amplitude and phase.

Typical uses
------------
* Seed initial guesses for a curve fit (e.g.
  :class:`scqat.math_tools.fit_multi_damped_oscillation.FitMultiDampedOscillation`
  or an exponential-decay fit).
* Standalone modal analysis of a ringdown / decay signal.

This lives in ``math_tools`` (a method, not an experiment) so any fitter,
protocol, or workflow can reuse it without inverting the dependency layering.
"""

import numpy as np
import scipy.linalg as la


def _build_hankel(y_data, L=None):
    """Build a Hankel matrix from the signal vector."""
    N = len(y_data)
    if L is None:
        L = N // 2
    c = y_data[: N - L]
    r = y_data[N - L - 1:]
    return la.hankel(c, r)


def _compute_svd(hankel_matrix):
    """Compute the economy SVD of the Hankel matrix."""
    try:
        U, s, Vh = la.svd(hankel_matrix, full_matrices=False)
    except la.LinAlgError as exc:
        raise ValueError(
            "SVD did not converge — check input data for NaNs or infinities."
        ) from exc
    return U, s, Vh


def _select_n_modes(s, method="relative", **kwargs):
    """
    Determine how many singular-value modes to retain.

    Parameters
    ----------
    s : 1-D array of singular values (descending).
    method : {'relative', 'absolute', 'energy', 'gap', 'diff_ratio', 'aic', 'mdl', 'fixed'}
    kwargs :
        threshold (float) – for 'relative' (default 2e-3), 'absolute' (default 0.1),
            and 'diff_ratio' (default 2.0: ratio s[i]/s[i+1] must exceed this value).
        energy_target (float) – for 'energy' (default 0.95).
        n_modes (int) – for 'fixed', the exact number of modes to use.
        n_samples (int) – rows of Hankel matrix, needed for 'aic'/'mdl'.
        n_features (int) – columns of Hankel matrix, needed for 'aic'/'mdl'.
    """
    if method == "relative":
        threshold = kwargs.get("threshold", 2e-3)
        n_modes = int(np.sum(s / s[0] > threshold))
    elif method == "absolute":
        threshold = kwargs.get("threshold", 0.1)
        n_modes = int(np.sum(s > threshold))
    elif method == "energy":
        energy_target = kwargs.get("energy_target", 0.95)
        cumulative = np.cumsum(s ** 2) / np.sum(s ** 2)
        n_modes = int(np.argmax(cumulative >= energy_target)) + 1
    elif method == "gap":
        # Largest ratio drop between consecutive singular values
        ratios = s[:-1] / np.maximum(s[1:], np.finfo(float).tiny)
        n_modes = int(np.argmax(ratios)) + 1
    elif method == "diff_ratio":
        # Threshold on the ratio r[i] = s[i] / s[i+1].
        # Keep all modes up to (and including) the last index where r[i] > threshold.
        threshold = kwargs.get("threshold", 1.5)
        ratios = s[:6] / s[1:7]
        above = np.where(ratios > threshold)[0]
        n_modes = int(above[-1]) + 1 if len(above) > 0 else 1
    elif method in ("aic", "mdl"):
        # Wax–Kailath information-theoretic model-order selection.
        n_samples = kwargs.get("n_samples")
        n_features = kwargs.get("n_features")
        if n_samples is None or n_features is None:
            raise ValueError(
                f"'{method}' mode selection requires 'n_samples' and "
                "'n_features' (Hankel matrix dimensions) in kwargs."
            )
        n = max(n_samples, n_features)
        p = len(s)
        eigenvalues = s ** 2
        best_k, best_criterion = 0, np.inf
        for k in range(p - 1):
            noise_eigs = eigenvalues[k + 1:]
            m = len(noise_eigs)
            arith_mean = np.mean(noise_eigs)
            if arith_mean <= 0:
                continue
            log_geo = np.mean(np.log(np.maximum(noise_eigs, np.finfo(float).tiny)))
            log_likelihood = -n * m * (log_geo - np.log(arith_mean))
            free_params = k * (2 * p - k)
            if method == "aic":
                criterion = 2 * log_likelihood + 2 * free_params
            else:  # mdl
                criterion = 2 * log_likelihood + free_params * np.log(n)
            if criterion < best_criterion:
                best_criterion = criterion
                best_k = k + 1
        n_modes = best_k
    elif method == "fixed":
        n_modes = kwargs.get("n_modes")
        if n_modes is None:
            raise ValueError("'fixed' mode selection requires 'n_modes' in kwargs.")
        n_modes = int(n_modes)
    else:
        raise ValueError(f"Unknown mode-selection method: '{method}'")
    return max(1, n_modes)


def _compute_eigenvalues(U, Vh, n_modes, method="mpm"):
    """Extract signal poles via the Matrix Pencil Method (MPM) or HSVD."""
    if method.lower() == "hsvd":
        U_filt = U[:, :n_modes]
        U1 = U_filt[:-1, :]
        U2 = U_filt[1:, :]
        Z, _, _, _ = la.lstsq(U1, U2)
    elif method.lower() == "mpm":
        V = Vh.conj().T
        V_filt = V[:, :n_modes]
        V1 = V_filt[:-1, :]
        V2 = V_filt[1:, :]
        Z = la.pinv(V1.conj().T) @ V2.conj().T
    else:
        raise ValueError("recon_method must be 'hsvd' or 'mpm'")
    return la.eigvals(Z)


def _compute_exponents(eigvals, dt, eigval_threshold=1e-3):
    """Convert z-plane eigenvalues to continuous-time exponents (poles)."""
    valid = eigvals[np.abs(eigvals) > eigval_threshold]
    return np.log(valid) / dt


def _compute_residues(exponents, y_data, tlist):
    """Solve for complex residues (amplitudes) of each exponential mode."""
    N = len(y_data)
    M = len(exponents)
    Z = np.zeros((N, M), dtype=complex)
    for i in range(M):
        Z[:, i] = np.exp(exponents[i] * tlist)
    residues, _, _, _ = la.lstsq(Z, y_data)
    return residues


def hankel_decompose(
    signal,
    time,
    *,
    mode_method="relative",
    recon_method="mpm",
    eigval_threshold=1e-3,
    **kwargs,
):
    """
    Decompose a 1-D signal into complex-exponential modes.

    Parameters
    ----------
    signal : array-like
        1-D signal samples (real or complex).
    time : array-like
        1-D sample times (same length as ``signal``); the sampling interval is
        taken as ``time[1] - time[0]``.
    mode_method : str
        Mode-selection strategy passed to :func:`_select_n_modes`
        ('relative', 'absolute', 'energy', 'gap', 'diff_ratio', 'aic', 'mdl',
        'fixed').
    recon_method : str
        Eigenvalue extraction algorithm ('mpm' or 'hsvd').
    eigval_threshold : float
        Minimum eigenvalue magnitude to keep (default 1e-3).
    **kwargs :
        Extra options forwarded to :func:`_select_n_modes` (e.g. ``threshold``,
        ``energy_target``, ``n_modes``).

    Returns
    -------
    dict
        ``modes`` : list of dicts (sorted by descending amplitude), each with
            ``freq_hz``, ``decay_rate``, ``time_constant``, ``amplitude``,
            ``phase``, ``complex_exponent``, ``complex_residue``.
        ``n_modes`` : number of retained singular-value modes.
        ``singular_values`` : the SVD spectrum.
        ``reconstruction`` : the real signal rebuilt from the modes.
    """
    y_data = np.asarray(signal).astype(complex)
    tlist = np.asarray(time).astype(float)
    dt = tlist[1] - tlist[0]

    # Steps 1–2: Hankel matrix + SVD
    hankel_matrix = _build_hankel(y_data)
    U, s, Vh = _compute_svd(hankel_matrix)

    # Step 3: mode selection
    n_modes = _select_n_modes(
        s, method=mode_method,
        n_samples=hankel_matrix.shape[0],
        n_features=hankel_matrix.shape[1],
        **kwargs,
    )

    # Steps 4–6: eigenvalues → exponents → residues
    eigvals = _compute_eigenvalues(U, Vh, n_modes, method=recon_method)
    exponents = _compute_exponents(eigvals, dt, eigval_threshold)
    residues = _compute_residues(exponents, y_data, tlist)

    # Package into a list of physical mode dictionaries
    modes = []
    for exp, res in zip(exponents, residues):
        freq_hz = float(np.imag(exp) / (2 * np.pi))
        decay_rate = float(np.real(exp))
        if freq_hz < -1e-5:
            continue
        modes.append({
            "freq_hz": freq_hz,
            "decay_rate": decay_rate,
            "time_constant": float(-1 / decay_rate) if decay_rate < -1e-10 else np.inf,
            "amplitude": float(np.abs(res)),
            "phase": float(np.angle(res)),
            "complex_exponent": complex(exp),
            "complex_residue": complex(res),
        })
    modes.sort(key=lambda m: m["amplitude"], reverse=True)

    # Reconstruct signal for diagnostics
    if modes:
        reconstruction = np.real(
            np.sum(
                [m["complex_residue"] * np.exp(m["complex_exponent"] * tlist) for m in modes],
                axis=0,
            )
        )
    else:
        reconstruction = np.zeros_like(tlist)

    return {
        "modes": modes,
        "n_modes": n_modes,
        "singular_values": s,
        "reconstruction": reconstruction,
    }
