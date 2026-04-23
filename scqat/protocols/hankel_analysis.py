"""
Hankel Analysis Protocol
========================
Extracts frequencies, decay rates, and amplitudes from time-series signals
using Hankel matrix SVD decomposition (MPM / HSVD methods).

Expected xarray.Dataset contract:
    Coordinates:
        - time : 1-D array of sample times
    Data variables:
        - signal : 1-D array (real or complex) sampled at `time`
"""

from typing import Any, Dict

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_analyzer import BaseAnalyzer


class HankelAnalyzer(BaseAnalyzer):
    """
    Analyzes time-series signals using Hankel matrix SVD decomposition,
    extracting frequencies, decay rates, and amplitudes via MPM or HSVD.
    """

    protocol_name = "hankel_analysis"

    # ------------------------------------------------------------------
    # Data validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if "time" not in dataset.coords:
            raise ValueError("HankelAnalyzer requires a 'time' coordinate.")
        if "signal" not in dataset.data_vars:
            raise ValueError("HankelAnalyzer requires a 'signal' data variable.")

    # ------------------------------------------------------------------
    # Private math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_hankel(y_data, L=None):
        """Build a Hankel matrix from the signal vector."""
        N = len(y_data)
        if L is None:
            L = N // 2
        c = y_data[: N - L]
        r = y_data[N - L - 1 :]
        return la.hankel(c, r)

    @staticmethod
    def _compute_svd(hankel_matrix):
        """Compute the economy SVD of the Hankel matrix."""
        try:
            U, s, Vh = la.svd(hankel_matrix, full_matrices=False)
        except la.LinAlgError as exc:
            raise ValueError(
                "SVD did not converge — check input data for NaNs or infinities."
            ) from exc
        return U, s, Vh

    @staticmethod
    def _select_n_modes(s, method="relative", **kwargs):
        """
        Determine how many singular-value modes to retain.

        Parameters
        ----------
        s : 1-D array of singular values (descending).
        method : {'relative', 'absolute', 'energy', 'gap', 'aic', 'mdl', 'fixed'}
        kwargs :
            threshold (float) – for 'relative' (default 2e-3) and 'absolute' (0.1).
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

    @staticmethod
    def _compute_eigenvalues(U, Vh, n_modes, method="mpm"):
        """
        Extract signal poles via the Matrix Pencil Method (MPM) or HSVD.
        """
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

    @staticmethod
    def _compute_exponents(eigvals, dt, eigval_threshold=1e-3):
        """Convert z-plane eigenvalues to continuous-time exponents (poles)."""
        valid = eigvals[np.abs(eigvals) > eigval_threshold]
        return np.log(valid) / dt

    @staticmethod
    def _compute_residues(exponents, y_data, tlist):
        """Solve for complex residues (amplitudes) of each exponential mode."""
        N = len(y_data)
        M = len(exponents)
        Z = np.zeros((N, M), dtype=complex)
        for i in range(M):
            Z[:, i] = np.exp(exponents[i] * tlist)
        residues, _, _, _ = la.lstsq(Z, y_data)
        return residues

    # ------------------------------------------------------------------
    # BaseAnalyzer interface
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Run the full Hankel / MPM pipeline and return extracted modal parameters.

        Kwargs
        ------
        mode_method : str
            Mode-selection strategy ('relative', 'absolute', 'energy',
            'gap', 'aic', 'mdl', 'fixed').
        recon_method : str
            Eigenvalue extraction algorithm ('mpm' or 'hsvd').
        threshold : float
            Passed to `_select_n_modes` for 'relative' / 'absolute'.
        energy_target : float
            Passed to `_select_n_modes` for 'energy'.
        n_modes : int
            Exact number of modes for 'fixed' mode selection.
        eigval_threshold : float
            Minimum eigenvalue magnitude to keep (default 1e-3).
        """
        y_data = dataset["signal"].values.astype(complex)
        tlist = dataset.coords["time"].values.astype(float)
        dt = tlist[1] - tlist[0]

        mode_method = kwargs.get("mode_method", "relative")
        recon_method = kwargs.get("recon_method", "mpm")
        eigval_threshold = kwargs.get("eigval_threshold", 1e-3)

        # Steps 1–2: Hankel matrix + SVD
        hankel_matrix = self._build_hankel(y_data)
        U, s, Vh = self._compute_svd(hankel_matrix)

        # Step 3: mode selection
        n_modes = self._select_n_modes(
            s, method=mode_method,
            n_samples=hankel_matrix.shape[0],
            n_features=hankel_matrix.shape[1],
            **kwargs,
        )

        # Steps 4–6: eigenvalues → exponents → residues
        eigvals = self._compute_eigenvalues(U, Vh, n_modes, method=recon_method)
        exponents = self._compute_exponents(eigvals, dt, eigval_threshold)
        residues = self._compute_residues(exponents, y_data, tlist)

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
        t_col = tlist[:, np.newaxis]
        reconstruction = np.real(
            np.sum(
                [res["complex_residue"] * np.exp(res["complex_exponent"] * tlist)
                 for res in modes],
                axis=0,
            )
        )

        return {
            "modes": modes,
            "n_modes": n_modes,
            "singular_values": s,
            "reconstruction": reconstruction,
        }

    def generate_figures(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Dict[str, plt.Figure]:
        """
        Generate two diagnostic figures:
        1. Singular-value spectrum (log scale) with the selected mode cutoff.
        2. Original signal vs. reconstructed signal overlay.
        """
        figs: Dict[str, plt.Figure] = {}

        s = results["singular_values"]
        n_modes = results["n_modes"]
        tlist = dataset.coords["time"].values
        y_data = np.real(dataset["signal"].values)
        reconstruction = results["reconstruction"]

        # --- Figure 1: Singular-value spectrum ---
        fig_sv, ax_sv = plt.subplots()
        ax_sv.semilogy(s, "o-", markersize=3)
        ax_sv.axvline(n_modes - 0.5, color="r", ls="--", label=f"cutoff = {n_modes}")
        ax_sv.set_xlabel("Index")
        ax_sv.set_ylabel("Singular value")
        ax_sv.set_title("Hankel SVD spectrum")
        ax_sv.legend()
        figs["singular_values"] = fig_sv

        # --- Figure 2: Signal reconstruction ---
        fig_sig, ax_sig = plt.subplots()
        ax_sig.plot(tlist, y_data, label="original", alpha=0.6)
        ax_sig.plot(tlist, reconstruction, "--", label="reconstruction")
        ax_sig.set_xlabel("Time")
        ax_sig.set_ylabel("Amplitude")
        ax_sig.set_title("Hankel signal reconstruction")
        ax_sig.legend()
        figs["reconstruction"] = fig_sig

        return figs
