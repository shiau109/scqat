"""
Resonator Spectroscopy Estimator (multi-method)
===============================================
One estimator, N extraction approaches ("methods"), selected per call:

* ``method="lorentzian"`` (default) — joint Lorentzian + polynomial-background
  fit of the power ``|IQ|^2`` (``scqat.tools.fit_lorentzian_bg``). Fast,
  magnitude-only; the background is fitted *with* the dip so it cannot bias
  the extracted centre/width.
* ``method="circle"`` — Probst notch-model circle fit of the complex S21
  (``scqat.tools.fit_notch_circle``). Calibrates the environment (amplitude,
  phase offset, cable delay) analytically; also yields Qi/Qc. Requires the
  ``full_freq`` coordinate (absolute frequency).

Result contract (two tiers)
---------------------------
COMMON keys — validated here, identical meaning/unit in every method; the only
keys downstream orchestration may rely on:

    detuning  : fitted resonance position, Hz relative to the LO
    fwhm      : total linewidth kappa_tot, Hz
    success   : bool — fit trustworthy
    full_freq : absolute resonance frequency, Hz (when the coord is present)
    detuning_err / fwhm_err : best-effort 1-sigma errors (NaN allowed)

Method extras (``amplitude``/``bg_c*`` for lorentzian; ``Ql``/``absQc``/
``Qc_dia_corr``/``Qi_dia_corr``/``phi0``/``delay``/... for circle) ride along
in the results/metadata; consumers use them only via ``if key in results``.
``results["method"]`` and ``plot_data.attrs["method"]`` record provenance; the
figure (key ``"resonator_spectroscopy"``) is drawn per-method by dispatching
on the plot_data attr, so saved plot data replots without re-fitting.

Expected xarray.Dataset contract
---------------------------------
Coordinates:
    - detuning : 1-D float array – readout-frequency detuning from the LO (Hz).
    - full_freq: (detuning,) absolute readout frequency (Hz). Optional for
                 ``lorentzian`` (reported when present); **required** for
                 ``circle``.
Data variables:
    - IQdata   : (detuning,) – complex demodulated signal (I + iQ), **or**
    - I, Q     : (detuning,) – the two quadratures, combined into IQdata.

The dataset should have the ``qubit`` dimension already removed (e.g. via
``repetition_data`` from ``scqat.parsers.qualibrate_parser``).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, with_iqdata
from scqat.estimators.resonator_spectroscopy.methods import METHODS
from scqat.tools.sweep_order import ascending

#: Tier-1 keys every method must return — the only keys orchestration may
#: rely on. Same name => same meaning and unit across methods.
COMMON_KEYS = frozenset({"detuning", "fwhm", "success"})


class ResonatorSpectroscopyEstimator(BaseEstimator):
    """Fit the resonator dip in 1-D resonator spectroscopy data.

    ``analyze(..., method=...)`` selects the extraction approach (see module
    docstring). Results are per-qubit; the COMMON keys are method-independent.
    """

    estimator_name = "resonator_spectroscopy"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _check_data(self, dataset: xr.Dataset) -> None:
        if "detuning" not in dataset.coords:
            raise ValueError(
                "ResonatorSpectroscopyEstimator requires a 'detuning' coordinate."
            )
        if "IQdata" not in dataset and not ("I" in dataset and "Q" in dataset):
            raise ValueError(
                "ResonatorSpectroscopyEstimator requires an 'IQdata' variable, or both 'I' and 'Q'."
            )

    @classmethod
    def _arrays(cls, dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Extract (detuning, complex iq, full_freq-or-None) once, uniformly -
        ascending in detuning, the ONE place both extract_parameters and
        build_plot_data read the sweep: its direction is provenance, never input
        (tools.sweep_order)."""
        ds = with_iqdata(ascending(dataset, "detuning"))
        detuning = ds.coords["detuning"].values.astype(float)
        iq = ds["IQdata"].values.ravel()
        full_freq = None
        if "full_freq" in ds.coords:
            full_freq = ds.coords["full_freq"].values.ravel().astype(float)
        return detuning, iq, full_freq

    # ------------------------------------------------------------------
    # Core extraction (method dispatch)
    # ------------------------------------------------------------------
    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Dispatch to the selected method and enforce the COMMON-key contract.

        Keyword arguments
        -----------------
        method : str, optional
            Extraction approach: ``"lorentzian"`` (default) or ``"circle"``.
        baseline_order : int, optional
            (lorentzian) polynomial background order 0/1/2, default 1.
        delay : float, optional
            (circle) fix the cable delay (seconds) instead of fitting it.
        """
        method_name = kwargs.pop("method", "lorentzian")
        if method_name not in METHODS:
            raise ValueError(
                f"Unknown method {method_name!r}; available: {sorted(METHODS)}"
            )
        m = METHODS[method_name]

        detuning, iq, full_freq = self._arrays(dataset)
        results = m.extract(detuning, iq, full_freq=full_freq, **kwargs)

        missing = COMMON_KEYS - results.keys()
        if missing:  # a method bug, not a data problem — fail loudly
            raise RuntimeError(
                f"method {method_name!r} violated the estimator contract: "
                f"missing common keys {sorted(missing)}"
            )
        results["method"] = method_name

        # Common post-step: absolute resonance frequency (method-agnostic;
        # a method that already knows it, e.g. circle, reports it itself)
        if full_freq is not None and "full_freq" not in results:
            order = np.argsort(detuning)
            results["full_freq"] = float(
                np.interp(results["detuning"], detuning[order], full_freq[order])
            )
        return results

    # ------------------------------------------------------------------
    # Metadata + plot data + figures (delegated per method)
    # ------------------------------------------------------------------
    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the scalar fit parameters; drop the method's bulky arrays
        (those belong in the plot data)."""
        drop = METHODS[results["method"]].bulky_keys
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Delegate to the method that produced ``results``. The returned
        Dataset is netCDF-safe and stamps ``attrs["method"]`` so the figure
        reconstructs offline from the saved file alone."""
        detuning, iq, full_freq = self._arrays(dataset)
        m = METHODS[results["method"]]
        plot_data = m.build_plot_data(detuning, iq, full_freq, results)
        if plot_data.attrs.get("method") != m.name:  # method bug — fail loudly
            raise RuntimeError(
                f"method {m.name!r} did not stamp plot_data.attrs['method']"
            )
        return plot_data

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """One figure, keyed ``"resonator_spectroscopy"`` regardless of method
        (stable artifact name). Dispatches on ``plot_data.attrs["method"]`` —
        never on estimator state — so ``replot(from_plotdata=...)`` draws the
        right per-method figure with zero recompute."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        name = str(plot_data.attrs.get("method", ""))
        if name not in METHODS:
            raise ValueError(
                f"plot_data carries unknown method {name!r}; available: {sorted(METHODS)}"
            )
        return {"resonator_spectroscopy": METHODS[name].plot(plot_data)}
