"""Composite resonator-spectroscopy-vs-flux analysis (subpackage).

Re-exports the estimator so external code can use
``from scqat.estimators.resonator_spectroscopy_flux import ResonatorSpectroscopyFluxEstimator``
regardless of the internal module layout.
"""

from .estimator import ResonatorSpectroscopyFluxEstimator

__all__ = ["ResonatorSpectroscopyFluxEstimator"]
