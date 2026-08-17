"""Method strategy registry for the readout_fidelity estimator.

Adding a way to summarise a sweep point = one module implementing
:class:`~scqat.estimators.readout_fidelity.methods.base.ReadoutFidelityMethod`
+ one entry here. Nothing else moves.
"""

from .base import ReadoutFidelityMethod
from .gmm import GmmMethod
from .average import AverageMethod

METHODS = {m.name: m for m in (GmmMethod(), AverageMethod())}

__all__ = ["ReadoutFidelityMethod", "GmmMethod", "AverageMethod", "METHODS"]
