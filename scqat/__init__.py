"""scqat — superconducting-qubit analysis tool.

This top-level package is intentionally kept import-light: ``import scqat`` must
have no side effects and must not pull in heavy/optional dependencies (e.g.
matplotlib, which ``core`` and ``estimators`` import). Import the layer you need
explicitly, for example::

    from scqat.parsers import load_xarray_h5
    from scqat.tools import get_fitter
    from scqat.estimators import RamseyEstimator

See ``MIGRATION.md`` for the QCAT→scqat feature backlog and porting recipe, and
``CLAUDE.md`` for the architecture.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
