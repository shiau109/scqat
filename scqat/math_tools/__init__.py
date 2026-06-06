"""
math_tools package — shared mathematical fitters and helpers.

Importing this package eagerly imports each fitter module so that they
register with the ``get_fitter`` factory. New fitters must be added here
to remain discoverable.
"""

from .function_fitting import FunctionFitting, register_fitter, get_fitter
from . import fit_abscos  # noqa: F401
from . import fit_cosine  # noqa: F401
from . import fit_damped_oscillation  # noqa: F401
from . import fit_damping_beat  # noqa: F401
from . import fit_exp_decay  # noqa: F401
from . import fit_multi_damped_oscillation  # noqa: F401
from . import fit_gaussian2d  # noqa: F401
from . import fit_lorentzian  # noqa: F401
from . import fit_powerlaw_base  # noqa: F401
from . import fit_qubit_decoherence  # noqa: F401
from . import fit_transmon_freq_flux  # noqa: F401
