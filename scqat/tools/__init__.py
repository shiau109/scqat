"""
tools package — shared mathematical fitters and helpers.

Importing this package eagerly imports each fitter module so that they
register with the ``get_fitter`` factory. New fitters must be added here
to remain discoverable.
"""

from .function_fitting import FunctionFitting, register_fitter, get_fitter
from .dip_fit import DIP_KNOBS, DIP_METHODS, fit_dip, validate_dip_kwargs
from .peak_fit import PEAK_KNOBS, fit_peaks, validate_peak_kwargs
from .peak_map import track_peaks
from .discriminate import (
    DISCRIMINATE_KNOBS,
    discriminate_states,
    validate_discriminate_kwargs,
)
from .ramsey_fit import RAMSEY_MODELS, fit_ramsey
from . import fit_abscos  # noqa: F401
from . import fit_cosine  # noqa: F401
from . import fit_damped_oscillation  # noqa: F401
from . import fit_damping_beat  # noqa: F401
from . import fit_exp_decay  # noqa: F401
from . import fit_multi_damped_oscillation  # noqa: F401
from . import fit_gaussian2d  # noqa: F401
from . import fit_lorentzian  # noqa: F401
from . import fit_lorentzian_bg  # noqa: F401
from . import fit_notch_circle  # noqa: F401
from . import fit_powerlaw_base  # noqa: F401
from . import fit_qubit_decoherence  # noqa: F401
from . import fit_transmon_freq_flux  # noqa: F401
