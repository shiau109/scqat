import numpy as np
from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, sin, abs, exp, max, min, mean, argmax, pi
from numpy import fft

from .function_fitting import (
    FunctionFitting, register_fitter, parse_xy, robust_dt, seed_amp_phase,
)


@register_fitter('damped_oscillation')
class FitDampedOscillation(FunctionFitting):
    """
    Fit a damped oscillation model:
        a * exp(-kappa * x) * trig(2*pi*f*x + phi) + c

    where ``trig`` is ``cos`` (default) or ``sin`` per the ``basis`` argument.
    The ``sin`` basis suits sequences whose fringe starts at zero amplitude and
    rises (e.g. a Ramsey x90 -> idle -> y90, whose phase is seeded at 0); the
    ``cos`` basis is the historical default and is unchanged for existing callers.

    Input DataArray must have a coordinate named 'x'.
    """

    def __init__(self, data: DataArray = None, x=None, basis: str = "cos"):
        if basis not in ("cos", "sin"):
            raise ValueError(f"basis must be 'cos' or 'sin', got {basis!r}.")
        self.basis = basis
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, kappa, f, phi, c):
        trig = sin if self.basis == "sin" else cos
        return a * exp(-kappa * x) * trig(2 * pi * f * x + phi) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = robust_dt(t)
        max_val = float(max(y))
        min_val = float(min(y))

        # FFT for frequency guess
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # remove DC
        power = abs(amp)

        f_guess = float(abs(freq[argmax(power)]))
        f_guess_dict = dict(value=f_guess, min=0.0, max=1.0 / dt / 2)

        kappa_guess = 1.0 / abs(t[-1] / 2) if abs(t[-1] / 2) > 0 else 1.0
        kappa_guess_dict = dict(value=kappa_guess, min=0.0, max=10 * kappa_guess)

        # Seed amplitude AND phase from the data (not a blind phi=0): a phi=0
        # seed on a near-anti-phase fringe would hit the a>=0 bound and collapse
        # to a flat line.
        peak_to_peak = max_val - min_val
        a_seed, phi_seed = seed_amp_phase(t, y, f_guess, kappa_guess, self.basis)
        if not np.isfinite(a_seed) or a_seed <= 0:
            a_seed = peak_to_peak / 2  # degenerate projection -> robust fallback
        a_max = float(np.maximum(peak_to_peak, a_seed * 2))
        a_guess_dict = dict(value=float(a_seed), min=0.0, max=a_max)
        phi_guess_dict = dict(value=float(phi_seed), min=-float(pi), max=float(pi))
        c_guess_dict = dict(value=float(mean(y)), min=min_val, max=max_val)

        self.params = self.model.make_params(
            a=a_guess_dict,
            kappa=kappa_guess_dict,
            f=f_guess_dict,
            phi=phi_guess_dict,
            c=c_guess_dict,
        )
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
