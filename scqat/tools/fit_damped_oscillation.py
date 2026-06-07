from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, abs, exp, max, min, mean, argmax, pi
from numpy import fft

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('damped_oscillation')
class FitDampedOscillation(FunctionFitting):
    """
    Fit a damped oscillation model:
        a * exp(-kappa * x) * cos(2*pi*f*x + phi) + c

    Input DataArray must have a coordinate named 'x'.
    """

    def __init__(self, data: DataArray = None, x=None):
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, kappa, f, phi, c):
        return a * exp(-kappa * x) * cos(2 * pi * f * x + phi) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))

        # FFT for frequency guess
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # remove DC
        power = abs(amp)

        f_guess = float(abs(freq[argmax(power)]))
        f_guess_dict = dict(value=f_guess, min=0.0, max=1.0 / dt / 2)

        phi_guess_dict = dict(value=0.0, min=-float(pi), max=float(pi))
        c_guess_dict = dict(value=float(mean(y)), min=min_val, max=max_val)

        a_guess = (max_val - min_val) / 2
        a_guess_dict = dict(value=a_guess, min=0.0, max=a_guess * 2)

        kappa_guess = 1.0 / abs(t[-1] / 2) if abs(t[-1] / 2) > 0 else 1.0
        kappa_guess_dict = dict(value=kappa_guess, min=0.0, max=10 * kappa_guess)

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
