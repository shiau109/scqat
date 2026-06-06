from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, abs, max, min, mean, argmax, pi
from numpy import fft

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('cosine')
class FitCosine(FunctionFitting):
    """
    Fit a (non-decaying) cosine model:
        a * cos(2*pi*f*x + phi) + c

    Accepts an ``xarray.DataArray`` with an ``'x'`` coordinate, raw ``(x, y)``
    arrays, or a bare ``y`` array (``x`` then defaults to the sample index), via
    the shared :func:`parse_xy` helper.

    Ported from qcat ``fit_cosine`` — generalised to the flexible :func:`parse_xy`
    input and the scqat ``fit()`` convention (``guess()`` runs only when
    parameters have not already been set, so a caller may tweak the guess).
    """

    def __init__(self, data: DataArray = None, x=None):
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, f, phi, c):
        return a * cos(2 * pi * f * x + phi) + c

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))

        # FFT for the frequency guess
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # remove DC
        power = abs(amp)
        f_guess = float(abs(freq[argmax(power)]))

        f_dict = dict(value=f_guess, min=0.0, max=1.0 / dt / 2)
        phi_dict = dict(value=0.0, min=-pi, max=pi)
        c_dict = dict(value=float(mean(y)), min=min_val, max=max_val)
        a_guess = (max_val - min_val) / 2
        a_dict = dict(value=a_guess, min=0.0, max=a_guess * 2 if a_guess > 0 else 1.0)

        self.params = self.model.make_params(a=a_dict, f=f_dict, phi=phi_dict, c=c_dict)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
