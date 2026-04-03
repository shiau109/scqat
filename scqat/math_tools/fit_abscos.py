from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, abs, pi, max, min, mean, argmin, argmax, fft

from .function_fitting import FunctionFitting, register_fitter


@register_fitter('abscos')
class FitAbsCos(FunctionFitting):
    """
    Fit an absolute cosine model:
        amplitude * |cos(2*pi*frequency*(x - phase))|

    Input DataArray must have a coordinate named 'x'.
    """

    def __init__(self, data: DataArray = None):
        self._data_parser(data)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray):
        if not isinstance(data, DataArray):
            raise ValueError("Input data must be an xarray.DataArray.")
        self.y = data.values
        self.x = data.coords["x"].values

    @staticmethod
    def model_function(x, amplitude, frequency, phase):
        return amplitude * abs(cos(2 * pi * frequency * (x - phase)))

    def guess(self):
        y = self.y
        x = self.x
        n = len(x)

        # Use average spacing over the full range (robust to duplicate x values
        # that arise when merging f_1 and f_2 data at the same charge gates).
        x_range = float(x[-1] - x[0])
        dt = x_range / (n - 1) if n > 1 and x_range > 0 else 1.0

        amplitude_guess = float(max(y))

        # |cos(2*pi*f*x)| has FFT energy at 2*f (the rectified cosine
        # doubles the apparent frequency).  Find the dominant FFT peak
        # and halve it to recover the |cos| frequency.
        spectrum = fft.fft(y)[: n // 2]
        freqs = fft.fftfreq(n, dt)[: len(spectrum)]
        spectrum[0] = 0  # remove DC
        power = abs(spectrum)
        fft_freq = float(abs(freqs[argmax(power)]))
        frequency_guess = fft_freq / 2.0 if fft_freq > 0 else 0.5 / dt

        # Phase guess: the minimum of |cos| occurs where
        # cos(2*pi*freq*(x - phase)) = 0, i.e. x_min = phase + 1/(4*freq)
        min_idx = int(argmin(y))
        x_min = float(x[min_idx])
        phase_guess = x_min - 1.0 / (4.0 * frequency_guess) if frequency_guess > 0 else 0.0

        self.params = self.model.make_params(
            amplitude=dict(value=amplitude_guess, min=0.0, max=amplitude_guess * 3),
            frequency=dict(value=frequency_guess, min=0.0),
            phase=dict(value=phase_guess),
        )
        return self.params

    def fit(self, data: DataArray = None) -> ModelResult:
        if data is not None:
            self._data_parser(data)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
