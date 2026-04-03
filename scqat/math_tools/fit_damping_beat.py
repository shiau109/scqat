from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import cos, abs, exp, max, min, mean, pi, nan
from numpy import fft, asarray
from scipy.signal import find_peaks

from .function_fitting import FunctionFitting, register_fitter


@register_fitter('damping_beat')
class FitDampingBeat(FunctionFitting):
    """
    Fit a damped beat model (two oscillation frequencies with independent decay):
        a_1*exp(-kappa_1*x)*cos(2*pi*f_1*x + phi_1)
      + a_2*exp(-kappa_2*x)*cos(2*pi*f_2*x + phi_2)
      + c

    If only one frequency is detected in the FFT, the second component is
    automatically frozen to zero, reducing to a single damped oscillation.

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

    def model_function(self, x, a_1, kappa_1, f_1, phi_1, a_2, kappa_2, f_2, phi_2, c):
        return (
            a_1 * exp(-kappa_1 * x) * cos(2 * pi * f_1 * x + phi_1)
            + a_2 * exp(-kappa_2 * x) * cos(2 * pi * f_2 * x + phi_2)
            + c
        )

    def guess(self):
        y = self.y
        t = self.x
        dt = float(t[1] - t[0])
        max_val = float(max(y))
        min_val = float(min(y))

        # FFT for frequency guesses
        amp = fft.fft(y)[: len(y) // 2]
        freq = fft.fftfreq(len(y), dt)[: len(amp)]
        amp[0] = 0  # remove DC
        power = abs(amp)
        freq = asarray(freq)

        # Find local maxima in the power spectrum
        local_peak_indices, properties = find_peaks(power, height=float(max(power)) * 0.1)
        if len(local_peak_indices) == 0:
            # Fallback: use the global max
            local_peak_indices = asarray([power.argmax()])

        # Sort local peaks by power (descending)
        sorted_local = local_peak_indices[power[local_peak_indices].argsort()[::-1]]
        f_1_idx = sorted_local[0]

        # Look for a second local peak with sufficient amplitude
        f_2_idx = None
        for idx in sorted_local[1:]:
            if power[idx] / power[f_1_idx] > 0.3:
                f_2_idx = idx
                break

        f_1_guess = float(abs(freq[f_1_idx]))
        a_1_guess = float(abs(amp[f_1_idx]))
        a_1_dict = dict(value=a_1_guess, min=0.0, max=a_1_guess * 2)
        f_1_dict = dict(value=f_1_guess, min=0.0, max=1.0 / dt / 2)
        phi_1_dict = dict(value=0.0, min=-float(pi), max=float(pi))

        kappa_1_guess = 1.0 / abs(t[-1] / 2) if abs(t[-1] / 2) > 0 else 1.0
        kappa_1_dict = dict(value=kappa_1_guess, min=0.0, max=10 * kappa_1_guess)

        if f_2_idx is None:
            # Single-frequency mode: freeze second component
            a_2_dict = dict(value=0, vary=False)
            f_2_dict = dict(value=0, vary=False)
            phi_2_dict = dict(value=0, vary=False)
            kappa_2_dict = dict(value=0, vary=False)
        else:
            f_2_guess = float(abs(freq[f_2_idx]))
            a_2_guess = float(abs(amp[f_2_idx]))
            a_2_dict = dict(value=a_2_guess, min=0.0, max=a_2_guess * 2)
            f_2_dict = dict(value=f_2_guess, min=0.0, max=1.0 / dt / 2)
            phi_2_dict = dict(value=0.0, min=-float(pi), max=float(pi))
            kappa_2_dict = dict(value=kappa_1_guess, min=0.0, max=10 * kappa_1_guess)

        c_dict = dict(value=float(mean(y)), min=min_val, max=max_val)

        self.params = self.model.make_params(
            a_1=a_1_dict,
            kappa_1=kappa_1_dict,
            f_1=f_1_dict,
            phi_1=phi_1_dict,
            a_2=a_2_dict,
            kappa_2=kappa_2_dict,
            f_2=f_2_dict,
            phi_2=phi_2_dict,
            c=c_dict,
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
