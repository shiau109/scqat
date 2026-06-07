from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


def lorentzian(x, x0, amplitude, gamma, offset):
    """
    Lorentzian peak (amplitude may be negative for dips):

        offset + amplitude / (1 + ((x - x0) / gamma)**2)
    """
    return offset + amplitude / (1 + ((x - x0) / gamma) ** 2)


@register_fitter('lorentzian')
class FitLorentzian(FunctionFitting):
    """
    Fit a Lorentzian peak/dip:
        offset + amplitude / (1 + ((x - x0)/gamma)**2)

    Input DataArray must have a coordinate named 'x'.

    Parameters
    ----------
    data : xarray.DataArray
        1-D signal with coordinate 'x'.
    inverted : bool, optional
        If True, the initial amplitude guess is forced negative
        (dip rather than peak). Default False.
    bounds : dict, optional
        Optional overrides for parameter bounds, e.g.
        ``{'x0': (xmin, xmax), 'gamma': (0, gamma_max)}``.
    """

    def __init__(self, data: DataArray = None, inverted: bool = False, bounds: dict = None, x=None):
        self._data_parser(data, x)
        self.inverted = inverted
        self.bounds = bounds or {}
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    @staticmethod
    def model_function(x, x0, amplitude, gamma, offset):
        return lorentzian(x, x0, amplitude, gamma, offset)

    def guess(self):
        x = self.x
        y = self.y

        offset_guess = float(np.median(y))
        deviations = y - offset_guess

        if self.inverted:
            idx = int(np.argmin(deviations))
        else:
            idx = int(np.argmax(np.abs(deviations)))

        x0_guess = float(x[idx])
        amp_guess = float(deviations[idx])
        if amp_guess == 0:
            amp_guess = float(np.max(np.abs(deviations))) or 1.0

        # FWHM rough estimate: width where |y - offset| > |amp|/2
        half_mask = np.abs(deviations) >= abs(amp_guess) / 2
        if half_mask.sum() >= 2:
            x_in = x[half_mask]
            gamma_guess = float((x_in.max() - x_in.min()) / 2)
        else:
            gamma_guess = float(abs(x[1] - x[0]) * 5) if len(x) > 1 else 1.0
        if gamma_guess <= 0:
            gamma_guess = float(abs(x[-1] - x[0])) / 10 if len(x) > 1 else 1.0

        x_lo, x_hi = float(x.min()), float(x.max())
        x_span = x_hi - x_lo if x_hi > x_lo else 1.0

        x0_bounds = self.bounds.get('x0', (x_lo, x_hi))
        gamma_bounds = self.bounds.get('gamma', (0.0, x_span))

        self.params = self.model.make_params(
            x0=dict(value=x0_guess, min=x0_bounds[0], max=x0_bounds[1]),
            amplitude=dict(value=amp_guess),
            gamma=dict(value=gamma_guess, min=gamma_bounds[0], max=gamma_bounds[1]),
            offset=dict(value=offset_guess),
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
