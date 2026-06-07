from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('transmon_freq_flux')
class FitTransmonFrequencyFlux(FunctionFitting):
    """
    Fit transmon frequency vs flux:

        f(x) = sqrt(8 * Ec * Ej_eff) - Ec
        Ej_eff = Ej_sum * sqrt(cos(pi*q)**2 + (d*sin(pi*q))**2),   q = (x - offset)/period

    By default ``Ec`` is fixed to ``Ec_design`` (0.2) and ``d`` is fixed to 0
    (symmetric transmon, ``Ej_eff = Ej_sum*|cos(pi*q)|``), as in qcat; free the
    parameters after :meth:`guess` if needed. Accepts an ``xarray.DataArray`` with
    an ``'x'`` coordinate, raw ``(x, y)`` arrays, or a bare ``y`` array, via
    :func:`parse_xy`.

    Ported from qcat ``fit_transmon_freqeuency_flux`` — generalised to the
    flexible :func:`parse_xy` input.
    """

    def __init__(self, data: DataArray = None, x=None, Ec_design: float = 0.2):
        self.Ec_design = Ec_design
        self._data_parser(data, x)
        self.params = None
        self.model = Model(self.model_function)

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, Ec, offset, period, Ej_sum, d):
        quan_flux = (x - offset) / period
        ej_eff = Ej_sum * np.sqrt(np.cos(np.pi * quan_flux) ** 2 + (d * np.sin(np.pi * quan_flux)) ** 2)
        return np.sqrt(8 * Ec * ej_eff) - Ec

    def guess(self):
        x = self.x
        y = self.y
        # Data-driven guesses: the arch spans ~one period, and the frequency
        # maximum (sweet spot) sits at the flux offset.
        span = float(np.max(x) - np.min(x))
        period_guess = span if span > 0 else 1.0
        offset_guess = float(x[int(np.argmax(y))])

        Ec_dict = dict(value=self.Ec_design, vary=False)
        period_dict = dict(value=period_guess, min=0.0)
        d_dict = dict(value=0.0, vary=False)
        offset_dict = dict(value=offset_guess,
                           min=offset_guess - period_guess / 2,
                           max=offset_guess + period_guess / 2)
        ej_sum_guess = float(np.max(y ** 2 / self.Ec_design / 8.0))
        ej_sum_dict = dict(value=ej_sum_guess, min=0.0)

        self.params = self.model.make_params(
            Ec=Ec_dict, offset=offset_dict, period=period_dict, Ej_sum=ej_sum_dict, d=d_dict,
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
