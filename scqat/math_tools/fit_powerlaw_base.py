from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import max, min, mean

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('powerlaw_base')
class FitBasePowerLaw(FunctionFitting):
    """
    Fit a base power-law model:
        a * (base ** x) + c

    Useful for sequence-length decays (e.g. randomized benchmarking), where
    ``base`` is the per-step depolarising parameter. Accepts an
    ``xarray.DataArray`` with an ``'x'`` coordinate, raw ``(x, y)`` arrays, or a
    bare ``y`` array (``x`` defaults to the sample index), via :func:`parse_xy`.

    Ported from qcat ``fit_powerlaw_base`` — generalised to the flexible
    :func:`parse_xy` input and the scqat ``fit()`` convention (``guess()`` runs
    only when parameters have not already been set).
    """

    def __init__(self, data: DataArray = None, x=None):
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, base, c):
        return a * (base ** x) + c

    def guess(self):
        y = self.y
        max_y = float(max(y))
        min_y = float(min(y))
        y_range = max_y - min_y

        base_dict = dict(value=0.9, min=0.0, max=1.0)
        c_dict = dict(value=float(mean(y)), min=min_y - y_range * 2, max=max_y)
        a_guess = y_range if y_range != 0 else 1.0
        a_dict = dict(value=a_guess, min=-a_guess * 2, max=a_guess * 2)

        self.params = self.model.make_params(a=a_dict, base=base_dict, c=c_dict)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
