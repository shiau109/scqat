from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
from numpy import exp, ptp

from .function_fitting import FunctionFitting, register_fitter, parse_xy


@register_fitter('exp_decay')
class FitExponentialDecay(FunctionFitting):
    """
    Fit an exponential decay model:
        a * exp(-x / tau) + c

    ``tau`` is the decay constant (e.g. a T1 relaxation time). Accepts an
    ``xarray.DataArray`` with an ``'x'`` coordinate, raw ``(x, y)`` arrays, or a
    bare ``y`` array (``x`` then defaults to the sample index), via the shared
    :func:`parse_xy` helper.

    Ported from qcat ``fit_exp_decay`` — generalised to the flexible
    :func:`parse_xy` input and the scqat ``fit()`` convention (``guess()`` runs
    only when parameters have not already been set, so a caller may tweak the
    guess before fitting).
    """

    def __init__(self, data: DataArray = None, x=None):
        self._data_parser(data, x)
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, a, tau, c):
        return a * exp(-x / tau) + c

    def guess(self):
        y = self.y
        x = self.x

        # Amplitude: difference between the first and last samples (sign-aware bounds)
        a_guess = float(y[0] - y[-1])
        if a_guess < 0:
            a_dict = dict(value=a_guess, min=2 * a_guess, max=0.0)
        elif a_guess > 0:
            a_dict = dict(value=a_guess, min=0.0, max=2 * a_guess)
        else:
            a_dict = dict(value=0.0)

        # Decay constant: order of half the swept span, bounded positive
        x_span = float(abs(x[-1] - x[0])) or 1.0
        tau_dict = dict(value=x_span / 2.0, min=0.0, max=x_span * 4.0)

        # Offset: the asymptotic (last) value, bounded within one amplitude of it.
        # When the first and last samples coincide (a_guess == 0 — flat or
        # otherwise degenerate data) that width collapses to zero and lmfit
        # rejects min == max at fit time; fall back to the data spread (then
        # |c_guess|, then 1.0) so the bound stays strictly open.
        c_guess = float(y[-1])
        c_halfwidth = abs(a_guess) or float(ptp(y)) or abs(c_guess) or 1.0
        c_dict = dict(value=c_guess, min=c_guess - c_halfwidth, max=c_guess + c_halfwidth)

        self.params = self.model.make_params(a=a_dict, tau=tau_dict, c=c_dict)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result
