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
        # Data-driven guesses: the frequency maximum (sweet spot) sits at the flux
        # offset, and Ej_sum inverts the arch top exactly: f_max = sqrt(8*Ec*Ej) - Ec
        # => Ej = (f_max + Ec)^2 / (8*Ec).
        span = float(np.max(x) - np.min(x))
        offset_guess = float(x[int(np.argmax(y))])
        f_max = float(np.max(y))
        ej_sum_guess = (f_max + self.Ec_design) ** 2 / (8.0 * self.Ec_design)

        # Period seed from the local curvature of the arch TOP: expanding the model
        # around the sweet spot, f(x) ~ f_max - (f_max+Ec)*(pi/P)^2*(x-x0)^2/2, so a
        # parabola fit y ~ a*(x-x0)^2 + f0 gives P = pi*sqrt((f0+Ec)/(2*|a|)). When
        # only the gentle top is inside the sweep (the common bring-up window) the
        # true period is far LARGER than the swept span, and seeding period=span
        # drops lmfit into a wrong basin (an oscillating arch through a gentle arc).
        # The seed is used ONLY in the top-only regime. Its tell-tale: the minimum
        # sits at a sweep EDGE. A deep INTERIOR minimum means a valley (half-period
        # point) is inside the window — the sweep spans a full arch feature, the
        # local-curvature expansion is invalid, and the span seed is the right one.
        period_guess = span if span > 0 else 1.0
        if x.size >= 5 and span > 0:
            order = np.argsort(x)
            y_sorted = y[order]
            y_range = float(np.max(y) - np.min(y))
            imin = int(np.argmin(y_sorted))
            deep_interior_min = (
                0 < imin < y_sorted.size - 1
                and y_sorted[imin] < min(y_sorted[0], y_sorted[-1]) - 0.1 * y_range
            )
            try:
                a = float(np.polyfit(x, y, 2)[0])
            except Exception:
                a = np.nan
            if np.isfinite(a) and a < 0 and y_range > 0 and not deep_interior_min:
                period_curv = float(np.pi * np.sqrt((f_max + self.Ec_design) / (2.0 * abs(a))))
                if np.isfinite(period_curv) and period_curv > 0:
                    period_guess = period_curv

        Ec_dict = dict(value=self.Ec_design, vary=False)
        period_dict = dict(value=period_guess, min=0.0)
        d_dict = dict(value=0.0, vary=False)
        offset_dict = dict(value=offset_guess,
                           min=offset_guess - period_guess / 2,
                           max=offset_guess + period_guess / 2)
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
