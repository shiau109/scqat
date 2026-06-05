"""
Fit a sum of damped oscillations seeded from Hankel modes.

Model
-----
    y(x) = sum_i a_i * exp(k_i * x) * cos(2*pi*f_i*x + phi_i) + c

where ``k_i`` is the continuous-time decay rate (typically negative for
decaying signals).  The number of components ``N`` and initial seeds for
``(a_i, k_i, f_i, phi_i)`` are taken from a list of Hankel mode dicts as
returned by :func:`scqat.math_tools.hankel.hankel_decompose` (each dict
provides ``amplitude``, ``decay_rate``, ``freq_hz``, ``phase``).

The constant offset ``c`` is a free parameter.  Frequency, decay rate,
amplitude and phase of every component are all free; the Hankel values
serve only as initial guesses.
"""

from typing import List, Dict, Any
import numpy as np
from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult

from .function_fitting import FunctionFitting, register_fitter, parse_xy


def _multi_damped_osc(x, params_dict, n_modes):
    """Evaluate the multi-damped-oscillation model from a flat dict of params."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for i in range(n_modes):
        a = params_dict[f"a{i}"]
        k = params_dict[f"k{i}"]
        f = params_dict[f"f{i}"]
        phi = params_dict[f"phi{i}"]
        y = y + a * np.exp(k * x) * np.cos(2.0 * np.pi * f * x + phi)
    return y + params_dict["c"]


def multi_damped_osc_eval(x, modes_params: List[Dict[str, float]], c: float = 0.0):
    """Public helper to evaluate the model from a list of per-mode dicts.

    Parameters
    ----------
    x : array-like
        Time/independent variable.
    modes_params : list of dict
        Each dict must contain keys ``a``, ``k``, ``f``, ``phi``.
    c : float
        Constant offset.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    for m in modes_params:
        y = y + m["a"] * np.exp(m["k"] * x) * np.cos(
            2.0 * np.pi * m["f"] * x + m["phi"]
        )
    return y + c


@register_fitter("multi_damped_oscillation")
class FitMultiDampedOscillation(FunctionFitting):
    """
    Fit a sum of N damped oscillations to data, seeded from Hankel modes.

    Parameters
    ----------
    data : xr.DataArray
        1-D data with a coordinate named ``'x'`` (time).
    modes : list of dict
        Hankel modes used as initial guesses.  Each dict must provide
        ``amplitude``, ``decay_rate``, ``freq_hz`` and ``phase`` (the
        format produced by :func:`scqat.math_tools.hankel.hankel_decompose`).
    """

    def __init__(self, data: DataArray = None, modes: List[Dict[str, Any]] = None, x=None):
        if modes is None or len(modes) == 0:
            raise ValueError("FitMultiDampedOscillation requires at least one mode.")
        self.modes = list(modes)
        self.n_modes = len(self.modes)
        self._data_parser(data, x)

        # Build a lmfit Model with a dynamic parameter list.
        param_names = []
        for i in range(self.n_modes):
            param_names.extend([f"a{i}", f"k{i}", f"f{i}", f"phi{i}"])
        param_names.append("c")
        self._param_names = param_names

        n_modes_local = self.n_modes

        # lmfit's Model inspects the function signature for parameter names,
        # so we build a real function with the right signature via exec().
        param_args = ", ".join(param_names)
        src = (
            f"def _model_fn(x, {param_args}):\n"
            f"    _kwargs = {{{', '.join(f'{n!r}: {n}' for n in param_names)}}}\n"
            f"    return _multi_damped_osc(x, _kwargs, {n_modes_local})\n"
        )
        ns = {"_multi_damped_osc": _multi_damped_osc}
        exec(src, ns)
        _model_fn = ns["_model_fn"]

        self.model = Model(_model_fn, independent_vars=["x"])
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, **kwargs):
        return _multi_damped_osc(x, kwargs, self.n_modes)

    def guess(self):
        y = self.y
        amp_max = 0.5 #float(np.max(np.abs(y))) if y.size else 1.0
        # c_guess = float(np.mean(y[-max(int(0.1 * y.size), 1):])) if y.size else 0.0
        c_guess = 0.0

        seeds = {}
        for i, m in enumerate(self.modes):
            a0 = float(m.get("amplitude", amp_max ))#/ self.n_modes))
            k0 = float(m.get("decay_rate", -1.0 / max(abs(self.x[-1] - self.x[0]), 1e-12)))
            f0 = float(m.get("freq_hz", 0.0))
            phi0 = float(m.get("phase", 0.0))

            # Reasonable bounds: decay must be non-positive, freq non-negative,
            # amplitude bounded by a generous multiple of the signal peak.
            seeds[f"a{i}"] = dict(value=a0, min=a0*0.1, max=a0*10)
            seeds[f"k{i}"] = dict(value=k0, min=k0*0.1, max=k0*10)
            if abs(f0) < 1e-12:
                seeds[f"f{i}"] = dict(value=0, min=0, max=0.005)
            else:
                seeds[f"f{i}"] = dict(value=abs(f0), min=0, max=0.005)
            seeds[f"phi{i}"] = dict(value=phi0, min=-2.0 * np.pi, max=2.0 * np.pi)

        seeds["c"] = dict(value=c_guess, min=-0.1, max=0.1)
        self.params = self.model.make_params(**seeds)
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()
        self.result = self.model.fit(self.y, self.params, x=self.x)
        return self.result

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def unpack_modes(self, result: ModelResult = None) -> List[Dict[str, float]]:
        """Return per-mode (a, k, f, phi) dicts from a fit result."""
        if result is None:
            result = getattr(self, "result", None)
        if result is None:
            raise RuntimeError("No fit result available; call fit() first.")
        out = []
        for i in range(self.n_modes):
            out.append({
                "a": float(result.params[f"a{i}"].value),
                "k": float(result.params[f"k{i}"].value),
                "f": float(result.params[f"f{i}"].value),
                "phi": float(result.params[f"phi{i}"].value),
                "a_err": (float(result.params[f"a{i}"].stderr)
                          if result.params[f"a{i}"].stderr is not None else float("nan")),
                "k_err": (float(result.params[f"k{i}"].stderr)
                          if result.params[f"k{i}"].stderr is not None else float("nan")),
                "f_err": (float(result.params[f"f{i}"].stderr)
                          if result.params[f"f{i}"].stderr is not None else float("nan")),
                "phi_err": (float(result.params[f"phi{i}"].stderr)
                            if result.params[f"phi{i}"].stderr is not None else float("nan")),
            })
        return out
