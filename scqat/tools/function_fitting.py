from abc import ABC, abstractmethod

import numpy as np
from xarray import DataArray


def parse_xy(data, x=None, dtype=float):
    """
    Normalize fitter input into ``(x, y)`` arrays so fitters can be reused by
    external (e.g. simulation) callers without wrapping their data.

    Accepts:
      * an ``xarray.DataArray`` with an ``'x'`` coordinate,
      * raw arrays — pass ``y`` as ``data`` and the grid as ``x``,
      * a bare ``y`` array — ``x`` then defaults to the sample index.
    """
    if data is None:
        raise ValueError("No input data provided to fitter.")
    if isinstance(data, DataArray):
        y = np.asarray(data.values, dtype=dtype)
        if "x" in data.coords:
            x_arr = np.asarray(data.coords["x"].values, dtype=dtype)
        elif x is not None:
            x_arr = np.asarray(x, dtype=dtype)
        else:
            x_arr = np.arange(y.shape[0], dtype=dtype)
        return x_arr, y
    y = np.asarray(data, dtype=dtype)
    x_arr = np.asarray(x, dtype=dtype) if x is not None else np.arange(y.shape[0], dtype=dtype)
    return x_arr, y


def robust_dt(t) -> float:
    """Average sample spacing of a 1-D axis, robust to a degenerate leading step.

    Returns the first step ``t[1]-t[0]`` when it is non-zero (preserving the
    historical behaviour for well-formed, uniformly sampled axes) and falls back
    to the average spacing ``(t[-1]-t[0])/(N-1)`` when the leading step is zero --
    e.g. a Ramsey idle-time sweep whose first points collapse onto the same value
    after 4 ns clock-cycle rounding. Raises ``ValueError`` if no non-zero spacing
    exists (fewer than two samples, or every position identical), so a divide by
    zero surfaces as a clear error instead of a ``ZeroDivisionError`` deep in FFT.
    """
    t = np.asarray(t)
    n = len(t)
    if n < 2:
        raise ValueError("Need at least 2 samples to determine a sample spacing.")
    dt = float(t[1] - t[0])
    if dt != 0:
        return dt
    span = float(t[-1] - t[0])
    if span != 0:
        return span / (n - 1)
    raise ValueError("Degenerate x axis: all sample positions are equal.")


def seed_amp_phase(t, y, freqs, kappas=None, basis="cos"):
    """Least-squares seed for the amplitude(s) and phase(s) of a damped sinusoid.

    Projects ``y`` onto the (envelope-weighted) cos/sin quadratures at each
    frequency in ``freqs``, plus a constant, so a fitter can seed
    ``a*exp(-kappa*x)*trig(2*pi*f*x + phi)`` with the phase the data actually
    carries instead of a blind ``phi=0``. A blind phase seed pins the model at a
    fixed phase; when the real fringe is near anti-phase there, the amplitude
    lower bound (``a>=0``) traps the optimiser at ``a->0`` — a flat line that
    "converges" but explains no oscillation. Seeding the phase from the data
    keeps the optimiser in the right basin so the amplitude stays meaningful.

    Parameters
    ----------
    t, y : 1-D arrays
        The sample grid and the signal.
    freqs : float or sequence of float
        One frequency (single component) or several (a beat).
    kappas : float or sequence of float, optional
        Per-frequency decay rate for the envelope weight; broadcast against
        ``freqs``. Defaults to 0 (no envelope), a robust seed regardless.
    basis : {"cos", "sin"}
        Trig basis of the target model, so the returned phase matches it.

    Returns
    -------
    (amp, phi) for a scalar ``freqs``, else a list of ``(amp, phi)`` pairs in
    the order of ``freqs``. ``amp`` is non-negative; ``phi`` is in ``[-pi, pi]``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    scalar = np.isscalar(freqs)
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    if kappas is None:
        kappas = np.zeros_like(freqs)
    kappas = np.broadcast_to(np.asarray(kappas, dtype=float), freqs.shape)

    columns = [np.ones_like(t)]
    for f, k in zip(freqs, kappas):
        theta = 2 * np.pi * f * t
        env = np.exp(-k * t)
        columns.append(env * np.cos(theta))
        columns.append(env * np.sin(theta))
    coef, *_ = np.linalg.lstsq(np.column_stack(columns), y, rcond=None)

    out = []
    for i in range(len(freqs)):
        c_cos = float(coef[1 + 2 * i])
        c_sin = float(coef[2 + 2 * i])
        amp = float(np.hypot(c_cos, c_sin))
        if basis == "sin":
            # a*sin(th+phi) = (a cos phi) sin th + (a sin phi) cos th
            phi = float(np.arctan2(c_cos, c_sin))
        else:
            # a*cos(th+phi) = (a cos phi) cos th - (a sin phi) sin th
            phi = float(np.arctan2(-c_sin, c_cos))
        out.append((amp, phi))
    return out[0] if scalar else out


class FunctionFitting(ABC):
    """
    Abstract base class for all function fitting routines.
    Child classes must implement model_function, guess, and fit methods.
    """
    def __init__(self):
        pass

    @abstractmethod
    def model_function(self, *args, **kwargs):
        pass

    @abstractmethod
    def guess(self):
        pass

    @abstractmethod
    def fit(self, data=None):
        pass

    def fitting_curve(self, x):
        """Return the model evaluated at x using current parameters."""
        return self.model(x)


# Registry for fitter classes
_FITTER_REGISTRY = {}

def register_fitter(name):
    """Decorator to register a fitter class by name for the factory."""
    def decorator(cls):
        _FITTER_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_fitter(name, *args, **kwargs):
    """
    Factory function to get a fitter class by name.
    Example: get_fitter('cosine', data)
    """
    cls = _FITTER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown fitter: {name}. Available: {list(_FITTER_REGISTRY.keys())}")
    return cls(*args, **kwargs)
