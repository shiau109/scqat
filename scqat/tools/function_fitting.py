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
