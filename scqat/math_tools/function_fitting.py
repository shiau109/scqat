from abc import ABC, abstractmethod

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
