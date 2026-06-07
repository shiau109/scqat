from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


# ----------------------------------------------------------------------
# Reparametrisation
# ----------------------------------------------------------------------
# The non-Markovian amplitude-damping channel is most naturally written in
# terms of the Lindbladian rates (gamma, lambda_):
#
#     gamma    = 2 * Lambda                  (relaxation rate)
#     lambda_  = sqrt(Gamma * Lambda / 2)    (coupling strength)
#
# which is equivalent to the older (Gamma, Lambda) form via
#
#     Lambda = gamma / 2
#     Gamma  = 4 * lambda_**2 / gamma
#
# Public model functions and fitter parameters use (gamma, lambda_).
# Internally we still compute G(t) using (Gamma, Lambda) since the closed
# form was derived in those variables -- this keeps the math transparent.


def decoherence_G(x, gamma, lambda_, Delta):
    """
    Non-Markovian amplitude-damping decoherence function G(t) with a
    detuning ``Delta``:

        Lambda = gamma / 2
        a      = Lambda - 1j*Delta
        d      = sqrt(a**2 - 2*Gamma*Lambda) = sqrt(a**2 - 4*lambda_**2)
        G(t)   = exp(-a*t/2) [cosh(d*t/2) + (a/d) sinh(d*t/2)]

    The independent variable is named ``x`` (= time) for compatibility with
    the FunctionFitting/lmfit convention.  Returns the complex G(t); callers
    take |G|^2 or Re(G) as appropriate for the observed component.
    """
    t = np.asarray(x, dtype=float)
    Lambda = gamma / 2.0
    a = Lambda - 1j * Delta
    # d^2 = a^2 - 2*Gamma*Lambda = a^2 - 4*lambda_^2
    d_sq = a * a - 4.0 * lambda_ * lambda_
    d = np.sqrt(np.complex128(d_sq))
    if np.abs(d) < 1e-15:
        # Critical-damping limit
        G = np.exp(-a * t / 2) * (1.0 + a * t / 2)
    else:
        arg = d * t / 2
        G = np.exp(-a * t / 2) * (
            np.cosh(arg) + (a / d) * np.sinh(arg)
        )
    return np.asarray(G, dtype=np.complex128)


def rho11_model(x, gamma, lambda_, Delta, rho_0):
    """rho_11(t) = |G(t)|^2 * rho_11(0)."""
    G = decoherence_G(x, gamma, lambda_, Delta)
    return (np.abs(G) ** 2 * rho_0).astype(float)


def rho10_model(x, gamma, lambda_, Delta, rho_0):
    """rho_10(t) = Re[G(t)] * rho_10(0)."""
    G = decoherence_G(x, gamma, lambda_, Delta)
    return (np.real(G) * rho_0).astype(float)


@register_fitter('qubit_decoherence')
class FitQubitDecoherence(FunctionFitting):
    """
    Fit non-Markovian amplitude-damping decoherence data.

    Two model components are supported via the ``component`` argument:
        - 'rho_11' : |G(t)|^2 * rho_0
        - 'rho_10' :  G(t)   * rho_0

    Fit parameters are (gamma, lambda_, rho_0).  Input DataArray must have a
    coordinate named 'x' (time).
    """

    def __init__(self, data: DataArray = None, component: str = "rho_11", fix_delta: bool = False, x=None):
        if component not in ("rho_11", "rho_10"):
            raise ValueError("component must be 'rho_11' or 'rho_10'.")
        self.component = component
        self.fix_delta = fix_delta
        self._data_parser(data, x)
        self._model_fn = rho11_model if component == "rho_11" else rho10_model
        self.model = Model(self._model_fn)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    def model_function(self, x, gamma, lambda_, Delta, rho_0):
        return self._model_fn(x, gamma, lambda_, Delta, rho_0)

    @staticmethod
    def _envelope_decay_time(t, y):
        """
        Estimate the 1/e decay time of the envelope of ``y`` (oscillatory or not).

        Robust to oscillations: uses |y - baseline| and finds the first time the
        envelope drops to 1/e of its initial value.  Falls back to a span-based
        estimate if the data does not decay enough within the observation window.
        """
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        if t.size < 2:
            return 1.0
        # Baseline: long-time mean of last 10% of points (handles rho_11 floor)
        tail = max(int(0.1 * y.size), 1)
        baseline = float(np.mean(y[-tail:]))
        env = np.abs(y - baseline)
        env0 = env[0] if env[0] > 0 else env.max()
        if env0 <= 0:
            return max(t[-1] - t[0], 1.0)
        target = env0 / np.e
        below = np.where(env <= target)[0]
        if below.size > 0 and below[0] > 0:
            return float(t[below[0]] - t[0])
        # Did not decay to 1/e in window – use full span as a lower bound
        return max(t[-1] - t[0], 1.0)

    def guess(self):
        t = self.x
        y = self.y

        tau_e = self._envelope_decay_time(t, y)
        # Build the initial guess in the (Gamma, Lambda) basis where the
        # closed-form reasoning is simpler, then convert to (gamma, lambda_).
        # At critical damping (Lambda = 2*Gamma) |G|^2 reaches 1/e near
        # t ~ 1.7/Lambda.  Multi-start in fit() spans the regimes.
        Lambda_guess = 1.7 / max(tau_e, 1e-12)
        Gamma_guess = 0.5 * Lambda_guess  # critical-damping seed

        gamma_guess = 2.0 * Lambda_guess
        lambda_guess = float(np.sqrt(Gamma_guess * Lambda_guess / 2.0))
        rho0_guess = float(y[0])

        delta_spec = dict(value=0.0, vary=False) if self.fix_delta else dict(value=0.0, min=-0.01, max=0.01)
        self.params = self.model.make_params(
            gamma=dict(value=gamma_guess, min=0.0, max=0.02),
            lambda_=dict(value=lambda_guess, min=0.0, max=0.02),
            Delta=delta_spec,
            rho_0=dict(value=rho0_guess),
        )
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
        if self.params is None:
            self.guess()

        # Recover Lambda0 from the current gamma seed and run multi-start over
        # (Gamma, Lambda) ratios spanning all damping regimes.
        gamma0 = float(self.params['gamma'].value)
        rho0 = float(self.params['rho_0'].value)
        Lambda0 = gamma0 / 2.0  # Lambda = gamma / 2

        # Gamma/Lambda ratios span deep-underdamped -> critical -> overdamped.
        gamma_over_Lambda_ratios = (0.5, 1.0, 1.5, 2)

        delta_spec = dict(value=0.0, vary=False) if self.fix_delta else dict(value=0.0, min=-0.01, max=0.01)
        best_result = None
        best_chi = np.inf
        for r in gamma_over_Lambda_ratios:
            Gamma_seed = r * Lambda0
            gamma_seed = 2.0 * Lambda0  # = gamma0
            lambda_seed = float(np.sqrt(Gamma_seed * Lambda0 / 2.0))
            seed = self.model.make_params(
                gamma=dict(value=gamma_seed, min=0.0, max=0.01),
                lambda_=dict(value=lambda_seed, min=0.0, max=0.02),
                Delta=delta_spec,
                rho_0=dict(value=rho0, min=-1.2, max=1.2),
            )
            try:
                res = self.model.fit(self.y, seed, x=self.x)
            except Exception:
                continue
            chi = res.chisqr if res.chisqr is not None else np.inf
            if res.success and chi < best_chi:
                best_chi = chi
                best_result = res

        if best_result is None:
            # Fall back to a single fit with the original guess so we still
            # return a ModelResult (may be unsuccessful).
            best_result = self.model.fit(self.y, self.params, x=self.x)

        self.result = best_result
        return best_result
