"""Notch-configuration resonator fit of complex S21.

Fits microwave resonators measured in the notch (hanger) geometry with the
model of

    S. Probst, F. B. Song, P. A. Bushev, A. V. Ustinov, M. Weides,
    "Efficient and robust analysis of complex scattering data under noise in
    microwave resonators", Rev. Sci. Instrum. 86, 024706 (2015).

Model::

    S21(f) = a * e^{i*alpha} * e^{-2*pi*i*f*delay}
             * [ 1 - (Ql/|Qc|) * e^{i*phi0} / (1 + 2i*Ql*(f/fr - 1)) ]

The environment (net amplitude ``a``, phase offset ``alpha``, cable delay
``delay``) is part of the model, so the "baseline" of a transmission scan is
calibrated out analytically instead of being estimated and subtracted.
``phi0`` accounts for impedance mismatch (a rotated resonance circle); the
diameter-corrected coupling quality factor is ``Qc = |Qc| / cos(phi0)`` and
``1/Qi = 1/Ql - 1/Qc``.

Pipeline (pure numpy/scipy; an independent implementation of the published
equations, NOT a port of any GPL code):

1. seed ``fr`` and ``Ql`` from a **power-Lorentzian** fit of ``|S21|^2``
   (``fit_lorentzian_bg`` — background-aware, so a sloped transmission baseline
   does not inflate the width the way a raw magnitude-FWHM does);
2. propose two cable-delay candidates — the full-band phase-slope regression and
   an algebraic-circle-tightness refinement — because neither is reliable in
   every regime (tightness wins for clean high-Q, the phase slope wins for noisy
   low-Q where the tightest circle is *not* the physical one);
3. for each delay candidate, seed ``a``/``alpha``/``|Qc|``/``phi0`` from a
   Taubin circle fit (G. Taubin, IEEE PAMI 13, 1991) and polish the remaining
   **six** parameters with a bounded complex least-squares fit of the S21 model,
   with the **delay held fixed** at the candidate;
4. keep the candidate with the smaller complex residual — the physically correct
   objective arbitrates the delay ambiguity — and read the coupling/internal
   quality factors and their errors from that fit's covariance.

Delay is fixed (not floated) inside each least-squares polish on purpose: over a
narrow relative bandwidth the cable delay, the global phase ``alpha`` and the
mismatch angle ``phi0`` are near-degenerate, so a free delay lets ``phi0``
collapse to zero. The two external delay heuristics bracket the true value and
the residual comparison selects between them.

The frequency axis must be the **absolute** frequency (Hz): the model term
``Ql*(f/fr - 1)`` is meaningless on a detuning axis that crosses zero.
"""

import numpy as np
from scipy.optimize import least_squares

from .function_fitting import FunctionFitting, register_fitter, parse_xy
from .fit_lorentzian_bg import FitLorentzianBG


def notch_s21(f, fr, Ql, absQc, phi0, a=1.0, alpha=0.0, delay=0.0):
    """Notch-geometry S21 model (see module docstring). Returns complex array."""
    ideal = 1.0 - (Ql / absQc) * np.exp(1j * phi0) / (1.0 + 2j * Ql * (f / fr - 1.0))
    return a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * delay) * ideal


def _span(f) -> float:
    """The swept frequency span by VALUE (``max - min``, 1.0 when degenerate).

    Never ``f[-1] - f[0]``: the sweep may run in either direction, and on a
    descending one that positional span is negative - it inverted the delay
    scan's ``minimize_scalar`` bracket (which raised) and the Lorentzian seed's
    width bound (which silently degraded the ``Ql`` seed).
    """
    return float(np.max(f) - np.min(f)) or 1.0


def _fit_circle_taubin(x, y):
    """Algebraic circle fit (Taubin SVD). Returns (xc, yc, r0)."""
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    z = u**2 + v**2
    zm = z.mean()
    if zm <= 0:  # all points coincide
        return float(xm), float(ym), 0.0
    zn = (z - zm) / (2.0 * np.sqrt(zm))
    A = np.column_stack((zn, u, v))
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    a0, b0, c0 = vt[-1]
    a0 = a0 / (2.0 * np.sqrt(zm))
    d0 = -zm * a0
    if a0 == 0:  # degenerate: points on a line
        return float(xm), float(ym), float(np.sqrt(zm))
    xc = -b0 / (2.0 * a0)
    yc = -c0 / (2.0 * a0)
    r0 = np.sqrt(b0**2 + c0**2 - 4.0 * a0 * d0) / (2.0 * abs(a0))
    return float(xc + xm), float(yc + ym), float(r0)


@register_fitter('notch_circle')
class FitNotchCircle(FunctionFitting):
    """
    Circle fit of complex notch-geometry S21 data (Probst model).

    Input DataArray must have a coordinate named 'x' holding the **absolute**
    frequency in Hz and complex values (I + iQ). Raw ``(x, y)`` arrays are
    accepted via ``parse_xy``.

    Parameters
    ----------
    data : xarray.DataArray or array-like (complex)
    delay : float, optional
        Fix the cable delay (seconds) instead of fitting it.
    x : array-like, optional
        Frequency grid when ``data`` is a bare array.

    ``fit()`` returns a plain dict (this is not an lmfit model)::

        fr, Ql, absQc, Qc_dia_corr, Qi_dia_corr, Qi_no_corr, phi0,
        delay, a, alpha, fr_err, Ql_err, absQc_err, phi0_err,
        Qc_dia_corr_err, Qi_dia_corr_err, chi_square, success,
        z_fit, z_norm, z_norm_fit  (complex arrays on the data grid)
    """

    #: full-model parameter order for the least-squares polish
    _PNAMES = ("fr", "Ql", "absQc", "phi0", "a", "alpha", "delay")

    def __init__(self, data=None, delay: float = None, x=None):
        self._data_parser(data, x)
        self.fixed_delay = delay
        self.results = None

    def _data_parser(self, data, x=None):
        x_arr, z_arr = parse_xy(data, x, dtype=complex)
        # parse_xy casts both arrays to the requested dtype; the frequency
        # grid is physically real — restore it.
        self.x = np.real(x_arr).astype(float)
        self.z = np.asarray(z_arr, dtype=complex)

    @staticmethod
    def model_function(f, fr, Ql, absQc, phi0, a=1.0, alpha=0.0, delay=0.0):
        return notch_s21(f, fr, Ql, absQc, phi0, a, alpha, delay)

    # ------------------------------------------------------------------
    # Seeds
    # ------------------------------------------------------------------
    def guess(self):
        """Seed ``fr``/``Ql`` from a power-Lorentzian fit and the cable delay
        from the full-band phase slope. Returned for inspection; ``fit()`` also
        tries an algebraic-circle delay candidate and picks the better one."""
        fr0, Ql0 = self._lorentzian_seed(self.x, self.z)
        phase = np.unwrap(np.angle(self.z))
        # centre f before the polyfit: absolute GHz-scale frequencies make the
        # raw Vandermonde ill-conditioned; shifting doesn't change the slope
        slope = np.polyfit(self.x - self.x.mean(), phase, 1)[0] if len(self.x) > 1 else 0.0
        return {"fr": fr0, "Ql": Ql0, "delay": -slope / (2.0 * np.pi)}

    @staticmethod
    def _lorentzian_seed(f, z):
        """Robust ``fr``/``Ql`` seed from a background-aware power-Lorentzian fit
        of ``|S21|^2`` (immune to the sloped transmission baseline that inflates
        a naive magnitude FWHM)."""
        from xarray import DataArray
        power = np.abs(z) ** 2
        # the span by VALUE: the sweep may run in either direction (see _span)
        span = _span(f)
        try:
            res = FitLorentzianBG(
                DataArray(power, coords={"x": f}, dims="x"),
                inverted=True, background_order=1,
                bounds={"x0": (float(f.min()), float(f.max())),
                        "gamma": (0.0, span)},
            ).fit()
            fr0 = float(res.params["x0"].value)
            fwhm0 = 2.0 * abs(float(res.params["gamma"].value))
        except Exception:
            fr0 = float(f[np.argmin(power)])
            fwhm0 = 0.0
        if not fwhm0 or not np.isfinite(fwhm0):
            fwhm0 = span / 10.0
        Ql0 = abs(fr0) / fwhm0 if fwhm0 else 1e4
        return fr0, Ql0

    def _fit_delay(self, delay0):
        """Algebraic-circle delay candidate: the delay-corrected data should lie
        on a circle, so minimize the radial scatter around the Taubin circle.

        Derivative-free (coarse bracket scan + bounded golden-section): the
        nested circle refit makes finite-difference gradients noisy, and the
        ~1e-8 s parameter scale stalls gradient methods."""
        from scipy.optimize import minimize_scalar

        f = self.x
        z = self.z / np.max(np.abs(self.z))

        def cost(tau):
            zc = z * np.exp(2j * np.pi * f * tau)
            xc, yc, r0 = _fit_circle_taubin(zc.real, zc.imag)
            return float(np.sum((np.hypot(zc.real - xc, zc.imag - yc) - r0) ** 2))

        span = _span(f)
        taus = delay0 + np.linspace(-0.6, 0.6, 13) / span
        tau_best = min(taus, key=cost)
        step = float(taus[1] - taus[0])
        sol = minimize_scalar(cost, bounds=(tau_best - 1.2 * step, tau_best + 1.2 * step),
                              method="bounded", options={"xatol": 1e-16})
        return float(sol.x)

    def _seeds_at_delay(self, f, z, delay, fr0, Ql0):
        """Circle-based seeds for ``a``, ``alpha``, ``|Qc|``, ``phi0`` at a
        given cable delay (with the Lorentzian ``fr``/``Ql`` seed)."""
        zd = z * np.exp(2j * np.pi * f * delay)
        xc, yc, r0 = _fit_circle_taubin(zd.real, zd.imag)
        zc = xc + 1j * yc
        ang_res = np.angle(zd[int(np.argmin(np.abs(z)))] - zc)   # resonance point
        off = zc + r0 * np.exp(1j * (ang_res + np.pi))           # opposite: off-res
        a_seed = float(np.abs(off)) or 1.0
        alpha_seed = float(np.angle(off))
        zn = zd / (a_seed * np.exp(1j * alpha_seed))
        xcn, ycn, r0n = _fit_circle_taubin(zn.real, zn.imag)
        phi0_seed = float(-np.arcsin(np.clip(ycn / r0n, -1.0, 1.0))) if r0n > 0 else 0.0
        absQc_seed = Ql0 / (2.0 * r0n) if r0n > 0 else Ql0 * 2.0
        return [fr0, Ql0, absQc_seed, phi0_seed, a_seed, alpha_seed, delay]

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, data=None, x=None) -> dict:
        if data is not None:
            self._data_parser(data, x)
        f, z = self.x, self.z

        fr0, Ql0 = self._lorentzian_seed(f, z)

        # delay candidates (the complex residual arbitrates between them); delay
        # is held FIXED inside each polish to avoid the delay/alpha/phi0
        # degeneracy over a narrow bandwidth
        if self.fixed_delay is not None:
            cands = [float(self.fixed_delay)]
        else:
            phase = np.unwrap(np.angle(z))
            d_lin = -np.polyfit(f - f.mean(), phase, 1)[0] / (2.0 * np.pi)
            d_circ = self._fit_delay(d_lin)
            cands = list({round(d_lin, 15), round(d_circ, 15)})

        # 6 free params: fr, Ql, absQc, phi0, a, alpha (delay fixed per candidate)
        lo = [f.min(), 1.0, 1.0, -np.pi, 0.0, -np.pi]
        hi = [f.max(), 1e8, 1e9, np.pi, np.inf, np.pi]

        best, delay = None, None
        for d in cands:
            def residual(p, _d=d):
                m = notch_s21(f, *p, _d)
                return np.concatenate([(m - z).real, (m - z).imag])
            p0 = self._seeds_at_delay(f, z, d, fr0, Ql0)[:6]
            p0 = [min(max(v, lo[i]), hi[i]) for i, v in enumerate(p0)]
            try:
                sol = least_squares(residual, p0, bounds=(lo, hi),
                                    method="trf", x_scale="jac", max_nfev=5000)
            except Exception:
                continue
            if best is None or sol.cost < best.cost:
                best, delay = sol, d

        if best is None:  # every candidate failed
            return self._failed_result(f, z, fr0, Ql0)

        fr, Ql, absQc, phi0, a, alpha = best.x

        # derived quality factors
        cosphi = np.cos(phi0)
        Qc_dia = absQc / cosphi if cosphi != 0 else np.inf
        Qi_dia = 1.0 / (1.0 / Ql - 1.0 / Qc_dia) if (1.0 / Ql - 1.0 / Qc_dia) != 0 else np.inf
        Qi_no = 1.0 / (1.0 / Ql - 1.0 / absQc) if (1.0 / Ql - 1.0 / absQc) != 0 else np.inf

        errs, chi2 = self._errors(best, Ql, absQc, phi0)

        z_fit = notch_s21(f, fr, Ql, absQc, phi0, a=a, alpha=alpha, delay=delay)
        env = a * np.exp(1j * alpha) * np.exp(-2j * np.pi * f * delay)
        z_norm = z / env
        z_norm_fit = notch_s21(f, fr, Ql, absQc, phi0)   # ideal (a=1, alpha=0, delay=0)

        # Circularity: a real resonance traces a 2-D arc in the complex plane, so
        # the normalized cloud has a substantial minor axis. Phase-less data
        # (e.g. a magnitude-only sweep with a noise Q quadrature) collapses onto a
        # line -> minor/major singular ratio ~ noise, and no meaningful Qi/Qc
        # exists. Isotropic noise only *raises* a line's ratio, so this is a safe
        # one-way gate.
        zc = z_norm - z_norm.mean()
        sv = np.linalg.svd(np.column_stack([zc.real, zc.imag]), compute_uv=False)
        circularity = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0

        in_span = float(f.min()) <= fr <= float(f.max())
        success = bool(in_span and np.isfinite(fr) and Ql > 0 and absQc > 0
                       and circularity > 0.3)

        self.results = {
            "fr": float(fr),
            "Ql": float(Ql),
            "absQc": float(absQc),
            "Qc_dia_corr": float(Qc_dia),
            "Qi_dia_corr": float(Qi_dia),
            "Qi_no_corr": float(Qi_no),
            "phi0": float(phi0),
            "delay": float(delay),
            "a": float(a),
            "alpha": float(alpha),
            "fr_err": errs["fr"],
            "Ql_err": errs["Ql"],
            "absQc_err": errs["absQc"],
            "phi0_err": errs["phi0"],
            "Qc_dia_corr_err": errs["Qc_dia"],
            "Qi_dia_corr_err": errs["Qi_dia"],
            "chi_square": chi2,
            "circularity": circularity,
            "success": success,
            "z_fit": z_fit,
            "z_norm": z_norm,
            "z_norm_fit": z_norm_fit,
        }
        return self.results

    # ------------------------------------------------------------------
    # Errors (covariance of the full-model least-squares fit)
    # ------------------------------------------------------------------
    @staticmethod
    def _errors(sol, Ql, absQc, phi0):
        nan = {k: float("nan") for k in ("fr", "Ql", "absQc", "phi0", "Qc_dia", "Qi_dia")}
        try:
            J = sol.jac
            n_res, n_par = J.shape
            dof = max(n_res - n_par, 1)
            chi2 = float(2.0 * sol.cost / dof)
            cov = np.linalg.inv(J.T @ J) * chi2
            sd = np.sqrt(np.abs(np.diag(cov)))
            fr_e, Ql_e, Qc_e, phi_e = sd[0], sd[1], sd[2], sd[3]

            # propagate into the diameter-corrected quantities (indices 1,2,3)
            cosphi, sinphi = np.cos(phi0), np.sin(phi0)
            sub = np.array([1, 2, 3])
            cov_s = cov[np.ix_(sub, sub)]
            g_qc = np.array([0.0, 1.0 / cosphi, absQc * sinphi / cosphi**2])
            qc_var = float(g_qc @ cov_s @ g_qc)
            qi = 1.0 / (1.0 / Ql - cosphi / absQc)
            g_qi = np.array([
                qi**2 / Ql**2,
                -qi**2 * cosphi / absQc**2,
                -qi**2 * sinphi / absQc,
            ])
            qi_var = float(g_qi @ cov_s @ g_qi)
            return {
                "fr": float(fr_e), "Ql": float(Ql_e), "absQc": float(Qc_e),
                "phi0": float(phi_e),
                "Qc_dia": float(np.sqrt(abs(qc_var))),
                "Qi_dia": float(np.sqrt(abs(qi_var))),
            }, chi2
        except Exception:
            return nan, float("nan")

    def _failed_result(self, f, z, fr0, Ql0):
        """Every least-squares candidate raised: return a non-success dict that
        still satisfies the return contract (seeds as best-effort values)."""
        nan = float("nan")
        return {
            "fr": float(fr0), "Ql": float(Ql0), "absQc": nan,
            "Qc_dia_corr": nan, "Qi_dia_corr": nan, "Qi_no_corr": nan,
            "phi0": nan, "delay": float(self.fixed_delay or 0.0), "a": nan, "alpha": nan,
            "fr_err": nan, "Ql_err": nan, "absQc_err": nan, "phi0_err": nan,
            "Qc_dia_corr_err": nan, "Qi_dia_corr_err": nan, "chi_square": nan,
            "success": False,
            "z_fit": np.full_like(z, nan), "z_norm": z.copy(),
            "z_norm_fit": np.full_like(z, nan),
        }
