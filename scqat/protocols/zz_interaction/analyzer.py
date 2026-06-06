from typing import Any, Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.core.base_analyzer import BaseAnalyzer
from scqat.math_tools.fit_damped_oscillation import FitDampedOscillation
from scqat.protocols.zz_interaction.visualization import (
    plot_raw_with_overlay,
    plot_zz_value,
)


class ZZInteractionEchoAnalyzer(BaseAnalyzer):
    """
    Analyze a ZZ-interaction echo experiment: a damped oscillation in free
    evolution ``time`` measured at every ``flux`` point.

    Expects an xarray.Dataset with:
        - Variable:    ``signal`` with dims (flux, time)
        - Coordinate:  ``time``  — free evolution time
        - Coordinate:  ``flux``  — the swept flux/coupler coordinate

    For each flux slice a :class:`FitDampedOscillation`
    (``a*exp(-kappa*x)*cos(2*pi*f*x + phi) + c``) is fitted. The oscillation
    frequency ``f`` is the ZZ strength and ``tau = 1/kappa`` is the echo decay
    (T2) — both reported as a function of flux.

    Ported from qcat ``zz_interaction``. Two differences from the original, both
    deliberate: the analyzer never mutates its input (qcat divided the ``time``
    coordinate by 1000 in place), and any unit rescaling of ``time`` is opt-in via
    the ``time_scale`` kwarg (default 1.0) rather than a hard-coded ms→µs factor.
    """

    protocol_name = "zz_interaction"

    # ------------------------------------------------------------------
    # BaseAnalyzer interface
    # ------------------------------------------------------------------

    def _check_data(self, dataset: xr.Dataset) -> None:
        if 'signal' not in dataset:
            raise ValueError("ZZ-interaction echo requires a 'signal' variable.")
        if 'time' not in dataset.coords:
            raise ValueError("ZZ-interaction echo requires a 'time' coordinate.")
        if 'flux' not in dataset.coords:
            raise ValueError("ZZ-interaction echo requires a 'flux' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """
        Fit a damped oscillation per flux point and collect the parameters.

        Kwargs:
            time_scale (float): Optional multiplier applied to the ``time`` axis
                before fitting (default 1.0, i.e. no change). Pass e.g. ``1e-3``
                to reproduce qcat's ms→µs conversion without mutating the input.

        Returns a dict of per-flux arrays:
            flux, a, kappa, f, phi, c, tau  (tau = 1/kappa).
        """
        time_scale = float(kwargs.get('time_scale', 1.0))

        flux = np.asarray(dataset.coords['flux'].values, dtype=float)
        time = np.asarray(dataset.coords['time'].values, dtype=float) * time_scale
        signal = dataset['signal'].transpose('flux', 'time').values

        n_flux = flux.shape[0]
        a = np.full(n_flux, np.nan)
        kappa = np.full(n_flux, np.nan)
        f = np.full(n_flux, np.nan)
        phi = np.full(n_flux, np.nan)
        c = np.full(n_flux, np.nan)

        for i in range(n_flux):
            da = xr.DataArray(np.asarray(signal[i]).squeeze(), coords={'x': time}, dims='x')
            try:
                fitter = FitDampedOscillation(da)
                fitter.guess()
                fitter.params['f'].set(min=0.0)
                result = fitter.fit()
                p = result.params
                a[i] = p['a'].value
                kappa[i] = p['kappa'].value
                f[i] = p['f'].value
                phi[i] = p['phi'].value
                c[i] = p['c'].value
            except Exception:
                continue

        with np.errstate(divide='ignore', invalid='ignore'):
            tau = np.where(kappa != 0, 1.0 / kappa, np.nan)

        return {
            'flux': flux,
            'a': a,
            'kappa': kappa,
            'f': f,
            'phi': phi,
            'c': c,
            'tau': tau,
        }

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        """
        Assemble the minimal arrays the two figures need into one self-sufficient
        Dataset, so the plots reconstruct from the saved ``*_plotdata.nc`` alone.

        Variables: ``signal`` (flux, time) for the raw colour-map, and ``f``/``tau``
        (flux) for the ZZ-strength and decay overlays.
        """
        time_scale = float(kwargs.get('time_scale', 1.0))
        flux = np.asarray(results['flux'], dtype=float)
        time = np.asarray(dataset.coords['time'].values, dtype=float) * time_scale
        signal = dataset['signal'].transpose('flux', 'time').values

        return xr.Dataset(
            {
                'signal': (['flux', 'time'], np.asarray(signal)),
                'f': ('flux', np.asarray(results['f'], dtype=float)),
                'tau': ('flux', np.asarray(results['tau'], dtype=float)),
            },
            coords={'flux': flux, 'time': time},
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Generate the raw colour-map (with ZZ-period / T2 overlays) and the
        ZZ-value figure, drawing strictly from ``plot_data``; rebuild it only when
        called outside ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return {
            'raw_data': plot_raw_with_overlay(plot_data),
            'zz_value': plot_zz_value(plot_data),
        }
