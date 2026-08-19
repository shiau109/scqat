"""AC-Stark phase echo estimator: the phase an off-resonant Stark tone imprints.

The probe runs a Hahn echo (``y90 - wait(D) - x180 - stark(D) - {x90 | -y90}``)
with an off-resonant Stark tone filling the SECOND free-evolution arm. The echo
refocuses static dephasing, so the only surviving phase is the AC-Stark phase
``phi = delta_stark * D`` accumulated in arm 2. That phase is read out in TWO
measurement bases (``meas_basis`` axis, 2 points):

* ``meas_basis = 0`` — close with ``x90``   -> <Z> = sin(phi)  (the Y quadrature)
* ``meas_basis = 1`` — close with ``-y90``  -> <Z> = cos(phi)  (the X quadrature)

so ``phi = atan2(sin, cos)`` — unambiguous over a full turn and sensitive
everywhere (a single closing pulse only sees ``cos phi``, blind near 0).

Dataset contract (per target; the ``target`` dim is split off upstream):
    * Variable:   ``signal`` (discriminated averaged population), OR ``IQdata``
      (complex), OR both ``I`` and ``Q`` to build it.
    * Coordinate: ``stark_amp``  — the swept Stark amplitude factor (outer).
    * Coordinate: ``meas_basis`` — the 2-point measurement basis [0, 1] (inner).

Method
------
Both bases are reduced with ONE shared axis (pooled ``positions``/PCA), so their
scale and offset are common: ``s_cos`` and ``s_sin`` orbit a circle whose center
is the readout offset. A least-squares circle fit finds that center, and the
phase is the angle of each point ABOUT IT,

    (cx, cy) = circle_fit(s_cos, s_sin)
    phi(a)   = unwrap(atan2(s_sin(a) - cy, s_cos(a) - cx)),  anchored to 0 at min |a|

Measuring the angle about the FITTED center (not by subtracting a single anchor
point) is robust to ANY constant phase offset at amp=0: a residual echo/readout
phase just moves where on the circle the anchor sits, it does not distort phi
(reading the offset off one anchor point silently breaks when phi(0) != 0). The
anchor (smallest ``|stark_amp|``, where ``phi ~ 0`` since ``phi ~ k*a**2`` is
minimized at ``a=0`` -- included by a one-sided 0..max or a symmetric -max..max
sweep) removes the constant offset; the readout contrast cancels in the angle and
a PCA sign flip only flips the winding direction. A linear fit of ``phi`` vs
``stark_amp**2`` reports the Stark coefficient ``k`` (slope).

Record-only: SCQO writes nothing to the device; ``stark_coeff`` / the per-amp
``phase`` land in the run record.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_positions, with_iqdata
from scqat.core.figures import render_figures
from scqat.estimators.qubit_stark_phase_echo.visualization import (
    plot_phase_vs_amp,
    plot_phasor,
    plot_quadratures,
)


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Algebraic (Kasa) least-squares circle fit -> (center_x, center_y, radius).

    Solves ``x**2 + y**2 = 2*cx*x + 2*cy*y + c`` (linear in cx, cy, c) for the
    center the (cos, sin) quadratures orbit. The center is the common readout
    offset; measuring the phase relative to it is robust to a constant phase
    offset at amp=0 (which a single-anchor offset estimate gets wrong)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    cx, cy, c = (float(v) for v in sol)
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
    return cx, cy, r


class QubitStarkPhaseEchoEstimator(BaseEstimator):
    """AC-Stark phase from a two-quadrature echo: phi(amp) and the Stark coefficient."""

    estimator_name = "qubit_stark_phase_echo"

    #: measurement-basis index -> closing pulse / quadrature it reads.
    SIN_BASIS = 0  # x90 close  -> <Z> = sin(phi)
    COS_BASIS = 1  # -y90 close -> <Z> = cos(phi)

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "stark_amp" not in dataset.coords:
            raise ValueError("qubit_stark_phase_echo requires a 'stark_amp' coordinate.")
        if "meas_basis" not in dataset.coords:
            raise ValueError("qubit_stark_phase_echo requires a 'meas_basis' coordinate.")
        if int(dataset.sizes.get("meas_basis", 0)) != 2:
            raise ValueError(
                "qubit_stark_phase_echo needs exactly 2 measurement bases "
                "(x90 and -y90); got "
                f"{int(dataset.sizes.get('meas_basis', 0))}."
            )
        if "signal" not in dataset and "IQdata" not in dataset and not (
            "I" in dataset and "Q" in dataset
        ):
            raise ValueError(
                "qubit_stark_phase_echo requires a 'signal' variable, an 'IQdata' "
                "variable, or both 'I' and 'Q'."
            )

    def _reduce(self, dataset: xr.Dataset) -> tuple[np.ndarray, str]:
        """Reduce to a real ``(stark_amp, meas_basis)`` signal with ONE shared axis.

        Discriminated data arrives pre-reduced as ``signal``; complex IQ is
        projected onto a single pooled ``|0>-|1>`` axis (so both bases share one
        scale/offset — the two-quadrature phase needs that).
        """
        if "signal" in dataset:
            sig = dataset["signal"].transpose("stark_amp", "meas_basis")
            return np.asarray(sig.values, dtype=float), "signal"

        from scqat.tools.iq_reduce import axial, axis_angle

        ds = with_iqdata(dataset)
        iq = ds["IQdata"].transpose("stark_amp", "meas_basis")
        I = np.real(iq.values)
        Q = np.imag(iq.values)
        pos = stored_positions(ds)
        # ONE axis for both bases: resolve it from the pooled cloud, then apply the
        # SAME fixed angle to every element (axial ravels; reshape restores 2-D).
        angle = axis_angle(I.ravel(), Q.ravel(), positions=pos)
        s = axial(I, Q, angle=angle).reshape(I.shape)
        return s, ("positions" if pos is not None else "pca")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        stark_amp = np.asarray(dataset.coords["stark_amp"].values, dtype=float)
        s, reduction_method = self._reduce(dataset)  # (n_amp, 2)

        s_sin = s[:, self.SIN_BASIS]  # x90  -> affine(sin phi)
        s_cos = s[:, self.COS_BASIS]  # -y90 -> affine(cos phi)

        # Fit the circle the (cos, sin) quadratures orbit, and measure the phase
        # RELATIVE TO ITS CENTER. This is robust to ANY constant phase offset at
        # amp=0 (a residual echo phase puts the anchor anywhere on the circle, not
        # at sin(phi)=0), where a single-anchor offset estimate would be wrong.
        cx, cy, r = _fit_circle(s_cos, s_sin)
        resid = np.hypot(s_cos - cx, s_sin - cy) - r
        rms_resid = float(np.sqrt(np.nanmean(resid ** 2))) if resid.size else float("inf")

        phase = np.unwrap(np.arctan2(s_sin - cy, s_cos - cx))
        # Anchor phi := 0 at the smallest |amplitude| (no stark drive -> no induced
        # phase, so this removes the constant readout/echo offset). phi ~ k*a**2 is
        # minimized at a=0, which a one-sided (0..max) or symmetric (-max..max)
        # sweep both include.
        i0 = int(np.argmin(np.abs(stark_amp)))
        phase = phase - phase[i0]

        amp_squared = stark_amp ** 2
        mask = np.isfinite(phase) & np.isfinite(amp_squared)
        if int(np.count_nonzero(mask)) >= 2:
            slope, intercept = np.polyfit(amp_squared[mask], phase[mask], 1)
        else:
            slope = intercept = np.nan

        # success: a real circle (its radius clears its own fit scatter, so the
        # phase is meaningful) plus a finite linear fit.
        success = bool(
            np.isfinite(slope)
            and int(np.count_nonzero(mask)) >= 2
            and r > 0.0
            and rms_resid < 0.5 * r
        )

        best_fit = slope * amp_squared + intercept if np.isfinite(slope) else np.full_like(
            amp_squared, np.nan
        )

        return {
            "stark_amp": stark_amp,
            "amp_squared": amp_squared,
            "phase": phase,
            "s_sin": np.asarray(s_sin, dtype=float),
            "s_cos": np.asarray(s_cos, dtype=float),
            "circle_cx": float(cx),
            "circle_cy": float(cy),
            "circle_r": float(r),
            "circle_rms_resid": rms_resid,
            "stark_coeff": float(slope),
            "intercept": float(intercept),
            "reduction_method": reduction_method,
            "success": success,
            "best_fit": np.asarray(best_fit, dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        drop = {"s_sin", "s_cos", "best_fit"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> xr.Dataset:
        stark_amp = np.asarray(results["stark_amp"], dtype=float)
        return xr.Dataset(
            {
                "s_sin": ("stark_amp", np.asarray(results["s_sin"], dtype=float)),
                "s_cos": ("stark_amp", np.asarray(results["s_cos"], dtype=float)),
                "phase": ("stark_amp", np.asarray(results["phase"], dtype=float)),
                "best_fit": ("stark_amp", np.asarray(results["best_fit"], dtype=float)),
                "amp_squared": ("stark_amp", np.asarray(results["amp_squared"], dtype=float)),
            },
            coords={"stark_amp": stark_amp},
            attrs={
                "stark_coeff": float(results["stark_coeff"]),
                "intercept": float(results["intercept"]),
                "circle_cx": float(results["circle_cx"]),
                "circle_cy": float(results["circle_cy"]),
                "circle_r": float(results["circle_r"]),
                "reduction_method": str(results["reduction_method"]),
                "success": int(bool(results["success"])),
            },
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results, **kwargs)
        return render_figures(
            {
                self.estimator_name: lambda: plot_phase_vs_amp(plot_data),
                "quadratures": lambda: plot_quadratures(plot_data),
                "phasor": lambda: plot_phasor(plot_data),
            },
            label=self.estimator_name,
        )
