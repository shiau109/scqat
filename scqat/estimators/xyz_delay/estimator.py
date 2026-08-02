from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator, stored_positions, with_iqdata
from scqat.tools.fit_triangle import fit_triangle
from scqat.tools.iq_reduce import axis_angle
from scqat.estimators.xyz_delay.visualization import plot_delay_scan


class XyzDelayEstimator(BaseEstimator):
    """Estimator for an XY-Z (flux-line) delay experiment: fits the ``|e> - |g>``
    readout-contrast trace versus the relative XY/Z timing to the triangle-overlap
    *model* and returns the peak position — the flux-line delay to apply.

    Expects an ``xarray.Dataset`` with (the ``target``/``qubit`` dim already
    removed):
        - Coordinate: ``'prepared_state'`` — the integer prep labels ``0`` (|g>)
          and ``1`` (|e>).
        - Coordinate: ``'relative_time'`` — the XY/Z relative shift (any unit;
          the fitted ``t0`` comes back in it).
        - Variable: ``'signal'`` (FPGA-discriminated averaged population),
          **or** complex ``'IQdata'``, **or** both ``'I'`` and ``'Q'``.

    The ``|e> - |g>`` difference isolates the timing-dependent contrast: the XY
    and Z pulses share a length, so their overlap versus relative shift is a
    triangle peaked at alignment. The heavy lifting — the guarded triangle fit
    with its argmax fallback — is :func:`scqat.tools.fit_triangle.fit_triangle`;
    this estimator resolves the dataset into the difference trace, forwards the
    flat kwarg surface, and owns the artifacts.

    The QM node's ``estimate`` step calls :meth:`analyze`; ``update`` then adds
    the returned ``t0_s`` to the flux channel's stored delay.
    """

    estimator_name = "xyz_delay"

    def _check_data(self, dataset: xr.Dataset) -> None:
        has_iq = "IQdata" in dataset.data_vars or ("I" in dataset.data_vars and "Q" in dataset.data_vars)
        if "signal" not in dataset.data_vars and not has_iq:
            raise ValueError(
                "XY-Z delay analysis requires a 'signal' variable, or complex 'IQdata', "
                "or both 'I' and 'Q'."
            )
        if "prepared_state" not in dataset.coords:
            raise ValueError("XY-Z delay analysis requires a 'prepared_state' coordinate ([0, 1]).")
        if "relative_time" not in dataset.coords:
            raise ValueError("XY-Z delay analysis requires a 'relative_time' coordinate.")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Reduce to the |e> - |g> difference trace and fit the triangle peak.

        Kwargs — flat and fully owned; unknown names raise:
            xy_duration_s (float, REQUIRED): the XY/Z pulse duration, in the unit
                of ``relative_time`` (seconds), seeding the triangle half-width
                and the wing/edge scale.
            min_snr (float): minimum peak significance for success (default 3.0).
            max_t0_std_s (float): maximum t0 uncertainty for success, same unit
                as ``relative_time`` (default: 0.125 * xy_duration_s).
            angle, positions: IQ->1-D axial-reduction axis knobs (see
                :func:`scqat.tools.iq_reduce.axis_angle`); ignored when the
                dataset already carries a real ``signal``. Absent both, the
                stored |g>/|e> blob centers (if attached) fix the axis, else PCA.

        Returns a dict with:
            t0_s, t0_std_s, amp, offset, half_width_s, snr, success,
            reduction_method, relative_time, difference, best_fit.
        """
        if "xy_duration_s" not in kwargs:
            raise ValueError("XyzDelayEstimator requires the 'xy_duration_s' kwarg (the pi pulse length).")
        xy_duration_s = float(kwargs.pop("xy_duration_s"))
        min_snr = float(kwargs.pop("min_snr", 3.0))
        max_t0_std = kwargs.pop("max_t0_std_s", None)
        max_t0_std = None if max_t0_std is None else float(max_t0_std)

        unknown = set(kwargs) - {"angle", "positions"}
        if unknown:
            raise ValueError(f"Unknown XY-Z delay kwarg(s) {sorted(unknown)}.")

        ds = dataset.transpose("prepared_state", "relative_time")
        t = np.asarray(ds.coords["relative_time"].values, dtype=float)

        if "signal" in ds.data_vars:
            sig = ds["signal"]
            excited = np.asarray(sig.sel(prepared_state=1).values, dtype=float)
            ground = np.asarray(sig.sel(prepared_state=0).values, dtype=float)
            difference = excited - ground
            reduction = "state"
        else:
            iq = with_iqdata(ds)["IQdata"]
            i_all = np.real(iq.values)
            q_all = np.imag(iq.values)
            angle = kwargs.get("angle")
            positions = kwargs.get("positions")
            if angle is None and positions is None:
                positions = stored_positions(ds)  # deterministic g->e axis when available
            # One shared projection axis for both prep slices, resolved over the
            # pooled cloud so the difference is a clean signed triangle.
            a = axis_angle(i_all.ravel(), q_all.ravel(), angle=angle, positions=positions)

            def project(z: np.ndarray) -> np.ndarray:
                return np.real(z) * np.cos(a) - np.imag(z) * np.sin(a)

            excited = project(iq.sel(prepared_state=1).values)
            ground = project(iq.sel(prepared_state=0).values)
            difference = np.asarray(excited - ground, dtype=float)
            # de-mean the raw-quadrature difference (matches the QM reference)
            difference = difference - float(np.mean(difference))
            # A PCA axis has no direction, so the projection sign is arbitrary;
            # orient the difference so its dominant excursion (the overlap peak)
            # is positive, which is what the triangle fit's amp > 0 gate expects.
            if angle is None and positions is None and difference.size:
                if abs(float(difference.min())) > abs(float(difference.max())):
                    difference = -difference
            reduction = "axial" if angle is None and positions is None else (
                "angle" if angle is not None else "positions")

        fit = fit_triangle(t, difference, xy_duration_s, min_snr=min_snr, max_t0_std=max_t0_std)

        return {
            "t0_s": float(fit["t0"]),
            "t0_std_s": float(fit["t0_std"]),
            "amp": float(fit["amp"]),
            "offset": float(fit["offset"]),
            "half_width_s": float(fit["half_width"]),
            "snr": float(fit["snr"]),
            "success": bool(fit["success"]),
            "reduction_method": reduction,
            "relative_time": t,
            "difference": np.asarray(difference, dtype=float),
            "best_fit": np.asarray(fit["best_fit"], dtype=float),
        }

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Persist the fitted scalars; drop the diagnostic arrays."""
        drop = {"relative_time", "difference", "best_fit"}
        return {k: v for k, v in results.items() if k not in drop}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        """Bundle the difference trace + best-fit curve over ``relative_time``;
        the fitted scalars live in ``.attrs`` so the figure needs no recompute."""
        relative_time = np.asarray(results["relative_time"], dtype=float)
        attrs = {
            "t0_s": float(results["t0_s"]),
            "t0_std_s": float(results["t0_std_s"]),
            "amp": float(results["amp"]),
            "offset": float(results["offset"]),
            "half_width_s": float(results["half_width_s"]),
            "snr": float(results["snr"]),
            "success": int(bool(results["success"])),
            "reduction_method": str(results.get("reduction_method", "state")),
        }
        return xr.Dataset(
            {
                "difference": ("relative_time", np.asarray(results["difference"], dtype=float)),
                "best_fit": ("relative_time", np.asarray(results["best_fit"], dtype=float)),
            },
            coords={"relative_time": relative_time},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        """Draw the delay-scan figure strictly from ``plot_data``; rebuild it
        only when called outside ``analyze()``."""
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return {"xyz_delay": plot_delay_scan(plot_data)}
