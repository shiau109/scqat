"""Load a SCQ.jl parametric-drive simulation into the estimator's Dataset format.

A SCQ.jl parametric-drive *frequency sweep* is a **virtual experiment**: it
synthesizes the same observable a real run measures -- the qubit excited-state
population rho_11(t) at each driving frequency -- so the **same**
``ParametricDriveDecoherenceEstimator`` analyzes it with no special-casing
(``scqat.estimators.ParametricDriveDecoherenceEstimator``). This loader is the
single bridge from a SCQ.jl HDF5 to that estimator's ``(driving_frequency,
driving_time, state)`` contract.

Canonical layout (written by the SCQ.jl study's fake-data export -- preferred,
unambiguous)::

    rho_11            : (n_freq, n_time) float  -- qubit |1> population
    driving_frequency : (n_freq,)        float  -- Hz
    driving_time      : (n_time,)        float  -- ns

Legacy SCQ.jl sweep files are also read::

    all_expect : (n_freq, n_time, 4) or (4, n_time, n_freq)
    frequency axis named  driving_frequency | omega_flux_vals | sweep_vals
    time axis named       driving_time | tlist

For ``all_expect`` the channel meaning is set by ``channel_convention``:

* ``"projectors"`` (SCQ.jl ``build_projectors`` order ``[|e,0>, |g,0>, |e,1>,
  |g,1>]``): ``rho_11 = ch0 + ch2`` (qubit excited, summed over the resonator).
* ``"moments"`` (``[rho_11, <sx>, <sy>, photon]``): ``rho_11 = ch0``.

Angular frequency axes (rad/ns, ``|f| < 1e6``) are converted to Hz via
``f / (2*pi) * 1e9``; values already at experiment scale (~1e8 Hz) pass through.
"""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np
import xarray as xr

_TWO_PI = 2.0 * np.pi
# Frequency axes below this magnitude are treated as angular (rad/ns) and
# converted to Hz; real experiment driving frequencies are ~1e8 Hz.
_ANGULAR_MAX = 1e6


def _to_hz(freq: np.ndarray) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    if freq.size and np.nanmax(np.abs(freq)) < _ANGULAR_MAX:
        return freq / _TWO_PI * 1e9
    return freq


def _orient2d(rho: np.ndarray, n_freq: int, n_time: int) -> np.ndarray:
    """Return ``rho`` oriented as ``(driving_frequency, driving_time)``."""
    rho = np.asarray(rho, dtype=float)
    if rho.shape == (n_freq, n_time):
        return rho
    if rho.shape == (n_time, n_freq):
        return rho.T
    raise ValueError(
        f"rho_11 array shape {rho.shape} matches neither "
        f"(n_freq={n_freq}, n_time={n_time}) nor its transpose."
    )


def _rho11_from_all_expect(
    A: np.ndarray, n_freq: int, n_time: int, channel_convention: str
) -> np.ndarray:
    """Reduce a 4-channel ``all_expect`` array to ``rho_11`` over (freq, time)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 3 or 4 not in A.shape:
        raise ValueError(
            f"all_expect must be 3-D with a length-4 channel axis; got {A.shape}."
        )
    # The size-4 axis is the channel axis. (n_freq/n_time == 4 is not expected
    # for real sweeps; guard against it for safety.)
    if n_freq == 4 or n_time == 4:
        raise ValueError(
            "Cannot disambiguate the channel axis when n_freq or n_time == 4."
        )
    A = np.moveaxis(A, A.shape.index(4), -1)  # -> (x, y, 4)
    if channel_convention == "projectors":
        rho = A[..., 0] + A[..., 2]  # P(|e,0>) + P(|e,1>) = qubit excited
    elif channel_convention == "moments":
        rho = A[..., 0]              # channel 0 is already rho_11
    else:
        raise ValueError(
            f"channel_convention must be 'projectors' or 'moments', got {channel_convention!r}."
        )
    return _orient2d(rho, n_freq, n_time)


def load_parametric_sim_h5(
    path: str,
    *,
    time_stride: int = 1,
    channel_convention: str = "projectors",
) -> xr.Dataset:
    """Read a SCQ.jl parametric-drive sim HDF5 into an estimator-ready Dataset.

    Parameters
    ----------
    path : str
        HDF5 file from the SCQ.jl parametric-drive study (canonical ``rho_11``
        layout, or a legacy ``all_expect`` sweep).
    time_stride : int, optional
        Keep every ``n``-th ``driving_time`` sample. Simulation time grids can
        have tens of thousands of points; downsampling keeps the per-frequency
        Hankel/decoherence fits tractable. Default 1 (no downsampling).
    channel_convention : {"projectors", "moments"}, optional
        Channel meaning when the file stores ``all_expect`` (ignored when a
        ``rho_11``/``state`` variable is present). Default ``"projectors"``
        (SCQ.jl ``build_projectors`` order).

    Returns
    -------
    xarray.Dataset
        ``state`` (= rho_11) over ``(driving_frequency [Hz], driving_time [ns])``,
        plus the file's root attributes. Feed straight to
        ``ParametricDriveDecoherenceEstimator.analyze(..., rho11_offset=0.0,
        rho11_scale=1.0)`` -- the simulated population needs no readout
        normalisation.
    """
    with h5py.File(path, "r") as f:
        keys = set(f.keys())
        attrs: dict[str, Any] = {
            k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()
        }

        time_key = "driving_time" if "driving_time" in keys else "tlist"
        if time_key not in keys:
            raise ValueError("No time axis found (expected 'driving_time' or 'tlist').")
        t = np.asarray(f[time_key][()], dtype=float)

        freq_raw = None
        for fk in ("driving_frequency", "omega_flux_vals", "sweep_vals"):
            if fk in keys:
                freq_raw = np.asarray(f[fk][()], dtype=float)
                break
        if freq_raw is None:
            raise ValueError(
                "No frequency axis found "
                "(expected 'driving_frequency', 'omega_flux_vals', or 'sweep_vals')."
            )

        if "rho_11" in keys:
            rho_src, from_channels = np.asarray(f["rho_11"][()], dtype=float), False
        elif "state" in keys:
            rho_src, from_channels = np.asarray(f["state"][()], dtype=float), False
        elif "all_expect" in keys:
            rho_src, from_channels = np.asarray(f["all_expect"][()], dtype=float), True
        else:
            raise ValueError(
                "No population data found (expected 'rho_11', 'state', or 'all_expect')."
            )

    freq = _to_hz(freq_raw)
    n_freq, n_time = freq.size, t.size
    rho = (
        _rho11_from_all_expect(rho_src, n_freq, n_time, channel_convention)
        if from_channels
        else _orient2d(rho_src, n_freq, n_time)
    )

    if time_stride > 1:
        t = t[::time_stride]
        rho = rho[:, ::time_stride]

    return xr.Dataset(
        {"state": (("driving_frequency", "driving_time"), rho)},
        coords={"driving_frequency": freq, "driving_time": t},
        attrs=attrs,
    )
