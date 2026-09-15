"""The exchange coupling J as a function of the COUPLER flux, from a fixed-time
swap map.

The fixed-duration sibling of ``pair_swap_chevron``: the same four joint
populations, drawn here over the member-flux x coupler-flux grid so the swap
**spot** and how the coupler bias moves it are visible. On top of that raw view
this estimator fits, per COUPLER column, the central peak of the transfer along
the member-flux axis and turns its height into an exchange angle — the coupler
flux is the ANGLE knob (SCQO ``TUTORIAL.md`` section 12), so the column-by-column
angle IS the ``J(Phi_c)`` calibration curve, and the polynomial through it
locates the decouple (off) point.

THE MODEL. With detuning ``delta`` and coupling ``J`` (Hz), the transfer after a
pulse of length ``t`` is ``(2J)^2/omega^2 * sin^2(pi*omega*t)`` with
``omega = sqrt(delta^2 + (2J)^2)``, so on resonance ``P_peak = sin^2(theta)``
with ``theta = 2*pi*J*t`` (a full swap is ``theta = pi/2``, i.e. ``t = 1/(4J)``).
The per-column reduction is :func:`scqat.tools.swap_lineshape.fit_swap_peak` — a
numerical routine in ``tools/``, reused freely; the MODEL claim (that this map's
columns are ``theta(coupler flux)``) is what this estimator owns.

WHY THE PEAK HEIGHT AND NOT THE WIDTH. Only the height carries J. Solving the
half-maximum condition of the model above gives ``hwhm_delta * t`` of about
0.44 / 0.43 / 0.40 / 0.32 at ``theta`` = pi/8 / pi/4 / pi/2 / 3pi/4 — the width
is set by the pulse's own Fourier limit ``~1/(2t)`` and NARROWS as the coupling
grows, so it is reported (``hwhm_qubit_flux_v``) as a diagnostic and never
converted into a coupling.

WHY THE HEIGHT IS READABLE AT ALL. The fitted trace is the single-excitation
normalization ``p_transfer / (p01 + p10)``
(:func:`~scqat.estimators._pair_swap_maps.pair_swap_normalized_transfer`): the
denominator carries the pi-pulse fidelity and the common T1 and does not depend
on the detuning, so the amplitude is pinned at 1 and the peak height is
``sin^2(theta)`` outright.

THE FOLD — READ THIS BEFORE TRUSTING A STRONG-COUPLING COLUMN. ``arcsin`` gives
the principal branch ``theta`` in [0, pi/2]. A map taken at a duration near a
full swap (which is how this map is normally run) pushes the strong-coupling end
of the coupler axis past pi/2, where ``sin^2`` folds back and J is UNDER-reported
and the ``J(Phi_c)`` curve doubles over on itself. The peak does not split when
that happens — for ``theta`` in (0, pi) the maximum stays exactly on resonance —
so nothing looks wrong. What does change is that the peak and the width fall
TOGETHER (below pi/2 a falling peak comes with a WIDENING one), and that is the
``branch_warn`` flag. Nothing is unfolded automatically: the fix is a shorter
``swap_time_ns``, so the whole coupler axis stays under a full swap.

Record-only: the SUCCESS / ``min_transfer`` verdict stays in SCQO, and nothing is
written to the device. In particular ``j_hz`` is NOT proposed to the pair — the J
measured here holds at the PULSED coupler/member flux working point, which is not
the standing idle point the catalog field names.

Dataset contract (the unified readout schema's joint form):
  vars   : ``joint_population`` — dims ``(joint_state, qubit_flux_v, coupler_flux_v)``
           in any order; ``joint_state`` labels ``"00"/"01"/"10"/"11"``
           (leftmost digit = the HIGH member)
  coords : ``joint_state`` / ``qubit_flux_v`` (V) / ``coupler_flux_v`` (V)
  kwargs : ``drive_side`` (``"high"`` | ``"low"``) selects the transfer partner;
           ``flux_side`` / ``high_name`` / ``low_name`` label the figure;
           ``swap_time_ns`` (float | None) is the duration the instrument
           ACTUALLY played — without it the angles are still reported and every
           ``j_hz`` is NaN; ``den_floor`` guards the normalization denominator;
           ``poly_degree`` / ``resonance_degree`` set the two conversion
           polynomials; ``window_factor`` / ``min_contrast`` / ``min_r_squared``
           are forwarded to the peak fit.

THE TWO CONVERSION FORMULAS (what a gate calibration actually consumes). Setting
a two-qubit gate means answering "for the angle I want, what do I set?" — and
neither answer is linear in the coupler flux. So both curves are published as
polynomial COEFFICIENTS, evaluable later without re-running anything:

* ``j2_poly_coeffs`` — the fit in ``J^2`` (not ``|J|``: a swap is even in J, so
  a linear zero crossing makes ``|J|`` a cusped V while ``J^2`` stays smooth).
  ``J = sqrt(max(poly, 0))`` and ``theta = 2*pi*J*t``. Invert it with
  :func:`scqat.tools.swap_lineshape.coupler_flux_for_theta`, which returns BOTH
  solutions — ``|J|`` has a minimum at the decouple point, so every reachable
  angle is delivered by one coupler flux on each side of it.
* ``resonance_poly_coeffs`` — the member flux that puts the pair on resonance,
  as a function of the coupler flux. The resonance point MOVES with the coupler
  bias (the coupler pulse pulls both members), so the angle solve above only
  becomes a setting once this is evaluated at the flux it returned.

``poly_window_v`` is the coupler range they were fitted over and is part of the
formula, not a footnote: outside it both are unsupported extrapolations, and on
the ``J^2`` parabola the unsupported arm climbs to a coupling nothing measured.
``j_formula`` / ``resonance_formula`` spell the same thing out as readable text.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators._pair_swap_maps import (
    pair_swap_normalized_transfer,
    pair_swap_plot_data,
    summarize_pair_swap,
)
from scqat.estimators.pair_swap_flux_map.visualization import (
    plot_coupling_curve,
    plot_pair_swap_flux_map,
    plot_transfer_fit,
)
from scqat.tools.swap_lineshape import fit_swap_peak, j_hz_from_theta

AXIS0 = "qubit_flux_v"
AXIS1 = "coupler_flux_v"

#: figure keys. ``save_figures`` prefixes the estimator name unless the key IS
#: it, so these land as ``pair_swap_flux_map.png``,
#: ``pair_swap_flux_map_coupling.png`` and ``pair_swap_flux_map_transfer_fit.png``.
FIG_MAP = "pair_swap_flux_map"
FIG_COUPLING = "coupling"
FIG_TRANSFER = "transfer_fit"

#: a fitted peak at or above this counts as having REACHED a full swap, which is
#: the only way an angle can get past pi/2 (see :func:`_branch_warnings`). Set
#: just under 1 rather than comfortably under it: the Lorentzian fit overshoots a
#: true peak by a few percent, so a lower threshold flags honest strong-coupling
#: columns that never folded, while a true full swap reads at least this high.
_FOLD_PEAK = 0.95

#: per-coupler-column curves, as plain lists in results and columns in plot_data.
_COLUMN_KEYS = (
    "theta_rad", "j_hz", "peak_transfer", "resonance_qubit_flux_v",
    "hwhm_qubit_flux_v", "fit_r_squared",
)
_COLUMN_FLAGS = ("fit_success", "branch_warn")

#: scalars build_plot_data stamps into attrs, with their NaN/0 defaults.
_SCALAR_ATTRS = {
    "swap_time_ns": float("nan"), "coupler_off_v": float("nan"),
    "j_at_off_hz": float("nan"), "j_max_hz": float("nan"),
    "j_max_coupler_flux_v": float("nan"), "theta_max_rad": float("nan"),
    "poly_prefer_v": float("nan"), "resonance_poly_r_squared": float("nan"),
    "poly_degree": 0, "resonance_poly_degree": 0,
    "off_is_interpolated": 0, "n_j_ok": 0, "n_branch_warn": 0,
    "n_poly_rows": 0,
}


def _as_column(values, size: int, dtype):
    """A per-coupler column of the right length, NaN/0-filled when absent.

    ``build_plot_data`` may be called standalone (the replot path) with a results
    dict that never held the fit, so a missing curve degrades rather than raising
    — rule 1 of "raw data must always be plottable".
    """
    if values is None:
        return np.full(size, np.nan if dtype is float else 0, dtype=dtype)
    return np.asarray(values, dtype=dtype)


def _branch_warnings(theta: np.ndarray, peak: np.ndarray, ok: np.ndarray,
                     fold_peak: float = _FOLD_PEAK) -> np.ndarray:
    """Flag columns whose angle may have folded past a full swap.

    The rule is structural, not a derivative test: ``theta`` can only exceed
    ``pi/2`` by first PASSING THROUGH a full swap, and a full swap is exactly
    ``peak = 1``. So walk outward from the weakest-coupling column in BOTH
    directions, and from the first column whose fitted peak reaches
    ``fold_peak`` onward, every column is suspect. (Walking outward is what the
    physics dictates: ``|J|`` has a minimum at the decouple point and rises on
    both sides of it, so both wings can fold independently.)

    The trigger column is flagged too: it sits AT the full swap, where
    ``dtheta/dpeak = 1/sin(2 theta)`` blows up, so its own angle is the least
    trustworthy number on the curve.

    What this CANNOT distinguish, because the data does not: a curve whose
    coupling peaks just short of a full swap and comes back down looks exactly
    like one that went past and folded. Both are flagged, which is the honest
    answer; resolving it takes a shorter pulse, not a cleverer flag.

    A derivative test on the WIDTH was tried first and rejected: at fixed
    duration the half-width moves by ~30% across a full angle sweep while the
    peak swings 0 to 1, so noise flips its sign and the flag fires at random.

    This is a WARNING, never a correction: nothing here rewrites ``theta``.
    """
    warn = np.zeros(peak.size, dtype=int)
    good = np.flatnonzero(ok & np.isfinite(theta) & np.isfinite(peak))
    if good.size < 2:
        return warn
    anchor = int(np.argmin(theta[good]))
    for direction in (1, -1):
        folded = False
        for pos in range(anchor, good.size if direction > 0 else -1, direction):
            i = int(good[pos])
            folded = folded or peak[i] >= fold_peak
            if folded:
                warn[i] = 1
    return warn


def _fit_j_vs_coupler(knob: np.ndarray, j_hz: np.ndarray, ok: np.ndarray,
                      degree: int) -> Dict[str, Any]:
    """The decouple point: the minimum of a polynomial through ``J^2(Phi_c)``.

    Fitted in ``J^2`` and not ``|J|`` on purpose. A swap transfer is even in J,
    so the measurement is ``|J|``; if J crosses zero roughly linearly in coupler
    flux then ``|J|`` is a V with a cusp — which a polynomial fits badly — while
    ``J^2`` is smooth and quadratic right there. Its minimum sits at the same
    coupler flux.

    Only successful columns are candidates. The polynomial's minimum is taken
    inside the swept window; when it falls outside, turns out to be a maximum, or
    there are too few good columns to constrain the degree, the answer degrades
    to the grid argmin with ``off_is_interpolated = 0`` — honest about a
    decouple point the sweep does not actually bracket. Never raises.
    """
    empty = {
        "coupler_off_v": float("nan"), "j_at_off_hz": float("nan"),
        "poly_degree": int(degree), "off_is_interpolated": 0,
        "poly_r_squared": float("nan"), "j2_poly_coeffs": [],
        "poly_window_v": [float("nan"), float("nan")],
        "poly_prefer_v": float("nan"), "_j_poly_curve": None,
    }
    good = np.flatnonzero(ok & np.isfinite(j_hz))
    if good.size == 0:
        return empty

    k, j2 = knob[good], j_hz[good] ** 2
    # The grid fallback is always available and is what an un-bracketed or
    # under-determined fit degrades to.
    i_min = int(np.argmin(j2))
    out = dict(empty)
    out["coupler_off_v"] = float(k[i_min])
    out["j_at_off_hz"] = float(np.sqrt(max(j2[i_min], 0.0)))
    # The window the polynomial is INTERPOLATING over, and a reference point
    # inside the measured support. Both are published because the polynomial is
    # the conversion formula a caller will invert later, and a two-branch curve
    # inverts to two fluxes: the window says where it may be trusted and the
    # reference says which branch the data actually came from.
    out["poly_window_v"] = [float(np.min(k)), float(np.max(k))]
    out["poly_prefer_v"] = float(np.median(k))

    if good.size < degree + 2:
        return out
    try:
        coeffs = np.polyfit(k, j2, int(degree))
    except Exception:  # noqa: BLE001 - a degenerate column set must not raise
        return out
    poly = np.poly1d(coeffs)
    out["j2_poly_coeffs"] = [float(c) for c in coeffs]

    ss_tot = float(np.sum((j2 - j2.mean()) ** 2))
    residual = float(np.sum((j2 - poly(k)) ** 2))
    out["poly_r_squared"] = 1.0 - residual / ss_tot if ss_tot > 0 else float("nan")
    # the curve over the FULL swept axis, for the figure
    out["_j_poly_curve"] = np.sqrt(np.clip(poly(knob), 0.0, None))

    lo, hi = float(np.min(k)), float(np.max(k))
    # Interior stationary points of the polynomial, plus nothing else: an
    # endpoint minimum is not a bracketed decouple point and keeps the grid answer.
    roots = poly.deriv().r if degree >= 2 else np.array([])
    interior = [float(r.real) for r in np.atleast_1d(roots)
                if abs(np.imag(r)) < 1e-12 and lo < r.real < hi
                and poly.deriv(2)(r.real) > 0]
    if not interior:
        return out
    best = min(interior, key=lambda v: float(poly(v)))
    out["coupler_off_v"] = float(best)
    out["j_at_off_hz"] = float(np.sqrt(max(float(poly(best)), 0.0)))
    out["off_is_interpolated"] = 1
    return out


def _poly_text(coeffs, variable: str = "V") -> str:
    """Render polynomial coefficients as readable algebra, highest power first."""
    coeffs = [float(c) for c in (coeffs or [])]
    if not coeffs:
        return ""
    order = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = order - i
        unit = "" if power == 0 else (f"*{variable}" if power == 1
                                      else f"*{variable}^{power}")
        terms.append(f"{c:+.6g}{unit}")
    return " ".join(terms).lstrip("+").strip()


def _conversion_formulas(results: Dict[str, Any], swap_time_ns) -> Dict[str, str]:
    """The two empirical conversions, as text a human can retype.

    The whole point of publishing coefficients is that the caller converts a
    WANTED angle into settings later, without re-running anything, so the
    formulas are spelled out in the metadata beside the numbers. Both are
    INTERPOLATING fits: the validity window is part of the formula, not a
    footnote.
    """
    out = {"j_formula": "", "resonance_formula": ""}
    window = results.get("poly_window_v") or []
    valid = (f"  [valid for coupler_flux_v in {window[0]:.6g}..{window[1]:.6g} V]"
             if len(window) == 2 and np.isfinite(window).all() else "")
    j2 = _poly_text(results.get("j2_poly_coeffs"))
    if j2:
        out["j_formula"] = f"J_hz(V) = sqrt(max(0, {j2})){valid}"
        if swap_time_ns:
            out["j_formula"] += (
                f";  theta_rad(V) = 2*pi*J_hz(V)*{float(swap_time_ns):g}e-9"
                f"  [a full swap is theta = pi/2]")
    resonance = _poly_text(results.get("resonance_poly_coeffs"))
    if resonance:
        out["resonance_formula"] = f"resonance_qubit_flux_v(V) = {resonance}{valid}"
    return out


def _fit_resonance_vs_coupler(knob: np.ndarray, resonance: np.ndarray,
                              peak: np.ndarray, hwhm: np.ndarray,
                              ok: np.ndarray, degree: int) -> Dict[str, Any]:
    """The second empirical curve: where to park the MEMBER flux, per coupler bias.

    Choosing a coupler flux is only half a gate setting — the pair also has to be
    brought onto resonance, and the resonance point MOVES with the coupler bias
    because the coupler pulse pulls both members. So the fitted ``x0`` per column
    is smoothed into a polynomial the caller can evaluate at whatever coupler
    flux the angle solve returned.

    WEIGHTED, because the raw curve is visibly noisiest near the decouple point:
    a peak locates its own centre to about ``hwhm / SNR``, and the height stands
    in for SNR, so the weight is ``peak / hwhm`` (numpy's ``polyfit`` wants
    ``1/sigma``). Without it the two or three near-zero-coupling columns, whose
    centres are barely determined at all, drag the whole curve.

    Never raises: too few columns, a degenerate fit or missing widths all degrade
    to an empty coefficient list.
    """
    out: Dict[str, Any] = {
        "resonance_poly_coeffs": [], "resonance_poly_degree": int(degree),
        "resonance_poly_r_squared": float("nan"), "_resonance_poly_curve": None,
    }
    usable = ok & np.isfinite(resonance) & np.isfinite(peak) & np.isfinite(hwhm)
    good = np.flatnonzero(usable & (hwhm > 0))
    if good.size < degree + 2:
        return out
    k, y = knob[good], resonance[good]
    weights = np.clip(peak[good], 0.0, None) / hwhm[good]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = None
    try:
        coeffs = np.polyfit(k, y, int(degree), w=weights)
    except Exception:  # noqa: BLE001 - a degenerate column set must not raise
        return out
    poly = np.poly1d(coeffs)
    out["resonance_poly_coeffs"] = [float(c) for c in coeffs]
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - poly(k)) ** 2))
    out["resonance_poly_r_squared"] = (1.0 - residual / ss_tot if ss_tot > 0
                                       else float("nan"))
    out["_resonance_poly_curve"] = poly(knob)
    return out


class PairSwapFluxMapEstimator(BaseEstimator):
    """Fit the swap angle per coupler-flux column and return the J(coupler) curve."""

    estimator_name = "pair_swap_flux_map"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError(
                "pair_swap_flux_map estimator requires the joint_population "
                f"variable (found data_vars: {list(dataset.data_vars)})"
            )
        for axis in ("joint_state", AXIS0, AXIS1):
            if axis not in dataset.coords:
                raise ValueError(
                    f"pair_swap_flux_map estimator requires a {axis!r} coordinate"
                )

    def extract_parameters(
        self, dataset: xr.Dataset, drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        swap_time_ns: Optional[float] = None, den_floor: float = 0.1,
        poly_degree: int = 2, resonance_degree: int = 2, **fit_knobs
    ) -> Dict[str, Any]:
        """Summarize the map, then fit one central peak per coupler column.

        ``fit_knobs`` are forwarded to
        :func:`~scqat.tools.swap_lineshape.fit_swap_peak` (``window_factor`` /
        ``min_contrast`` / ``min_r_squared``) and validated ONCE here, before the
        column loop, so a typo raises instead of being swallowed per column.
        """
        allowed = {"window_factor", "min_contrast", "min_r_squared"}
        unknown = set(fit_knobs) - allowed
        if unknown:
            raise ValueError(
                f"Unknown knob(s) {sorted(unknown)} for the swap peak fit; "
                f"valid: {sorted(allowed)}"
            )

        # The map summary (where transfer peaks, the marginal ranges) is the same
        # reduction every pair-swap map reports; only the fit below is new.
        results = summarize_pair_swap(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        projected = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        transfer = pair_swap_normalized_transfer(
            projected, drive_side, den_floor=den_floor)          # (qubit, coupler)
        qubit_flux = np.asarray(dataset[AXIS0].values, dtype=float)
        knob = np.asarray(dataset[AXIS1].values, dtype=float)

        # One peak fit per COUPLER column, along the member-flux axis. A column
        # that cannot be fitted degrades to NaN with success False -- the map
        # still draws.
        theta = np.full(knob.size, np.nan)
        peak = np.full(knob.size, np.nan)
        resonance = np.full(knob.size, np.nan)
        hwhm = np.full(knob.size, np.nan)
        r_squared = np.full(knob.size, np.nan)
        ok = np.zeros(knob.size, dtype=bool)
        best_fit = np.full(transfer.shape, np.nan)
        for i in range(knob.size):
            fit = fit_swap_peak(qubit_flux, transfer[:, i], **fit_knobs)
            theta[i] = fit["theta_rad"]
            peak[i] = fit["peak"]
            resonance[i] = fit["x0"]
            hwhm[i] = fit["hwhm"]
            r_squared[i] = fit["r_squared"]
            ok[i] = bool(fit["success"])
            best_fit[:, i] = fit["best_fit"]

        j_hz = np.asarray(j_hz_from_theta(theta, swap_time_ns), dtype=float)
        j_hz = np.where(ok, j_hz, np.nan)
        warn = _branch_warnings(theta, peak, ok)
        # A folded column reports an angle it does not have, so it may not enter
        # the polynomial or set the maximum — that would corrupt the decouple
        # point with the very rows the flag exists to distrust. `n_j_ok` and
        # `n_branch_warn` show the split, so nothing is dropped silently.
        quotable = ok & (warn == 0)
        poly = _fit_j_vs_coupler(knob, j_hz, quotable, poly_degree)
        # The resonance line is fitted over every SUCCESSFUL column, not just the
        # quotable ones: a folded column still locates its own resonance
        # perfectly well — folding is an ambiguity in the HEIGHT, not the centre.
        resonance_poly = _fit_resonance_vs_coupler(
            knob, resonance, peak, hwhm, ok, resonance_degree)

        quoted = np.where(quotable, j_hz, np.nan)
        i_max = int(np.nanargmax(quoted)) if np.isfinite(quoted).any() else None
        results.update({
            "swap_time_ns": (float(swap_time_ns) if swap_time_ns is not None
                             else float("nan")),
            "den_floor": float(den_floor),
            "n_j_ok": int(ok.sum()),
            "n_branch_warn": int(warn.sum()),
            "n_poly_rows": int(quotable.sum()),
            "theta_max_rad": (float(np.nanmax(theta[quotable])) if quotable.any()
                              else float("nan")),
            "j_max_hz": float(j_hz[i_max]) if i_max is not None else float("nan"),
            "j_max_coupler_flux_v": (float(knob[i_max]) if i_max is not None
                                     else float("nan")),
            **{k: v for k, v in poly.items() if not k.startswith("_")},
            **{k: v for k, v in resonance_poly.items() if not k.startswith("_")},
        })
        results.update(_conversion_formulas(results, swap_time_ns))
        # The curves themselves, as plain lists so the metadata JSON stays portable.
        for key, values in (("theta_rad", theta), ("j_hz", j_hz),
                            ("peak_transfer", peak),
                            ("resonance_qubit_flux_v", resonance),
                            ("hwhm_qubit_flux_v", hwhm),
                            ("fit_r_squared", r_squared)):
            results[key] = [float(v) for v in values]
        results["fit_success"] = [int(v) for v in ok]
        results["branch_warn"] = [int(v) for v in warn]
        # underscore-prefixed: bulky arrays kept for build_plot_data only
        results["_transfer_norm"] = transfer
        results["_transfer_fit"] = best_fit
        results["_j_poly_curve"] = poly["_j_poly_curve"]
        results["_resonance_poly_curve"] = resonance_poly["_resonance_poly_curve"]
        return results

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Drop the bulky per-column maps; the J curve is small and stays."""
        return {k: v for k, v in results.items() if not k.startswith("_")}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], drive_side: str = "low",
        flux_side: Optional[str] = None,
        high_name: Optional[str] = None, low_name: Optional[str] = None,
        den_floor: float = 0.1, **kwargs
    ) -> Optional[xr.Dataset]:
        out = pair_swap_plot_data(
            dataset, AXIS0, AXIS1, drive_side, flux_side, high_name, low_name
        )
        dims = (AXIS0, AXIS1)
        transfer = results.get("_transfer_norm")
        if transfer is None:
            transfer = pair_swap_normalized_transfer(
                out, drive_side, den_floor=den_floor)
        out["transfer_norm"] = (dims, np.asarray(transfer, dtype=float))
        best_fit = results.get("_transfer_fit")
        if best_fit is None:
            best_fit = np.full(np.shape(transfer), np.nan)
        # The fit degrades to NaN, never to absent, so the raw maps still draw
        # when every column failed.
        out["transfer_fit"] = (dims, np.asarray(best_fit, dtype=float))

        n_knob = int(out.sizes[AXIS1])
        for key in _COLUMN_KEYS:
            out[key] = ((AXIS1,), _as_column(results.get(key), n_knob, float))
        for key in _COLUMN_FLAGS:
            out[key] = ((AXIS1,), _as_column(results.get(key), n_knob, int))
        out["j_poly_curve"] = ((AXIS1,),
                               _as_column(results.get("_j_poly_curve"), n_knob, float))
        out["resonance_poly_curve"] = (
            (AXIS1,), _as_column(results.get("_resonance_poly_curve"), n_knob, float))
        out.attrs.update({
            key: type(default)(results.get(key, default))
            for key, default in _SCALAR_ATTRS.items()
        })
        # The two conversion formulas travel WITH the plot data, so a saved
        # plotdata.nc answers "what flux for this angle?" on its own. netCDF has
        # no empty-list attribute, so an unfitted polynomial is stored as [nan].
        for key in ("j2_poly_coeffs", "resonance_poly_coeffs", "poly_window_v"):
            values = [float(v) for v in (results.get(key) or [])]
            out.attrs[key] = values or [float("nan")]
        return out

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(
                dataset, results,
                drive_side=kwargs.get("drive_side", "low"),
                flux_side=kwargs.get("flux_side"),
                high_name=kwargs.get("high_name"),
                low_name=kwargs.get("low_name"),
                den_floor=kwargs.get("den_floor", 0.1),
            )
        return render_figures(
            {
                FIG_MAP: lambda: plot_pair_swap_flux_map(plot_data),
                FIG_COUPLING: lambda: plot_coupling_curve(plot_data),
                FIG_TRANSFER: lambda: plot_transfer_fit(plot_data),
            },
            label=self.estimator_name,
        )
