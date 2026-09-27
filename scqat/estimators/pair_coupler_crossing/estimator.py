"""Where a tunable coupler crosses its two neighbours - and the coupler arch behind it.

Dataset contract (one qubit pair; the acquisition layer splits targets off):

    coords   ``coupler_flux_v`` (V)  - the coupler flux-pulse amplitude, in ANY order
                                       (the probe's realized order; canonicalized on
                                       entry, so every artifact is ascending)
             ``joint_state``        - ``"00"/"01"/"10"/"11"``, leftmost digit = the
                                       HIGH member
    vars     ``joint_population`` (joint_state, coupler_flux_v)

The estimator is blind to the flux FRAME: every position is reported on the
``coupler_flux_v`` axis as given, and ``bracket_at`` (default 0.0 - the idle point on
an idle-relative axis) is where the crossings are paired around. The acquisition
layer re-references.

The signal
----------
Each measured member got an x180 at its own idle frequency while the coupler sat at
the swept bias. Where the coupler crosses that member, the two hybridize and the
x180 misses: the member's marginal (``P_high = P10 + P11``, ``P_low = P01 + P11``)
DIPS. Normalized by its baseline (a high percentile of the trace), a dip is a run of
points below ``dip_threshold``; its center is the (1 - s)-weighted centroid.

The crossings nearest ``bracket_at`` on either side pair up into a symmetry point
``c_k`` and a half-separation ``d_k`` per member. Both members' symmetry points are
the same point of the coupler arch - where the coupler is at an extremum:

* high member INNER (smaller ``d``) -> the coupler sits ABOVE both at idle and the
  symmetry point is its APEX (sweet spot);
* low member inner -> the coupler sits BELOW both and it is the ANTI-apex.

Model
-----
The coupler arch ``f_c(x) = (F + Ec) sqrt(|cos(pi (x - x0) / P)|) - Ec`` crosses a
member at ``f_k`` a distance ``d_k`` from the apex with
``cos(pi d_k / P) = ((f_k + Ec) / (F + Ec))^2``, or a distance ``e_k`` from the
anti-apex with ``sin(pi e_k / P)`` equal to the same ratio. Two members give two
equations for ``(F, P)``: the ratio of the two distances depends on ``F`` alone and
is monotonic in it, so ``F`` is a 1-D root and ``P`` follows. Errors propagate
through a numerical Jacobian over ``(center, d_high, d_low)``.

The result does not depend on the order of the flux axis.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import brentq

from scqat.core.base_estimator import BaseEstimator
from scqat.core.figures import render_figures
from scqat.estimators.pair_coupler_crossing.visualization import (
    plot_arch,
    plot_crossings,
)
from scqat.tools.sweep_order import ascending

AXIS = "coupler_flux_v"
ROLES = ("high", "low")
MEASURE = ("both", "high", "low")
COUPLER_SIDES = ("auto", "above", "below")

#: joint-state labels whose digit for the role is 1 (label order: high, low)
_EXCITED = {"high": ("10", "11"), "low": ("01", "11")}

#: the arch solve looks for f_c_max up to here; a ratio that needs more is unsolved
F_MAX_HZ = 50e9
#: a dip needs at least this many points below the threshold
MIN_DIP_POINTS = 2
#: two runs separated by at most this many points above the threshold are one dip
MERGE_GAP_POINTS = 2

_NAN = float("nan")


# ---------------------------------------------------------------- dip finding

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, stop) index pairs of the True runs of ``mask``."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _merge(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in runs:
        if out and lo - out[-1][1] - 1 <= gap:
            out[-1] = (out[-1][0], hi)
        else:
            out.append((lo, hi))
    return out


def find_dips(x: np.ndarray, s: np.ndarray, *, threshold: float, sigma_s: float,
              step: float) -> list[Dict[str, Any]]:
    """Dips of a normalized trace ``s`` (baseline ~1) on an ASCENDING axis ``x``.

    A dip is a run of ``s < threshold`` (runs ``MERGE_GAP_POINTS`` apart merge),
    at least ``MIN_DIP_POINTS`` long. Its center is the (1 - s)-weighted centroid of
    the run; the stderr combines the noise of the weights (``sigma_s`` per point)
    with a sampling floor of ``step / sqrt(12)``. A run touching either end of the
    axis is kept but marked ``touches_edge`` - its centroid is truncated.
    """
    below = np.isfinite(s) & (s < threshold)
    dips = []
    for lo, hi in _merge(_runs(below), MERGE_GAP_POINTS):
        idx = np.arange(lo, hi + 1)
        idx = idx[np.isfinite(s[idx])]
        if np.count_nonzero(s[idx] < threshold) < MIN_DIP_POINTS:
            continue
        w = np.clip(1.0 - s[idx], 0.0, None)
        total = float(w.sum())
        if total <= 0:
            continue
        xc = float(np.sum(x[idx] * w) / total)
        noise = sigma_s * np.sqrt(float(np.sum((x[idx] - xc) ** 2))) / total
        dips.append({
            "center_v": xc,
            "center_stderr_v": float(np.sqrt(noise ** 2 + step ** 2 / 12.0)),
            "lower_edge_v": float(x[lo]),
            "upper_edge_v": float(x[hi]),
            "width_v": float(x[hi] - x[lo]),
            "min_s": float(np.nanmin(s[idx])),
            "touches_edge": int(lo == 0 or hi == len(x) - 1),
        })
    return dips


def bracket(dips: list[Dict[str, Any]], at: float):
    """The nearest usable dip on each side of ``at``: (lower, upper) or None.

    Edge-truncated dips are not usable, and a dip whose run CONTAINS ``at`` (the
    member is already on a crossing at the bracketing point) means there is no
    clean bracket at all.
    """
    if any(d["lower_edge_v"] <= at <= d["upper_edge_v"] for d in dips):
        return None
    usable = [d for d in dips if not d["touches_edge"]]
    left = [d for d in usable if d["center_v"] < at]
    right = [d for d in usable if d["center_v"] > at]
    if not left or not right:
        return None
    return max(left, key=lambda d: d["center_v"]), min(right, key=lambda d: d["center_v"])


# ---------------------------------------------------------------- the arch

def arch_frequency(x, *, f_max_hz: float, apex_v: float, period_v: float,
                   ec_hz: float) -> np.ndarray:
    """``f_c(x) = (F + Ec) sqrt(|cos(pi (x - x0) / P)|) - Ec`` (symmetric SQUID)."""
    phase = np.pi * (np.asarray(x, dtype=float) - apex_v) / period_v
    return (f_max_hz + ec_hz) * np.sqrt(np.abs(np.cos(phase))) - ec_hz


def _ratio(f_hz: float, f_max_hz: float, ec_hz: float) -> float:
    return ((f_hz + ec_hz) / (f_max_hz + ec_hz)) ** 2


def solve_arch(kind: str, d_high: float, d_low: float, f_high_hz: float,
               f_low_hz: float, ec_hz: float) -> Optional[tuple[float, float]]:
    """``(f_max_hz, period_v)`` from the two half-separations, or None.

    ``kind='apex'``: the distances are from the apex, ``cos(pi d/P) = ratio``;
    ``kind='anti_apex'``: from the anti-apex, ``sin(pi d/P) = ratio``. The ratio of
    the two distances fixes ``F`` alone (monotonic), so ``F`` is one bracketed root.
    """
    if not (np.isfinite(d_high) and np.isfinite(d_low) and d_high > 0 and d_low > 0
            and f_high_hz > f_low_hz > -ec_hz):
        return None
    inv = np.arccos if kind == "apex" else np.arcsin
    target = d_high / d_low if kind == "apex" else d_low / d_high

    def g(f_max: float) -> float:
        ih = inv(min(_ratio(f_high_hz, f_max, ec_hz), 1.0))
        il = inv(min(_ratio(f_low_hz, f_max, ec_hz), 1.0))
        return (ih / il if kind == "apex" else il / ih) - target

    lo, hi = f_high_hz * (1 + 1e-12) + 1.0, F_MAX_HZ
    try:
        glo, ghi = g(lo), g(hi)
    except (ValueError, ZeroDivisionError):
        return None
    if not (np.isfinite(glo) and np.isfinite(ghi)) or glo * ghi > 0:
        return None
    f_max = float(brentq(g, lo, hi, xtol=1.0, rtol=1e-12))
    period = float(np.pi * d_high / inv(_ratio(f_high_hz, f_max, ec_hz)))
    return f_max, period


def _arch_point(kind: str, center: float, d_high: float, d_low: float,
                f_high_hz: float, f_low_hz: float, ec_hz: float, at: float):
    """(apex, F, P, f_c(at)) for one set of inputs, or None."""
    sol = solve_arch(kind, d_high, d_low, f_high_hz, f_low_hz, ec_hz)
    if sol is None:
        return None
    f_max, period = sol
    if kind == "apex":
        apex = center
    else:  # the apex image nearest the bracketing point
        apex = center - period / 2 if center > at else center + period / 2
    f_at = float(arch_frequency(at, f_max_hz=f_max, apex_v=apex, period_v=period,
                                ec_hz=ec_hz))
    return np.array([apex, f_max, period, f_at])


def arch_with_errors(kind: str, center: float, center_err: float, d_high: float,
                     d_high_err: float, d_low: float, d_low_err: float,
                     f_high_hz: float, f_low_hz: float, ec_hz: float, at: float):
    """(values, stderrs) of (apex, F, P, f_c(at)) - a central-difference Jacobian
    over the three independent inputs (center, d_high, d_low)."""
    args = (f_high_hz, f_low_hz, ec_hz, at)
    base = _arch_point(kind, center, d_high, d_low, *args)
    if base is None:
        return None
    inputs = np.array([center, d_high, d_low])
    sigmas = np.array([center_err, d_high_err, d_low_err])
    var = np.zeros(4)
    for i in range(3):
        h = max(abs(inputs[i]) * 1e-6, 1e-9)
        up, dn = inputs.copy(), inputs.copy()
        up[i] += h
        dn[i] -= h
        p_up = _arch_point(kind, *up, *args)
        p_dn = _arch_point(kind, *dn, *args)
        if p_up is None or p_dn is None:
            return base, np.full(4, _NAN)
        var += ((p_up - p_dn) / (2 * h) * sigmas[i]) ** 2
    return base, np.sqrt(var)


# ---------------------------------------------------------------- the estimator

class PairCouplerCrossingEstimator(BaseEstimator):
    """Coupler crossings with both neighbours -> apex, period and f_c_max."""

    estimator_name = "pair_coupler_crossing"

    def _check_data(self, dataset: xr.Dataset) -> None:
        if "joint_population" not in dataset.data_vars:
            raise ValueError("pair_coupler_crossing requires a 'joint_population' variable.")
        for coord in (AXIS, "joint_state"):
            if coord not in dataset.coords:
                raise ValueError(f"pair_coupler_crossing requires a '{coord}' coordinate.")
        missing = {"00", "01", "10", "11"} - {str(v) for v in dataset["joint_state"].values}
        if missing:
            raise ValueError(f"pair_coupler_crossing: joint_state lacks {sorted(missing)}")

    def extract_parameters(self, dataset: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Dips per measured member -> crossings -> symmetry point -> arch.

        Kwargs - flat and fully owned; unknown names raise:
            measure (str): ``'both'`` (default) / ``'high'`` / ``'low'`` - which
                members got the x180 (only those are analyzed).
            coupler_side (str): ``'auto'`` (default) / ``'above'`` / ``'below'`` -
                where the coupler sits relative to the members at the bracketing
                point. ``auto`` decides from which member's crossings are inner
                (both measured) or leaves the center kind unknown (one measured).
            f_high_hz, f_low_hz (float, optional): the members' frequencies the
                x180s were played at; the arch solve needs both.
            ec_hz (float): the coupler charging energy for the arch (default 0.2 GHz).
            bracket_at (float): the axis value the crossings are paired around
                (default 0.0 - the idle point on an idle-relative axis).
            dip_threshold (float): normalized level a dip must go below (default 0.5).
            baseline_percentile (float): the percentile taken as a member's
                baseline (default 90).
            center_tolerance_v (float, optional): largest allowed difference of the
                two members' symmetry points (default max(5 mV, 2 grid steps)).
            high_name, low_name (str): member names for the figure labels.
        """
        measure = str(kwargs.pop("measure", "both"))
        side = str(kwargs.pop("coupler_side", "auto"))
        f_high = kwargs.pop("f_high_hz", None)
        f_low = kwargs.pop("f_low_hz", None)
        ec_hz = float(kwargs.pop("ec_hz", 0.2e9))
        at = float(kwargs.pop("bracket_at", 0.0))
        threshold = float(kwargs.pop("dip_threshold", 0.5))
        percentile = float(kwargs.pop("baseline_percentile", 90.0))
        tolerance = kwargs.pop("center_tolerance_v", None)
        names = {"high": str(kwargs.pop("high_name", "high")),
                 "low": str(kwargs.pop("low_name", "low"))}
        if kwargs:
            raise ValueError(f"pair_coupler_crossing: unknown kwargs {sorted(kwargs)}")
        if measure not in MEASURE:
            raise ValueError(f"measure must be one of {MEASURE}, got {measure!r}")
        if side not in COUPLER_SIDES:
            raise ValueError(f"coupler_side must be one of {COUPLER_SIDES}, got {side!r}")
        if not 0 < threshold < 1:
            raise ValueError(f"dip_threshold must be in (0, 1), got {threshold}")
        # the realized sweep order is provenance, never input (tools.sweep_order)
        dataset = ascending(dataset, AXIS)

        x = np.asarray(dataset[AXIS].values, dtype=float)
        joint = dataset["joint_population"].transpose("joint_state", AXIS)
        labels = [str(v) for v in joint["joint_state"].values]
        jp = np.asarray(joint.values, dtype=float)
        step = float(np.median(np.diff(x))) if x.size > 1 else _NAN
        tol = (max(5e-3, 2 * step) if tolerance is None else float(tolerance))
        measured = ROLES if measure == "both" else (measure,)

        res: Dict[str, Any] = {
            "measure": measure, "coupler_side": side, "ec_hz": ec_hz,
            "f_high_hz": _NAN if f_high is None else float(f_high),
            "f_low_hz": _NAN if f_low is None else float(f_low),
            "bracket_at": at, "dip_threshold": threshold,
            "baseline_percentile": percentile, "center_tolerance_v": tol,
            "high_name": names["high"], "low_name": names["low"],
            "n_points": int(x.size), "step_v": step,
            "coupler_flux_v": x.tolist(),
            "joint_state": labels, "joint_population": jp,
        }
        per_role: Dict[str, Dict[str, Any]] = {}
        for role in ROLES:
            p = sum(jp[labels.index(lbl)] for lbl in _EXCITED[role])
            finite = p[np.isfinite(p)]
            baseline = float(np.percentile(finite, percentile)) if finite.size else _NAN
            s = p / baseline if np.isfinite(baseline) and baseline > 0 else np.full_like(p, _NAN)
            info: Dict[str, Any] = {"marginal": p, "normalized": s, "baseline": baseline,
                                    "dips": [], "in_dip": np.zeros(x.size, dtype=int)}
            res[f"baseline_{role}"] = baseline
            res[f"measured_{role}"] = int(role in measured)
            for key in ("lower", "upper"):
                for q in ("v", "stderr_v", "width_v", "min_s"):
                    res[f"crossing_{role}_{key}_{q}"] = _NAN
            res.update({f"symmetry_{role}_v": _NAN, f"symmetry_{role}_stderr_v": _NAN,
                        f"half_separation_{role}_v": _NAN,
                        f"half_separation_{role}_stderr_v": _NAN,
                        f"crossings_not_bracketed_{role}": int(role in measured)})
            if role in measured and np.isfinite(baseline) and baseline > 0:
                ok = np.isfinite(s)
                clear = ok & (s >= threshold)
                sigma = (1.4826 * float(np.median(np.abs(s[clear] - np.median(s[clear]))))
                         if np.count_nonzero(clear) >= 3 else 0.0)
                dips = find_dips(x, s, threshold=threshold, sigma_s=sigma, step=step)
                info["dips"] = dips
                for d in dips:
                    info["in_dip"][(x >= d["lower_edge_v"]) & (x <= d["upper_edge_v"])] = 1
                pair = bracket(dips, at)
                if pair is not None:
                    lower, upper = pair
                    for key, d in (("lower", lower), ("upper", upper)):
                        res[f"crossing_{role}_{key}_v"] = d["center_v"]
                        res[f"crossing_{role}_{key}_stderr_v"] = d["center_stderr_v"]
                        res[f"crossing_{role}_{key}_width_v"] = d["width_v"]
                        res[f"crossing_{role}_{key}_min_s"] = d["min_s"]
                    err = 0.5 * float(np.hypot(lower["center_stderr_v"], upper["center_stderr_v"]))
                    res.update({
                        f"symmetry_{role}_v": 0.5 * (lower["center_v"] + upper["center_v"]),
                        f"symmetry_{role}_stderr_v": err,
                        f"half_separation_{role}_v": 0.5 * (upper["center_v"] - lower["center_v"]),
                        f"half_separation_{role}_stderr_v": err,
                        f"crossings_not_bracketed_{role}": 0,
                    })
            res[f"dip_centers_{role}_v"] = [d["center_v"] for d in info["dips"]]
            res[f"dip_touches_edge_{role}"] = [d["touches_edge"] for d in info["dips"]]
            per_role[role] = info
        res["marginal"] = np.stack([per_role[r]["marginal"] for r in ROLES])
        res["normalized"] = np.stack([per_role[r]["normalized"] for r in ROLES])
        res["in_dip"] = np.stack([per_role[r]["in_dip"] for r in ROLES])

        bracketed = [r for r in measured if not res[f"crossings_not_bracketed_{r}"]]
        res.update(center_from_idle_v=_NAN, center_from_idle_stderr_v=_NAN,
                   center_kind="unknown", center_mismatch=0, side_conflict=0,
                   arch_unsolved=1, apex_from_idle_v=_NAN, apex_from_idle_stderr_v=_NAN,
                   f_c_max_hz=_NAN, f_c_max_stderr_hz=_NAN, period_v=_NAN,
                   period_stderr_v=_NAN, f_c_at_idle_hz=_NAN, f_c_at_idle_stderr_hz=_NAN,
                   apex_identified=0)
        if bracketed:
            c = np.array([res[f"symmetry_{r}_v"] for r in bracketed])
            e = np.array([res[f"symmetry_{r}_stderr_v"] for r in bracketed])
            w = 1.0 / np.where(e > 0, e, np.nanmin(e[e > 0]) if np.any(e > 0) else 1.0) ** 2
            res["center_from_idle_v"] = float(np.sum(w * c) / np.sum(w))
            res["center_from_idle_stderr_v"] = float(1.0 / np.sqrt(np.sum(w)))

        expected = {"above": "apex", "below": "anti_apex"}.get(side)
        kind = "unknown"
        if len(bracketed) == 2:
            dh, dl = res["half_separation_high_v"], res["half_separation_low_v"]
            if abs(res["symmetry_high_v"] - res["symmetry_low_v"]) > tol:
                res["center_mismatch"] = 1
            elif dh != dl:
                kind = "apex" if dh < dl else "anti_apex"
                if expected is not None and expected != kind:
                    res["side_conflict"] = 1
        elif len(bracketed) == 1 and expected is not None:
            kind = expected
        if not res["center_mismatch"]:
            res["center_kind"] = kind

        if (kind in ("apex", "anti_apex") and len(bracketed) == 2
                and not res["center_mismatch"] and not res["side_conflict"]
                and f_high is not None and f_low is not None):
            solved = arch_with_errors(
                kind, res["center_from_idle_v"], res["center_from_idle_stderr_v"],
                res["half_separation_high_v"], res["half_separation_high_stderr_v"],
                res["half_separation_low_v"], res["half_separation_low_stderr_v"],
                float(f_high), float(f_low), ec_hz, at)
            if solved is not None:
                (apex, f_max, period, f_at), (s_apex, s_f, s_p, s_at) = solved
                res.update(arch_unsolved=0, apex_from_idle_v=float(apex),
                           apex_from_idle_stderr_v=float(s_apex), f_c_max_hz=float(f_max),
                           f_c_max_stderr_hz=float(s_f), period_v=float(period),
                           period_stderr_v=float(s_p), f_c_at_idle_hz=float(f_at),
                           f_c_at_idle_stderr_hz=float(s_at))
        if (res["center_kind"] == "apex" and not res["side_conflict"]
                and np.isfinite(res["center_from_idle_v"])):
            res["apex_identified"] = 1
            if res["arch_unsolved"]:  # the center IS the apex, with or without the arch
                res["apex_from_idle_v"] = res["center_from_idle_v"]
                res["apex_from_idle_stderr_v"] = res["center_from_idle_stderr_v"]
        elif res["center_kind"] == "anti_apex" and not res["arch_unsolved"]:
            res["apex_identified"] = 1

        res["success"] = bool(
            all(not res[f"crossings_not_bracketed_{r}"] for r in measured)
            and not res["center_mismatch"] and not res["side_conflict"])
        return res

    def extract_metadata(self, results: Dict[str, Any]) -> Dict[str, Any]:
        bulky = ("joint_population", "marginal", "normalized", "in_dip")
        return {k: v for k, v in results.items() if k not in bulky}

    def build_plot_data(
        self, dataset: xr.Dataset, results: Dict[str, Any], **kwargs
    ) -> Optional[xr.Dataset]:
        x = np.asarray(results["coupler_flux_v"], dtype=float)
        solved = not results["arch_unsolved"]
        if solved and x.size:
            fine = np.linspace(float(np.min(x)), float(np.max(x)), 401)
            arch = arch_frequency(fine, f_max_hz=results["f_c_max_hz"],
                                  apex_v=results["apex_from_idle_v"],
                                  period_v=results["period_v"], ec_hz=results["ec_hz"])
        else:
            fine = (np.linspace(float(np.min(x)), float(np.max(x)), 2) if x.size
                    else np.zeros(2))
            arch = np.full(fine.size, _NAN)
        scalar_keys = [k for k, v in results.items()
                       if isinstance(v, (int, float, str, bool, np.floating, np.integer))
                       and k != "success"]
        attrs = {k: results[k] for k in scalar_keys}
        attrs["success"] = int(bool(results["success"]))
        for role in ROLES:
            attrs[f"dip_centers_{role}_v"] = np.asarray(results[f"dip_centers_{role}_v"],
                                                        dtype=float)
        return xr.Dataset(
            {
                "joint_population": (("joint_state", AXIS),
                                     np.asarray(results["joint_population"], dtype=float)),
                "marginal": (("member", AXIS), np.asarray(results["marginal"], dtype=float)),
                "normalized": (("member", AXIS),
                               np.asarray(results["normalized"], dtype=float)),
                "in_dip": (("member", AXIS), np.asarray(results["in_dip"], dtype=np.int32)),
                "arch_frequency_hz": ("flux_fine", np.asarray(arch, dtype=float)),
            },
            coords={AXIS: x, "joint_state": list(results["joint_state"]),
                    "member": list(ROLES), "flux_fine": fine},
            attrs=attrs,
        )

    def generate_figures(
        self,
        dataset: xr.Dataset,
        results: Dict[str, Any],
        plot_data: Optional[xr.Dataset] = None,
        **kwargs,
    ) -> Dict[str, plt.Figure]:
        if plot_data is None:
            plot_data = self.build_plot_data(dataset, results)
        return render_figures({
            "crossings": lambda: plot_crossings(plot_data),
            "arch": lambda: plot_arch(plot_data),
        }, label=self.estimator_name)
