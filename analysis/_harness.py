"""
analysis/_harness.py — reusable offline harness for working with saved estimator data.

Why
---
``BaseEstimator.analyze(ds) -> (results, figs)`` is uniform across every estimator,
so ONE set of helpers covers the offline use cases:

* **try a new approach** — compare a custom method against the estimator (``compare``);
* **test parameters**    — run the estimator across a grid of kwargs (``estimator_method`` + ``compare``);
* **replot**             — regenerate the estimator's own figures from saved raw data
                           (e.g. an LCHQM run that stored ds_raw but skipped plotting) (``replot``).

The only per-experiment differences are how a saved ``ds_raw.h5`` is shaped into what the
estimator expects (``prep``) and how a ``results`` dict maps to normalized plot fields
(``adapt``) — both supplied by the thin per-experiment ``# %%`` script. Keep this module
logic-only so it stays git-friendly.

Normalized "method"
-------------------
A *method* is any callable ``sq (xr.Dataset) -> dict`` returning::

    {"label": str, "ok": bool,
     "detuning": float,  # Hz
     "fwhm": float,      # Hz
     "x", "y", "fit_x", "fit_y": 1-D arrays}   # x in Hz; omit the rest when ok is False

``adapt`` (per experiment)
--------------------------
``adapt(results: dict, sq: xr.Dataset) -> dict | None`` extracts the normalized fields
(detuning/fwhm/x/y/fit_x/fit_y) from an estimator ``results`` dict, or ``None`` if the
fit found nothing. ``estimator_method`` uses it to turn any estimator+kwargs into a method.
"""
from __future__ import annotations

import glob
import os
from typing import Callable

import xarray as xr
import matplotlib.pyplot as plt

from scqat.parsers import load_xarray_h5, repetition_data

__all__ = ["load", "slices", "compare", "estimator_method", "replot"]

Method = Callable[[xr.Dataset], dict]
Prep = Callable[[xr.Dataset], xr.Dataset]
Adapt = Callable[[dict, xr.Dataset], "dict | None"]


def load(path: str) -> xr.Dataset:
    """Load a saved ``ds_raw.h5`` (or any xarray HDF5) into memory."""
    return load_xarray_h5(path)


def slices(ds: xr.Dataset, prep: Prep | None = None, dim: str = "qubit"):
    """Split ``ds`` along ``dim`` into per-unit slices, applying ``prep`` (e.g. build
    IQdata / assign full_freq) to each. Returns ``[(name, sq), ...]``."""
    out = []
    for sq in repetition_data(ds, repetition_dim=dim):
        name = sq[dim].values.item()
        out.append((name, prep(sq) if prep is not None else sq))
    return out


def estimator_method(estimator, adapt: Adapt, label: str | None = None, **analyze_kwargs) -> Method:
    """Turn ``estimator`` + ``analyze_kwargs`` into a comparable *method* (``sq -> dict``).

    Use it for the baseline, and for parameter testing pass one per kwarg set, e.g.::

        [estimator_method(est, adapt, label=f"prom={p}", prominence=p) for p in (...)]
    """
    name = label or getattr(estimator, "estimator_name", type(estimator).__name__)

    def method(sq: xr.Dataset) -> dict:
        results = estimator.analyze(sq, output_dir=None, skip_figures=True, **analyze_kwargs)[0]
        fields = adapt(results, sq)
        if fields is None:
            return {"label": name, "ok": False}
        return {"label": name, "ok": True, **fields}

    method.__name__ = name
    return method


def compare(slices_, methods, out_png=None, freq_scale: float = 1e6, freq_unit: str = "MHz"):
    """Run each method on each ``(name, sq)`` slice; print a detuning/FWHM table and
    draw a rows=slices x cols=methods grid (signal + fit + centre line). Saves a PNG if
    ``out_png`` is given. Returns ``(rows, fig)``.

    ``methods`` is a list of callables ``sq -> normalized dict`` (see module docstring).
    """
    n_rows, n_cols = len(slices_), max(len(methods), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.2 * n_rows),
                             squeeze=False, dpi=110)
    head = f"{'unit':6} {'method':30} {'detuning/' + freq_unit:>16} {'FWHM/' + freq_unit:>14}"
    print(head)
    print("-" * len(head))
    rows = []
    for r, (name, sq) in enumerate(slices_):
        for c, method in enumerate(methods):
            out = method(sq)
            label = out.get("label") or getattr(method, "__name__", f"method{c}")
            ax = axes[r][c]
            ax.set_xlabel(f"detuning ({freq_unit})")
            if not out.get("ok", False):
                ax.set_title(f"{name} — {label}: NO FIT")
                print(f"{name:6} {label:30} {'--':>16} {'--':>14}")
                rows.append({"unit": name, "method": label, "detuning": None, "fwhm": None})
                continue
            ax.plot(out["x"] / freq_scale, out["y"], ".", ms=3, label="signal")
            ax.plot(out["fit_x"] / freq_scale, out["fit_y"], "-", lw=1.6, color="C1", label="fit")
            ax.axvline(out["detuning"] / freq_scale, color="C3", ls=":", lw=1.0)
            ax.set_title(f"{name} — {label}")
            ax.legend(fontsize=8)
            print(f"{name:6} {label:30} {out['detuning'] / freq_scale:>16.3f} {out['fwhm'] / freq_scale:>14.3f}")
            rows.append({"unit": name, "method": label,
                         "detuning": out["detuning"], "fwhm": out["fwhm"]})
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"\nsaved comparison -> {out_png}")
    return rows, fig


def _plotdata_name(path: str) -> str:
    """Unit name from a saved plot-data filename: ``plotdata_q1.h5`` -> ``q1``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[len("plotdata_"):] if stem.startswith("plotdata_") else stem


def _iter_plotdata(spec):
    """Yield ``(name, plot_data)`` from a ``{name: path}`` mapping, a directory of
    ``plotdata_*.h5`` (e.g. an LCHQM run folder saved with ``save_plot_data=True``), or a
    single saved file path."""
    if isinstance(spec, dict):
        items = list(spec.items())
    elif os.path.isdir(spec):
        items = [(_plotdata_name(p), p) for p in sorted(glob.glob(os.path.join(spec, "plotdata_*.h5")))]
    else:
        items = [(_plotdata_name(spec), spec)]
    for name, path in items:
        yield name, load_xarray_h5(path)


def _save_figs(figs, name, out_dir):
    if not out_dir:
        return
    for fig_name, fig in figs.items():
        path = os.path.join(out_dir, f"{name}__{fig_name}.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"   saved {path}")


def replot(estimator, slices_=None, out_dir=None, from_plotdata=None,
           print_metadata=True, **analyze_kwargs):
    """Regenerate the estimator's OWN figures per unit and (optionally) save them as
    ``<out_dir>/<name>__<figname>.png``. Returns ``{name: {figname: Figure}}``.

    Two modes — pass exactly one of:

    * ``slices_`` (the ``(name, sq)`` list): **re-fit** the saved raw data — the figures an
      LCHQM run would have produced had ``plot`` not been skipped. Also prints the
      persisted metadata so you can eyeball the extracted parameters.
    * ``from_plotdata``: **no re-fit** — a saved ``plotdata_<unit>.h5`` path, a directory of
      them (e.g. an LCHQM run folder saved with ``save_plot_data=True``), or a
      ``{name: path}`` mapping. Each is loaded and drawn via
      ``estimator.generate_figures(None, None, plot_data=...)`` (plot-data-contract
      estimators only — they draw using only ``plot_data``).

    Set ``print_metadata=False`` to suppress the per-unit metadata line in the re-fit
    mode — handy when an estimator's metadata is bulky (e.g. state discrimination's GMM
    arrays) and the caller prints its own summary instead.
    """
    if (slices_ is None) == (from_plotdata is None):
        raise ValueError("pass exactly one of `slices_` or `from_plotdata`")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    figures = {}
    if from_plotdata is not None:
        for name, plot_data in _iter_plotdata(from_plotdata):
            figs = estimator.generate_figures(None, None, plot_data=plot_data)
            figures[name] = figs
            print(f"{name}: replotted {list(figs)} from saved plot-data (no re-fit)")
            _save_figs(figs, name, out_dir)
        return figures

    for name, sq in slices_:
        results = estimator.analyze(sq, output_dir=None, skip_figures=True, **analyze_kwargs)[0]
        figs = estimator.generate_figures(sq, results)
        figures[name] = figs
        if print_metadata:
            print(f"{name}: figures={list(figs)}  metadata={estimator.extract_metadata(results)}")
        else:
            print(f"{name}: figures={list(figs)}")
        _save_figs(figs, name, out_dir)
    return figures
