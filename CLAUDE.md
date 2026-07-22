# AI Assistant Guidelines: scqat

This repository (`scqat`) analyzes superconducting-qubit data. The data may come
from **either an experiment or a simulation** — estimators must be blind to which.
It strictly follows a decoupled, Domain-Driven architecture using
`xarray.Dataset` as the universal data transfer object.

> Cross-repo terminology (Experiment = probe + **estimator**) is defined in `D:\github\SCQO\CLAUDE.md` → Terminology. This repo implements the **estimator** + **tool/fitter** half (the analysis side).

## Purpose & Consumption
`scqat` is a **pip-installable library** (`pyproject.toml`) meant to be imported
by other repositories:
- A **simulation** or **experiment** repo can `from scqat.tools import ...`
  to reuse the shared algorithms, or `from scqat.parsers import ...` to load a
  file it produced into the universal `xarray.Dataset`.
- Those repos hand a `Dataset` (or a file path) to an estimator in
  `scqat.estimators` and get back metadata + plot-reconstruction data.

So every layer must stay importable in isolation and free of side effects at
import time.

## Core Architectural Rules
1. **Universal Data Container:** All data passed from parsers to estimators MUST
   be an `xarray.Dataset`. Raw data arrays become `DataArray` variables, sweep
   parameters become `Coordinates`, and instrument/simulation settings become
   `Attributes` (`.attrs`).
2. **Strict Separation of Concerns:**
   - `scqat/parsers/`: ONLY for reading raw files (HDF5, CSV, API payloads) from
     a given path — whether produced by an experiment or a simulation — and
     converting them to `xarray.Dataset`. NEVER put physics analysis or fitting
     logic here.
   - `scqat/estimators/`: ONE estimator per experiment (e.g. T1, Ramsey,
     MIST). They accept an `xarray.Dataset`, process it, and output derived
     metadata and figures. NEVER put file I/O or raw data loading here.
   - `scqat/tools/`: Shared, **pure** mathematical algorithms (fitting,
     FFTs, Hankel analysis, analytical solvers). Anything used by more than one
     estimator lives here.
   - `scqat/workflows/`: Multi-estimator **orchestration pipelines** that chain
     parsers → several estimators/tools for a higher-level analysis (e.g.
     `ep_pipeline.py`). Pipelines return plain data structures; plotting is left
     to the caller. NEVER put file I/O of raw inputs anywhere except via parsers.
   - `scqat/core/base_estimator.py`: Defines the `BaseEstimator` ABC and the
     saving/loading helpers. The `analyze()` orchestrator optionally invokes
     these when `output_dir` is provided. Estimators MUST NOT perform file I/O
     outside this mechanism.
3. **Dependency direction (keeps the core reusable):** The import arrow points
   **one way only**: `workflows → estimators → tools`, and
   `parsers → (nothing in scqat)`. In particular `tools` MUST NOT import
   from `estimators`, `parsers`, or `workflows`. This is what lets an external
   simulation repo reuse `tools` without dragging in experiment logic.
4. **Format / provenance agnosticism:** Estimators must be completely blind to
   whether data came from simulation or experiment. They interact only with the
   `xarray.Dataset` API.

## Workflow Rules
1. **Plan Before Implementation:** When asked to generate code, ALWAYS explain
   your implementation plan first. Do NOT modify any existing code until
   receiving explicit approval from the user.

## Release checklist
Cutting a release = **two steps, in this order**:
1. Bump `version` in `pyproject.toml` to `X.Y.Z` and commit.
2. Tag that commit `vX.Y.Z`.

The pyproject version MUST equal the tag: dependents (SCQO) declare real floors
like `scqat>=0.1.4`, resolved from package metadata (`importlib.metadata`), so a
tag whose tree still carries the old version breaks every downstream install.
Never retag or rewrite an existing tag — if a tagged tree has the wrong version,
cut the next number.

## Estimator Output Contract
**One compute, two projections:** `extract_parameters()` runs the heavy work
**once** and returns the rich `results` — the single source of truth.
`extract_metadata()` and `build_plot_data()` are pure projections of it (key
scalars vs. plot arrays), so nothing is recomputed. Estimators transpose the
input dataset **by coordinate name** (order-invariant), so callers may pass sweep
axes in any order.

From that, an estimator produces **one mandatory artifact and two optional ones**:

1. **Metadata (mandatory)** — the *key physical parameters* (e.g. `T1`,
   `frequency`, `fwhm`), small and JSON-serializable, persisted as
   `<estimator_name>_metadata.json` (stamped with `estimator_name`).
   `extract_metadata(results)` projects the subset to keep; it defaults to the
   identity (a simple estimator's `results` *is* the metadata) and is overridden
   only to drop bulky intermediates. The parameter set may evolve (rarely), and
   JSON's flexible schema absorbs that.
2. **Plot data (optional)** — the *minimal arrays needed to redraw every figure
   with zero recalculation* (only trivial unit conversion, e.g. Hz→MHz, is
   allowed downstream). Returned by `build_plot_data()` as a single
   `xarray.Dataset` and saved as `<estimator_name>_plotdata.nc` (netCDF). Default
   is `None` (no plot-data artifact).
3. **Figures (optional)** — returned by `generate_figures()`. Default is `{}`.
   Because figures draw only from plot data, **providing figures implies
   providing plot data.**

The metadata/plot-data split exists so a *different* repo (possibly not even
Python) can reload the plot data and reconstruct the figures without rerunning
any analysis and without unpickling. JSON + netCDF are both self-describing and
language-agnostic — never use `pickle` for these artifacts.

**Self-enforcing rule:** `generate_figures()` must draw using **only** the
`plot_data` Dataset, never the raw input `dataset` or the working `results`. If a
figure can be drawn from `plot_data` alone, an external consumer can too —
anything a figure needs therefore has to be put into `build_plot_data()`.
(During migration the orchestrator still passes `dataset` and `results` to
`generate_figures` so older estimators keep working; new/migrated estimators must
ignore them.)

## Multi-method estimators (N approaches, one physics)

When more than one analysis approach can extract the same physical parameters
(reference implementation: `estimators/resonator_spectroscopy/` — `lorentzian`
joint-background fit vs `circle` Probst notch fit), structure it as follows:

1. **Still ONE estimator per experiment.** Approaches are *method strategy
   objects* in a `methods/` subpackage (`methods/base.py` ABC + one module per
   method + a `METHODS` registry in `methods/__init__.py`); heavy math stays in
   `tools/` fitters. Selection is a plain `method=` kwarg on
   `extract_parameters` (default = the cheap/robust method).
2. **Two-tier result contract.** The estimator declares `COMMON_KEYS` (same
   name ⇒ same meaning AND unit in every method) and validates them right after
   `extract()`; orchestration (SCQO) may rely only on those. Everything else is
   method-owned extras, consumed downstream only via `if key in results`. Never
   reuse a key name across methods with a different meaning; error keys are
   common-but-best-effort (NaN allowed).
3. **Provenance.** `results["method"]` (→ metadata JSON) and
   `plot_data.attrs["method"]` are always stamped.
4. **Plots are method-dependent; artifacts are not.** Each method owns its
   plot_data variables and figure layout, but the figure dict key (and hence the
   PNG name) is identical for all methods, plot_data stays netCDF-safe (no
   complex variables — store I/Q float pairs), and `generate_figures` dispatches
   on `plot_data.attrs["method"]` — never on estimator state — so a saved
   `plotdata.nc` replots with zero re-fit.
5. **Adding method #N** = one module in `methods/` + one registry entry (+ one
   Literal value in the SCQO experiment's Parameters). Nothing else moves.
6. **Cross-method test.** The estimator's test suite includes an agreement test:
   same synthetic data → same COMMON physics within tolerance (a method changes
   robustness, not physics).

## Layered analyses (an experiment that contains another experiment's fit)

**Estimators never call estimators.** When an analysis needs another
experiment's fit as an inner step (the vs-flux/vs-power maps fit a resonator
dip per slice), the shared piece is a *pure per-trace reduction* — and by rule
it lives in `tools/` (references: `tools/dip_fit.py`, `fit_dip()` +
`DIP_METHODS` + `DIP_KNOBS`, for the resonator family; `tools/peak_fit.py`,
`fit_peaks()` + `PEAK_KNOBS`, plus the generic 2-D tracker
`tools/peak_map.py::track_peaks` for the qubit family), consumed by every
estimator in the family. When two sweep experiments share the whole map
reduction (vs-flux and parametric-drive), the tracker is generic-keyed
(`x`/`y`) in `tools/` and each estimator relabels into its own vocabulary; an
experiment's *Dataset-shaped* stage helpers live in its own subpackage
(reference: `resonator_spectroscopy_flux/dips.py`,
`qubit_spectroscopy_flux/peaks.py`) and may be imported as plain functions by
a downstream composite estimator of the same family or by control repos.
Estimator→estimator calls conflate the experiment-level contract
(metadata/plot_data/figures) with plain math, and their flat `**kwargs`
namespace makes inner options unreachable or silently mis-routed.

1. **"Same experiment + extra sweep axis" shares the reduction, not the
   estimator.** Only the per-trace fit is common; how it is driven (candidate
   seeding from the 2-D map, local windows, fallbacks), the cross-axis
   acceptance gates, the second-stage model, and the artifacts are all
   sweep-specific and belong to the swept experiment's own estimator.
2. **Flat, fully-owned kwarg surface.** The estimator's primary method axis is
   `method`; a secondary axis is `<thing>_method` (e.g. `dip_method`); every
   kwarg is documented in `extract_parameters`'s docstring. No prefixes, no
   forwarding of "whatever is left over".
3. **Validate before loops.** tools expose their valid-knob sets
   (`validate_dip_kwargs`); callers validate ONCE before any per-slice loop, so
   a typo'd kwarg raises instead of being swallowed by per-slice `try/except`
   fallbacks.
4. **Required keys only.** A caller of a tools reduction may rely only on its
   documented required keys (`detuning`/`fwhm`/`success` for `fit_dip`);
   method-owned extras (e.g. `amplitude`) via `.get` with defined degraded
   behavior.
5. **Provenance.** Every resolved `*_method` value is stamped in results, the
   metadata JSON, and `plot_data.attrs` (strings; bools as int 0/1; absent
   axes omitted). SCQO mirrors each axis as a `Literal` kept equal to the
   registry by a sync test (SCQO `tests/test_estimator_method_sync.py`).

## Implementation Guide

### Adding a new data format
Create a new script in `parsers/`. Write a function that takes a file path or API
payload (from experiment or simulation) and returns an `xarray.Dataset`.

### Adding a new estimator
Create a new class in `estimators/` that inherits from `BaseEstimator` (found in
`scqat/core/base_estimator.py`).

Every subclass MUST:
1. Set the class attribute `estimator_name` (str) — controls default output
   filenames.
2. Implement `extract_parameters(dataset, **kwargs) -> Dict[str, Any]` — the
   heavy compute returning the full `results`. This is the **only** required
   method; for a simple estimator its return *is* the metadata.

Every subclass SHOULD also:
3. Override `_check_data(dataset)` to validate that all required coordinates and
   variables are present. Raise `ValueError` with a descriptive message on
   failure.
4. Document the **dataset contract** — the exact variable, coordinate, and
   attribute names the estimator expects — in the module or class docstring.

A subclass MAY (optional — each has a safe default; semantics per **Estimator
Output Contract** above):
5. Override `extract_metadata(results) -> Dict[str, Any]` (default: identity).
6. Override `build_plot_data(dataset, results, **kwargs) -> xr.Dataset`
   (default: `None`).
7. Override `generate_figures(dataset, results, plot_data=None, **kwargs)
   -> Dict[str, plt.Figure]` (default: `{}`; requires `build_plot_data`).

The inherited `analyze()` method orchestrates:
`_check_data` → `extract_parameters` → `extract_metadata` → `build_plot_data` →
(optional save of metadata + plot data) → `generate_figures` → (optional save of
figures).

- **Every estimator is a subpackage** — one uniform layout, so there is no
  per-estimator "should I fold this into a single module?" judgment call:
  ```
  estimators/<name>/
      __init__.py      # re-exports the estimator class
      estimator.py     # the BaseEstimator subclass
      visualization.py # estimator-specific plotting helpers (consume plot_data)
  ```
  `visualization.py` is present whenever the estimator draws figures (almost
  always); a pure-fit estimator with no figures omits it but stays a subpackage,
  so adding a plot later is a new file, not a module→package restructure. The
  `__init__.py` MUST re-export the estimator class so external code can always use
  `from scqat.estimators.<name> import <Estimator>`.

- **`estimators/__init__.py` aggregation:** Every new estimator MUST also be
  imported in `scqat/estimators/__init__.py` so all estimators are available via
  `from scqat.estimators import <Estimator>`.

### Adding a new fitter to `tools/`
All fitters inherit from `FunctionFitting` (in
`scqat/tools/function_fitting.py`) and must:
1. Decorate the class with `@register_fitter('<name>')` so it is discoverable via
   the `get_fitter('<name>')` factory.
2. Accept flexible input so external (simulation) callers can use it without
   wrapping: an `xarray.DataArray` with an `'x'` coordinate, **or** raw `(x, y)`
   arrays, **or** a bare `y` array. Use the shared `parse_xy` helper in
   `function_fitting.py` to normalize the input.
3. Implement `model_function`, `guess`, and `fit` methods.
4. Write `pytest` tests in `tests/` for the new fitter (estimator-level tests are
   optional).

## Offline analysis on saved data (`analysis/`)
Iterate on estimators against **real saved runs** (`ds_raw.h5` / `plotdata_*.h5` produced by a
driver such as LCHQM) without re-running hardware. The reusable engine is
`analysis/_harness.py`; per-experiment entry points are thin `# %%`-cell scripts
(`analysis/try_<experiment>.py`).

1. **`.py` + `# %%` cells, not `.ipynb`.** Notebook UX in VS Code (run-cell, inline figures) but
   git-friendly files (clean diffs, reviewable, importable). Real notebooks stay in `notebooks/`.
2. **Reuse the engine; never re-implement load/slice/plot per file.** `_harness.py` provides
   `load(path)`, `slices(ds, prep)`, `compare(slices, methods)`,
   `estimator_method(est, adapt, **kwargs)`, and `replot(est, slices_ | from_plotdata=…)`. A
   per-experiment `try_<exp>.py` sets only: the data path, `prep` (raw → estimator input),
   `adapt` (`results` → normalized plot fields), and the methods to compare.
3. **Three uses, one engine:** (A) try a new approach — compare a custom method against the
   estimator; (B) test parameters — the estimator across a kwarg grid (e.g. `min_snr` /
   `prominence`); (C) replot a plot-skipped run — re-fit from `ds_raw`, or with **no re-fit**
   from a saved `plotdata_*.h5` via `replot(..., from_plotdata=<run dir or file>)`.
4. **Validate estimator changes against real saved data before committing**, stating the
   expected truth (e.g. a noise sweep → 0 peaks; a two-transition sweep → both peaks). But
   **never put external/absolute data paths in `tests/`** — tests use synthetic data or a small
   committed fixture (e.g. `notebooks/charge_gate_ramsey_plot_payload.h5`); `analysis/` is the
   only place path-based exploration lives.

A saved `plotdata_*.h5` *is* the estimator-native `build_plot_data` Dataset (see **Estimator
Output Contract**): reload it with `from scqat.parsers import load_xarray_h5` and draw via
`estimator.generate_figures(None, None, plot_data=…)` — no re-fit, no parsing.
