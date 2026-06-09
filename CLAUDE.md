# AI Assistant Guidelines: SCqubit-analysis-tool

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

## Estimator Output Contract
An estimator produces **one mandatory artifact and two optional ones**:

1. **Metadata (mandatory)** — the *key physical parameters* extracted from the
   data (e.g. `T1`, `frequency`, `fwhm`). Small and JSON-serializable. The heavy
   compute `extract_parameters()` returns the full `results`; `extract_metadata(
   results)` projects the subset to persist as `<estimator_name>_metadata.json`.
   `extract_metadata` defaults to the identity (so a simple estimator's `results`
   *is* the metadata) and is overridden only to drop bulky intermediates. The
   file is stamped with `estimator_name`; the parameter set may evolve over time
   (rarely), and JSON's flexible schema absorbs that.
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

**One compute, two projections:** `extract_parameters` runs the heavy work once
and returns the rich `results`; `extract_metadata` and `build_plot_data` are pure
projections of it (key scalars vs. plot arrays). This avoids recomputation and
keeps `results` the single source of truth.

**Self-enforcing rule:** `generate_figures()` must draw using **only** the
`plot_data` Dataset, never the raw input `dataset` or the working `results`. If a
figure can be drawn from `plot_data` alone, an external consumer can too —
anything a figure needs therefore has to be put into `build_plot_data()`.
(During migration the orchestrator still passes `dataset` and `results` to
`generate_figures` so older estimators keep working; new/migrated estimators must
ignore them.)

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

A subclass MAY (optional, each has a safe default):
5. Override `extract_metadata(results) -> Dict[str, Any]` to drop bulky
   intermediates from the persisted metadata (default: identity).
6. Override `build_plot_data(dataset, results, **kwargs) -> xr.Dataset` to emit
   the **minimal arrays to redraw the figures** (default: `None`).
7. Override `generate_figures(dataset, results, plot_data=None, **kwargs)
   -> Dict[str, plt.Figure]`, drawing **only** from `plot_data` (default: `{}`;
   requires `build_plot_data`). `dataset`/`results` are passed for
   not-yet-migrated estimators; new code must ignore them.

The inherited `analyze()` method orchestrates:
`_check_data` → `extract_parameters` → `extract_metadata` → `build_plot_data` →
(optional save of metadata + plot data) → `generate_figures` → (optional save of
figures).

- **Every estimator is a subpackage** — one uniform layout, so there is no
  per-estimator judgment call about whether to "fold". The structure keys on stable
  identity (it *is* an estimator), not on the mutable question of whether it
  currently draws a figure:
  ```
  estimators/<name>/
      __init__.py      # re-exports the estimator class
      estimator.py     # the BaseEstimator subclass
      visualization.py # estimator-specific plotting helpers (consume plot_data)
  ```
  `visualization.py` is present **whenever the estimator draws figures** — i.e.
  almost always, since `generate_figures` requires `plot_data` and every estimator
  to date emits both. A genuine pure-fit estimator with no figures simply omits
  `visualization.py` while staying a subpackage, so that *adding* a plot later is a
  new file rather than a module→package restructure. The subpackage `__init__.py`
  MUST re-export the estimator class so external code can always use
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
