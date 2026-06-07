# AI Assistant Guidelines: SCqubit-analysis-tool

This repository (`scqat`) analyzes superconducting-qubit data. The data may come
from **either an experiment or a simulation** — analyzers must be blind to which.
It strictly follows a decoupled, Domain-Driven architecture using
`xarray.Dataset` as the universal data transfer object.

## Purpose & Consumption
`scqat` is a **pip-installable library** (`pyproject.toml`) meant to be imported
by other repositories:
- A **simulation** or **experiment** repo can `from scqat.math_tools import ...`
  to reuse the shared algorithms, or `from scqat.parsers import ...` to load a
  file it produced into the universal `xarray.Dataset`.
- Those repos hand a `Dataset` (or a file path) to an analyzer in
  `scqat.protocols` and get back metadata + plot-reconstruction data.

So every layer must stay importable in isolation and free of side effects at
import time.

## Core Architectural Rules
1. **Universal Data Container:** All data passed from parsers to protocols MUST
   be an `xarray.Dataset`. Raw data arrays become `DataArray` variables, sweep
   parameters become `Coordinates`, and instrument/simulation settings become
   `Attributes` (`.attrs`).
2. **Strict Separation of Concerns:**
   - `scqat/parsers/`: ONLY for reading raw files (HDF5, CSV, API payloads) from
     a given path — whether produced by an experiment or a simulation — and
     converting them to `xarray.Dataset`. NEVER put physics analysis or fitting
     logic here.
   - `scqat/protocols/`: ONE analyzer per experiment/protocol (e.g. T1, Ramsey,
     MIST). They accept an `xarray.Dataset`, process it, and output derived
     metadata and figures. NEVER put file I/O or raw data loading here.
   - `scqat/math_tools/`: Shared, **pure** mathematical algorithms (fitting,
     FFTs, Hankel analysis, analytical solvers). Anything used by more than one
     protocol lives here.
   - `scqat/workflows/`: Multi-protocol **orchestration pipelines** that chain
     parsers → several protocols/math_tools for a higher-level analysis (e.g.
     `ep_pipeline.py`). Pipelines return plain data structures; plotting is left
     to the caller. NEVER put file I/O of raw inputs anywhere except via parsers.
   - `scqat/core/base_analyzer.py`: Defines the `BaseAnalyzer` ABC and the
     saving/loading helpers. The `analyze()` orchestrator optionally invokes
     these when `output_dir` is provided. Analyzers MUST NOT perform file I/O
     outside this mechanism.
3. **Dependency direction (keeps the core reusable):** The import arrow points
   **one way only**: `workflows → protocols → math_tools`, and
   `parsers → (nothing in scqat)`. In particular `math_tools` MUST NOT import
   from `protocols`, `parsers`, or `workflows`. This is what lets an external
   simulation repo reuse `math_tools` without dragging in experiment logic.
4. **Format / provenance agnosticism:** Analyzers must be completely blind to
   whether data came from simulation or experiment. They interact only with the
   `xarray.Dataset` API.

## Workflow Rules
1. **Plan Before Implementation:** When asked to generate code, ALWAYS explain
   your implementation plan first. Do NOT modify any existing code until
   receiving explicit approval from the user.

## Analyzer Output Contract
An analyzer produces **one mandatory artifact and two optional ones**:

1. **Metadata (mandatory)** — the *key physical parameters* extracted from the
   data (e.g. `T1`, `frequency`, `fwhm`). Small and JSON-serializable. The heavy
   compute `extract_parameters()` returns the full `results`; `extract_metadata(
   results)` projects the subset to persist as `<protocol_name>_metadata.json`.
   `extract_metadata` defaults to the identity (so a simple protocol's `results`
   *is* the metadata) and is overridden only to drop bulky intermediates. The
   file is stamped with `protocol_name`; the parameter set may evolve over time
   (rarely), and JSON's flexible schema absorbs that.
2. **Plot data (optional)** — the *minimal arrays needed to redraw every figure
   with zero recalculation* (only trivial unit conversion, e.g. Hz→MHz, is
   allowed downstream). Returned by `build_plot_data()` as a single
   `xarray.Dataset` and saved as `<protocol_name>_plotdata.nc` (netCDF). Default
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
`generate_figures` so older protocols keep working; new/migrated protocols must
ignore them.)

## Implementation Guide

### Adding a new data format
Create a new script in `parsers/`. Write a function that takes a file path or API
payload (from experiment or simulation) and returns an `xarray.Dataset`.

### Adding a new protocol
Create a new class in `protocols/` that inherits from `BaseAnalyzer` (found in
`scqat/core/base_analyzer.py`).

Every subclass MUST:
1. Set the class attribute `protocol_name` (str) — controls default output
   filenames.
2. Implement `extract_parameters(dataset, **kwargs) -> Dict[str, Any]` — the
   heavy compute returning the full `results`. This is the **only** required
   method; for a simple protocol its return *is* the metadata.

Every subclass SHOULD also:
3. Override `_check_data(dataset)` to validate that all required coordinates and
   variables are present. Raise `ValueError` with a descriptive message on
   failure.
4. Document the **dataset contract** — the exact variable, coordinate, and
   attribute names the protocol expects — in the module or class docstring.

A subclass MAY (optional, each has a safe default):
5. Override `extract_metadata(results) -> Dict[str, Any]` to drop bulky
   intermediates from the persisted metadata (default: identity).
6. Override `build_plot_data(dataset, results, **kwargs) -> xr.Dataset` to emit
   the **minimal arrays to redraw the figures** (default: `None`).
7. Override `generate_figures(dataset, results, plot_data=None, **kwargs)
   -> Dict[str, plt.Figure]`, drawing **only** from `plot_data` (default: `{}`;
   requires `build_plot_data`). `dataset`/`results` are passed for
   not-yet-migrated protocols; new code must ignore them.

The inherited `analyze()` method orchestrates:
`_check_data` → `extract_parameters` → `extract_metadata` → `build_plot_data` →
(optional save of metadata + plot data) → `generate_figures` → (optional save of
figures).

- **Simple protocol** (no dedicated visualization): a single file, e.g.
  `protocols/t1_inversion_recovery.py`.
- **Complex protocol** (with its own visualization helpers): a subpackage, e.g.:
  ```
  protocols/state_discrimination/
      __init__.py      # re-exports the analyzer class
      analyzer.py      # the BaseAnalyzer subclass
      visualization.py # protocol-specific plotting helpers (consume plot_data)
  ```
  The subpackage `__init__.py` MUST re-export the analyzer so external code can
  use `from scqat.protocols.<name> import <Analyzer>` regardless of whether it is
  a flat module or a subpackage.

- **`protocols/__init__.py` aggregation:** Every new analyzer MUST also be
  imported in `scqat/protocols/__init__.py` so all analyzers are available via
  `from scqat.protocols import <Analyzer>`.

### Adding a new fitter to `math_tools/`
All fitters inherit from `FunctionFitting` (in
`scqat/math_tools/function_fitting.py`) and must:
1. Decorate the class with `@register_fitter('<name>')` so it is discoverable via
   the `get_fitter('<name>')` factory.
2. Accept flexible input so external (simulation) callers can use it without
   wrapping: an `xarray.DataArray` with an `'x'` coordinate, **or** raw `(x, y)`
   arrays, **or** a bare `y` array. Use the shared `parse_xy` helper in
   `function_fitting.py` to normalize the input.
3. Implement `model_function`, `guess`, and `fit` methods.
4. Write `pytest` tests in `tests/` for the new fitter (protocol-level tests are
   optional).
