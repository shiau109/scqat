# AI Assistant Guidelines: SCqubit-analysis-tool

This repository (`scqat`) processes superconducting qubit experimental data. It strictly follows a decoupled, Domain-Driven architecture using `xarray.Dataset` as the universal data transfer object.

## Core Architectural Rules:
1. **Universal Data Container:** All data passed from parsers to protocols MUST be an `xarray.Dataset`. Raw data arrays become `DataArray` variables, sweep parameters become `Coordinates`, and instrument settings become `Attributes` (`.attrs`).
2. **Strict Separation of Concerns:**
   - `scqat/parsers/`: ONLY for reading raw files (HDF5, CSV, API payloads) and converting them to `xarray.Dataset`. NEVER put physics analysis or fitting logic here.
   - `scqat/protocols/`: ONLY for protocol-specific unpacking and analysis (e.g., T1, Ramsey, MIST). They accept an `xarray.Dataset`, process it, and output derived metadata and figures. NEVER put file I/O or raw data loading here.
   - `scqat/math_tools/`: Shared mathematical functions (fitting, FFTs, analytical solvers). If an operation is used by multiple protocols, it lives here.
   - `scqat/core/base_analyzer.py`: Defines the `BaseAnalyzer` ABC and contains the saving/loading helpers (`save_metadata`, `save_figures`, `load_metadata`). The `analyze()` orchestrator optionally invokes these when `output_dir` is provided. Analyzers themselves MUST NOT perform file I/O outside this mechanism.
3. **Format Agnosticism:** Analyzers must be completely blind to data provenance. They should only interact with the `xarray.Dataset` API.

## Workflow Rules:
1. **Plan Before Implementation:** When asked to generate code, ALWAYS explain your implementation plan first. Do NOT modify any existing code until receiving explicit approval from the user.

## Implementation Guide:

### Adding a new data format
Create a new script in `parsers/`. Write a function that takes a file path or API payload and returns an `xarray.Dataset`.

### Adding a new protocol
Create a new class in `protocols/` that inherits from `BaseAnalyzer` (found in `scqat/core/base_analyzer.py`).

Every subclass MUST:
1. Set the class attribute `protocol_name` (str) — controls default output filenames.
2. Override `_check_data(dataset)` to validate that all required coordinates and variables are present. Raise `ValueError` with a descriptive message on failure.
3. Implement `extract_parameters(dataset, **kwargs) -> Dict[str, Any]` for the physics computation.
4. Implement `generate_figures(dataset, results, **kwargs) -> Dict[str, plt.Figure]` for visualization.
5. Document the **dataset contract** — the exact variable names, coordinate names, and attribute names the protocol expects — in the module or class docstring.

The inherited `analyze()` method orchestrates: `_check_data` → `extract_parameters` → `generate_figures` (with optional save).

- **Simple protocol** (no dedicated visualization): a single file, e.g. `protocols/t1_inversion_recovery.py`.
- **Complex protocol** (with its own visualization helpers): a subpackage, e.g.:
  ```
  protocols/state_discrimination/
      __init__.py      # re-exports the analyzer class
      analyzer.py      # the BaseAnalyzer subclass
      visualization.py # protocol-specific plotting helpers
  ```
  The subpackage `__init__.py` MUST re-export the analyzer so that external code can use `from scqat.protocols.<name> import <Analyzer>` regardless of whether it is a flat module or a subpackage.

- **`protocols/__init__.py` aggregation:** Every new analyzer MUST also be imported in `scqat/protocols/__init__.py` so that all analyzers are available via `from scqat.protocols import <Analyzer>`.

### Adding a new fitter to `math_tools/`
All fitters inherit from `FunctionFitting` (in `scqat/math_tools/function_fitting.py`) and must:
1. Decorate the class with `@register_fitter('<name>')` so it is discoverable via the `get_fitter('<name>')` factory.
2. Accept an `xarray.DataArray` with a coordinate named `'x'` as input data.
3. Implement `model_function`, `guess`, and `fit` methods.
4. Write `pytest` tests in `tests/` for the new fitter (protocol-level tests are optional).

### Configuration
Use `scqat/config.py` for all hardcoded physical constants and standard plotting formats. Do not hardcode magic numbers in the protocols.