# AI Assistant Guidelines: SCqubit-analysis-tool

This repository (`scqat`) processes superconducting qubit experimental data. It strictly follows a decoupled, Domain-Driven architecture using `xarray.Dataset` as the universal data transfer object.

## Core Architectural Rules:
1. **Universal Data Container:** All data passed from parsers to protocols MUST be an `xarray.Dataset`. Raw data arrays become `DataArray` variables, sweep parameters become `Coordinates`, and instrument settings become `Attributes` (`.attrs`).
2. **Strict Separation of Concerns:**
   - `scqat/parsers/`: ONLY for reading raw files (HDF5, CSV, API payloads) and converting them to `xarray.Dataset`. NEVER put physics analysis or fitting logic here.
   - `scqat/protocols/`: ONLY for protocol-specific unpacking and analysis (e.g., T1, Ramsey, MIST). They accept an `xarray.Dataset`, process it, and output derived metadata and figures. NEVER put file I/O or raw data loading here.
   - `scqat/math_tools/`: Shared mathematical functions (fitting, FFTs, analytical solvers). If an operation is used by multiple protocols, it lives here.
   - `scqat/core/exporters.py`: Handles saving the resulting dictionaries and figures. Analyzers MUST NOT save their own files.
3. **Format Agnosticism:** Analyzers must be completely blind to data provenance. They should only interact with the `xarray.Dataset` API.

## Workflow Rules:
1. **Plan Before Implementation:** When asked to generate code, ALWAYS explain your implementation plan first. Do NOT modify any existing code until receiving explicit approval from the user.

## Implementation Guide:
- **Adding a new data format:** Create a new script in `parsers/`. Write a function that takes a file path or API payload and returns an `xarray.Dataset`.
- **Adding a new protocol:** Create a new class in `protocols/` that inherits from `BaseAnalyzer` (found in `scqat/core/base_analyzer.py`). Implement `extract_parameters(dataset, **kwargs) -> Dict[str, Any]` for computation and `generate_figures(dataset, results, **kwargs) -> Dict[str, plt.Figure]` for visualization. The inherited `analyze()` method orchestrates both steps.
  - **Simple protocol** (no dedicated visualization): a single file, e.g. `protocols/t1_inversion_recovery.py`.
  - **Complex protocol** (with its own visualization helpers): a subpackage, e.g.:
    ```
    protocols/state_discrimination/
        __init__.py      # re-exports the analyzer class
        analyzer.py      # the BaseAnalyzer subclass
        visualization.py # protocol-specific plotting helpers
    ```
    The `__init__.py` MUST re-export the analyzer so that external code can use `from scqat.protocols.<name> import <Analyzer>` regardless of whether it is a flat module or a subpackage.
- **Configuration:** Use `scqat/config.py` for all hardcoded physical constants and standard plotting formats. Do not hardcode magic numbers in the protocols.
- **Testing:** Write `pytest` functions in the `tests/` directory whenever adding a new algorithm to `math_tools/`.