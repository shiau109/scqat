# QCAT → scqat migration

`scqat` is the modern, well-structured replacement for the lab's older analysis tool
**`qcat`** (`D:\github\QCAT`, package `qcat`, layout `src/qcat/...`). Both fit analysis
models to superconducting-qubit data — from **experiment or simulation** — and extract
physical parameters. This document is the single source of truth for **(1)** the structural
health of `scqat` and **(2)** which `qcat` features still need to be ported, in what order,
and how. Update it as features land.

Recorded 2026-06-06 from the working tree. Scope of the current pass: **scqat only** —
no changes to any consumer repo (e.g. `LCHQMDriver`'s `LCH_*` nodes).

---

## Architecture recap

`parsers → protocols → math_tools`, with `workflows` orchestrating on top and a
`BaseAnalyzer` ABC (`scqat/core/base_analyzer.py`) defining the output contract. The import
arrow points one way only; `math_tools` must not import from `protocols`/`parsers`/`workflows`.
Every analyzer produces **two artifacts**:

- **metadata** — key physical parameters, JSON-serializable, from `extract_parameters()`,
  saved as `<protocol_name>_metadata.json`.
- **plot data** — the minimal arrays to redraw every figure with no recalculation, a single
  `xarray.Dataset` from `build_plot_data()`, saved as `<protocol_name>_plotdata.nc`.

**Self-enforcing rule:** `generate_figures()` must draw from **`plot_data` only**, never the
raw `dataset` or `results`. See `.github/copilot-instructions.md` for the full rules.

---

## Structure health (track 1)

The architecture is sound — do **not** restructure it. Findings from this pass:

### Fixed in this pass
- **`scqat/parsers/__init__.py` was empty** while the design doc advertised
  `from scqat.parsers import ...`. Now re-exports `load_xarray_h5`, `repetition_data`,
  `parse_timestamp`.
- **`scqat/__init__.py` was empty.** Now exposes `__version__` (kept import-light on
  purpose — `import scqat` stays side-effect-free and does not pull in matplotlib).

### plot_data-contract compliance audit (the main structural debt)
The flagship contract is implemented by **2 of 7** protocols. The base class tolerates this
during migration, but until the rest are retrofitted, language-agnostic figure reconstruction
holds for those two only. **New ports MUST implement the contract** (see recipe) so they don't
copy the non-compliant shape.

| Protocol | `build_plot_data`? | Notes |
|---|---|---|
| `charge_gate_ramsey` | ✅ | reference implementation — copy this shape |
| `ramsey` | ✅ | retrofitted (figures rebuild from netCDF `plot_data` alone) |
| `state_discrimination` | ❌ | |
| `single_state_outlier` | ❌ | some figures commented out |
| `qubit_spectroscopy` | ❌ | |
| `qubit_decoherence` | ❌ | |
| `hankel_analysis` | ❌ | |

### Non-issues (noted, intentionally unchanged)
- Flat-module vs subpackage protocol layout is sanctioned by the doc (simple vs complex).
- Fitter registry is complete: `fit_gaussian2d.py` registers both `gaussian2d` and
  `multi_gaussian2d`; all fitters are eagerly imported in `math_tools/__init__.py`.
- The note elsewhere that "scqat has no parser yet" is **stale** — parsers exist (below).

---

## Feature-gap inventory (track 2)

### Analyzers — `qcat/src/qcat/analysis/` vs `scqat/protocols/`

| qcat analyzer | scqat status | Disposition |
|---|---|---|
| `ramsey.RamseyAnalysis` | ✅ `RamseyAnalyzer` | present — verify beat-model parity (`f_1/f_2/a_1/a_2/kappa`) |
| `state_discrimination.StateDiscrimination` | ✅ `StateDiscriminationAnalyzer` | present (GMM); metadata schema differs from qcat — see risk below |
| `charge_gate_ramsey.ChargeGateRamseyAnalysis` | ✅ `ChargeGateRamseyAnalyzer` | present (reference impl) |
| `readout_power.ROFidelityPower` | ❌ missing | **port** — orchestrates state-disc over `amp_prefactor` sweep |
| `readout_freq.ROFidelityFreq` | ❌ missing | **port** — orchestrates state-disc over a frequency sweep |
| `zz_interaction.ZZinteractionEcho` | ✅ `ZZInteractionEchoAnalyzer` | ported, plot_data-compliant (damped-oscillation per flux; reuses `damped_oscillation`) |
| `ac_stark_shift` (functional) | ❌ missing | **port** — depends on qubit-spectroscopy fitting |
| `readout_pulse_photon` (functional) | ❌ missing | **port** — qubit-spectroscopy fit vs pulse delay |
| `conditional_phase` | empty stub | won't port |
| `plot_ds_raw_scatter` | utility | optional helper, low priority |

scqat-only protocols with no qcat analysis-module equivalent (added during EP/MIST work,
keep): `qubit_spectroscopy` (lived in `NCU` in qcat), `qubit_decoherence`, `hankel_analysis`,
`single_state_outlier`.

### Fitters — `qcat/utilities/function_fitting/` vs `scqat/math_tools/`

| qcat fitter | scqat status |
|---|---|
| `fit_damped_oscillation` | ✅ `damped_oscillation` |
| `fit_damping_beat` | ✅ `damping_beat` |
| `fit_gaussian2d` (single + multi) | ✅ `gaussian2d`, `multi_gaussian2d` |
| `fit_cosine` | ❌ port |
| `fit_exp_decay` | ✅ `exp_decay` (ported, tested; flexible `parse_xy` input) |
| `fit_powerlaw_base` | ❌ port |
| `fit_transmon_freqeuency_flux` | ❌ port |

scqat-only fitters (keep): `abscos`, `lorentzian`, `multi_damped_oscillation`,
`qubit_decoherence`.

### Other qcat modules

| qcat module | Disposition |
|---|---|
| `parser/qm_reader.py` (`load_xarray_h5`, `repetition_data`) | ✅ already ported to `scqat/parsers/` |
| `common_calculator/analytical.py` (`Relax_cal`, `n_predict`, `resonator_freq_response`) | **deferred** — port only when a protocol needs it |
| `common_calculator/convertor.py` (`PetoT`, `PtoV`, `VtoN`, `NtoV`) | **deferred** — same |
| `utilities/data_processing.py` (`rot`, `IQ_data_dis`, `find_nearest`) | port on demand into `math_tools` |
| `NCU/Fit_library.py`, `NCU/Visualized_library.py` (Rabi/T1/T2/resonator spectroscopy, ~350 KB) | **do NOT port wholesale** — extract individual models into `math_tools`/`protocols` on demand |

---

## Suggested order

1. **Retrofit the remaining 5 non-compliant protocols to the `plot_data` contract** (locks the
   pattern before new code copies the old shape). `ramsey` is done — use it (and
   `charge_gate_ramsey`) as the template; remaining: `state_discrimination`,
   `single_state_outlier`, `qubit_spectroscopy`, `qubit_decoherence`, `hankel_analysis`.
2. ~~**`ZZinteractionEcho`**~~ — **done** (`ZZInteractionEchoAnalyzer`); validated the recipe
   end-to-end (reuses `damped_oscillation`).
3. **`ROFidelityPower` / `ROFidelityFreq`** — reuse `StateDiscriminationAnalyzer` (resolve the
   schema risk below first). ← next feature.
4. **`ac_stark_shift`** — reuse `QubitSpectroscopyAnalyzer`.
5. **`readout_pulse_photon`**.

Port the remaining missing fitters (`cosine`, `powerlaw_base`, `transmon_freq_vs_flux`) as the
dependent analyzer needs them, each with a `pytest`. (`exp_decay` is done — `tests/test_fit_exp_decay.py`.)

---

## Porting recipe (qcat class → scqat protocol)

qcat analyzers follow an ad-hoc `__init__/_import_data/_start_analysis/_plot_results` shape and
store results on `self`. scqat analyzers are **stateless** `BaseAnalyzer` subclasses that take
the `dataset` as a method argument. Map idioms as follows:

| qcat | scqat (`BaseAnalyzer` subclass) |
|---|---|
| `__init__(data)` + `_import_data` (stores `self.data`) | `_check_data(dataset)` — **validate required coords/vars only**, raise `ValueError`; store nothing |
| `_start_analysis()` (fits; stores `self.analysis_result` / `self.summary_dataset`) | `extract_parameters(dataset, **kw) -> dict` — return JSON-safe **metadata** |
| `_plot_results()` (imports viz, returns figs) | split into `build_plot_data(dataset, results) -> xr.Dataset` **and** `generate_figures(dataset, results, plot_data) -> dict[str, plt.Figure]` drawing **only from `plot_data`** |
| manual `.to_netcdf(...)` / `_export_result` | rely on `BaseAnalyzer.save_*`; call `analyze(dataset, output_dir=...)` |
| `FunctionFitting` subclass in `utilities/function_fitting/` | `math_tools/fit_<name>.py`: subclass `FunctionFitting`, decorate `@register_fitter('<name>')`, accept flexible input via `parse_xy`, add the import to `math_tools/__init__.py`, add a `pytest` in `tests/` |

Checklist for each new protocol:
1. Set `protocol_name`.
2. Implement `_check_data`, `extract_parameters`, `build_plot_data`, `generate_figures`
   (figures from `plot_data` only).
3. Document the dataset contract (required variables/coordinates/attrs) in the class docstring.
4. Flat module if simple; subpackage (`analyzer.py` + `visualization.py` + `__init__.py`
   re-export) if it has dedicated plotting.
5. Re-export the analyzer in `scqat/protocols/__init__.py`.
6. Add a notebook and/or `pytest` exercising it.

### Known porting risk — readout-fidelity sweeps
`ROFidelityPower`/`ROFidelityFreq` in qcat loop over a sweep and call `StateDiscrimination`,
consuming `analysis_result` keys: `trained_paras{std, mean}`, `outlier_probability`,
`gaussian_norms`, `direct_counts`, `norm_res`. scqat's `StateDiscriminationAnalyzer` returns a
**different metadata schema**. Before porting these two, verify the exact keys
`StateDiscriminationAnalyzer.extract_parameters` returns and adapt the orchestration — do
**not** assume drop-in reuse.
