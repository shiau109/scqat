# Parametric-drive estimator examples

Runnable examples for the two parametric-drive estimators (the ones the renewed
LCHQMDriver `LCH_qubit_parametric_drive_*` nodes call). Each script loads data,
runs the estimator per qubit, prints a one-line summary, and saves the metadata,
plot-data, and figures under `output/<tag>/<qubit>/`.

| Script | Estimator | Source | Data shape |
|---|---|---|---|
| `run_decoherence_estimator.py` | `ParametricDriveDecoherenceEstimator` | `LCH_qubit_parametric_drive_freq_time` (`tomography` flag) | `driving_time` × `driving_frequency` (× `basis`) |
| `run_decoherence_on_sim.py` | `ParametricDriveDecoherenceEstimator` | SCQ.jl parametric-drive sim (virtual experiment) | `driving_time` × `driving_frequency` |
| `run_resonance_estimator.py` | `ParametricDriveResonanceEstimator` | `LCH_qubit_parametric_drive_fixed_time` | `amplitude_ratio` × `driving_frequency` |

## Decoherence (existing data, both layouts)

Per `driving_frequency` it rebuilds rho_11(t), fits the non-Markovian
amplitude-damping model, and reports γ, λ, Δ and the exceptional-point figure of
merit `8·λ²/γ²`. The same node produces both layouts via its `tomography` flag, and
two real datasets are wired in and auto-detected:

- **rho_11-only** (`tomography=False`) — `#1354_..._parametric_drive_time_...`
- **tomography** (`tomography=True`, basis = X/Y/Z) — `#1518_..._parametric_drive_time_tomo_...`

```bash
# run every example dataset that exists on disk
python examples/parametric_drive/run_decoherence_estimator.py

# or analyse one specific acquisition
python examples/parametric_drive/run_decoherence_estimator.py path/to/ds_raw.h5
```

Output figures per qubit: `..._decoherence_params.png` (γ, λ, |Δ|, 8λ²/γ² vs
driving frequency) and `..._rho11_fits.png` (rho_11(t) data + fit, coloured by
frequency).

If your readout zero / contrast differ from the defaults, set `rho11_offset` /
`rho11_scale` in `ESTIMATOR_KWARGS` at the top of the script (these feed the
`rho_11 = (state − offset) / scale` normalisation).

## Simulation (virtual experiment)

A SCQ.jl parametric-drive frequency sweep synthesizes the same observable the
real node measures -- rho_11(t) per driving frequency -- so the **same**
estimator analyses it unchanged. `load_parametric_sim_h5`
(`scqat.workflows.parametric_sim`) reads a SCQ.jl sim HDF5 into the estimator's
`(driving_frequency, driving_time, state)` contract; the only difference from the
experiment path is `rho11_offset=0, rho11_scale=1` (a simulated population needs
no readout correction).

```bash
python examples/parametric_drive/run_decoherence_on_sim.py                 # bundled SCQ.jl sweep
python examples/parametric_drive/run_decoherence_on_sim.py path/to/sim.h5  # your sweep
```

The canonical SCQ.jl export (`studies/driven_q/parametric_drive`,
`export_fake_data`) writes `rho_11` / `driving_frequency` (Hz) / `driving_time`
(ns) directly; legacy `all_expect` sweeps (projector or moment channels) are also
read. Large sim time grids are thinned with `time_stride`.

## Resonance (fixed-time map)

Fits a Lorentzian per `amplitude_ratio` slice and returns the cleaned
resonance-peak point-cloud over the 2-D map. No fixed-time dataset was provided,
so it runs a **synthetic** drifting-ridge map by default; pass a real `ds_raw.h5`
to analyse measured data:

```bash
python examples/parametric_drive/run_resonance_estimator.py                 # synthetic demo
python examples/parametric_drive/run_resonance_estimator.py path/to/ds_raw.h5
```

## Notes

- The scripts insert the repo root on `sys.path`, so they run without installing
  scqat. Run them with the `qcat` environment (or any env with scqat's deps).
- The decoherence path reuses `scqat.workflows.ep_pipeline`; the full
  multi-stage notebook view is still in `notebooks/EP/view_single_raw.ipynb`.
