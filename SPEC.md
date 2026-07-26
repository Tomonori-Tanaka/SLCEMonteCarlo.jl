# SLCEMonteCarlo.jl — specification

Full classical spin Monte Carlo for fitted SCE models from `SLCE.jl`.
Self-contained core (no Carlo.jl), Threads parallelism, own binning analysis.
This file tracks the architecture and public API; the decision records live in
`docs/specs/*.md`. Validated end-to-end on the Nd₂Fe₁₄B l02 refit (68 atoms,
4692 terms): 4×4×4 tiling in 0.01 s / 7.8 MB of index arrays, machine-precision
training-cell and periodic-replication gates, and an 8-rung PT run recovering the
ferrimagnetic Nd-vs-Fe order at 250 K.

## Module layout

| File | Contents |
|---|---|
| `src/units.jl` | `KB_EV`, `resolve_kt` (kelvin XOR model-energy-unit control) |
| `src/hamiltonian.jl` | `TermSlot`, `ScaledTerm`, `TiledHamiltonian` (supercell tiling, CSR instance/site adjacency, `site_active`/`n_active` for scheduling and `site_has_spin`/`n_spin_active` for the spin channel — non-magnetic sites are frozen and excluded, precompiled sparse contraction programs, conflict-graph coloring for parallel sweeps), `site_index` |
| `src/energy.jl` | the energy contract: `total_energy`, `site_coeffs!`, `delta_energy`, `site_gradient`, all-site `energy_gradient!` (program kernels + bitwise-gated rank-generic reference kernels), joint `total_energy(H, config, disps)` and the displacement row filler |
| `src/binning.jl` | `LogBinner`, `BinStore`, `jackknife` |
| `src/observables.jl` | `Observable`, `Evaluable`, standard sets |
| `src/state.jl` | `SpinConfig`, `ChainState` (spin config + displacements + per-site RNG streams + the two proposal widths), `SweepScratch`, `_recenter!` (per-component centre-of-mass projection, called from `_renormalize!`) |
| `src/updates.jl` | Metropolis (adaptive step), overrelaxation, displacement Metropolis, compound sweeps — color-ordered, serial or `sweep_tasks`-parallel with bit-identical results |
| `src/gpu/*.jl` | GPU Metropolis prototype (KernelAbstractions, backend supplied by the caller): `philox.jl` keyed Philox4x32-10 stream, `zlm_device.jl` bitwise device tesseral row, `gpu_hamiltonian.jl`/`gpu_state.jl` device tables + chain state, `gpu_sweep.jl` fused kernel + drivers + keyed serial reference |
| `src/minimize.jl` | `minimize_energy` (on-sphere BB descent), `find_ground_state` (multi-start anneal + polish), `GroundStateResult` |
| `src/run.jl` | `run_mc` (single T + annealing), `TempResult`, `MCResult` |
| `src/pt.jl` | `run_pt` (replica exchange over threads), `PTResult` |
| `src/checkpoint.jl` | JLD2 schema v3, `resume` |
| `src/geometry.jl` | `supercell_crystal`, `to_matrix`/`from_matrix` |
| `src/reduce.jl` | `reduce_cell`/`ReducedCell` — verified re-expression of a supercell-fitted model in a user-chosen smaller cell |

## Dependency boundary

Reads a fitted model only through `SLCE`'s public surface
(`decorated_terms`, `row_layout`/`row_index`/`site_rows!`, `restrict`,
`multipole_terms`, `n_atoms`, `intercept`, `SLCE.load`, `Lattice`/`Crystal`,
`SLCE.Harmonics`, `SLCE.SolidHarmonics`). The per-term scale — the general
`(4π)^(n_spin_slots/2)` carried by `DecoratedTerm.scale`, which reduces to
`(4π)^(body/2)` on the frozen pure-spin surface — is applied exactly once,
in the `TiledHamiltonian` constructor. Row numbering is **not** invented here: it is
`SLCE.row_layout(model)`, so a pure-spin model's row tables are the pre-M4 ones. The MC core is geometry-free (integer site
topology only); geometry helpers take an explicit `Crystal`.

## Public API

Exported: `KB_EV`, `TiledHamiltonian`, `n_sites`, `total_energy`, `Observable`,
`Evaluable`, `ObservableStat`, `standard_observables`, `standard_evaluables`,
`run_mc`, `MCResult`, `TempResult`, `run_pt`, `PTResult`, `minimize_energy`,
`find_ground_state`, `GroundStateResult`, `resume`, `supercell_crystal`,
`ReducedCell`, `reduce_cell`, and (since 2026-07-19, after the A100 GO and the
l02/l044 production validations) the GPU sweep API: `GPUTiledHamiltonian`,
`GPUChainState`, `gpu_metropolis_sweep!`, `gpu_run_sweeps!`, `to_host!`.

Public, unexported (`SLCEMonteCarlo.<name>`): `resolve_kt`, `ScaledTerm`,
`TermSlot`, `has_disp`, `SpinConfig`, `site_index`, `site_atom`, `site_coeffs!`, `delta_energy`,
`site_gradient`, `energy_gradient`, `energy_gradient!` (all-site exact
tangent-projected `∂E/∂e_s`, bit-identical for any `ntasks` — the field/torque
entry point for dependent packages such as spin dynamics),
`LogBinner`, `std_error`, `tau_int`, `BinStore`, `bin_means`,
`jackknife`, `ChainState`, `SweepScratch`, `metropolis_sweep!`, `overrelaxation_sweep!`,
`displacement_sweep!`,
`to_matrix`, `from_matrix`, `philox_block`,
`philox_normal2` (the counter-based RNG as a stable contract for dependent
packages; consumers must claim a nonzero `ctr[4]` domain tag — MC streams use 0),
`model_fingerprint` (the checkpoint format's stable FNV-1a model identity, shared
by dependent packages' checkpoint files), `GPUGradientScratch`,
`gpu_energy_gradient!`, `gpu_zlm_rows!` (phase 2: the device all-site
tangent-projected SCE gradient — the GPU field entry point for dependent
packages' dynamics; bitwise-gated against its lane reference, G7).

## Design record index

- `docs/specs/hamiltonian-tiling.md` — supercell unfolding, CSR memory layout
- `docs/specs/updates-stationarity.md` — Metropolis/OR stationarity, adaptive-step freeze
- `docs/specs/binning-observables.md` — C/χ/U conventions (authoritative), log-binning, jackknife
- `docs/specs/pt-threads-determinism.md` — lane/RNG discipline, bit-reproducibility
- `docs/specs/checkpoint-schema.md` — JLD2 schema v3
- `docs/specs/cell-reduction.md` — verified reduction to a user-chosen smaller cell
- `docs/specs/ground-state-search.md` — on-sphere descent, thermal cycling, multi-start determinism
- `docs/specs/gpu-feasibility.md` — GPU-port assessment: strategy, measured baseline, go/no-go
- `docs/specs/gpu-prototype.md` — GPU Metropolis prototype: keyed RNG layout, determinism contract, kernel shape, A100 readout
