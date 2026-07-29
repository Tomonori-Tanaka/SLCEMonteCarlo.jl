# SLCEMonteCarlo.jl — specification

Full classical spin Monte Carlo for fitted SLCE models from `SLCE.jl`.
Self-contained core (no Carlo.jl), Threads parallelism, own binning analysis.
This file tracks the architecture and public API; the decision records live in
`docs/specs/*.md`. Validated end-to-end on the Nd₂Fe₁₄B l02 refit (68 atoms,
4692 terms): 4×4×4 tiling in 0.01 s / 7.8 MB of index arrays, machine-precision
training-cell and periodic-replication gates, and an 8-rung PT run recovering the
ferrimagnetic Nd-vs-Fe order at 250 K.

## Module layout

| File | Contents |
|---|---|
| `src/units.jl` | re-exports `SLCE`'s `KB_EV` / `resolve_kt` (kelvin XOR model-energy-unit control); the definitions live upstream, never here |
| `src/hamiltonian.jl` | `TermSlot`, `ScaledTerm`, `TiledHamiltonian` (supercell tiling, CSR instance/site adjacency, `site_active`/`n_active` for scheduling and `site_has_spin`/`n_spin_active` for the spin channel — non-magnetic sites are frozen and excluded, precompiled sparse contraction programs, conflict-graph coloring for parallel sweeps), `site_index` |
| `src/coefficients.jl` | `set_coefficients!` — coefficient hot-swap over the factored weight stream (`sent_w == term_coef[sent_term] · sent_base`), for the strain outer move and the active-learning hook: 0.033 ms vs 13–96 ms for a rebuild, and independent of `dims` |
| `src/strain.jl` | `StrainSchedule` — the sampler's form of a `SLCE.StrainedModels` K(ε) volume grid: polynomial coefficients over the term list a Hamiltonian was built from, converted ONCE (so a proposal costs a Horner pass plus an in-place `set_coefficients!`, not a `model_at` rebuild), plus `j0(s)` per training cell, the supercell volume, `N_mob = H.n_disp_active` (whose `V^{N_mob}` is the NPT volume power — NOT the COM-reduced `D/3`; re-corrected 2026-07-29) and the sampled dimension `d_dim` for diagnostics. Construction validates that every grid point's term list agrees atom-for-atom and image-for-image with the others AND with the Hamiltonian, refusing by name a Hamiltonian built without `keep_zero_terms` — the index map has to be a property of the basis, not of the fit — and, whenever the Hamiltonian re-centres any direction, that every grid node is flat along those directions at the Hamiltonian's OWN dims, one-sidedly (the ASR hard-error; flatness is linear in the coefficients, so the nodes certify the family). Also the NPT energy contract (`strain_delta_energy` — `ΔE_config + n_cells·Δj0 + P·ΔV`, `j0(ε)` the ONLY elastic source, no elastic-term keyword anywhere by design), the paired proposal arms (`:logvolume` / `:scale`, ln-weight coefficient `3·N_mob + 3` / `3·N_mob + 2` branching on the same symbol as the draw), and `strain_move!` + `StrainScratch` — the outer Metropolis move (affine displacement rescale, in-place coefficient swap, bit-identical Horner restore on reject, reject-never-clamp outside the grid). Record: `docs/specs/strain-move.md` |
| `src/energy.jl` | the energy contract: `total_energy`, `site_coeffs!`, `delta_energy`, `site_gradient`, all-site `energy_gradient!` (program kernels + bitwise-gated rank-generic reference kernels), joint `total_energy(H, config, disps)` and the displacement row filler |
| `src/binning.jl` | `LogBinner`, `BinStore`, `jackknife` |
| `src/observables.jl` | `MCView` (the single argument every observable receives), `Observable`, `Evaluable` (with its `scope` site-count declaration), standard sets — spin, and the centre-of-mass-free displacement ones on a joint model |
| `src/state.jl` | `SpinConfig`, `ChainState` (spin config + displacements + the cell scale `strain` + per-site RNG streams + the two proposal widths), `SweepScratch`, `_recenter!` (per-component centre-of-mass projection, called from `_renormalize!`) |
| `src/updates.jl` | Metropolis (adaptive step), overrelaxation, displacement Metropolis, compound sweeps — color-ordered, serial or `sweep_tasks`-parallel with bit-identical results |
| `src/gpu/*.jl` | GPU Metropolis prototype (KernelAbstractions, backend supplied by the caller): `philox.jl` keyed Philox4x32-10 stream, `zlm_device.jl` bitwise device tesseral row, `disp_device.jl` device solid-harmonic / displacement rows, `gpu_hamiltonian.jl`/`gpu_state.jl` device tables (both channels' color schedules + the layout's displacement blocks) + chain state (spins, displacements, the full row table), `gpu_sweep.jl` the two fused kernels (spin + displacement), the compound driver, and their keyed serial references. Both sweeps are channel-restricted (row-range ΔE, per-channel schedule); the gradient entry points are spin-only |
| `src/minimize.jl` | `minimize_energy` (on-sphere BB descent), `find_ground_state` (multi-start anneal + polish), `GroundStateResult` |
| `src/stability.jl` | `force_constant_matrix` / `harmonic_stability` — the displacement Hessian of the TILED Hamiltonian by central differences, and its spectrum: the before-the-run screen complementing the after-the-fact escape detector |
| `src/run.jl` | `run_mc` (single T + annealing; NPT via `strain = StrainSchedule(...)` + `pressure_GPa` XOR `pressure`, resolved once through `GPA_PER_EV_A3`), `TempResult`, `MCResult` |
| `src/pt.jl` | `run_pt` (replica exchange over threads), `PTResult` |
| `src/checkpoint.jl` | JLD2 schema v4 (strain channel; reference-scale model fingerprint + grid fingerprint), `resume` |
| `src/geometry.jl` | `supercell_crystal`, `to_matrix`/`from_matrix` |
| `src/reduce.jl` | `reduce_cell`/`ReducedCell{T}` — verified re-expression of a supercell-fitted model (pure-spin `SpinMultipoleTerm` or joint `DecoratedTerm`) in a user-chosen smaller cell |

## Dependency boundary

Reads a fitted model only through `SLCE`'s public surface
(`decorated_terms`, `row_layout`/`row_index`/`site_rows!`, `restrict`,
`spin_multipole_terms`, `n_atoms`, `intercept`, `SLCE.load`, `Lattice`/`Crystal`,
`SLCE.Harmonics`, `SLCE.SolidHarmonics`). The per-term scale — the general
`(4π)^(n_spin_slots/2)` carried by `DecoratedTerm.scale`, which reduces to
`(4π)^(body/2)` on the frozen pure-spin surface — is applied exactly once,
in the `TiledHamiltonian` constructor. Row numbering is **not** invented here: it is
`SLCE.row_layout(model)`, so a pure-spin model's row tables are the pre-M4 ones. The MC core is geometry-free (integer site
topology only); geometry helpers take an explicit `Crystal`.

## Public API

Exported: `KB_EV`, `TiledHamiltonian`, `n_sites`, `total_energy`,
`set_coefficients!`, `Observable`,
`Evaluable`, `ObservableStat`, `standard_observables`, `standard_evaluables`,
`run_mc`, `MCResult`, `TempResult`, `run_pt`, `PTResult`, `minimize_energy`,
`find_ground_state`, `GroundStateResult`, `resume`, `supercell_crystal`,
`ReducedCell`, `reduce_cell`, and (since 2026-07-19, after the A100 GO and the
l02/l044 production validations) the GPU sweep API: `GPUTiledHamiltonian`,
`GPUChainState`, `gpu_metropolis_sweep!`, `gpu_displacement_sweep!`,
`gpu_run_sweeps!`, `to_host!`, and `sync_coefficients!` (the `sent_w`-only device re-upload after a host coefficient swap).

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
tangent-projected SLCE gradient — the GPU field entry point for dependent
packages' dynamics; bitwise-gated against its lane reference, G7),
`StrainSchedule`'s accessors (`strain_domain`, `in_strain_domain`,
`strain_coefficients[!]`, `strain_j0`, `strain_volume`), the NPT energy contract
`strain_delta_energy`, the move pair `StrainScratch` / `strain_move!`, the
§8(ζ) mechanical-equilibrium diagnostic (`energy_volume_derivative` — exact
`dE_total/dV` via differentiated Horner + Euler's theorem on the displacement
degrees — and `pressure_diagnostics`, the `:pressure` evaluable whose mean must
equal the applied pressure on an equilibrated NPT chain), the per-measurement
`has_strain` / `strain` view accessors, and `GPA_PER_EV_A3`
(GPa per eV/Å³ — the pressure analogue of `KB_EV`, applied exactly once at
keyword resolution).

## Design record index

- `docs/specs/hamiltonian-tiling.md` — supercell unfolding, CSR memory layout
- `docs/specs/updates-stationarity.md` — Metropolis/OR stationarity, adaptive-step freeze
- `docs/specs/binning-observables.md` — C/χ/U conventions (authoritative), log-binning, jackknife
- `docs/specs/pt-threads-determinism.md` — lane/RNG discipline, bit-reproducibility
- `docs/specs/checkpoint-schema.md` — JLD2 schema v4
- `docs/specs/cell-reduction.md` — verified reduction to a user-chosen smaller cell
- `docs/specs/ground-state-search.md` — on-sphere descent, thermal cycling, multi-start determinism
- `docs/specs/gpu-feasibility.md` — GPU-port assessment: strategy, measured baseline, go/no-go
- `docs/specs/gpu-prototype.md` — GPU Metropolis prototype: keyed RNG layout, determinism contract, kernel shape, A100 readout
- `docs/specs/strain-move.md` — the outer NPT strain move: energy contract, proposal arms, measured Jacobian exponent, driver wiring, schema v4
