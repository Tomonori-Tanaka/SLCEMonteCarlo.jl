# SLCEMonteCarlo.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://tomonori-tanaka.github.io/SLCEMonteCarlo.jl/dev/)
[![CI](https://github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl/actions/workflows/CI.yml)

**Documentation:** <https://tomonori-tanaka.github.io/SLCEMonteCarlo.jl/dev/>

Classical Monte Carlo for fitted SLCE (spin–lattice cluster expansion)
models from [SLCE.jl](https://github.com/Tomonori-Tanaka/SLCE.jl) — spins, atomic
displacements, and the cell volume.

- **Supercell tiling** — replicate the fitted training-cell Hamiltonian onto an
  `N₁ × N₂ × N₃` supercell from SLCE's public introspection surface alone
  (`decorated_terms` / `row_layout` / `restrict`).
- **Cell reduction** — re-express a supercell-fitted model in a user-chosen smaller
  cell (`reduce_cell`, verified — structure and couplings must actually have that
  periodicity), so MC sizes are not locked to training-cell multiples.
- **Updates** — single-spin Metropolis with an adaptive proposal step, plus
  overrelaxation sweeps (involutive reflection + Metropolis correction, exact for
  any body order).
- **Joint spin–lattice models** — displacement sweeps with their own adaptive
  width, a centre-of-mass-free frame, a harmonic screen (`harmonic_stability`)
  and a run-time detector for the truncated expansion's unbounded directions.
- **NPT** — an outer strain move over a `SLCE.StrainedModels` volume grid
  (`StrainSchedule`), hydrostatic, with the mechanical-equilibrium check
  (`pressure_diagnostics`) and the isobaric `W` observables (`npt_observables`).
- **Runs** — single temperature, warm-started annealing sweeps (`run_mc`), and
  replica exchange over threads (`run_pt`, NPT-aware), bit-reproducible for a
  fixed seed (within one package + Julia version, independent of the thread
  count — this is a testing discipline, not a cross-version guarantee; see
  `docs/specs/pt-threads-determinism.md` P6).
- **Ground states** — deterministic on-sphere gradient descent
  (`minimize_energy`) and multi-start annealing + polish with optional thermal
  cycling (`find_ground_state`), threads-parallel and seed-reproducible.
- **Observables** — energy, specific heat, `|m|`, susceptibility, Binder cumulant,
  per-sublattice magnetization, mean-square displacements per sublattice
  (the Debye–Waller input), the isobaric enthalpy and heat capacity, and
  user-defined observables/evaluables, with autocorrelation-aware log-binning
  errors and jackknifed derived quantities.
- **Checkpoint/restart** — versioned JLD2 schema; a resumed run is bit-identical
  to an uninterrupted one.
- **GPU sweeps** — chain-level device Metropolis sweeps, spin and displacement,
  on any KernelAbstractions backend (`GPUTiledHamiltonian` / `GPUChainState` /
  `gpu_run_sweeps!` / `to_host!`; A100-validated, 30.1× at the 8³ go/no-go bar
  and up to 38.1× on the heaviest production model). Metropolis only — no
  overrelaxation, no replica exchange, no strain move; `run_mc`/`run_pt` remain
  CPU drivers; no CUDA dependency (the caller passes the backend).

Temperatures are absolute, under exactly one of two keywords: `temperature`
(kelvin) or `kT` (the model's energy units); pressure follows the same rule,
under `pressure_GPa` or `pressure` (eV/Å³).

```julia
using SLCEMonteCarlo, SLCE

model = SLCE.load(SLCEModel, "model.toml")
H = TiledHamiltonian(model; dims = (4, 4, 4))

result = run_mc(H; temperature = [1200, 900, 600, 300],   # annealing ladder
                sweeps_therm = 2_000, sweeps_measure = 20_000, seed = 1)

pt = run_pt(H; temperature = range(200, 1400; length = 16), seed = 1)

gs = find_ground_state(H; nstarts = 16, seed = 1)   # ground-state search
gs = find_ground_state(H; inits = pt.final_configs, anneal_sweeps = 0)  # PT polish
```

Development status: v0. See `SPEC.md` and `docs/` for the architecture and
decision records.
