# SLCEMonteCarlo.jl

Classical spin Monte Carlo for fitted SCE (symmetry-adapted cluster expansion) models
from `SLCE.jl`: tile the fitted training-cell Hamiltonian
onto an `N₁ × N₂ × N₃` supercell and sample it with single-spin Metropolis (adaptive
step) and overrelaxation sweeps — single temperature, annealing sweeps, or replica
exchange over threads — with composable observables, autocorrelation-aware binning
errors, and bit-reproducible checkpoint/restart.

## Where it sits in the SCE family

| Package | Role |
|---|---|
| `SLCE.jl` | fits the SCE model (the input here) |
| `SLCETools.jl` | single-training-cell *configuration sampling* (mean-field + light MC) |
| **`SLCEMonteCarlo.jl`** | full supercell MC: observables ``E, C, |m|, χ, U``, annealing, parallel tempering |

The fitted model is read **only** through `SLCE`'s public introspection surface
(`multipole_terms`, `n_atoms`, `intercept`, `SLCE.Harmonics`); the per-term
``(4π)^{N/2}`` scale is applied exactly once, in the `TiledHamiltonian` constructor.

## Temperature convention

Absolute temperatures under exactly one of two keywords, everywhere:

- `temperature` — kelvin, converted internally with [`KB_EV`](@ref)
  (assumes an eV-fitted model, the package convention for DFT-fitted models);
- `kT` — ``k_B T`` directly in the model's energy units (theory / test runs).

## Reading order

[Getting started](getting_started.md) → the guides ([running](guide/running.md),
[parallel tempering](guide/parallel_tempering.md),
[ground states](guide/ground_states.md), [parallelism](guide/parallelism.md),
[observables](guide/observables.md), [checkpointing](guide/checkpointing.md)) →
the theory notes ([updates](theory/updates.md), [binning](theory/binning.md)).
Design decision records live in `docs/specs/` inside the repository.
