# SLCEMonteCarlo.jl

Classical Monte Carlo for fitted SLCE (spin–lattice cluster expansion) models from
`SLCE.jl`: tile the fitted training-cell Hamiltonian onto an ``N_1 × N_2 × N_3``
supercell and sample it with the exact ``ΔE`` of the fitted model at any body
order — single-spin Metropolis (adaptive step) and overrelaxation sweeps, atomic
displacement sweeps on a joint model, and an outer isothermal–isobaric strain move
when the cell volume is sampled too. Runs come as a single temperature, an
annealing ladder, or replica exchange over threads, with composable observables,
autocorrelation-aware binning errors, and bit-reproducible checkpoint/restart.

## What can be sampled

| field | move | guide |
|---|---|---|
| spin directions | Metropolis rotation + antipodal flip; overrelaxation | [running chains](guide/running.md) |
| atomic displacements | Gaussian shift per displacement-active site | [joint spin–lattice models](guide/joint.md) |
| cell volume (NPT) | outer strain move over a `SLCE.StrainedModels` grid | [NPT and strain moves](guide/npt.md) |

Ground states are found numerically with `minimize_energy` (deterministic
on-sphere gradient descent) and `find_ground_state` (multi-start annealing +
polish). A chain-level device (GPU) sweep sits underneath the CPU drivers.

## Where it sits in the SLCE family

| Package | Role |
|---|---|
| `SLCE.jl` | fits the SLCE model (the input here) |
| `SLCETools.jl` | single-training-cell *configuration sampling* (mean-field + light MC) |
| **`SLCEMonteCarlo.jl`** | full supercell MC: observables ``E, C, |m|, χ, U``, displacements, NPT, annealing, parallel tempering |
| `SLCEDynamics.jl` | spin dynamics (LLG) on the same fitted models |

The fitted model is read **only** through `SLCE`'s public introspection surface
(`decorated_terms`, `row_layout`, `restrict`, `spin_multipole_terms`, `n_atoms`,
`intercept`, `SLCE.Harmonics`); the per-term ``(4π)^{n_\mathrm{spin\,slots}/2}``
scale — which reduces to ``(4π)^{N/2}`` on the pure-spin surface — is applied
exactly once, in the `TiledHamiltonian` constructor, and is never re-derived from
the cluster shape.

## Temperature convention

Absolute temperatures under exactly one of two keywords, everywhere:

- `temperature` — kelvin, converted internally with `KB_EV`
  (assumes an eV-fitted model, the package convention for DFT-fitted models);
- `kT` — ``k_B T`` directly in the model's energy units (theory / test runs).

Pressure follows the same discipline: exactly one of `pressure_GPa` or `pressure`
(eV/Å³), never a shared keyword.

## Reading order

[Getting started](getting_started.md) → the [cubic Heisenberg
tutorial](tutorials/cubic_heisenberg.md) → the guides, in the order they appear in
the sidebar: the chain mechanics ([running](guide/running.md)), the two extra
sampled fields ([joint models](guide/joint.md), [NPT](guide/npt.md)), then the
sampling strategies ([parallel tempering](guide/parallel_tempering.md), [ground
states](guide/ground_states.md)), what to measure and how to survive a walltime
kill ([observables](guide/observables.md),
[checkpointing](guide/checkpointing.md)), and finally the execution layers
([parallelism](guide/parallelism.md), [GPU](guide/gpu.md)) → the theory notes
([updates](theory/updates.md), [binning](theory/binning.md)).

Design decision records live in `docs/specs/` inside the repository.
