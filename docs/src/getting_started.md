# Getting started

```@meta
CurrentModule = SLCEMonteCarlo
```

## Install

The package lives alongside its model source `SLCE.jl`; during development
both are path-devs:

```julia
using Pkg
Pkg.develop(path = "path/to/SLCE.jl")
Pkg.develop(path = "path/to/SLCEMonteCarlo.jl")
```

## From a fitted model to observables

```julia
using SLCEMonteCarlo, SLCE

model = SLCE.load(SLCEModel, "model.toml")   # a fitted SCE
H = TiledHamiltonian(model; dims = (4, 4, 4))         # training cell → 4×4×4 supercell

result = run_mc(H; temperature = 300, seed = 1)       # kelvin (or kT = ... in eV)
result                                                 # summary table

p = result.points[1]
p.stats[:energy]           # ⟨E⟩ ± binning error, τ_int
p.stats[:specific_heat]    # jackknifed C/k_B per site
p.stats[:sublattice_m]     # per-sublattice magnetization vectors
```

The temperature rule (everywhere in the package): **exactly one** of

- `temperature` — kelvin, converted with [`KB_EV`](@ref) (assumes an eV-fitted
  model, the convention for DFT-fitted models), or
- `kT` — ``k_B T`` directly in the model's energy units (theory / test runs).

Both accept a scalar or a collection. Passing both, or `temperature = 0.02`
(meant as eV) by accident, is an error — the two units never share a keyword.

## Simulation cells finer-grained than the training cell

`dims` counts multiples of the cell the terms are expressed in — by default the
training cell. If the model was fitted on a supercell (say a 4×4×4 bcc conventional
cell, 128 atoms), that makes finite-size checks jump in ×4 steps. When the structure
and the fit actually have the periodicity of a smaller cell, [`reduce_cell`](@ref)
re-expresses the Hamiltonian in a cell **you** specify — after *verifying* that the
lattice relation is integer, the atoms map onto each other, and every fitted term has
its full set of translation copies (anything else is a hard error, never a silent
symmetrization):

```julia
red = reduce_cell(model, crystal, Matrix(crystal.lattice.vectors) / 4)  # 2-atom cube
H   = TiledHamiltonian(red; dims = (6, 6, 6))       # 432 sites — not a ×4 multiple
out = supercell_crystal(red.crystal, (6, 6, 6))     # matching geometry for I/O
```

The chosen cell need not be primitive (a bcc *conventional* cube under a
primitive-compatible model is fine), and non-diagonal relations between the two
cells are supported. Details and the verification contract:
`docs/specs/cell-reduction.md`.

## An annealing run and a parallel-tempering run

```julia
# warm-started ladder: high → low = annealing
ann = run_mc(H; temperature = [1200, 900, 600, 450, 300], seed = 1)

# replica exchange over threads (start Julia with -t N)
pt = run_pt(H; temperature = range(250, 1300; length = 16), seed = 1)
pt.swap_acceptance         # the ladder diagnostic — aim for O(0.2–0.5)
```

## A worked example with figures

The [cubic Heisenberg tutorial](tutorials/cubic_heisenberg.md) runs this whole
pipeline through a known transition (``k_BT_c/|J| \approx 1.443``) and plots the
energy, specific heat, magnetization, susceptibility, and Binder-cumulant crossing,
plus a user-defined staggered-magnetization observable on the antiferromagnetic
counterpart — all computed at docs-build time.

## Where things are defined

- Model → Hamiltonian: [`TiledHamiltonian`](@ref) (tiling, memory layout —
  `docs/specs/hamiltonian-tiling.md`).
- Runs: [`run_mc`](@ref) (single/annealing), [`run_pt`](@ref) (replica exchange),
  [`resume`](@ref) (checkpoint restart).
- Observables: [`Observable`](@ref) / [`Evaluable`](@ref) and the standard sets;
  conventions in `docs/specs/binning-observables.md`.
- Geometry for I/O: [`supercell_crystal`](@ref),
  [`to_matrix`](@ref) / [`from_matrix`](@ref).
