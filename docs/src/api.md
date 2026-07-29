# API reference

```@meta
CurrentModule = SLCEMonteCarlo
```

## Module

```@docs
SLCEMonteCarlo
```

## Hamiltonian

```@docs
TiledHamiltonian
n_sites
site_index
site_atom
ScaledTerm
TermSlot
has_disp(::TiledHamiltonian)
set_coefficients!
SpinConfig
```

## Volume grids (K(ε))

The sampler's form of a `SLCE.StrainedModels` grid: polynomial coefficients over the term
list a [`TiledHamiltonian`](@ref) was built from, converted once so that installing the
coefficients for a new cell volume is a Horner pass plus an in-place
[`set_coefficients!`](@ref) rather than a model rebuild.

Build the Hamiltonian with `keep_zero_terms = true` — both introspection surfaces upstream
prune exactly-zero coefficients by default, which makes their index → SALC map a function
of the fit rather than of the basis, and two grid points whose sparse fits zero different
keys would otherwise produce term lists of equal length with shifted maps.

```@docs
StrainSchedule
strain_domain
strain_coefficients
strain_coefficients!
strain_j0
strain_volume
strain_delta_energy
StrainScratch
strain_move!
```

## Cell reduction

```@docs
reduce_cell
ReducedCell
```

## Energy contract

```@docs
total_energy
site_coeffs!
delta_energy
site_gradient
```

## Running

```@docs
run_mc
MCResult
TempResult
run_pt
PTResult
resume
```

## Ground-state search

```@docs
minimize_energy
find_ground_state
GroundStateResult
```

## Harmonic stability

```@docs
force_constant_matrix
harmonic_stability
```

## Chain internals

```@docs
ChainState
SweepScratch
metropolis_sweep!
overrelaxation_sweep!
displacement_sweep!
```

## GPU

The chain-level device sweep (see the [GPU guide](guide/gpu.md)). The gradient
tier (`SLCEMonteCarlo.gpu_energy_gradient!`, `SLCEMonteCarlo.GPUGradientScratch`,
`SLCEMonteCarlo.gpu_zlm_rows!`) is public but unexported — the inter-package seam
for dependent packages' GPU dynamics.

```@docs
GPUTiledHamiltonian
GPUChainState
gpu_metropolis_sweep!
gpu_displacement_sweep!
gpu_run_sweeps!
to_host!
sync_coefficients!
```

## Observables

```@docs
MCView
has_strain
strain
Observable
Evaluable
ObservableStat
standard_observables
standard_evaluables
```

## Binning

```@docs
LogBinner
std_error
tau_int
BinStore
bin_means
jackknife
```

## Geometry / I/O

```@docs
supercell_crystal
to_matrix
from_matrix
```

## Units

`KB_EV` and `resolve_kt` are **defined in `SLCE`** and re-exported here unchanged —
one definition of the kelvin ↔ model-energy conversion for the whole family, so two
copies cannot drift apart. They are documented in
[SLCE.jl's API reference](https://tomonori-tanaka.github.io/SLCE.jl/dev/api/#Units).

| name | what it is |
|:--|:--|
| `KB_EV` | Boltzmann's constant in eV/K, the exact CODATA ratio `1.380649e-23 / 1.602176634e-19`. `kT = KB_EV * temperature`, assuming an **eV-fitted** model. |
| `SLCEMonteCarlo.resolve_kt(temperature, kT)` | resolves exactly one of the two controls — scalar or collection — into a validated `k_B·T` vector in the model's energy units. |
