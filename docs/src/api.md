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
SpinConfig
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

## Chain internals

```@docs
ChainState
SweepScratch
metropolis_sweep!
overrelaxation_sweep!
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
gpu_run_sweeps!
to_host!
```

## Observables

```@docs
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

```@docs
KB_EV
resolve_kt
```
