"""
    SLCEMonteCarlo

Classical spin Monte Carlo for fitted SLCE (spin–lattice cluster expansion) models
from `SLCE.jl`: tile the fitted training-cell Hamiltonian onto an
`N₁ × N₂ × N₃` supercell ([`TiledHamiltonian`](@ref)) — optionally after a verified
re-expression in a user-chosen smaller cell ([`reduce_cell`](@ref)) — and sample it
with single-spin
Metropolis (adaptive step) and overrelaxation sweeps — single temperature, annealing
sweeps ([`run_mc`](@ref)), or replica exchange over threads ([`run_pt`](@ref)) — with
composable observables, autocorrelation-aware binning errors, and bit-reproducible
checkpoint/restart ([`resume`](@ref)) — reproducibility meaning: deterministic for
a fixed seed within one package + Julia version, independent of the thread count
(the scope is `docs/specs/pt-threads-determinism.md` P6). Ground states are found
numerically with
[`minimize_energy`](@ref) (deterministic on-sphere gradient descent) and
[`find_ground_state`](@ref) (multi-start annealing + polish).

The fitted model is read **only** through `SLCE`'s public introspection surface
(`decorated_terms`, `row_layout`, `restrict`, `spin_multipole_terms`, `n_atoms`,
`intercept`, `SLCE.Harmonics`); the per-term scale — the general
`(4π)^(n_spin_slots/2)` a `DecoratedTerm` carries as a field, which reduces to
`(4π)^(body/2)` on the frozen pure-spin surface — is applied exactly once, in the
[`TiledHamiltonian`](@ref) constructor, and is never re-derived from the cluster shape.
Temperatures are absolute, under exactly one of two keywords:
`temperature` [kelvin, converted with `KB_EV`] or `kT` [model energy units].
"""
module SLCEMonteCarlo

using Adapt: Adapt
using JLD2: jldopen
using KernelAbstractions: KernelAbstractions, @kernel, @index, @localmem,
                          @synchronize, @groupsize, @Const, Backend
using LinearAlgebra
using Printf: @sprintf, @printf
using Random: Random, AbstractRNG, Xoshiro
using StaticArrays
using Statistics: Statistics, mean

using SLCE: SLCE, SLCEModel, SpinMultipoleTerm, spin_multipole_terms, intercept,
                  DecoratedTerm, decorated_terms, restrict,
                  RowLayout, row_layout, row_index,
                  Lattice, Crystal, cartesian_positions
# Extended here rather than redefined: `n_atoms` for `ReducedCell`, `has_disp` for
# `TiledHamiltonian`. Both ask the core's question at this package's granularity — a
# second generic of the same name would make `SLCE.has_disp` and
# `SLCEMonteCarlo.has_disp` different functions for a user who loads both.
import SLCE: n_atoms, has_disp
import SLCE.Harmonics
import SLCE.SolidHarmonics

include("units.jl")
include("hamiltonian.jl")
include("coefficients.jl")
# The sampler's form of a K(ε) volume grid: polynomial coefficients over the term list a
# Hamiltonian was built from, converted once so a strain move costs a Horner pass and an
# in-place rewrite rather than a model rebuild.
include("strain.jl")
include("energy.jl")
include("binning.jl")
include("observables.jl")
include("state.jl")
include("updates.jl")
include("gpu/philox.jl")
include("gpu/zlm_device.jl")
include("gpu/disp_device.jl")
include("gpu/grad_device.jl")
include("gpu/gpu_hamiltonian.jl")
include("gpu/gpu_state.jl")
include("gpu/gpu_sweep.jl")
include("gpu/gpu_gradient.jl")
include("minimize.jl")
include("stability.jl")
include("run.jl")
include("pt.jl")
include("checkpoint.jl")
include("geometry.jl")
include("reduce.jl")

export KB_EV
export TiledHamiltonian, n_sites, total_energy, set_coefficients!
# K(ε) volume grids, sampler side (the move itself lands with the driver)
export StrainSchedule
public strain_domain, in_strain_domain, strain_coefficients, strain_coefficients!,
       strain_j0, strain_volume
public has_strain, strain
export MCView, Observable, Evaluable, ObservableStat, standard_observables,
       standard_evaluables
export run_mc, MCResult, TempResult
export run_pt, PTResult
export minimize_energy, find_ground_state, GroundStateResult
export force_constant_matrix, harmonic_stability
export resume
export supercell_crystal
export ReducedCell, reduce_cell

public resolve_kt
public ScaledTerm, TermSlot, SpinConfig, site_index, site_atom, has_disp
public site_coeffs!, delta_energy, site_gradient, energy_gradient, energy_gradient!
public philox_block, philox_normal2
public model_fingerprint
public LogBinner, BinStore, jackknife, std_error, tau_int, bin_means
public ChainState, SweepScratch, metropolis_sweep!, overrelaxation_sweep!,
       displacement_sweep!
public to_matrix, from_matrix
# GPU sweep API (exported 2026-07-19: A100 GO 30.1x + l02/l044 production
# validation landed; the gradient tier below stays public-unexported — it is the
# inter-package seam consumed by SLCEDynamics, not an end-user surface)
export GPUTiledHamiltonian, GPUChainState, gpu_metropolis_sweep!,
       gpu_displacement_sweep!, gpu_run_sweeps!, to_host!
public GPUGradientScratch, gpu_energy_gradient!, gpu_zlm_rows!

end # module SLCEMonteCarlo
