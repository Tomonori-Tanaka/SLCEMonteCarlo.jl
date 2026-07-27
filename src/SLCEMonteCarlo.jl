"""
    SLCEMonteCarlo

Classical spin Monte Carlo for fitted SCE (symmetry-adapted cluster expansion) models
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
(`decorated_terms`, `row_layout`, `restrict`, `multipole_terms`, `n_atoms`,
`intercept`, `SLCE.Harmonics`); the per-term scale — the general
`(4π)^(n_spin_slots/2)` a `DecoratedTerm` carries as a field, which reduces to
`(4π)^(body/2)` on the frozen pure-spin surface — is applied exactly once, in the
[`TiledHamiltonian`](@ref) constructor, and is never re-derived from the cluster shape.
Temperatures are absolute, under exactly one of two keywords:
`temperature` [kelvin, converted with [`KB_EV`](@ref)] or `kT` [model energy units].
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

using SLCE: SLCE, SLCEModel, MultipoleTerm, multipole_terms, intercept,
                  DecoratedTerm, decorated_terms, restrict,
                  RowLayout, row_layout, row_index,
                  Lattice, Crystal, cartesian_positions
import SLCE: n_atoms                  # extended for ReducedCell
import SLCE.Harmonics
import SLCE.SolidHarmonics

include("units.jl")
include("hamiltonian.jl")
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
export TiledHamiltonian, n_sites, total_energy
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
export GPUTiledHamiltonian, GPUChainState, gpu_metropolis_sweep!, gpu_run_sweeps!,
       to_host!
public GPUGradientScratch, gpu_energy_gradient!, gpu_zlm_rows!

end # module SLCEMonteCarlo
