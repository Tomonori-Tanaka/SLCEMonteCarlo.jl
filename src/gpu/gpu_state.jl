# Device-resident chain state for the GPU Metropolis path (decision record
# docs/specs/gpu-prototype.md G1/G2/G8). Deliberately NOT derived from a `ChainState`
# RNG: the GPU path draws from the keyed Philox stream (seed, site, sweep) — a
# different Markov chain than any CPU run (the P6-breaking line in the CHANGELOG).

"""
    GPUChainState

Mutable chain state on a KernelAbstractions backend: device `config` (vector of
unit `SVector{3,Float64}`), `disps` (Cartesian displacements, model length units —
all-zero and never read on a pure-spin Hamiltonian), `zrows` (the full
`nrows × n_sites` basis-row table: the tesseral SPIN block and, on a joint model,
the displacement blocks), per-site `dE` staging and `acc` accept flags (Int32 0/1 —
no atomics anywhere), plus host-side bookkeeping: incremental `energy` (model units,
`j0` excluded), the keyed-RNG `seed`, the number of completed sweeps `sweep_index`
(the RNG counter word — every sweep's draws are a pure function of `(seed, site,
sweep_index + 1)`), the two fixed proposal widths `step` (radians) and `step_u`
(length; no on-device adaptation in this prototype), acceptance counters per move
type, and preallocated host staging buffers for the per-sweep `dE`/`acc` copy-back.

`step_u`, `acc_disp` and `att_disp` are carried but **not yet used**: no device sweep
moves the displacements until G8 phase 3, so `acc_disp / att_disp` is `0/0` on every
run this version can produce.
"""
mutable struct GPUChainState{VC<:AbstractVector{SVector{3,Float64}},
                             MF<:AbstractMatrix{Float64},
                             VF<:AbstractVector{Float64},VI<:AbstractVector{Int32}}
    const config::VC
    const disps::VC
    const zrows::MF
    const dE::VF
    const acc::VI
    energy::Float64
    const seed::UInt64
    sweep_index::Int
    step::Float64
    step_u::Float64
    acc_metro::Int
    att_metro::Int
    acc_disp::Int
    att_disp::Int
    const h_dE::Vector{Float64}
    const h_acc::Vector{Int32}
end

"""
    GPUChainState(gH::GPUTiledHamiltonian, st::ChainState;
                  seed::Integer = rand(UInt64)) -> GPUChainState

Upload `st`'s configuration, displacements, basis rows, energy, and proposal widths
to `gH.backend`. `seed` keys the Philox stream (recorded in the state; reuse it to
reproduce the device trajectory bitwise on the same backend).
"""
function GPUChainState(gH::GPUTiledHamiltonian, st::ChainState;
                       seed::Integer = rand(UInt64))
    backend = gH.backend
    n = n_sites(gH)
    config = KernelAbstractions.allocate(backend, SVector{3,Float64}, n)
    copyto!(config, st.config)
    disps = KernelAbstractions.allocate(backend, SVector{3,Float64}, n)
    copyto!(disps, st.disps)
    zrows = KernelAbstractions.allocate(backend, Float64, gH.host.nrows, n)
    copyto!(zrows, st.zrows)
    dE = KernelAbstractions.zeros(backend, Float64, n)
    acc = KernelAbstractions.zeros(backend, Int32, n)
    return GPUChainState(config, disps, zrows, dE, acc, st.energy, UInt64(seed), 0,
                         st.step, st.step_u, 0, 0, 0, 0,
                         zeros(Float64, n), zeros(Int32, n))
end

"""
    to_host!(st::ChainState, gst::GPUChainState) -> ChainState

Copy the device configuration, displacements, basis rows, and incremental energy
back into `st` — after which every host facility (renormalization, `_total_energy`,
observables, checkpointing) applies unchanged. `st`'s RNGs, proposal widths, and
counters are untouched (the device chain does not use them).
"""
function to_host!(st::ChainState, gst::GPUChainState)::ChainState
    copyto!(st.config, gst.config)
    copyto!(st.disps, gst.disps)
    copyto!(st.zrows, gst.zrows)
    st.energy = gst.energy
    return st
end

# Inverse of `to_host!` for the renormalization round-trip: push the (host-
# renormalized) config / disps / zrows / energy back to the device without touching
# the keyed-RNG bookkeeping.
function _from_host!(gst::GPUChainState, st::ChainState)::GPUChainState
    copyto!(gst.config, st.config)
    copyto!(gst.disps, st.disps)
    copyto!(gst.zrows, st.zrows)
    gst.energy = st.energy
    return gst
end
