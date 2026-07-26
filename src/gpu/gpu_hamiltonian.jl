# Device-resident Hamiltonian tables — the subset of `TiledHamiltonian` the fused
# Metropolis kernel walks (docs/specs/gpu-prototype.md G4). Skipped deliberately:
# `terms` (reference kernels only), the energy programs (total energy stays on the
# host), `site_slot` (not read by `site_coeffs!`), `site_has_l1` (overrelaxation
# only), `site_active` (the coloring already excludes inactive sites), and the
# geometry scalars.

"""
    _GPUTables

The flat CSR tables of one [`TiledHamiltonian`](@ref) as backend arrays, in the
exact layout `site_coeffs!` walks (see `hamiltonian.jl` for field semantics).
`Adapt.@adapt_structure` makes the whole struct kernel-passable on any
KernelAbstractions backend.
"""
struct _GPUTables{VI<:AbstractVector{Int32},VB<:AbstractVector{Int8},
                  VF<:AbstractVector{Float64}}
    # per-site adjacency and instance membership (general path)
    site_ptr::VI
    site_inst::VI
    inst_ptr::VI
    inst_sites::VI
    # site contraction programs (pair / triplet / general)
    site_prog::VI
    sprog_ptr::VI
    sent_w::VF
    sent_tgt::VI
    sfac_ptr::VI
    sfac_row::VI
    sfac_slot::VB
    site_col::VI
    site_col2::VI
    pent_row::VI
    pent_row2::VI
    # coloring (class-major site list; the per-color ranges live on the host)
    color_sites::VI
end

Adapt.@adapt_structure _GPUTables

"""
    _to_device(backend, x::Array) -> backend array

Allocate on `backend` and copy `x` (identity-cost on the CPU backend).
"""
function _to_device(backend::Backend, x::Array{T}) where {T}
    d = KernelAbstractions.allocate(backend, T, size(x)...)
    copyto!(d, x)
    return d
end

"""
    GPUTiledHamiltonian(backend, H::TiledHamiltonian) -> GPUTiledHamiltonian

Upload the Metropolis-kernel tables of `H` to `backend` (a
`KernelAbstractions.Backend` — `CPU()` for the host-array reference backend, or
e.g. `CUDABackend()` supplied by the caller; the package itself never references
a GPU runtime). Keeps `H` alongside for host-side bookkeeping (`color_ptr`
launch ranges, the fixed-order ΔE reduction, renormalization, total energy).
"""
struct GPUTiledHamiltonian{B<:Backend,D<:_GPUTables}
    backend::B
    host::TiledHamiltonian
    dev::D

    function GPUTiledHamiltonian(backend::Backend, H::TiledHamiltonian)
        _require_spin_only(H, "the GPU device path")
        H.lmax <= 6 || throw(ArgumentError(
            "lmax = $(H.lmax) unsupported on the device path (gated ≤ 6)"))
        pr = H.progs
        dev = _GPUTables(_to_device(backend, H.site_ptr),
                         _to_device(backend, H.site_inst),
                         _to_device(backend, H.inst_ptr),
                         _to_device(backend, H.inst_sites),
                         _to_device(backend, pr.site_prog),
                         _to_device(backend, pr.sprog_ptr),
                         _to_device(backend, pr.sent_w),
                         _to_device(backend, pr.sent_tgt),
                         _to_device(backend, pr.sfac_ptr),
                         _to_device(backend, pr.sfac_row),
                         _to_device(backend, pr.sfac_slot),
                         _to_device(backend, pr.site_col),
                         _to_device(backend, pr.site_col2),
                         _to_device(backend, pr.pent_row),
                         _to_device(backend, pr.pent_row2),
                         _to_device(backend, H.color_sites))
        return new{typeof(backend),typeof(dev)}(backend, H, dev)
    end
end

"""
    n_sites(gH::GPUTiledHamiltonian) -> Int

Number of supercell sites of the wrapped [`TiledHamiltonian`](@ref).
"""
n_sites(gH::GPUTiledHamiltonian)::Int = gH.host.n_sites
