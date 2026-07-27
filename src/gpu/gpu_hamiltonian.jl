# Device-resident Hamiltonian tables — the subset of `TiledHamiltonian` the fused
# Metropolis kernel walks (docs/specs/gpu-prototype.md G4/G8). Skipped deliberately:
# `terms` (reference kernels only), the energy programs (total energy stays on the
# host), `site_slot` (not read by `site_coeffs!`), `site_has_l1` (overrelaxation
# only), `site_active` (superseded by the per-channel color lists below), and the
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
    # coloring, SPLIT BY CHANNEL (class-major site lists; the per-color ranges live
    # on the host). One coloring, two schedules: on a joint model a site may be
    # active in one channel only — a force-constant-only ligand carries no spin — and
    # a sweep that visited it anyway would be always-accepted noise, exactly as on the
    # host (`metropolis_sweep!`/`displacement_sweep!` skip on `site_has_spin` /
    # `site_has_disp`). On a pure-spin Hamiltonian `spin_sites == H.color_sites`
    # verbatim and `disp_sites` is empty.
    spin_sites::VI
    disp_sites::VI
    # The `RowLayout` displacement blocks, flattened (`_disp_layout_tables`); empty on
    # a pure-spin layout. Uploaded but read by no kernel yet — the displacement sweep
    # is G8 phase 3; only the shape assertions of the joint testset cover them.
    fac_k::VI
    fac_l::VI
    fac_start::VI
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

# One channel's slice of the coloring: the color classes in order, each restricted to
# the sites `keep` flags. `keep === H.site_active` reproduces `color_ptr`/`color_sites`
# exactly, and so does `H.site_has_spin` on a pure-spin Hamiltonian (there the two
# predicates coincide) — which is what keeps the pure-spin launch schedule, and with it
# every pre-M4 bitwise gate, unchanged.
function _channel_colors(H::TiledHamiltonian, keep::Vector{Bool})
    ptr = Vector{Int32}(undef, H.n_colors + 1)
    sites = Int32[]
    sizehint!(sites, count(keep))       # `_metropolis_sweep_keyed_ref!` rebuilds the
    ptr[1] = Int32(1)                   #   tables once per sweep (test tier)
    for c = 1:H.n_colors
        for q = Int(H.color_ptr[c]):(Int(H.color_ptr[c + 1]) - 1)
            s = Int(H.color_sites[q])
            keep[s] && push!(sites, Int32(s))
        end
        ptr[c + 1] = Int32(length(sites) + 1)
    end
    return ptr, sites
end

# The host arrays of `_GPUTables`, in field order, plus the two per-color launch
# ranges. The ONE place the table list is written down: the device upload and the
# host-array reference tables (`_host_tables`, used by `_metropolis_sweep_keyed_ref!`
# and the walk tests) both build from it, so a new field cannot land on one side only.
function _table_arrays(H::TiledHamiltonian)
    pr = H.progs
    spin_ptr, spin_sites = _channel_colors(H, H.site_has_spin)
    disp_ptr, disp_sites = _channel_colors(H, H.site_has_disp)
    fac_k, fac_l, fac_start = _disp_layout_tables(H.layout)
    arrays = (H.site_ptr, H.site_inst, H.inst_ptr, H.inst_sites,
              pr.site_prog, pr.sprog_ptr, pr.sent_w, pr.sent_tgt, pr.sfac_ptr,
              pr.sfac_row, pr.sfac_slot, pr.site_col, pr.site_col2,
              pr.pent_row, pr.pent_row2,
              spin_sites, disp_sites, fac_k, fac_l, fac_start)
    return arrays, spin_ptr, disp_ptr
end

# `_GPUTables` over `H`'s own host arrays — the reference/test side of the same
# tables, with no copy.
_host_tables(H::TiledHamiltonian) = _GPUTables(_table_arrays(H)[1]...)

"""
    GPUTiledHamiltonian(backend, H::TiledHamiltonian) -> GPUTiledHamiltonian

Upload the Metropolis-kernel tables of `H` to `backend` (a
`KernelAbstractions.Backend` — `CPU()` for the host-array reference backend, or
e.g. `CUDABackend()` supplied by the caller; the package itself never references
a GPU runtime). Keeps `H` alongside for host-side bookkeeping (the per-color
launch ranges, the fixed-order ΔE reduction, renormalization, total energy).

Joint (spin + displacement) Hamiltonians are accepted: the tables carry both
channels' color lists and the layout's displacement blocks. The **gradient** entry
points remain spin-only (`gpu_energy_gradient!` throws on a joint `H`).
"""
struct GPUTiledHamiltonian{B<:Backend,D<:_GPUTables}
    backend::B
    host::TiledHamiltonian
    dev::D
    spin_ptr::Vector{Int32}          # color c: dev.spin_sites[ptr[c]:ptr[c+1]-1]
    disp_ptr::Vector{Int32}          # color c: dev.disp_sites[ptr[c]:ptr[c+1]-1]

    function GPUTiledHamiltonian(backend::Backend, H::TiledHamiltonian)
        H.lmax <= 6 || throw(ArgumentError(
            "lmax = $(H.lmax) unsupported on the device path (gated ≤ 6)"))
        # Gated here rather than at the device displacement row, because THIS is where
        # the layout's displacement blocks are uploaded: a table the device row cannot
        # consume has no business on the device.
        H.disp_lmax <= 6 || throw(ArgumentError(
            "disp_lmax = $(H.disp_lmax) unsupported on the device path (gated ≤ 6)"))
        # The kernel spells the SPIN block width as `(LMAX + 1)²` — the `@localmem`
        # trial row, the ΔE row range, and the write-back extent all — while the host
        # says `H.nlm`. `row_layout` makes those equal, but `TiledHamiltonian` accepts a
        # caller-supplied `RowLayout` and only checks `disp_offset` against the slot
        # rows, so a padded one would give a truncated device ΔE and a partial write
        # into the displacement block. Assert the identity where the kernel is built.
        H.nlm == (H.lmax + 1)^2 || throw(ArgumentError(
            "this Hamiltonian's layout has a SPIN block of $(H.nlm) rows but " *
            "lmax = $(H.lmax), i.e. $((H.lmax + 1)^2) tesseral rows; the device " *
            "kernel derives the block width from lmax and cannot address a padded " *
            "layout"))
        arrays, spin_ptr, disp_ptr = _table_arrays(H)
        dev = _GPUTables(map(a -> _to_device(backend, a), arrays)...)
        return new{typeof(backend),typeof(dev)}(backend, H, dev, spin_ptr, disp_ptr)
    end
end

"""
    n_sites(gH::GPUTiledHamiltonian) -> Int

Number of supercell sites of the wrapped [`TiledHamiltonian`](@ref).
"""
n_sites(gH::GPUTiledHamiltonian)::Int = gH.host.n_sites
