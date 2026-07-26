# The device all-site SCE gradient (decision record docs/specs/gpu-prototype.md
# G7) — phase 2 of the GPU path, the entry point SLCEDynamics' GPU LLG
# consumes. One workgroup per site; lane 1 fills the site's gradient-row table
# (the `znew` analog — `∇Z(e_s)` depends only on `e_s`, fixed during the pass);
# all lanes walk the site's adjacency entries with the sweep's exact three-way
# dispatch and zero-skips, folding a 3-vector partial each; lane 1 does the
# lane-ordered fold. The pass is read-only in `config`/`zrows` and writes only
# `G[s]` per workgroup, so there is NO coloring — one launch over all sites.
#
# Determinism (G3): bitwise reproducible for fixed (backend, workgroupsize,
# package + Julia version); the whole pipeline (row + walk + fold) is `+ − * /`
# and correctly-rounded `sqrt` — no libm, no RNG — so device output is expected
# to match the serial lane reference bitwise on EVERY backend (verified on the
# CPU backend in CI; claimed and gated on CUDA in the A100 smoke, with the
# documented fallback of a scaled-tolerance gate should FMA contraction ever
# appear). Keep `muladd`/`@fastmath` out of `_grad_zlm_device` /
# `_entry_walk_grad` / the folds.
#
# COUPLED SITES: `_grad_kernel!` and `_gradient_lane_ref!` (bottom of this file)
# implement one arithmetic contract — the gradient-row table (`grad_device.jl`),
# `_entry_walk_grad`'s dispatch/skips (structurally `_entry_walk_partial` with
# the ΔE dot replaced by the ∇Z 3-vector), and the lane-ordered component fold.
# The bitwise gate in test/unit/test_gpu.jl compares them on the CPU backend;
# change either side and the other plus the G7 record move with it.
# `_gradient_lane_ref!` is also the reference SLCEDynamics' composite GPU-LLG
# gate calls by qualified name — renaming it is a cross-package break.

"""
    GPUGradientScratch(gH::GPUTiledHamiltonian) -> GPUGradientScratch

Device workspace for [`gpu_energy_gradient!`](@ref): the `nlm × n_sites`
tesseral-row matrix rebuilt from the configuration on every refresh. Allocate
once per run and reuse across calls (the layout is an implementation detail of
the gradient path — callers never read it).
"""
struct GPUGradientScratch{MF<:AbstractMatrix{Float64}}
    zrows::MF
end

GPUGradientScratch(gH::GPUTiledHamiltonian) =
    GPUGradientScratch(KernelAbstractions.allocate(gH.backend, Float64,
                                                   gH.host.nlm, gH.host.n_sites))

# One thread per site: rebuild the site's tesseral row from its spin (the value
# row is bitwise-identical to the host `_zlm_row!` by the G4 row identity, so a
# rebuilt matrix matches host `_zrows` bitwise).
@kernel function _zlm_rows_kernel!(zrows, @Const(config), ::Val{LMAX}) where {LMAX}
    s = @index(Global, Linear)
    @inbounds begin
        col = @view zrows[:, s]
        _zlm_row_device!(col, config[s], Val(LMAX))
    end
end

"""
    gpu_zlm_rows!(gsc::GPUGradientScratch, gH::GPUTiledHamiltonian, dconfig;
                  workgroupsize = 128, synchronize = true) -> gsc

Rebuild the scratch's tesseral rows from `dconfig` (all sites — the dynamics
mode, where every spin moved). Bitwise-identical to the host `_zrows`.
"""
function gpu_zlm_rows!(gsc::GPUGradientScratch, gH::GPUTiledHamiltonian, dconfig;
                       workgroupsize::Integer = 128, synchronize::Bool = true)
    H = gH.host
    n = H.n_sites
    length(dconfig) == n || throw(DimensionMismatch(
        "config has $(length(dconfig)) sites; the Hamiltonian has $n"))
    size(gsc.zrows) == (H.nlm, n) || throw(DimensionMismatch(
        "scratch zrows has size $(size(gsc.zrows)); expected ($(H.nlm), $n)"))
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    kern = _zlm_rows_kernel!(gH.backend, ws)
    # invokelatest: the same static-analysis launch barrier as the sweep path
    Base.invokelatest(kern, gsc.zrows, dconfig, Val(H.lmax); ndrange = n)
    synchronize && KernelAbstractions.synchronize(gH.backend)
    return gsc
end

# One lane's strided share of the gradient entry walk: structurally
# `_entry_walk_partial` (same three-way `site_col` dispatch, same `z == 0.0` /
# `p == 0.0` skips — part of the bitwise contract) with the ΔE dot product
# replaced by the ∇Z(e_s) 3-vector of the entry's target row.
@inline function _entry_walk_grad(tb::_GPUTables, zrows::AbstractMatrix{Float64},
                                  grow::AbstractVector{Float64}, s::Int,
                                  lane::Int, ws::Int)::SVector{3,Float64}
    a = SVector{3,Float64}(0.0, 0.0, 0.0)
    @inbounds for j = (Int(tb.site_ptr[s]) + lane - 1):ws:(Int(tb.site_ptr[s + 1]) - 1)
        pid = Int(tb.site_prog[j])
        col = Int(tb.site_col[j])
        if col > 0
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                z = zrows[Int(tb.pent_row[e]), col]
                z == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                w = tb.sent_w[e] * z
                a += SVector{3,Float64}(w * grow[3 * tgt - 2], w * grow[3 * tgt - 1],
                                        w * grow[3 * tgt])
            end
        elseif col < 0
            col2 = Int(tb.site_col2[j])
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                p = zrows[Int(tb.pent_row[e]), -col] * zrows[Int(tb.pent_row2[e]), col2]
                p == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                w = tb.sent_w[e] * p
                a += SVector{3,Float64}(w * grow[3 * tgt - 2], w * grow[3 * tgt - 1],
                                        w * grow[3 * tgt])
            end
        else
            off = Int(tb.inst_ptr[Int(tb.site_inst[j])]) - 1
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                p = 1.0
                for f = Int(tb.sfac_ptr[e]):(Int(tb.sfac_ptr[e + 1]) - 1)
                    m = Int(tb.inst_sites[off + tb.sfac_slot[f]])
                    p *= zrows[Int(tb.sfac_row[f]), m]
                end
                p == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                w = tb.sent_w[e] * p
                a += SVector{3,Float64}(w * grow[3 * tgt - 2], w * grow[3 * tgt - 1],
                                        w * grow[3 * tgt])
            end
        end
    end
    return a
end

# The all-site gradient kernel: workgroup g ↔ site g (no coloring — read-only
# pass, per-site disjoint writes). Same localmem-across-@synchronize discipline
# as `_metro_kernel!`.
@kernel function _grad_kernel!(G, @Const(config), @Const(zrows), @Const(tb),
                               ::Val{LMAX}) where {LMAX}
    grow = @localmem Float64 3 * (LMAX + 1) * (LMAX + 1)
    partials = @localmem Float64 3 * prod(@groupsize())

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        _grad_zlm_row_device!(grow, config[g], Val(LMAX))
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds begin
        # Int(g)/Int(lane): the CUDA backend's @index returns Int32 (the CPU
        # backend's returns Int) and the walk's signature is Int-typed
        a = _entry_walk_grad(tb, zrows, grow, Int(g), Int(lane),
                             prod(@groupsize()))
        partials[3 * Int(lane) - 2] = a[1]
        partials[3 * Int(lane) - 1] = a[2]
        partials[3 * Int(lane)] = a[3]
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for t = 1:prod(@groupsize())                 # lane-ordered fold (G4/G7)
            gx += partials[3 * t - 2]
            gy += partials[3 * t - 1]
            gz += partials[3 * t]
        end
        G[g] = SVector{3,Float64}(gx, gy, gz)
    end
end

"""
    gpu_energy_gradient!(dG, gH::GPUTiledHamiltonian, dconfig,
                         gsc::GPUGradientScratch;
                         workgroupsize = 128, refresh_zrows = true,
                         synchronize = true) -> dG

Device twin of [`energy_gradient!`](@ref): the all-site, tangent-projected SCE
gradient `dG[s] = ∂E/∂e_s` (`e·G = 0` to rounding, inactive sites exactly zero
— their adjacency range is empty — model energy units, `j0` excluded, exact at
any body order). `dG` and `dconfig` are caller-owned device vectors of
`SVector{3,Float64}` on `gH.backend`; `gsc` is allocated once per run.

`refresh_zrows = true` first rebuilds the scratch's tesseral rows from
`dconfig` — the mode a dynamics caller wants (every spin moved); pass `false`
only when the rows are already `Z(dconfig)` (the MC invariant — see the
`GPUChainState` convenience method). `synchronize = false` only enqueues (KA
queue order still serializes subsequent launches) — for callers chaining the
gradient into a longer per-step launch sequence.

Bitwise reproducible for fixed (backend, `workgroupsize`); the arithmetic
contract is pinned by `_gradient_lane_ref!` (test tier).
"""
function gpu_energy_gradient!(dG::AbstractVector{SVector{3,Float64}},
                              gH::GPUTiledHamiltonian,
                              dconfig::AbstractVector{SVector{3,Float64}},
                              gsc::GPUGradientScratch;
                              workgroupsize::Integer = 128,
                              refresh_zrows::Bool = true,
                              synchronize::Bool = true)
    H = gH.host
    n = H.n_sites
    length(dG) == n || throw(DimensionMismatch(
        "G has $(length(dG)) sites; the Hamiltonian has $n"))
    length(dconfig) == n || throw(DimensionMismatch(
        "config has $(length(dconfig)) sites; the Hamiltonian has $n"))
    size(gsc.zrows) == (H.nlm, n) || throw(DimensionMismatch(
        "scratch zrows has size $(size(gsc.zrows)); expected ($(H.nlm), $n)"))
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    refresh_zrows && gpu_zlm_rows!(gsc, gH, dconfig; workgroupsize = ws,
                                   synchronize = false)
    kern = _grad_kernel!(gH.backend, ws)
    Base.invokelatest(kern, dG, dconfig, gsc.zrows, gH.dev, Val(H.lmax);
                      ndrange = n * ws)
    synchronize && KernelAbstractions.synchronize(gH.backend)
    return dG
end

"""
    gpu_energy_gradient!(dG, gst::GPUChainState, gH::GPUTiledHamiltonian,
                         gsc::GPUGradientScratch; kwargs...) -> dG

MC-side convenience: the chain state's rows are already `Z(config)` (the sweep
invariant), so the rebuild is skipped and the sweep's own row matrix is read.
"""
gpu_energy_gradient!(dG::AbstractVector{SVector{3,Float64}}, gst::GPUChainState,
                     gH::GPUTiledHamiltonian, gsc::GPUGradientScratch;
                     kwargs...) =
    _gpu_gradient_rows!(dG, gH, gst.config, gst.zrows; kwargs...)

# Shared launch on an explicit row matrix (the chain-state method's body).
function _gpu_gradient_rows!(dG, gH::GPUTiledHamiltonian, dconfig, zrows;
                             workgroupsize::Integer = 128,
                             synchronize::Bool = true)
    H = gH.host
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    kern = _grad_kernel!(gH.backend, ws)
    Base.invokelatest(kern, dG, dconfig, zrows, gH.dev, Val(H.lmax);
                      ndrange = H.n_sites * ws)
    synchronize && KernelAbstractions.synchronize(gH.backend)
    return dG
end

# ---------------------------------------------------------------------------
# Reference implementation (the readable spec of the kernel's arithmetic, in the
# `_metropolis_sweep_keyed_ref!` tradition): a plain serial host pass sharing
# `_grad_zlm_row_device_dyn!` and `_entry_walk_grad` — same strided lane shares,
# same lane-ordered component fold — so the kernel must match it bitwise on the
# CPU backend (and, being libm-free, is expected to on CUDA too). Test-only on
# this side; SLCEDynamics' composite GPU-LLG gate calls it by qualified name.
# ---------------------------------------------------------------------------
function _gradient_lane_ref!(G::Vector{SVector{3,Float64}}, H::TiledHamiltonian,
                             config::SpinConfig, zrows::Matrix{Float64},
                             ws::Int)::Vector{SVector{3,Float64}}
    # Test-only, but `_gradient_lane_ref!` is called by qualified name from
    # SLCEDynamics' GPU-LLG composite gate, so it is a real cross-package entry
    # point: it sizes its buffers from `H.nlm`/`H.lmax` and would silently drop a
    # displacement channel rather than throw.
    _require_spin_only(H, "_gradient_lane_ref!")
    tb = _GPUTables(H.site_ptr, H.site_inst, H.inst_ptr, H.inst_sites,
                    H.progs.site_prog, H.progs.sprog_ptr, H.progs.sent_w,
                    H.progs.sent_tgt, H.progs.sfac_ptr, H.progs.sfac_row,
                    H.progs.sfac_slot, H.progs.site_col, H.progs.site_col2,
                    H.progs.pent_row, H.progs.pent_row2, H.color_sites)
    grow = Vector{Float64}(undef, 3 * H.nlm)
    partials = Vector{SVector{3,Float64}}(undef, ws)
    for s = 1:H.n_sites
        _grad_zlm_row_device_dyn!(grow, config[s], H.lmax)
        for lane = 1:ws
            partials[lane] = _entry_walk_grad(tb, zrows, grow, s, lane, ws)
        end
        gx = 0.0
        gy = 0.0
        gz = 0.0
        for t = 1:ws
            gx += partials[t][1]
            gy += partials[t][2]
            gz += partials[t][3]
        end
        G[s] = SVector{3,Float64}(gx, gy, gz)
    end
    return G
end
