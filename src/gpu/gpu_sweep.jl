# The fused device Metropolis sweep (decision record docs/specs/gpu-prototype.md
# G4): color-serial kernel launches, one workgroup per active site of the current
# color, threads term-parallel over the site's adjacency entries with a direct-ΔE
# accumulation (no materialized coefficient vector — `znew` depends only on the
# proposal, so ΔE = Σ_entries w·p·(znew[tgt] − zrows[tgt, s]) folds into the walk).
#
# Determinism contract (G3): a run is bitwise reproducible for a fixed (seed,
# backend, workgroup size, package + Julia version) and is scheduling-independent
# by construction — every random draw is a pure function of (seed, site, sweep)
# via the keyed Philox stream, all device writes are per-site disjoint, and the
# in-workgroup reduction has a fixed, lane-ordered structure. Bitwise identity
# across backends holds for the algebraic kernels only (the Box–Muller and accept
# `exp`/`log`/`cos`/`sin` are backend-libm); the cross-backend gates are the
# incremental-energy drift gate and statistics.
#
# COUPLED SITES: the kernel and `_metropolis_sweep_keyed_ref!` (bottom of this
# file) implement one arithmetic contract — the proposal slot map (philox.jl),
# `_entry_walk_partial`, the lane-ordered partial fold, and the accept rule. The
# full-sweep bitwise gate in test/unit/test_gpu.jl compares them on the CPU
# backend; change either side (or the slot map, or the skip predicates) and the
# other side plus the G-record move with it.

"""
    _keyed_proposal(seed, site, sweep, e, step) -> (e2, u_acc)

The Metropolis proposal of `_attempt_metro!`, drawn from the keyed Philox slots
(G2): slot 0 → flip uniform (vs `_FLIP_FRACTION`) and accept uniform; slots 1–2 →
the rotation axis normals and the Gaussian angle `step · n₄`. Branch-dependent
*consumption* is meaningless here — every slot's value is fixed by `(seed, site,
sweep)` whether or not it is evaluated.
"""
@inline function _keyed_proposal(seed::UInt64, site::Int32, sweep::Int32,
                                 e::SVector{3,Float64},
                                 step::Float64)::Tuple{SVector{3,Float64},Float64}
    u_flip, u_acc = _philox_uniform2(_philox_block(seed, site, sweep, _SLOT_FLIP_ACC))
    u_flip < _FLIP_FRACTION && return -e, u_acc
    n1, n2 = _philox_normal2(_philox_block(seed, site, sweep, _SLOT_AXIS12))
    n3, n4 = _philox_normal2(_philox_block(seed, site, sweep, _SLOT_AXIS3_ANGLE))
    axis = normalize(SVector{3,Float64}(n1, n2, n3))
    return _rotate(e, axis, step * n4), u_acc
end

# One lane's strided share of the direct-ΔE entry walk: the exact three-way
# dispatch and zero-skips of `site_coeffs!` (energy.jl — the skips are part of
# the bitwise contract: adding an exact 0.0 could flip a −0.0 partial), with the
# ΔE dot product folded in per entry. `tb` is a `_GPUTables` of device arrays
# (inside the kernel) or of host arrays (the keyed reference) — same code path.
@inline function _entry_walk_partial(tb::_GPUTables, zrows::AbstractMatrix{Float64},
                                     znew::AbstractVector{Float64}, s::Int,
                                     lane::Int, ws::Int)::Float64
    a = 0.0
    @inbounds for j = (Int(tb.site_ptr[s]) + lane - 1):ws:(Int(tb.site_ptr[s + 1]) - 1)
        pid = Int(tb.site_prog[j])
        col = Int(tb.site_col[j])
        if col > 0
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                z = zrows[Int(tb.pent_row[e]), col]
                z == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                a += tb.sent_w[e] * z * (znew[tgt] - zrows[tgt, s])
            end
        elseif col < 0
            col2 = Int(tb.site_col2[j])
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                p = zrows[Int(tb.pent_row[e]), -col] * zrows[Int(tb.pent_row2[e]), col2]
                p == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                a += tb.sent_w[e] * p * (znew[tgt] - zrows[tgt, s])
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
                a += tb.sent_w[e] * p * (znew[tgt] - zrows[tgt, s])
            end
        end
    end
    return a
end

# The fused kernel: one workgroup per site of the launched color slice. Lane 1
# owns the proposal, the trial tesseral row, the lane-ordered ΔE fold, the accept
# decision, and every state write; the entry walk runs on all lanes. Everything
# lane 1 needs across a `@synchronize` lives in `@localmem` (KA's CPU backend
# does not carry private variables across synchronization points).
@kernel function _metro_kernel!(config, zrows, dE, acc, @Const(tb), color_lo::Int,
                                β::Float64, step::Float64, seed::UInt64,
                                sweep::Int32, ::Val{LMAX}) where {LMAX}
    znew = @localmem Float64 (LMAX + 1) * (LMAX + 1)
    stash = @localmem Float64 4                     # e2 components + accept uniform
    partials = @localmem Float64 prod(@groupsize())

    # NOTE: plain locals do not survive `@synchronize` on the CPU backend (the
    # body is split into blocks there), so `lane`/`s` are recomputed at the top
    # level of every inter-sync segment — value-identical; only `@localmem`
    # carries state across syncs, and `@index` must sit at segment top level for
    # the CPU-backend code transform to rewrite it.
    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        s = Int(tb.color_sites[color_lo + g - 1])
        e2, u_acc = _keyed_proposal(seed, Int32(s), sweep, config[s], step)
        _zlm_row_device!(znew, e2, Val(LMAX))
        stash[1] = e2[1]
        stash[2] = e2[2]
        stash[3] = e2[3]
        stash[4] = u_acc
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds begin
        s = Int(tb.color_sites[color_lo + g - 1])
        # Int(lane): the CUDA backend's @index returns Int32 (the CPU backend's
        # returns Int) and the walk's signature is Int-typed
        partials[lane] = _entry_walk_partial(tb, zrows, znew, s, Int(lane),
                                             prod(@groupsize()))
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        s = Int(tb.color_sites[color_lo + g - 1])
        ΔE = 0.0
        for t = 1:prod(@groupsize())                 # lane-ordered fold (G4)
            ΔE += partials[t]
        end
        if ΔE <= 0.0 || stash[4] < exp(-β * ΔE)
            config[s] = SVector{3,Float64}(stash[1], stash[2], stash[3])
            for k = 1:(LMAX + 1) * (LMAX + 1)
                zrows[k, s] = znew[k]
            end
            dE[s] = ΔE
            acc[s] = Int32(1)
        end
    end
end

"""
    gpu_metropolis_sweep!(gst::GPUChainState, gH::GPUTiledHamiltonian, β::Float64;
                          workgroupsize::Integer = 128) -> Int

One compound Metropolis sweep on the device — every active site once, in
color-serial launches (launches on one backend queue are ordered; a single
`synchronize` follows the color loop). Returns the number of accepted moves.
The per-site ΔE staging is copied back and folded on the host in the fixed
color order of `_reduce_dE` (deterministic). `workgroupsize` must be a power of
two and is part of the determinism scope — the pinned default is 128 (G3/G4).
"""
function gpu_metropolis_sweep!(gst::GPUChainState, gH::GPUTiledHamiltonian,
                               β::Float64; workgroupsize::Integer = 128)::Int
    H = gH.host
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    gst.sweep_index < typemax(Int32) - 1 ||
        throw(ArgumentError("sweep_index exhausted the 32-bit RNG counter word"))
    sweep = Int32(gst.sweep_index + 1)
    fill!(gst.dE, 0.0)
    fill!(gst.acc, Int32(0))
    kern = _metro_kernel!(gH.backend, ws)
    for c = 1:H.n_colors
        lo = Int(H.color_ptr[c])
        n = Int(H.color_ptr[c + 1]) - lo
        n == 0 && continue
        # invokelatest: a launch barrier only for static analysis — with an
        # abstract-Backend signature the GPU half of the kernel-invocation union
        # has no method until a GPU package is loaded (JET false positive); one
        # dynamic dispatch per color launch is noise next to the kernel itself.
        Base.invokelatest(kern, gst.config, gst.zrows, gst.dE, gst.acc, gH.dev,
                          lo, β, gst.step, gst.seed, sweep, Val(H.lmax);
                          ndrange = n * ws)
    end
    KernelAbstractions.synchronize(gH.backend)
    copyto!(gst.h_dE, gst.dE)
    copyto!(gst.h_acc, gst.acc)
    gst.energy += _reduce_dE(H, gst.h_dE)
    nacc = Int(sum(gst.h_acc))
    gst.acc_metro += nacc
    gst.att_metro += H.n_spin_active
    gst.sweep_index += 1
    return nacc
end

"""
    gpu_run_sweeps!(gst::GPUChainState, gH::GPUTiledHamiltonian, st::ChainState,
                    β::Float64, nsweeps::Integer; renorm_interval::Integer = 1_000,
                    workgroupsize::Integer = 128) -> GPUChainState

Run `nsweeps` device sweeps, renormalizing on the host every `renorm_interval`
sweeps (download → `_renormalize!` — drift check and energy re-anchor — →
re-upload; `renorm_interval ≤ 0` disables). The renormalized rows re-upload
seamlessly: `normalize` is IEEE-exact arithmetic and the host `_zlm_row!` is
bitwise-identical to the device row by design (G4). Downloads the final state
into `st` before returning (without a trailing renormalization).
"""
function gpu_run_sweeps!(gst::GPUChainState, gH::GPUTiledHamiltonian,
                         st::ChainState, β::Float64, nsweeps::Integer;
                         renorm_interval::Integer = 1_000,
                         workgroupsize::Integer = 128)::GPUChainState
    for i = 1:nsweeps
        gpu_metropolis_sweep!(gst, gH, β; workgroupsize = workgroupsize)
        if renorm_interval > 0 && i % renorm_interval == 0
            to_host!(st, gst)
            _renormalize!(st, gH.host)
            _from_host!(gst, st)
        end
    end
    to_host!(st, gst)
    return gst
end

# ---------------------------------------------------------------------------
# Reference implementation (the readable spec of the kernel's arithmetic, in the
# `energy.jl` reference-kernel tradition): a plain serial host sweep of the SAME
# keyed scheme — same proposal slots, same `_entry_walk_partial` strided shares,
# same lane-ordered fold, same accept rule — so on the CPU backend (same libm)
# the kernel must match it bitwise. Test-only; never called by the drivers.
# ---------------------------------------------------------------------------
function _metropolis_sweep_keyed_ref!(config::SpinConfig, zrows::Matrix{Float64},
                                      dE::Vector{Float64}, acc::Vector{Int32},
                                      H::TiledHamiltonian, β::Float64,
                                      step::Float64, seed::UInt64, sweep::Int32,
                                      ws::Int)::Int
    # Test-only reference, but it sizes its buffers from `H.nlm`/`H.lmax` and would
    # silently drop a displacement channel rather than throw.
    _require_spin_only(H, "_metropolis_sweep_keyed_ref!")
    tb = _GPUTables(H.site_ptr, H.site_inst, H.inst_ptr, H.inst_sites,
                    H.progs.site_prog, H.progs.sprog_ptr, H.progs.sent_w,
                    H.progs.sent_tgt, H.progs.sfac_ptr, H.progs.sfac_row,
                    H.progs.sfac_slot, H.progs.site_col, H.progs.site_col2,
                    H.progs.pent_row, H.progs.pent_row2, H.color_sites)
    nlm = H.nlm
    znew = Vector{Float64}(undef, nlm)
    partials = Vector{Float64}(undef, ws)
    fill!(dE, 0.0)
    fill!(acc, Int32(0))
    nacc = 0
    for q in eachindex(H.color_sites)
        s = Int(H.color_sites[q])
        e2, u_acc = _keyed_proposal(seed, Int32(s), sweep, config[s], step)
        _zlm_row_device_dyn!(znew, e2, H.lmax)
        for lane = 1:ws
            partials[lane] = _entry_walk_partial(tb, zrows, znew, s, lane, ws)
        end
        ΔE = 0.0
        for t = 1:ws
            ΔE += partials[t]
        end
        if ΔE <= 0.0 || u_acc < exp(-β * ΔE)
            config[s] = e2
            copyto!(view(zrows, :, s), znew)
            dE[s] = ΔE
            acc[s] = Int32(1)
            nacc += 1
        end
    end
    return nacc
end

# Runtime-lmax dispatch onto the Val-specialized device row (reference/test use).
function _zlm_row_device_dyn!(z::AbstractVector{Float64}, u::SVector{3,Float64},
                              lmax::Int)::Nothing
    if lmax == 0
        _zlm_row_device!(z, u, Val(0))
    elseif lmax == 1
        _zlm_row_device!(z, u, Val(1))
    elseif lmax == 2
        _zlm_row_device!(z, u, Val(2))
    elseif lmax == 3
        _zlm_row_device!(z, u, Val(3))
    elseif lmax == 4
        _zlm_row_device!(z, u, Val(4))
    elseif lmax == 5
        _zlm_row_device!(z, u, Val(5))
    elseif lmax == 6
        _zlm_row_device!(z, u, Val(6))
    else
        throw(ArgumentError("lmax = $lmax unsupported on the device path (≤ 6)"))
    end
    return nothing
end
