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
# other side plus the G-record move with it. Both move kinds run this way: the spin
# pair (`_metro_kernel!` / `_metropolis_sweep_keyed_ref!`) and the displacement pair
# (`_disp_kernel!` / `_displacement_sweep_keyed_ref!`). The walk's `lo:hi` row range
# IS `delta_energy(c, zold, znew, lo, hi)` (energy.jl), full-length buffer and all,
# and the channel schedules are `_channel_colors` (gpu_hamiltonian.jl) — the host
# sweeps' `site_has_spin` / `site_has_disp` skips.

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
#
# `lo:hi` is the moved channel's ROW RANGE, and this is `delta_energy(c, zold, znew,
# lo, hi)` (energy.jl) in every respect — same entries, same FULL-LENGTH `znew` indexed
# absolutely. A single-channel move rewrites one block of the site's row table, the
# rows it did not touch contribute `c_k · 0`, and entries targeting another block are
# skipped rather than read, which keeps the untouched rows out of the arithmetic
# instead of merely multiplying them by zero.
#
# One convention, not two: `znew` is `H.nrows` long in the host sweeps
# (`SweepScratch.znew`) and `@localmem Float64 NROWS` in both kernels, and only
# `lo:hi` of it is ever written or read. Sizing the kernel buffer to the moved block
# alone would save a handful of floats per workgroup and buy a silent hazard — a
# displacement kernel handing a block-relative buffer to a `lo = nlm+1` walk reads the
# SPIN block as displacement rows, with no shape error to catch it.
#
# On a pure-spin Hamiltonian `lo:hi` is the whole table, every entry passes, and the
# walk is the pre-M4 one bit for bit.
@inline function _entry_walk_partial(tb::_GPUTables, zrows::AbstractMatrix{Float64},
                                     znew::AbstractVector{Float64}, s::Int,
                                     lane::Int, ws::Int, lo::Int, hi::Int)::Float64
    a = 0.0
    @inbounds for j = (Int(tb.site_ptr[s]) + lane - 1):ws:(Int(tb.site_ptr[s + 1]) - 1)
        pid = Int(tb.site_prog[j])
        col = Int(tb.site_col[j])
        if col > 0
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                z = zrows[Int(tb.pent_row[e]), col]
                z == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                (lo <= tgt <= hi) || continue
                a += tb.sent_w[e] * z * (znew[tgt] - zrows[tgt, s])
            end
        elseif col < 0
            col2 = Int(tb.site_col2[j])
            for e = Int(tb.sprog_ptr[pid]):(Int(tb.sprog_ptr[pid + 1]) - 1)
                p = zrows[Int(tb.pent_row[e]), -col] * zrows[Int(tb.pent_row2[e]), col2]
                p == 0.0 && continue
                tgt = Int(tb.sent_tgt[e])
                (lo <= tgt <= hi) || continue
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
                (lo <= tgt <= hi) || continue
                a += tb.sent_w[e] * p * (znew[tgt] - zrows[tgt, s])
            end
        end
    end
    return a
end

# The fused kernel: one workgroup per site of the launched color slice — of the SPIN
# schedule `tb.spin_sites`, so a joint model's displacement-only sites are never
# visited (host parity: `metropolis_sweep!` skips on `site_has_spin`). Lane 1 owns the
# proposal, the trial tesseral row, the lane-ordered ΔE fold, the accept decision, and
# every state write; the entry walk runs on all lanes. Everything lane 1 needs across a
# `@synchronize` lives in `@localmem` (KA's CPU backend does not carry private
# variables across synchronization points).
#
# `(LMAX + 1)²` IS `H.nlm` (`RowLayout.disp_offset` — asserted in the constructor), so
# it is the walk's row range `1:nlm` and the write-back extent: both touch the SPIN
# block only, and the site's displacement rows stay exactly as they are — they are
# constant factors already folded into the entry weights. The trial row is `NROWS`
# long all the same, so it is indexed like every other row table (see the walk).
@kernel function _metro_kernel!(config, zrows, dE, acc, @Const(tb), color_lo::Int,
                                β::Float64, step::Float64, seed::UInt64,
                                sweep::Int32, ::Val{LMAX},
                                ::Val{NROWS}) where {LMAX,NROWS}
    znew = @localmem Float64 NROWS
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
        s = Int(tb.spin_sites[color_lo + g - 1])
        e2, u_acc = _keyed_proposal(seed, Int32(s), sweep, config[s], step)
        # The displacement half of the buffer is never read (the walk's range check)
        # and never written back, but `@localmem` is genuinely uninitialized on a GPU,
        # so leave it zero rather than let a future range mistake read garbage instead
        # of tripping a shape error. Empty loop on a pure-spin model.
        for k = ((LMAX + 1) * (LMAX + 1) + 1):NROWS
            znew[k] = 0.0
        end
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
        s = Int(tb.spin_sites[color_lo + g - 1])
        # Int(lane): the CUDA backend's @index returns Int32 (the CPU backend's
        # returns Int) and the walk's signature is Int-typed
        partials[lane] = _entry_walk_partial(tb, zrows, znew, s, Int(lane),
                                             prod(@groupsize()), 1,
                                             (LMAX + 1) * (LMAX + 1))
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        s = Int(tb.spin_sites[color_lo + g - 1])
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
    _keyed_disp_proposal(seed, site, sweep, u, step_u) -> (u2, u_acc)

The displacement proposal of `_attempt_disp!`, drawn from the keyed Philox slots
(G2): slot 3 → the accept uniform; slots 4–5 → the isotropic Gaussian shift
`step_u · (g₁, g₂, g₃)`. Symmetric, so the acceptance is the bare Boltzmann ratio.
Slot 5's second Box–Muller normal is discarded — a spare half-block is cheaper than
a mixed accessor, and every slot's value stays a pure function of
`(seed, site, sweep)`.
"""
@inline function _keyed_disp_proposal(seed::UInt64, site::Int32, sweep::Int32,
                                      u::SVector{3,Float64},
                                      step_u::Float64)::Tuple{SVector{3,Float64},
                                                              Float64}
    _, u_acc = _philox_uniform2(_philox_block(seed, site, sweep, _SLOT_DISP_ACC))
    g1, g2 = _philox_normal2(_philox_block(seed, site, sweep, _SLOT_DISP12))
    g3, _ = _philox_normal2(_philox_block(seed, site, sweep, _SLOT_DISP3))
    return u + step_u * SVector{3,Float64}(g1, g2, g3), u_acc
end

# The displacement kernel: `_metro_kernel!` with the other channel's proposal, row
# filler, row range and write-back — same workgroup shape, same lane-1 ownership, same
# lane-ordered fold, same accept rule. It walks `tb.disp_sites`, so a joint model's
# spin-only sites are never visited (host parity: `displacement_sweep!` skips on
# `site_has_disp`).
#
# `rbuf` is the solid-harmonic batch workspace of `_disp_rows_device!` — one batch per
# site serves every `(k, l)` block, exactly as on the host. Both it and the trial row
# are `@localmem`, so nothing crosses a `@synchronize` in a private variable.
@kernel function _disp_kernel!(disps, zrows, dE, acc, @Const(tb), color_lo::Int,
                               β::Float64, step_u::Float64, seed::UInt64,
                               sweep::Int32, nlm::Int, ::Val{DLMAX},
                               ::Val{NROWS}) where {DLMAX,NROWS}
    znew = @localmem Float64 NROWS
    rbuf = @localmem Float64 (DLMAX + 1) * (DLMAX + 1)
    stash = @localmem Float64 4                     # u2 components + accept uniform
    partials = @localmem Float64 prod(@groupsize())

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        s = Int(tb.disp_sites[color_lo + g - 1])
        u2, u_acc = _keyed_disp_proposal(seed, Int32(s), sweep, disps[s], step_u)
        # `tb.fac_start` are the layout's ABSOLUTE block offsets, so this fills
        # `znew[nlm+1:NROWS]` and leaves the spin block untouched (never read).
        _disp_rows_device!(znew, rbuf, u2, tb.fac_k, tb.fac_l, tb.fac_start,
                           Val(DLMAX))
        stash[1] = u2[1]
        stash[2] = u2[2]
        stash[3] = u2[3]
        stash[4] = u_acc
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds begin
        s = Int(tb.disp_sites[color_lo + g - 1])
        partials[lane] = _entry_walk_partial(tb, zrows, znew, s, Int(lane),
                                             prod(@groupsize()), nlm + 1, NROWS)
    end
    @synchronize

    lane = @index(Local, Linear)
    g = @index(Group, Linear)
    @inbounds if lane == 1
        s = Int(tb.disp_sites[color_lo + g - 1])
        ΔE = 0.0
        for t = 1:prod(@groupsize())                 # lane-ordered fold (G4)
            ΔE += partials[t]
        end
        if ΔE <= 0.0 || stash[4] < exp(-β * ΔE)
            disps[s] = SVector{3,Float64}(stash[1], stash[2], stash[3])
            for k = (nlm + 1):NROWS
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

One compound Metropolis sweep on the device — every **spin-active** site once, in
color-serial launches (launches on one backend queue are ordered; a single
`synchronize` follows the color loop). Returns the number of accepted moves.
The per-site ΔE staging is copied back and folded on the host in the fixed
color order of `_reduce_dE` (deterministic). `workgroupsize` must be a power of
two and is part of the determinism scope — the pinned default is 128 (G3/G4).

On a joint Hamiltonian this is the SPIN half of the sweep: the displacements and
their basis rows are read but never written, and sites with no spin axis are not
visited at all. Pair it with [`gpu_displacement_sweep!`](@ref) — or let
[`gpu_run_sweeps!`](@ref) compound them, which is what it does by default.
"""
function gpu_metropolis_sweep!(gst::GPUChainState, gH::GPUTiledHamiltonian,
                               β::Float64; workgroupsize::Integer = 128)::Int
    H = gH.host
    _require_spin_sites(H, "gpu_metropolis_sweep!")
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    gst.sweep_index < typemax(Int32) - 1 ||
        throw(ArgumentError("sweep_index exhausted the 32-bit RNG counter word"))
    sweep = Int32(gst.sweep_index + 1)
    fill!(gst.dE, 0.0)
    fill!(gst.acc, Int32(0))
    kern = _metro_kernel!(gH.backend, ws)
    for c = 1:H.n_colors
        lo = Int(gH.spin_ptr[c])
        n = Int(gH.spin_ptr[c + 1]) - lo
        n == 0 && continue
        # invokelatest: a launch barrier only for static analysis — with an
        # abstract-Backend signature the GPU half of the kernel-invocation union
        # has no method until a GPU package is loaded (JET false positive); one
        # dynamic dispatch per color launch is noise next to the kernel itself.
        Base.invokelatest(kern, gst.config, gst.zrows, gst.dE, gst.acc, gH.dev,
                          lo, β, gst.step, gst.seed, sweep, Val(H.lmax),
                          Val(H.nrows); ndrange = n * ws)
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
    gpu_displacement_sweep!(gst::GPUChainState, gH::GPUTiledHamiltonian, β::Float64;
                            workgroupsize::Integer = 128) -> Int

One single-site displacement Metropolis sweep on the device — every
**displacement-active** site once, in the same color-serial launch structure as
[`gpu_metropolis_sweep!`](@ref) and with the same determinism scope. The proposal is
the isotropic Gaussian shift of width `gst.step_u` (a **length**, in the model's
units) from the keyed Philox slots 3–5, and the ΔE is the exact
`c_s·(R(u′) − R(u))` over the site's displacement rows. Returns the number of
accepted moves.

The SPIN block is read but never written: the site's own tesseral row is a constant
factor already folded into the entry weights. Throws on a Hamiltonian with no
displacement-active site.

Its RNG counter is `gst.disp_index`, **separate** from the spin sweep's
`gst.sweep_index`: a compound sweep may run several displacement passes per
Metropolis pass, and sharing a counter would make every pass in one step propose the
identical shift.
"""
function gpu_displacement_sweep!(gst::GPUChainState, gH::GPUTiledHamiltonian,
                                 β::Float64; workgroupsize::Integer = 128)::Int
    H = gH.host
    _require_disp(H, "gpu_displacement_sweep!")
    ws = Int(workgroupsize)
    ispow2(ws) || throw(ArgumentError("workgroupsize must be a power of two (got $ws)"))
    gst.disp_index < typemax(Int32) - 1 ||
        throw(ArgumentError("disp_index exhausted the 32-bit RNG counter word"))
    sweep = Int32(gst.disp_index + 1)
    fill!(gst.dE, 0.0)
    fill!(gst.acc, Int32(0))
    kern = _disp_kernel!(gH.backend, ws)
    for c = 1:H.n_colors
        lo = Int(gH.disp_ptr[c])
        n = Int(gH.disp_ptr[c + 1]) - lo
        n == 0 && continue
        Base.invokelatest(kern, gst.disps, gst.zrows, gst.dE, gst.acc, gH.dev,
                          lo, β, gst.step_u, gst.seed, sweep, H.nlm,
                          Val(H.disp_lmax), Val(H.nrows); ndrange = n * ws)
    end
    KernelAbstractions.synchronize(gH.backend)
    copyto!(gst.h_dE, gst.dE)
    copyto!(gst.h_acc, gst.acc)
    gst.energy += _reduce_dE(H, gst.h_dE)
    nacc = Int(sum(gst.h_acc))
    gst.acc_disp += nacc
    gst.att_disp += H.n_disp_active
    gst.disp_index += 1
    return nacc
end

"""
    gpu_run_sweeps!(gst::GPUChainState, gH::GPUTiledHamiltonian, st::ChainState,
                    β::Float64, nsweeps::Integer; renorm_interval::Integer = 1_000,
                    workgroupsize::Integer = 128,
                    disp_per_metropolis = nothing) -> GPUChainState

Run `nsweeps` compound device sweeps, renormalizing on the host every
`renorm_interval` sweeps (download → `_renormalize!` — drift check, displacement
re-centring, and energy re-anchor — → re-upload; `renorm_interval ≤ 0` disables).
The renormalized SPIN rows re-upload seamlessly: `normalize` is IEEE-exact arithmetic
and the host `_zlm_row!` is bitwise-identical to the device row by design (G4). The
displacement rows are bitwise only for `k = 0` blocks — a `k ≥ 1` row can move by up
to `kmax + 1` ulp across the round-trip (G8: `r2`'s FP-contraction differs between a
host call and an inlined device batch). Both sides stay deterministic, so
reproducibility is unaffected; what the seam costs is the *bitwise* claim, and the
drift gate is what covers it.

Downloads the final state into `st` before returning (without a trailing
renormalization).

One compound sweep is one [`gpu_metropolis_sweep!`](@ref) followed by
`disp_per_metropolis` [`gpu_displacement_sweep!`](@ref)s. `nothing` means "whatever
this Hamiltonian needs" — one pass on a joint model, none on a pure-spin one — the
same rule (and the same `_resolve_disp_passes`) the host drivers use. A plain default
of 0 would be the silent-wrong-ensemble trap: a joint model sampled at frozen `u`
gives the conditional `π(ê | u)`, not the joint distribution, and every displacement
observable off such a run is conditional on a lattice nobody equilibrated. Passing
`0` explicitly is how you ask for that conditional on purpose.

Two Hamiltonians get fewer passes than that rule alone would suggest, both because a
channel has no *site* to attempt anything at:

  * a model with displacement rows but **no displacement-active site** (every
    displacement coupling fitted to zero) runs zero displacement passes — with
    nothing depending on `u`, the frozen-`u` conditional is the joint distribution;
  * a model with **no spin-active site** (a lattice-only one) skips the Metropolis
    pass rather than erroring on it.

A model with nothing to sweep in either channel throws. So does an explicit
displacement pass on a Hamiltonian with no displacement rows at all.

Note that `_renormalize!` re-centres each displacement-coupling component, so
`st.disps` carries a per-component gauge: the physical displacement is
`disps[s] + com_removed[c]`.
"""
function gpu_run_sweeps!(gst::GPUChainState, gH::GPUTiledHamiltonian,
                         st::ChainState, β::Float64, nsweeps::Integer;
                         renorm_interval::Integer = 1_000,
                         workgroupsize::Integer = 128,
                         disp_per_metropolis::Union{Nothing,Integer} =
                             nothing)::GPUChainState
    H = gH.host
    # Resolve FIRST — that is where "a displacement pass on a Hamiltonian that has no
    # displacement rows" is refused — and clamp after. `_resolve_disp_passes` gates on
    # `has_disp` (a property of the row LAYOUT) while the sweep gates on
    # `n_disp_active` (the sites), and the two differ on a joint basis whose
    # displacement couplings all fitted to zero: layout rows, but no site whose energy
    # depends on any `u`. Running zero passes there is not a wrong-ensemble risk, it is
    # the opposite — with nothing depending on `u`, the frozen-`u` conditional IS the
    # joint distribution — and it is what the host does too (`displacement_sweep!` over
    # an empty site list, a documented no-op).
    npass = _resolve_disp_passes(H, disp_per_metropolis)
    H.n_disp_active == 0 && (npass = 0)
    nspin = H.n_spin_active
    nspin > 0 || npass > 0 || throw(ArgumentError(
        "this run would attempt no move at all: the Hamiltonian has no spin-active " *
        "site and no displacement pass to run"))
    _warn_gpu_escape_cadence(H, Int(nsweeps), Int(renorm_interval), npass)
    for i = 1:nsweeps
        nspin > 0 && gpu_metropolis_sweep!(gst, gH, β; workgroupsize = workgroupsize)
        for _ = 1:npass
            gpu_displacement_sweep!(gst, gH, β; workgroupsize = workgroupsize)
        end
        if renorm_interval > 0 && i % renorm_interval == 0
            to_host!(st, gst)
            _renormalize!(st, gH.host)
            _from_host!(gst, st)
        end
    end
    to_host!(st, gst)
    return gst
end

# The device counterpart of `_warn_escape_cadence` (run.jl). The escape detector is
# the only diagnostic that measures displacement RECURRENCE, and it runs inside
# `_renormalize!` — so a device joint run's screening is entirely a function of
# `renorm_interval`, which this driver lets the caller set to 0. Warn up front, while
# the caller can still change it, rather than let `escaped == false` mean "not
# screened" and read as "clean". Same block-test arithmetic as the host, so the two
# warnings agree on what counts as enough.
function _warn_gpu_escape_cadence(H::TiledHamiltonian, nsweeps::Int,
                                  renorm_interval::Int, npass::Int)
    npass > 0 || return nothing
    need = _escape_min_checks()
    if renorm_interval <= 0
        @warn "this joint device run has renormalization disabled " *
              "(renorm_interval = $(renorm_interval)), so the displacement channel " *
              "gets NO escape screening at all and — on a model with a flat " *
              "uniform-shift direction — no re-centring either, letting the frame " *
              "random-walk. Set renorm_interval to " *
              "$(max(1, fld(nsweeps, need))) or below to arm the detector." maxlog = 1
        return nothing
    end
    checks = fld(nsweeps, renorm_interval)
    checks >= need && return nothing
    @warn "this joint device run gets $(checks) renormalization checks " *
          "(nsweeps = $(nsweeps), renorm_interval = $(renorm_interval)), but the " *
          "escape detector's block test needs $(need) before it can raise " *
          "$(_ESCAPE_STRIKES) consecutive strikes. Only the absolute " *
          "$(_ESCAPE_ABSOLUTE)× guard is live, which catches a fast escape but not a " *
          "slow one. Lower renorm_interval to $(max(1, fld(nsweeps, need))) or " *
          "below to arm it." maxlog = 1
    return nothing
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
    tb = _host_tables(H)
    nlm = H.nlm
    znew = Vector{Float64}(undef, H.nrows)      # full row table, as in the kernel
    partials = Vector{Float64}(undef, ws)
    fill!(dE, 0.0)
    fill!(acc, Int32(0))
    nacc = 0
    for q in eachindex(tb.spin_sites)
        s = Int(tb.spin_sites[q])
        e2, u_acc = _keyed_proposal(seed, Int32(s), sweep, config[s], step)
        _zlm_row_device_dyn!(znew, e2, H.lmax)
        for lane = 1:ws
            partials[lane] = _entry_walk_partial(tb, zrows, znew, s, lane, ws, 1, nlm)
        end
        ΔE = 0.0
        for t = 1:ws
            ΔE += partials[t]
        end
        if ΔE <= 0.0 || u_acc < exp(-β * ΔE)
            config[s] = e2
            copyto!(view(zrows, 1:nlm, s), view(znew, 1:nlm))
            dE[s] = ΔE
            acc[s] = Int32(1)
            nacc += 1
        end
    end
    return nacc
end

# The displacement half of the same reference: `_disp_kernel!`'s arithmetic, serial.
function _displacement_sweep_keyed_ref!(disps::Vector{SVector{3,Float64}},
                                        zrows::Matrix{Float64}, dE::Vector{Float64},
                                        acc::Vector{Int32}, H::TiledHamiltonian,
                                        β::Float64, step_u::Float64, seed::UInt64,
                                        sweep::Int32, ws::Int)::Int
    tb = _host_tables(H)
    nlm = H.nlm
    nrows = H.nrows
    znew = Vector{Float64}(undef, nrows)
    rbuf = Vector{Float64}(undef, max(1, (H.disp_lmax + 1)^2))
    partials = Vector{Float64}(undef, ws)
    fill!(dE, 0.0)
    fill!(acc, Int32(0))
    nacc = 0
    for q in eachindex(tb.disp_sites)
        s = Int(tb.disp_sites[q])
        u2, u_acc = _keyed_disp_proposal(seed, Int32(s), sweep, disps[s], step_u)
        _disp_rows_device_dyn!(znew, rbuf, u2, tb.fac_k, tb.fac_l, tb.fac_start,
                               H.disp_lmax)
        for lane = 1:ws
            partials[lane] = _entry_walk_partial(tb, zrows, znew, s, lane, ws,
                                                 nlm + 1, nrows)
        end
        ΔE = 0.0
        for t = 1:ws
            ΔE += partials[t]
        end
        if ΔE <= 0.0 || u_acc < exp(-β * ΔE)
            disps[s] = u2
            copyto!(view(zrows, (nlm + 1):nrows, s), view(znew, (nlm + 1):nrows))
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
