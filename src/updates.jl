# Update schemes (stationarity arguments: `docs/specs/updates-stationarity.md`).
#
# Sweeps scan the Hamiltonian's color classes in order (color-major, sites
# ascending within a class — `TiledHamiltonian.color_ptr`/`color_sites`; inactive
# sites are uncolored and never visited). Sites in one class share no cluster
# instance, so their single-spin kernels are exactly independent: updating a class
# in ANY execution order — including concurrently — is the same Markov chain.
# Three ingredients make one sweep bit-deterministic regardless of how many tasks
# execute it (the `sweep_tasks` option of run_mc/run_pt/find_ground_state):
#   1. every site owns its proposal/accept RNG stream (`ChainState.site_rngs`),
#   2. accepted ΔE are staged per site (`SweepScratch.dE`, write-disjoint) and
#      reduced in the fixed class order (`_reduce_dE` — one shared loop, so the
#      serial and parallel paths sum in the identical order),
#   3. acceptance counters are integers (order-free sums).
# Gate: test_parallel.jl (serial ≡ parallel bitwise). β enters ONLY in the accept
# steps; coefficients and energies stay in the model's energy units.

# Antipodal-flip fraction of the Metropolis proposal: the Rodrigues rotation alone
# mixes slowly between the ±lobes of a strongly bimodal single-site potential.
const _FLIP_FRACTION = 0.2

# Rodrigues rotation of `e` about the unit `axis` by angle `θ` (an isometry; the
# proposal is symmetric because `θ ~ step·randn` is sign-symmetric and the axis is
# uniform).
@inline function _rotate(e::SVector{3,Float64}, axis::SVector{3,Float64},
                         θ::Float64)::SVector{3,Float64}
    c, s = cos(θ), sin(θ)
    return c * e + s * cross(axis, e) + (1 - c) * dot(axis, e) * axis
end

# One Metropolis attempt at site `s`: symmetric two-component proposal (antipodal
# flip with probability 0.2, else a Rodrigues rotation by `step·randn` about a
# uniform axis) from the site's own stream, exact `ΔE = c_s·ΔZ`, accept with
# `ΔE ≤ 0 || rand < exp(−β·ΔE)` (the uniform is drawn ONLY when ΔE > 0 — the
# RNG-consumption contract). On accept, updates config/zrows and stages ΔE in
# `dE[s]` (zeroed at sweep start). Returns 1 on accept.
@inline function _attempt_metro!(config::SpinConfig, zrows::Matrix{Float64},
                                 H::TiledHamiltonian, β::Float64, step::Float64,
                                 s::Int, sc::SweepScratch, rng::Xoshiro,
                                 dE::Vector{Float64})::Int
    fill!(sc.c, 0.0)
    site_coeffs!(sc.c, H, s, zrows)
    e = config[s]
    e2 = if rand(rng) < _FLIP_FRACTION
        -e
    else
        _rotate(e, _random_unit(rng), step * randn(rng))
    end
    _zlm_row!(sc.znew, e2, H.lmax, sc.plm)
    ΔE = delta_energy(sc.c, view(zrows, :, s), sc.znew)
    if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
        config[s] = e2
        copyto!(view(zrows, :, s), sc.znew)
        @inbounds dE[s] = ΔE
        return 1
    end
    return 0
end

# One overrelaxation attempt at site `s`: reflect about the local l = 1 field axis
# — `e′ = 2(e·ĥ)ĥ − e`, ĥ read off the l = 1 leave-one-out coefficients (tesseral
# Z₁ₘ ∝ (y, z, x) ⇒ ĥ = (c₄, c₂, c₃); a convention drift upstream only rotates the
# axis — correctness is unaffected, the pure-l=1 ΔE ≡ 0 gate pins it) — plus the
# Metropolis correction on the exact ΔE. Returns (attempted, accepted) as 0/1; a
# vanishing local field does not count as an attempt.
@inline function _attempt_or!(config::SpinConfig, zrows::Matrix{Float64},
                              H::TiledHamiltonian, β::Float64, s::Int,
                              sc::SweepScratch, rng::Xoshiro,
                              dE::Vector{Float64})::Tuple{Int,Int}
    fill!(sc.c, 0.0)
    site_coeffs!(sc.c, H, s, zrows)
    h = SVector(sc.c[4], sc.c[2], sc.c[3])
    hn = norm(h)
    hn < 1e-12 && return (0, 0)
    axis = h / hn
    e = config[s]
    e2 = 2 * dot(e, axis) * axis - e
    _zlm_row!(sc.znew, e2, H.lmax, sc.plm)
    ΔE = delta_energy(sc.c, view(zrows, :, s), sc.znew)
    if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
        config[s] = e2
        copyto!(view(zrows, :, s), sc.znew)
        @inbounds dE[s] = ΔE
        return (1, 1)
    end
    return (1, 0)
end

# Deterministic energy reduction: read the staged per-site ΔE back in the fixed
# class order. The one loop both execution paths share — identical summation order
# is what keeps `st.energy` independent of the task count.
function _reduce_dE(H::TiledHamiltonian, dE::Vector{Float64})::Float64
    ΔE = 0.0
    @inbounds for q in eachindex(H.color_sites)
        ΔE += dE[H.color_sites[q]]
    end
    return ΔE
end

# Sense-reversing spin barrier separating color classes. Two robustness valves:
# the yield backoff keeps a briefly-oversubscribed task pool safe (a pure spin
# could starve the task that must arrive last), and the `poisoned` flag makes the
# barrier exception-safe — a task that dies (a bug, or Ctrl-C injecting an
# InterruptException) poisons the barrier before rethrowing, so the surviving
# tasks exit instead of spinning forever on a release that can never come, and the
# enclosing @sync can propagate the real exception. Returns `false` when poisoned
# (the caller abandons the sweep; the chain state is torn mid-sweep, as after any
# exception).
mutable struct _SweepBarrier
    const count::Threads.Atomic{Int}
    const gen::Threads.Atomic{Int}
    const poisoned::Threads.Atomic{Bool}
    const n::Int
end
_SweepBarrier(n::Int) =
    _SweepBarrier(Threads.Atomic{Int}(0), Threads.Atomic{Int}(0),
                  Threads.Atomic{Bool}(false), n)

function _barrier_wait!(b::_SweepBarrier)::Bool
    b.poisoned[] && return false
    g = b.gen[]
    if Threads.atomic_add!(b.count, 1) == b.n - 1
        b.count[] = 0
        Threads.atomic_add!(b.gen, 1)
    else
        spins = 0
        while b.gen[] == g
            b.poisoned[] && return false
            ccall(:jl_cpu_pause, Cvoid, ())
            GC.safepoint()
            spins += 1
            spins > (1 << 14) && (yield(); spins = 0)
        end
    end
    return !b.poisoned[]
end

"""
    metropolis_sweep!(st::ChainState, H::TiledHamiltonian, β, sc) -> Int

One single-spin Metropolis lattice sweep: one attempt per **active** site, scanned
in the Hamiltonian's color-class order, each site using its own RNG stream
(`st.site_rngs[s]`) with the symmetric two-component proposal (antipodal flip with
probability 0.2, else a Rodrigues rotation by `st.step · randn` about a uniform
axis) and the exact `ΔE = c_s·ΔZ` from [`site_coeffs!`](@ref). Accepts with
`ΔE ≤ 0 || rand < exp(−β·ΔE)` (the uniform is drawn **only** when `ΔE > 0`).
Inactive sites are skipped (spin-independent energy — an attempt there is always
accepted noise that would waste RNG and bias the acceptance statistics).

`sc` is one [`SweepScratch`](@ref) (serial execution) or a `Vector{SweepScratch}`
(one per task — the class slices are executed by `length(sc)` concurrent tasks,
with the identical, bit-deterministic result: sites in one class share no
instance, every site has its own RNG stream, and the accepted ΔE are reduced in
the fixed class order). Mutates the config/rows/energy in place; returns the
number of accepted moves.
"""
function metropolis_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                           sc::SweepScratch)::Int
    fill!(sc.dE, 0.0)
    nacc = 0
    @inbounds for q in eachindex(H.color_sites)
        s = Int(H.color_sites[q])
        H.site_has_spin[s] || continue     # colored for scheduling, but not magnetic
        nacc += _attempt_metro!(st.config, st.zrows, H, β, st.step, s, sc,
                                st.site_rngs[s], sc.dE)
    end
    st.energy += _reduce_dE(H, sc.dE)
    st.acc_metro += nacc
    st.att_metro += H.n_spin_active
    return nacc
end

function metropolis_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                           scs::Vector{SweepScratch})::Int
    isempty(scs) && throw(ArgumentError("scs must hold at least one SweepScratch"))
    length(scs) == 1 && return metropolis_sweep!(st, H, β, scs[1])
    ntasks = length(scs)
    dE = scs[1].dE
    fill!(dE, 0.0)
    bar = _SweepBarrier(ntasks)
    nacc = Threads.Atomic{Int}(0)
    @sync for t = 1:ntasks
        Threads.@spawn begin
            sc = scs[t]
            a = 0
            try
                for c = 1:H.n_colors
                    q1 = Int(H.color_ptr[c + 1]) - 1
                    for q = (Int(H.color_ptr[c]) + t - 1):ntasks:q1
                        s = Int(H.color_sites[q])
                        H.site_has_spin[s] || continue
                        a += _attempt_metro!(st.config, st.zrows, H, β, st.step,
                                             s, sc, st.site_rngs[s], dE)
                    end
                    _barrier_wait!(bar) || break
                end
            catch
                bar.poisoned[] = true
                rethrow()
            end
            Threads.atomic_add!(nacc, a)
        end
    end
    st.energy += _reduce_dE(H, dE)
    st.acc_metro += nacc[]
    st.att_metro += H.n_spin_active
    return nacc[]
end

"""
    overrelaxation_sweep!(st::ChainState, H::TiledHamiltonian, β, sc) -> Int

One overrelaxation lattice sweep, scanned in the color-class order of
[`metropolis_sweep!`](@ref) (and accepting the same serial `SweepScratch` /
parallel `Vector{SweepScratch}` forms, with the same bit-deterministic result):
at each site, reflect the spin about its local `l = 1` field axis —
`e′ = 2(e·ĥ)ĥ − e`, with `ĥ` read off the `l = 1` components of the leave-one-out
coefficients (independent of `e` itself) — and accept with the standard
Metropolis rule on the **exact** ΔE.

The proposal is a deterministic involution (`S∘S = id`, an isometry of the sphere)
whose axis depends only on the other spins, so Metropolis acceptance gives detailed
balance outright; for a **pure-`l=1`** site channel the reflection conserves the
site energy exactly (`ΔE ≡ 0`, always accepted) and the move degenerates to
classical microcanonical overrelaxation — the accept step corrects the `l ≥ 2` /
multi-body remainder exactly (see `docs/specs/updates-stationarity.md`).

Not ergodic on its own (in the pure case it conserves `e·h` per move) — always mix
with Metropolis sweeps (`or_per_metropolis` in [`run_mc`](@ref)). Sites with no
`l = 1` channel (or a vanishing local field) are skipped and do not count as
attempts. Returns the number of accepted moves.
"""
function overrelaxation_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                               sc::SweepScratch)::Int
    fill!(sc.dE, 0.0)
    nacc = 0
    natt = 0
    @inbounds for q in eachindex(H.color_sites)
        s = Int(H.color_sites[q])
        H.site_has_l1[s] || continue
        at, ac = _attempt_or!(st.config, st.zrows, H, β, s, sc, st.site_rngs[s],
                              sc.dE)
        natt += at
        nacc += ac
    end
    st.energy += _reduce_dE(H, sc.dE)
    st.acc_or += nacc
    st.att_or += natt
    return nacc
end

function overrelaxation_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                               scs::Vector{SweepScratch})::Int
    isempty(scs) && throw(ArgumentError("scs must hold at least one SweepScratch"))
    length(scs) == 1 && return overrelaxation_sweep!(st, H, β, scs[1])
    ntasks = length(scs)
    dE = scs[1].dE
    fill!(dE, 0.0)
    bar = _SweepBarrier(ntasks)
    nacc = Threads.Atomic{Int}(0)
    natt = Threads.Atomic{Int}(0)
    @sync for t = 1:ntasks
        Threads.@spawn begin
            sc = scs[t]
            a = 0
            n = 0
            try
                for c = 1:H.n_colors
                    q1 = Int(H.color_ptr[c + 1]) - 1
                    for q = (Int(H.color_ptr[c]) + t - 1):ntasks:q1
                        s = Int(H.color_sites[q])
                        H.site_has_l1[s] || continue
                        at, ac = _attempt_or!(st.config, st.zrows, H, β, s, sc,
                                              st.site_rngs[s], dE)
                        n += at
                        a += ac
                    end
                    _barrier_wait!(bar) || break
                end
            catch
                bar.poisoned[] = true
                rethrow()
            end
            Threads.atomic_add!(nacc, a)
            Threads.atomic_add!(natt, n)
        end
    end
    st.energy += _reduce_dE(H, dE)
    st.acc_or += nacc[]
    st.att_or += natt[]
    return nacc[]
end

# Multiplicative step adaptation toward the target acceptance, on the current
# counter window (then resets it). Thermalization only — once `st.frozen` the
# transition kernel must stay fixed (a history-dependent step is a bias source and
# breaks bit-reproducible resume), so this is a no-op.
function _adapt_step!(st::ChainState, target::Float64)::Float64
    st.frozen && return st.step
    if st.att_metro > 0
        a = st.acc_metro / st.att_metro
        st.step = clamp(st.step * exp(0.5 * (a - target)), 1e-3, Float64(π))
    end
    st.acc_metro = 0
    st.att_metro = 0
    return st.step
end
