# Replica exchange (parallel tempering) over threads
# (design + determinism guarantees: `docs/specs/pt-threads-determinism.md`).
#
# Lane r owns rung r of the temperature ladder: its ChainState, scratch, RNG, and
# accumulators. Between segments (every `exchange_interval` compound sweeps), the
# coordinator attempts adjacent-pair swaps of the chain *payload* (config / rows /
# energy) — RNG, step, and accumulators stay with the lane, so every lane's
# measurement stream is a fixed-temperature marginal and the adapted step remains
# per-temperature.
#
# Determinism: every random decision is attributed to a specific RNG whose
# consumption order is fixed by the segment schedule, never by thread timing — lane
# RNGs are consumed only inside that lane's sweeps, the dedicated exchange RNG only
# in the serial pre-draw (one uniform *unconditionally* per attempted pair,
# boundary-major, ascending pair order), so the async pairwise-handshake schedule
# below reproduces the serial reference bit for bit. Results are bit-identical for
# a fixed seed regardless of `ntasks` / `JULIA_NUM_THREADS` (gated).

"""
    PTResult

Result of [`run_pt`](@ref): `points` (one [`TempResult`](@ref) per ladder rung, in
ladder order), the adjacent-pair `swap_acceptance` fractions (length
`n_rungs − 1`; over the whole run), each lane's `final_config` and `final_disps`, and
the run `seed`. Prints as a summary table.

`final_disps[r]` is lane `r`'s last displacement configuration in the
centre-of-mass-free frame (empty on a pure-spin model). Note it is the *lane's*
payload at the end of the run — a replica exchange moves whole physical states
between lanes, so it is a sample of rung `r`'s marginal, not the continuation of one
replica's trajectory.
"""
struct PTResult
    points::Vector{TempResult}
    swap_acceptance::Vector{Float64}
    final_configs::Vector{SpinConfig}
    final_disps::Vector{Vector{SVector{3,Float64}}}
    seed::UInt64
end

Base.show(io::IO, r::PTResult) =
    print(io, "PTResult(", length(r.points), " rungs, ",
          length(first(r.final_configs)), " sites)")

function Base.show(io::IO, ::MIME"text/plain", r::PTResult)
    println(io, "PTResult: ", length(r.points), " rungs, ",
            length(first(r.final_configs)), " sites, seed ", r.seed)
    _print_points_table(io, r.points, length(first(r.final_configs)))
    print(io, "  swap acceptance: ")
    println(io, join([@sprintf("%.2f", a) for a in r.swap_acceptance], " "))
    return nothing
end

# One parallel-tempering lane: a rung of the ladder with its chain, scratch, and
# (during measurement) accumulators.
mutable struct _PTLane
    const st::ChainState
    const scs::Vector{SweepScratch}
    const kt::Float64
    const β::Float64
    accs::Vector{ObsAccumulator}
    phase_sweeps::Int              # sweeps done in the current phase
end

# Swap the replica payload between two chains (reference swaps — O(1)).
function _swap_payload!(a::ChainState, b::ChainState)
    a.config, b.config = b.config, a.config
    # The displacements and their accumulated re-centring record travel with the
    # configuration: a replica exchange moves a whole physical state between lanes,
    # and the frame that state has been pinned to is part of it. The RNG streams and
    # the proposal widths stay with the lane.
    a.disps, b.disps = b.disps, a.disps
    a.com_removed, b.com_removed = b.com_removed, a.com_removed
    # The escape-detector accumulators do NOT travel: like the acceptance counters they
    # describe the lane, and a lane is a fixed temperature. The question the detector
    # asks — does the r.m.s. displacement of this temperature's marginal stay flat —
    # is answered by the lane's own series, not by following one replica up and down
    # the ladder.
    a.zrows, b.zrows = b.zrows, a.zrows
    a.energy, b.energy = b.energy, a.energy
    # The cell scale is payload too — it labels the physical state, exactly like the
    # displacements it scales. In v0 both lanes are always at 1.0 (PT + strain is
    # refused: one shared Hamiltonian cannot carry two coefficient sets), so this is
    # the future-proof no-op, not a live channel.
    a.strain, b.strain = b.strain, a.strain
    return nothing
end

# Run `n` compound sweeps of one lane (thread-confined: touches only lane state).
# In the measurement phase, adaptation is off (frozen) and measurements fire every
# `measure_interval` sweeps.
function _lane_segment!(lane::_PTLane, H::TiledHamiltonian, plan::UpdatePlan,
                        n::Int, measure::Bool)
    st = lane.st
    for _ = 1:n
        lane.phase_sweeps += 1
        _compound_sweep!(st, H, lane.β, lane.scs, plan)
        measure || (lane.phase_sweeps % plan.adapt_interval == 0 &&
                    _adapt_step!(st, plan.adapt_target))
        lane.phase_sweeps % plan.renorm_interval == 0 &&
            _renormalize!(st, H, lane.scs[1])
        if measure && lane.phase_sweeps % plan.measure_interval == 0
            view = MCView(H, st.config, st.disps, st.energy)
            for acc in lane.accs
                _measure!(acc, view)
            end
        end
    end
    return nothing
end

# One adjacent-pair swap attempt (the Metropolis rule on the payloads; `u` is the
# pre-attributed uniform). Shared by the serial and the async boundary code so the
# arithmetic can never drift apart.
@inline function _attempt_swap!(a::_PTLane, b::_PTLane, i::Int, u::Float64,
                                swap_att::Vector{Int}, swap_acc::Vector{Int})
    swap_att[i] += 1
    logw = (1 / a.kt - 1 / b.kt) * (a.st.energy - b.st.energy)
    if u < exp(min(0.0, logw))
        _swap_payload!(a.st, b.st)
        swap_acc[i] += 1
    end
    return nothing
end

# --- pairwise boundary synchronization (async lane schedule) --------------------------
#
# Between global sync points, every lane runs as its own task and an exchange
# boundary only synchronizes the two lanes of each attempted pair: the lower lane
# (the *performer*) waits for its partner to arrive, applies the swap attempt, and
# releases it — no lane ever waits for the whole ladder, so a straggler (an E-core
# lane, a renormalization) stalls its neighbors instead of every rung. The uniforms
# are pre-drawn per block in the serial consumption order, so the trajectory is the
# serial reference's, bit for bit (pt-threads-determinism.md P2/P3).
struct _PairSync
    conds::Vector{Threads.Condition}   # one per lane; conds[r] guards arrival[r]
                                       #   and (for the pair below r) released[r−1]
    arrival::Vector{Int}               # last boundary lane r announced (as responder)
    released::Vector{Int}              # last boundary pair i completed its swap
    failed::Threads.Atomic{Bool}       # poison flag — a dying task aborts the block
end

_PairSync(R::Int) = _PairSync([Threads.Condition() for _ = 1:R], zeros(Int, R),
                              zeros(Int, max(R - 1, 0)), Threads.Atomic{Bool}(false))

# Poison the block: wake every parked lane so it can observe `failed` and bail out
# (the @sync then surfaces the original exception — wrapped in the usual
# CompositeException/TaskFailedException — instead of livelocking).
function _poison!(ps::_PairSync)
    ps.failed[] = true
    for c in ps.conds
        lock(c)
        try
            notify(c)
        finally
            unlock(c)
        end
    end
    return nothing
end

# Lane `r`'s side of exchange boundary `k` (parity `p`, pre-drawn uniforms `u` for
# the attempted pairs in ascending order). Returns `false` when the block was
# poisoned (the caller exits quietly). Memory ordering: every cross-lane read
# (partner energy, swapped payload) happens after observing the partner's counter
# under that lane's condition lock.
function _boundary!(ps::_PairSync, lanes::Vector{_PTLane}, r::Int, k::Int, p::Int,
                    u::Vector{Float64}, swap_att::Vector{Int},
                    swap_acc::Vector{Int})::Bool
    R = length(lanes)
    if r <= R - 1 && mod(r - 1 - p, 2) == 0        # performer of pair (r, r + 1)
        c = ps.conds[r + 1]
        lock(c)
        try
            while ps.arrival[r + 1] < k
                ps.failed[] && return false
                wait(c)
            end
        finally
            unlock(c)
        end
        ps.failed[] && return false
        # the partner is parked waiting on `released` — its payload is quiescent
        _attempt_swap!(lanes[r], lanes[r + 1], r, u[(r - 1 - p) ÷ 2 + 1],
                       swap_att, swap_acc)
        lock(c)
        try
            ps.released[r] = k
            notify(c)
        finally
            unlock(c)
        end
    elseif r >= 2 && mod(r - 2 - p, 2) == 0        # responder of pair (r − 1, r)
        c = ps.conds[r]
        lock(c)
        try
            ps.arrival[r] = k
            notify(c)
            while ps.released[r - 1] < k
                ps.failed[] && return false
                wait(c)
            end
        finally
            unlock(c)
        end
    end                                            # edge lanes idle through this one
    return true
end

# Run every lane through one async block (`blk` sweeps in segments of `seglen`,
# pairwise handshakes at the first `nbound` segment ends). All lanes are globally
# in sync again when this returns — the caller may checkpoint.
function _pt_block_async!(lanes::Vector{_PTLane}, H::TiledHamiltonian,
                          plan::UpdatePlan, blk::Int, seglen::Int, nbound::Int,
                          measure::Bool, us::Vector{Vector{Float64}}, parity0::Int,
                          swap_att::Vector{Int}, swap_acc::Vector{Int})
    ps = _PairSync(length(lanes))
    @sync for r = 1:length(lanes)
        Threads.@spawn begin
            try
                left = blk
                k = 0
                while left > 0
                    n = min(seglen, left)
                    _lane_segment!(lanes[r], H, plan, n, measure)
                    left -= n
                    k += 1
                    k <= nbound || continue
                    _boundary!(ps, lanes, r, k, (parity0 + k - 1) % 2, us[k],
                               swap_att, swap_acc) || break
                end
            catch
                _poison!(ps)
                rethrow()
            end
        end
    end
    return nothing
end

# Sweeps until the next global sync point (checkpoint write or phase end): the
# smallest whole number of segments after which the checkpointer's `since`
# arithmetic (`_ck_pt!`) triggers a write, capped at the rest of the phase.
function _pt_block_sweeps(ck, left::Int, seglen::Int)::Int
    (ck === nothing || ck.interval <= 0) && return left
    return min(left, max(1, cld(ck.interval - ck.since, seglen)) * seglen)
end

# Run all lanes for one phase (`total` sweeps each) in segments of `seglen` sweeps,
# with adjacent-pair exchange attempts between segments (alternating even/odd pair
# parity, one uniform per attempted pair in ascending order). `ntasks == 1` is the
# serial reference schedule; `ntasks ≥ 2` runs one task per lane with pairwise
# boundary handshakes, globally re-syncing only at checkpoint writes and phase ends
# — bit-identical to serial (the uniforms are pre-drawn in the serial order and the
# boundary energies are chain-determined). `done0` resumes the phase mid-flight
# from a checkpoint; `ck` writes periodic checkpoints at segment boundaries.
# Returns the exchange parity to carry into the next phase.
function _run_pt_phase!(lanes::Vector{_PTLane}, H::TiledHamiltonian,
                        plan::UpdatePlan, total::Int, seglen::Int, measure::Bool,
                        exchange_rng::Xoshiro, swap_att::Vector{Int},
                        swap_acc::Vector{Int}, ntasks::Int, parity::Int;
                        done0::Int = 0, ck = nothing)::Int
    R = length(lanes)
    done = done0
    while done < total
        if ntasks <= 1
            n = min(seglen, total - done)
            for lane in lanes
                _lane_segment!(lane, H, plan, n, measure)
            end
            done += n
            if done < total
                for i = (1 + parity):2:(R - 1)
                    u = rand(exchange_rng)  # drawn unconditionally — determinism
                    _attempt_swap!(lanes[i], lanes[i + 1], i, u, swap_att, swap_acc)
                end
                parity = 1 - parity
            end
            _ck_pt!(ck, n, H, lanes, measure ? :measure : :therm, done, parity,
                    exchange_rng, swap_att, swap_acc)
        else
            blk = _pt_block_sweeps(ck, total - done, seglen)
            nbound = cld(blk, seglen) - (blk == total - done ? 1 : 0)
            # pre-draw the uniforms in the serial consumption order (boundary-
            # major, attempted pairs ascending) — the async schedule never
            # touches the stream, so the trajectory stays the serial one
            us = Vector{Float64}[[rand(exchange_rng)
                                  for _ = (1 + (parity + k - 1) % 2):2:(R - 1)]
                                 for k = 1:nbound]
            _pt_block_async!(lanes, H, plan, blk, seglen, nbound, measure, us,
                             parity, swap_att, swap_acc)
            done += blk
            parity = (parity + nbound) % 2
            _ck_pt!(ck, blk, H, lanes, measure ? :measure : :therm, done, parity,
                    exchange_rng, swap_att, swap_acc)
        end
    end
    return parity
end

# The shared phase driver of `run_pt` and a "pt"-kind `resume`.
function _pt_run!(lanes::Vector{_PTLane}, H::TiledHamiltonian, plan::UpdatePlan,
                  observables::Vector{Observable}, evaluables::Vector{Evaluable},
                  exchange_interval::Int, nt::Int, exchange_rng::Xoshiro,
                  swap_att::Vector{Int}, swap_acc::Vector{Int}, phase0::Symbol,
                  done0::Int, parity0::Int, ck)::PTResult
    parity = parity0
    mdone0 = 0
    if phase0 === :therm
        parity = _run_pt_phase!(lanes, H, plan, plan.sweeps_therm,
                                exchange_interval, false, exchange_rng, swap_att,
                                swap_acc, nt, parity; done0 = done0, ck = ck)
        planned = fld(plan.sweeps_measure, plan.measure_interval)
        for lane in lanes
            _renormalize!(lane.st, H, lane.scs[1])
            _warn_step_u_saturated(lane.st, H, plan)
            _freeze_and_reset!(lane.st)
            lane.accs = [ObsAccumulator(o, planned, plan.nbins)
                         for o in observables]
            lane.phase_sweeps = 0
        end
        # boundary checkpoint: the measurement phase starts fresh from this state
        ck === nothing ||
            _write_ckpt_pt(ck, H, lanes, :measure, 0, parity, exchange_rng,
                           swap_att, swap_acc)
    else
        mdone0 = done0
    end
    _run_pt_phase!(lanes, H, plan, plan.sweeps_measure, exchange_interval, true,
                   exchange_rng, swap_att, swap_acc, nt, parity; done0 = mdone0,
                   ck = ck)
    R = length(lanes)
    joint = has_disp(H)
    points = [let st = lane.st, s = _chain_summary(st, joint)
                  TempResult(lane.kt, lane.kt / KB_EV,
                             _finalize_stats(lane.accs, evaluables, lane.kt,
                                             H.n_spin_active, H.n_active),
                             s.acc_m, s.acc_o, s.acc_d, st.step, s.step_u,
                             st.max_drift, s.disp_rms, s.disp_max, s.disp_checks,
                             s.escaped)
              end
              for lane in lanes]
    swaps = [swap_att[i] == 0 ? NaN : swap_acc[i] / swap_att[i] for i = 1:(R - 1)]
    return PTResult(points, swaps, [copy(lane.st.config) for lane in lanes],
                    [_final_disps(H, lane.st) for lane in lanes], plan.seed)
end

"""
    run_pt(H::TiledHamiltonian; temperature = nothing, kT = nothing,
           exchange_interval = 10, ntasks = nothing, kwargs...) -> PTResult

Replica-exchange (parallel-tempering) Monte Carlo: one chain (**lane**) per rung of
a strictly monotone temperature ladder (**exactly one** of `temperature` [kelvin] /
`kT` [model energy units], length ≥ 2), all lanes sweeping concurrently over
threads. Every `exchange_interval` compound sweeps, adjacent rungs attempt to swap
their chain payloads with probability `min(1, exp((βᵢ−βⱼ)(Eᵢ−Eⱼ)))` (alternating
even/odd pairs) — so cold rungs keep escaping metastable basins through the hot end
of the ladder. Exchanges run during thermalization and measurement alike.

`ntasks = 1` runs the serial reference schedule; any `ntasks ≥ 2` (default when
threads are available) runs **every lane as its own task**, and an exchange
boundary synchronizes only the two lanes of each attempted pair — a straggling
lane stalls its neighbors, not the whole ladder (the ladder globally re-syncs only
at checkpoint writes and phase ends). `sweep_tasks` additionally parallelizes each
lane's own sweeps (color-parallel updates — keep `ntasks · sweep_tasks` within the
thread count; useful for short ladders on many cores). Results are **bit-identical
for a fixed seed regardless of `ntasks`, `sweep_tasks`, and the thread count** —
every random decision has a dedicated RNG whose consumption order is fixed by the
segment schedule (per-site RNGs inside sweeps; the exchange uniforms are pre-drawn
in serial order, one per attempted pair).

`checkpoint` / `checkpoint_interval` write restartable checkpoints at segment
boundaries (interval in sweeps, `0` ⇒ only at the thermalization→measurement
boundary); continue with [`resume`](@ref) — bit-identical to an uninterrupted run.

Everything else — `sweeps_therm`, `sweeps_measure`, `measure_interval`,
`or_per_metropolis`, `disp_per_metropolis`, `step` / `step_u` / `adapt_target` /
`adapt_interval` (adaptation is per-lane, thermalization-only), `renorm_interval`,
`nbins`, `observables`, `evaluables`, `init` / `disps` (every lane starts from them;
default: independent random spins at the clamped-ion displacements),
`seed` — as in [`run_mc`](@ref). Lane `r`'s statistics land in `points[r]`
(ladder order); adjacent-pair swap acceptances (diagnostic: aim for O(0.2–0.5),
tighten the ladder where they collapse) in `swap_acceptance`.
"""
function run_pt(H::TiledHamiltonian; temperature = nothing, kT = nothing,
                exchange_interval::Integer = 10,
                ntasks::Union{Nothing,Integer} = nothing,
                sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                measure_interval::Integer = 1, or_per_metropolis::Integer = 0,
                disp_per_metropolis::Union{Nothing,Integer} = nothing,
                step::Real = 0.6, step_u::Real = _DEFAULT_STEP_U,
                adapt_target::Real = 0.5,
                adapt_interval::Integer = 50, renorm_interval::Integer = 1_000,
                nbins::Integer = 32,
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(H),
                init = nothing, disps = nothing, sweep_tasks::Integer = 1,
                seed::Integer = rand(UInt64),
                checkpoint::Union{Nothing,AbstractString} = nothing,
                checkpoint_interval::Integer = 0)::PTResult
    kts = resolve_kt(temperature, kT)
    R = length(kts)
    R >= 2 || throw(ArgumentError("parallel tempering needs a ladder of ≥ 2 " *
                                  "temperatures; got $R"))
    (all(diff(kts) .> 0) || all(diff(kts) .< 0)) || throw(ArgumentError(
        "the temperature ladder must be strictly monotone; got $kts"))
    exchange_interval >= 1 || throw(ArgumentError(
        "exchange_interval must be ≥ 1; got $exchange_interval"))
    nt = ntasks === nothing ? min(R, Threads.nthreads()) : Int(ntasks)
    nt >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $nt"))
    ndisp = _resolve_disp_passes(H, disp_per_metropolis)
    plan = UpdatePlan(kts; sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = or_per_metropolis,
                      disp_per_metropolis = ndisp, step = step, step_u = step_u,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = false, sweep_tasks = sweep_tasks, seed = seed)
    _check_observables(observables)
    _warn_escape_cadence(H, plan)
    nt * sweep_tasks > Threads.nthreads() && @warn(
        "ntasks · sweep_tasks = $(nt * sweep_tasks) exceeds the " *
        "$(Threads.nthreads()) available threads; the run stays correct and " *
        "bit-identical but oversubscribed", maxlog = 1)
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "pt", Int(exchange_interval))

    # RNG discipline: master → one Xoshiro per lane (fixed order), then the
    # exchange RNG; initial configs come from each lane's own RNG.
    master = Xoshiro(plan.seed)
    lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                         rand(master, UInt64), rand(master, UInt64)) for _ = 1:R]
    exchange_rng = Xoshiro(rand(master, UInt64), rand(master, UInt64),
                           rand(master, UInt64), rand(master, UInt64))
    lanes = [_PTLane(ChainState(H, _initial_config(H, init, lane_rngs[r]),
                                lane_rngs[r], plan.step0; disps = disps,
                                step_u = plan.step_u0),
                     [SweepScratch(H) for _ = 1:plan.sweep_tasks], kts[r],
                     1.0 / kts[r], ObsAccumulator[], 0)
             for r = 1:R]
    swap_att = zeros(Int, R - 1)
    swap_acc = zeros(Int, R - 1)
    return _pt_run!(lanes, H, plan, observables, evaluables,
                    Int(exchange_interval), nt, exchange_rng, swap_att, swap_acc,
                    :therm, 0, 0, ck)
end
