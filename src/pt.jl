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
`n_rungs − 1`; over the whole run), each lane's `final_config` and `final_disps`,
the per-lane `final_strains`, and the run `seed`. Prints as a summary table.

`final_disps[r]` is lane `r`'s last displacement configuration in the
centre-of-mass-free frame (empty on a pure-spin model). Note it is the *lane's*
payload at the end of the run — a replica exchange moves whole physical states
between lanes, so it is a sample of rung `r`'s marginal, not the continuation of one
replica's trajectory.

`final_strains` is the per-lane vector of end-of-run cell scales on a strained (NPT)
run and `nothing` on a fixed-cell one (**`nothing` is not `ones(R)`**, the
[`MCView`](@ref) discipline). `final_disps[r]`'s absolute lengths are expressed at
`final_strains[r]`. A ladder warm start —
`run_pt(H; strain = sch, strain_init = r.final_strains, init = ..., ...)` (a scalar
`strain_init` broadcasts) — is fully self-consistent for at most ONE rung: per-lane
`init`/`disps` are not wired, so every other lane pairs its own start scale with a
shared configuration expressed at a different lane's scale (legal — the chain
re-equilibrates — just not the `run_mc` triple). The self-consistent per-lane
continuation is [`resume`](@ref).
"""
struct PTResult
    points::Vector{TempResult}
    swap_acceptance::Vector{Float64}
    final_configs::Vector{SpinConfig}
    final_disps::Vector{Vector{SVector{3,Float64}}}
    final_strains::Union{Nothing,Vector{Float64}}
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
    r.final_strains === nothing ||
        println(io, "  final cell scales s = ",
                join([@sprintf("%.6g", s) for s in r.final_strains], " "))
    return nothing
end

# One parallel-tempering lane: a rung of the ladder with its chain, scratch, and
# (during measurement) accumulators.
#
# `H` is the Hamiltonian the lane sweeps. On a fixed-cell run every lane holds the
# SAME object (the caller's), and the field never changes. On a strained run each
# lane holds its own coefficient clone (`_coefficient_clone`) — the lanes sit at
# different cell scales concurrently, so per-lane coefficient state is what removes
# the shared-`H` data race that used to make PT + strain a refusal — and an accepted
# exchange swaps the two lanes' `H` REFERENCES together with the chain payload: the
# installed coefficients describe the payload's cell scale, so they travel with it,
# and no reinstall is ever needed at a swap. `sctx` is the lane's strain context
# (`nothing` = fixed cell): the schedule is shared (immutable), the scratch per lane.
mutable struct _PTLane
    const st::ChainState
    const scs::Vector{SweepScratch}
    const kt::Float64
    const β::Float64
    accs::Vector{ObsAccumulator}
    phase_sweeps::Int              # sweeps done in the current phase
    H::TiledHamiltonian
    const sctx::Union{Nothing,Tuple{StrainSchedule,StrainScratch}}
end

# Swap the full replica bundle between two lanes: the chain payload AND the lanes'
# Hamiltonian references. The two halves are ONE operation on purpose — the installed
# coefficients describe the payload's cell scale, and swapping either half alone
# re-opens exactly the coefficients-vs-scale desync the per-lane clones exist to
# prevent. (On a fixed-cell run every lane holds the caller's `H` and the reference
# swap is a same-object no-op.)
function _swap_lanes!(a::_PTLane, b::_PTLane)
    _swap_payload!(a.st, b.st)
    a.H, b.H = b.H, a.H
    return nothing
end

# Swap the replica payload between two chains (reference swaps — O(1)).
# Call through `_swap_lanes!` — on a strained run the lane's Hamiltonian reference
# must travel with the scale this function moves.
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
    # displacements it scales. On a strained run the lane's Hamiltonian reference
    # travels with it (`_attempt_swap!`), keeping the installed coefficients paired
    # with the scale they describe.
    a.strain, b.strain = b.strain, a.strain
    return nothing
end

# Run `n` compound sweeps of one lane (thread-confined: touches only lane state —
# `lane.H` included, which on a strained run is the lane's own coefficient clone).
# In the measurement phase, adaptation is off (frozen) and measurements fire every
# `measure_interval` sweeps. The strain move mirrors `_run_temperature!`'s order
# exactly (compound sweep → strain → adaptation → renormalization → measurement)
# and draws only from the lane-owned `st.rng`, so it changes nothing about the
# exchange stream's consumption order.
function _lane_segment!(lane::_PTLane, plan::UpdatePlan, n::Int, measure::Bool)
    st = lane.st
    for _ = 1:n
        lane.phase_sweeps += 1
        _compound_sweep!(st, lane.H, lane.β, lane.scs, plan)
        lane.sctx !== nothing && lane.phase_sweeps % plan.strain_interval == 0 &&
            strain_move!(st, lane.H, lane.sctx[1], lane.sctx[2], lane.kt;
                         pressure = plan.pressure, step = plan.strain_step,
                         proposal = plan.strain_proposal)
        measure || (lane.phase_sweeps % plan.adapt_interval == 0 &&
                    _adapt_step!(st, plan.adapt_target))
        lane.phase_sweeps % plan.renorm_interval == 0 &&
            _renormalize!(st, lane.H, lane.scs[1])
        if measure && lane.phase_sweeps % plan.measure_interval == 0
            # a strained lane's view carries the cell scale, exactly as in `run_mc`
            view = lane.sctx === nothing ?
                   MCView(lane.H, st.config, st.disps, st.energy) :
                   MCView(lane.H, st.config, st.disps, st.energy, st.strain)
            for acc in lane.accs
                _measure!(acc, view)
            end
        end
    end
    return nothing
end

# The difference `W_a − W_b` of the lanes' β-conjugate bundle weights in the
# exchange ratio. Fixed cell: the configurational energies, as always. Strained:
# `W = E + n_cells·j0(s) + P·V(s)` — configurational plus the elastic `n_cells·j0`
# (a constant to the sweeps, but the lanes sit at different scales, so it stops
# being one here — same reason as `strain_move!`'s ΔE) plus the pressure term.
# Formed as the sum of the three DIFFERENCES, as `strain_delta_energy` does, never
# per-lane totals differenced: `n_cells·j0` and `P·V` are extensive while `ΔW` is
# not, so per-lane totals would compute the difference at `ulp(|W|)` granularity —
# a conditioning loss growing linearly with `n_cells` (the bracket gate's
# hand-derived logw mirrors this exact association). The `V^{N_mob}` proposal
# Jacobian does NOT appear: it is a β-independent property of the bundle itself, so
# it cancels exactly between the swapped and unswapped assignments — only
# β-conjugate content survives in `(β_a − β_b)·(W_a − W_b)`.
@inline function _swap_dweight(a::_PTLane, b::_PTLane, pressure::Float64)::Float64
    de = a.st.energy - b.st.energy
    a.sctx === nothing && return de
    # One schedule serves the whole ladder (the drivers hand every lane the same
    # object), so `a`'s is deliberately applied to `b`'s scale too — which also
    # keeps the partner's `sctx` out of the cross-lane read set at a boundary.
    sch = a.sctx[1]
    return de +
           sch.n_cells * (strain_j0(sch, a.st.strain) - strain_j0(sch, b.st.strain)) +
           pressure * (strain_volume(sch, a.st.strain) -
                       strain_volume(sch, b.st.strain))
end

# One adjacent-pair swap attempt (the Metropolis rule on the payloads; `u` is the
# pre-attributed uniform). Shared by the serial and the async boundary code so the
# arithmetic can never drift apart. On a fixed-cell run `_swap_dweight` reduces to
# the chain-energy difference and the rule is the NVT one, bit for bit.
@inline function _attempt_swap!(a::_PTLane, b::_PTLane, i::Int, u::Float64,
                                swap_att::Vector{Int}, swap_acc::Vector{Int},
                                pressure::Float64)
    swap_att[i] += 1
    logw = (1 / a.kt - 1 / b.kt) * _swap_dweight(a, b, pressure)
    if u < exp(min(0.0, logw))
        _swap_lanes!(a, b)
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
# (partner energy and strain, swapped payload) AND the performer's cross-lane
# writes (the partner's payload and its `H` reference in `_swap_lanes!`) happen
# between observing the partner's `arrival` counter and publishing `released`,
# both under that lane's condition lock.
function _boundary!(ps::_PairSync, lanes::Vector{_PTLane}, r::Int, k::Int, p::Int,
                    u::Vector{Float64}, swap_att::Vector{Int},
                    swap_acc::Vector{Int}, pressure::Float64)::Bool
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
                       swap_att, swap_acc, pressure)
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
function _pt_block_async!(lanes::Vector{_PTLane}, plan::UpdatePlan, blk::Int,
                          seglen::Int, nbound::Int, measure::Bool,
                          us::Vector{Vector{Float64}}, parity0::Int,
                          swap_att::Vector{Int}, swap_acc::Vector{Int})
    ps = _PairSync(length(lanes))
    @sync for r = 1:length(lanes)
        Threads.@spawn begin
            try
                left = blk
                k = 0
                while left > 0
                    n = min(seglen, left)
                    _lane_segment!(lanes[r], plan, n, measure)
                    left -= n
                    k += 1
                    k <= nbound || continue
                    _boundary!(ps, lanes, r, k, (parity0 + k - 1) % 2, us[k],
                               swap_att, swap_acc, plan.pressure) || break
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
function _run_pt_phase!(lanes::Vector{_PTLane}, plan::UpdatePlan, total::Int,
                        seglen::Int, measure::Bool, exchange_rng::Xoshiro,
                        swap_att::Vector{Int}, swap_acc::Vector{Int}, ntasks::Int,
                        parity::Int; done0::Int = 0, ck = nothing)::Int
    R = length(lanes)
    done = done0
    while done < total
        if ntasks <= 1
            n = min(seglen, total - done)
            for lane in lanes
                _lane_segment!(lane, plan, n, measure)
            end
            done += n
            if done < total
                for i = (1 + parity):2:(R - 1)
                    u = rand(exchange_rng)  # drawn unconditionally — determinism
                    _attempt_swap!(lanes[i], lanes[i + 1], i, u, swap_att, swap_acc,
                                   plan.pressure)
                end
                parity = 1 - parity
            end
            _ck_pt!(ck, n, lanes, measure ? :measure : :therm, done, parity,
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
            _pt_block_async!(lanes, plan, blk, seglen, nbound, measure, us,
                             parity, swap_att, swap_acc)
            done += blk
            parity = (parity + nbound) % 2
            _ck_pt!(ck, blk, lanes, measure ? :measure : :therm, done, parity,
                    exchange_rng, swap_att, swap_acc)
        end
    end
    return parity
end

# The shared phase driver of `run_pt` and a "pt"-kind `resume`.
function _pt_run!(lanes::Vector{_PTLane}, plan::UpdatePlan,
                  observables::Vector{Observable}, evaluables::Vector{Evaluable},
                  exchange_interval::Int, nt::Int, exchange_rng::Xoshiro,
                  swap_att::Vector{Int}, swap_acc::Vector{Int}, phase0::Symbol,
                  done0::Int, parity0::Int, ck)::PTResult
    parity = parity0
    mdone0 = 0
    if phase0 === :therm
        parity = _run_pt_phase!(lanes, plan, plan.sweeps_therm,
                                exchange_interval, false, exchange_rng, swap_att,
                                swap_acc, nt, parity; done0 = done0, ck = ck)
        planned = fld(plan.sweeps_measure, plan.measure_interval)
        for lane in lanes
            _renormalize!(lane.st, lane.H, lane.scs[1])
            _warn_step_u_saturated(lane.st, lane.H, plan)
            _freeze_and_reset!(lane.st)
            lane.accs = [ObsAccumulator(o, planned, plan.nbins)
                         for o in observables]
            lane.phase_sweeps = 0
        end
        # boundary checkpoint: the measurement phase starts fresh from this state
        ck === nothing ||
            _write_ckpt_pt(ck, lanes, :measure, 0, parity, exchange_rng,
                           swap_att, swap_acc)
    else
        mdone0 = done0
    end
    _run_pt_phase!(lanes, plan, plan.sweeps_measure, exchange_interval, true,
                   exchange_rng, swap_att, swap_acc, nt, parity; done0 = mdone0,
                   ck = ck)
    R = length(lanes)
    joint = has_disp(lanes[1].H)
    points = [let st = lane.st, s = _chain_summary(st, joint)
                  TempResult(lane.kt, lane.kt / KB_EV,
                             _finalize_stats(lane.accs, evaluables, lane.kt,
                                             lane.H.n_spin_active, lane.H.n_active),
                             s.acc_m, s.acc_o, s.acc_d, s.acc_s, st.step, s.step_u,
                             st.max_drift, s.disp_rms, s.disp_max, s.disp_checks,
                             s.escaped)
              end
              for lane in lanes]
    swaps = [swap_att[i] == 0 ? NaN : swap_acc[i] / swap_att[i] for i = 1:(R - 1)]
    return PTResult(points, swaps, [copy(lane.st.config) for lane in lanes],
                    [_final_disps(lane.H, lane.st) for lane in lanes],
                    first(lanes).sctx === nothing ? nothing :
                    [lane.st.strain for lane in lanes], plan.seed)
end

"""
    run_pt(H::TiledHamiltonian; temperature = nothing, kT = nothing,
           exchange_interval = 10, ntasks = nothing, kwargs...) -> PTResult

Replica-exchange (parallel-tempering) Monte Carlo: one chain (**lane**) per rung of
a strictly monotone temperature ladder (**exactly one** of `temperature` [kelvin] /
`kT` [model energy units], length ≥ 2), all lanes sweeping concurrently over
threads. Every `exchange_interval` compound sweeps, adjacent rungs attempt to swap
their chain payloads with probability `min(1, exp((βᵢ−βⱼ)(Wᵢ−Wⱼ)))` (alternating
even/odd pairs), where `W` is the configurational energy on a fixed-cell run and
the full `E + n_cells·j0(s) + P·V(s)` on a strained one — so cold rungs keep
escaping metastable basins through the hot end of the ladder. Exchanges run during
thermalization and measurement alike.

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

# NPT (fluctuating cell)

Passing a [`StrainSchedule`](@ref) as `strain` makes every lane an
isothermal–isobaric chain at the **same** hydrostatic pressure (`pressure_GPa` XOR
`pressure`, exactly as in [`run_mc`](@ref), together with `strain_interval` /
`strain_proposal` / `strain_step`; `strain_init` additionally takes a length-`R`
vector — one warm-start scale per lane, a previous run's
`PTResult.final_strains` — with a scalar broadcasting to the whole ladder).
Each lane then sweeps its own **coefficient
clone** of `H` — the lanes sit at different cell scales concurrently, so per-lane
coefficient state is what makes this sound — and an accepted exchange moves the
clone reference together with the payload, so a lane's installed coefficients
always describe its chain's scale. The caller's `H` itself is installed at the
schedule's reference scale `s = 1` up front (the identity checkpoints store) and is
never touched again: it is handed back at the reference, like `run_mc`.

The exchange rule generalizes, it is not re-derived per channel: the swap ratio of
two bundles `(config, u, s)` between rungs keeps only β-conjugate content — the
bundle's `V^{N_mob}` measure factor is β-independent and cancels exactly — leaving
`(βᵢ−βⱼ)(Wᵢ−Wⱼ)` with `W = E + n_cells·j0(s) + P·V(s)`.

[`pressure_diagnostics`](@ref) is **not usable** under `run_pt` (its per-instance
scratch assumes one serial chain, and its views come from the lane clones), and is
refused by name at entry; run the identity on an NPT `run_mc` chain instead. And as
on every strained run, `:energy` / `:specific_heat` are **configurational-only**:
they omit the fluctuating `n_cells·j0(s) + P·V(s)`, so the reported `C` is neither
`C_V` nor the NPT `C_P` — append [`npt_observables`](@ref) (same schedule and
pressure as the run; its pure closures are lane-clone-safe) and read `:enthalpy` /
`:npt_specific_heat` instead.
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
                checkpoint_interval::Integer = 0,
                strain::Union{Nothing,StrainSchedule} = nothing,
                strain_interval::Union{Nothing,Integer} = nothing,
                strain_proposal::Symbol = :logvolume,
                strain_step::Union{Nothing,Real} = nothing,
                strain_init::Union{Nothing,Real,AbstractVector{<:Real}} = nothing,
                pressure_GPa::Union{Nothing,Real} = nothing,
                pressure::Union{Nothing,Real} = nothing)::PTResult
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
    nstrain = _resolve_strain_moves(strain, strain_interval)
    p_model = _resolve_pressure(strain, pressure_GPa, pressure)
    s0s = _resolve_strain_init_pt(strain, strain_init, R)
    sstep = strain === nothing ? 0.0 :
            strain_step === nothing ? _default_strain_step(strain, strain_proposal) :
            Float64(strain_step)
    plan = UpdatePlan(kts; sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = or_per_metropolis,
                      disp_per_metropolis = ndisp, step = step, step_u = step_u,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = false, sweep_tasks = sweep_tasks, seed = seed,
                      strain_interval = nstrain, strain_proposal = strain_proposal,
                      strain_step = sstep, pressure = p_model)
    _check_observables(observables)
    strain === nothing && _refuse_npt_observables(observables)
    _warn_escape_cadence(H, plan)
    nt * sweep_tasks > Threads.nthreads() && @warn(
        "ntasks · sweep_tasks = $(nt * sweep_tasks) exceeds the " *
        "$(Threads.nthreads()) available threads; the run stays correct and " *
        "bit-identical but oversubscribed", maxlog = 1)
    if strain !== nothing
        # Refuse `pressure_diagnostics` at ENTRY, not at the first measurement: the
        # per-view identity check would throw anyway (the lanes measure through
        # coefficient clones), but only after the whole thermalization phase is
        # already spent. By observable name — the two raw diagnostics are the
        # scratch carriers.
        for o in observables
            o.name in (:strain_dEdV, :strain_invV) && throw(ArgumentError(
                "pressure_diagnostics cannot run under run_pt: its scratch assumes " *
                "one serial chain, and the lanes measure through per-lane " *
                "coefficient clones its identity check refuses. Run the " *
                "mechanical-equilibrium identity on an NPT run_mc chain instead."))
        end
        _check_strain_pairing(H, strain)
        # Install the reference into the CALLER's `H` (with the one-time flatness
        # recheck — this is the run's anchor, as in `run_mc`), then clone per lane.
        # `H` itself never enters a lane, so it stays at the reference for the whole
        # run — which is both the identity the checkpointer must capture below and
        # the state `run_mc` hands back.
        set_coefficients!(H, strain_coefficients(strain, 1.0))
    end
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "pt", Int(exchange_interval);
                            grid_fp = strain === nothing ? UInt64(0) :
                                      _grid_fingerprint(strain))

    # RNG discipline: master → one Xoshiro per lane (fixed order), then the
    # exchange RNG; initial configs come from each lane's own RNG.
    master = Xoshiro(plan.seed)
    lane_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                         rand(master, UInt64), rand(master, UInt64)) for _ = 1:R]
    exchange_rng = Xoshiro(rand(master, UInt64), rand(master, UInt64),
                           rand(master, UInt64), rand(master, UInt64))
    lanes = [let Hl = strain === nothing ? H : _coefficient_clone(H),
                 sctx_l = strain === nothing ? nothing : (strain, StrainScratch(H))
                 # the warm-start scale goes into the LANE's clone before its chain
                 # computes the initial energy (the (H, chain) contract from sweep
                 # one); the caller's `H` — and the checkpointer's fingerprint —
                 # stay at the reference installed above
                 s0s === nothing ||
                     set_coefficients!(Hl, strain_coefficients!(sctx_l[2].coef,
                                                                strain, s0s[r]);
                                       recheck_translation = false)
                 st = ChainState(Hl, _initial_config(H, init, lane_rngs[r]),
                                 lane_rngs[r], plan.step0; disps = disps,
                                 step_u = plan.step_u0)
                 s0s === nothing || (st.strain = s0s[r])
                 _PTLane(st, [SweepScratch(H) for _ = 1:plan.sweep_tasks], kts[r],
                         1.0 / kts[r], ObsAccumulator[], 0, Hl, sctx_l)
             end
             for r = 1:R]
    swap_att = zeros(Int, R - 1)
    swap_acc = zeros(Int, R - 1)
    return _pt_run!(lanes, plan, observables, evaluables,
                    Int(exchange_interval), nt, exchange_rng, swap_att, swap_acc,
                    :therm, 0, 0, ck)
end
