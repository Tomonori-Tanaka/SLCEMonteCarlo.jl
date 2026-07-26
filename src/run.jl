# The single-chain run driver: one temperature or a warm-started sweep (annealing).

# Validated run configuration (internal; also the checkpointable description of a
# run). `kts` is the resolved k_B·T ladder in model energy units.
struct UpdatePlan
    kts::Vector{Float64}
    sweeps_therm::Int
    sweeps_measure::Int
    measure_interval::Int
    or_per_metropolis::Int
    disp_per_metropolis::Int
    step0::Float64
    step_u0::Float64
    adapt_target::Float64
    adapt_interval::Int
    renorm_interval::Int
    nbins::Int
    carryover::Bool
    sweep_tasks::Int
    seed::UInt64

    function UpdatePlan(kts::Vector{Float64}; sweeps_therm::Integer,
                        sweeps_measure::Integer, measure_interval::Integer,
                        or_per_metropolis::Integer, disp_per_metropolis::Integer,
                        step::Real, step_u::Real, adapt_target::Real,
                        adapt_interval::Integer, renorm_interval::Integer,
                        nbins::Integer, carryover::Bool, seed::Integer,
                        sweep_tasks::Integer = 1)
        isempty(kts) && throw(ArgumentError("the temperature ladder is empty"))
        sweeps_therm >= 0 ||
            throw(ArgumentError("sweeps_therm must be ≥ 0; got $sweeps_therm"))
        sweeps_measure >= 1 ||
            throw(ArgumentError("sweeps_measure must be ≥ 1; got $sweeps_measure"))
        1 <= measure_interval <= sweeps_measure || throw(ArgumentError(
            "measure_interval must be in 1:sweeps_measure; got $measure_interval"))
        or_per_metropolis >= 0 || throw(ArgumentError(
            "or_per_metropolis must be ≥ 0; got $or_per_metropolis"))
        disp_per_metropolis >= 0 || throw(ArgumentError(
            "disp_per_metropolis must be ≥ 0; got $disp_per_metropolis"))
        step > 0 || throw(ArgumentError("step must be > 0; got $step"))
        step_u > 0 || throw(ArgumentError("step_u must be > 0; got $step_u"))
        0 < adapt_target < 1 || throw(ArgumentError(
            "adapt_target must be in (0, 1); got $adapt_target"))
        adapt_interval >= 1 ||
            throw(ArgumentError("adapt_interval must be ≥ 1; got $adapt_interval"))
        renorm_interval >= 1 ||
            throw(ArgumentError("renorm_interval must be ≥ 1; got $renorm_interval"))
        nbins >= 2 || throw(ArgumentError("nbins must be ≥ 2; got $nbins"))
        seed >= 0 || throw(ArgumentError("seed must be ≥ 0; got $seed"))
        sweep_tasks >= 1 ||
            throw(ArgumentError("sweep_tasks must be ≥ 1; got $sweep_tasks"))
        return new(kts, sweeps_therm, sweeps_measure, measure_interval,
                   or_per_metropolis, disp_per_metropolis, Float64(step),
                   Float64(step_u), Float64(adapt_target),
                   adapt_interval, renorm_interval, nbins, carryover,
                   Int(sweep_tasks), UInt64(seed))
    end
end

"""
    TempResult

One temperature's finalized results: the `kT` / `temperature` labels (model energy
units / kelvin), the observable `stats` (`Dict{Symbol,ObservableStat}` — raw
observables with binning errors and `τ_int`, evaluables jackknifed), the measured
Metropolis / overrelaxation / displacement acceptance fractions (measurement phase
only; `NaN` where not applicable), the frozen proposal widths `final_step` (radians)
and `final_step_u` (model length units), the worst incremental-energy `max_drift`
seen at measurement-phase renormalization points, and the displacement escape
diagnostics — `disp_rms`, `disp_max`, and the number `disp_checks` of measurements
behind them.

The three displacement diagnostics have **different temporal semantics**, and reading
one as another is the easy mistake:

- `disp_rms` is the phase **average** `√(⟨|u|²⟩)` over the `disp_checks`
  measurements, in the centre-of-mass-free frame — an equilibrium estimate (with no
  error bar: it is a diagnostic, not an observable; a proper `⟨u²⟩` with binning
  errors belongs in the observable list);
- `disp_max` is the largest single-site displacement seen **anywhere** in the phase —
  an extreme-value statistic, so its expectation grows with `disp_checks` and it is
  not comparable between runs of different length;
- `disp_checks` is how many renormalization points the phase contained. All of the
  above are `NaN` when it is zero, which happens when the measurement phase is
  shorter than `renorm_interval` — and it is also the number that says whether
  `escaped` means anything.

`escaped == true` means the escape detector (`docs/specs/updates-stationarity.md`
U8) found the r.m.s. displacement growing rather than stationary during the
**measurement** phase — evidence that the model's displacement energy is unbounded
below, in which case no displacement observable of this point means anything. It is
a diagnostic, not a hard failure: the run finishes and the flag travels with the
result (and through checkpoints) so a post-hoc analysis can screen on it.

`escaped == false` is only a verdict when `disp_checks` is large enough for the
detector to have been able to speak (the drivers warn up front when it is not — see
[`run_mc`](@ref)'s `renorm_interval`). With too few checks it means *not screened*,
not *clean*.
"""
struct TempResult
    kT::Float64
    temperature::Float64
    stats::Dict{Symbol,ObservableStat}
    acceptance_metropolis::Float64
    acceptance_or::Float64
    acceptance_disp::Float64
    final_step::Float64
    final_step_u::Float64
    max_drift::Float64
    disp_rms::Float64
    disp_max::Float64
    disp_checks::Int
    escaped::Bool
end

# The measurement-phase summary a finished chain contributes to its TempResult.
# One helper for `run_mc` and `run_pt` alike: a diagnostic that only one driver
# reports is a diagnostic that silently disappears from half the runs.
function _chain_summary(st::ChainState, joint::Bool)
    measured = joint && st.disp_checks > 0
    return (; acc_m = st.att_metro == 0 ? NaN : st.acc_metro / st.att_metro,
            acc_o = st.att_or == 0 ? NaN : st.acc_or / st.att_or,
            acc_d = st.att_disp == 0 ? NaN : st.acc_disp / st.att_disp,
            step_u = joint ? st.step_u : NaN,
            # the PHASE AVERAGE, not the last snapshot: on a handful of sites the
            # single-check r.m.s. scatters by ~1/√(6·n_disp) and would be read off the
            # summary table as an equilibrium value
            disp_rms = measured ? sqrt(st.disp_ms_sum / st.disp_checks) : NaN,
            disp_max = measured ? st.disp_max : NaN,
            disp_checks = joint ? st.disp_checks : 0,
            escaped = st.escape_warned)
end

Base.show(io::IO, p::TempResult) =
    print(io, "TempResult(kT=", @sprintf("%.6g", p.kT), ", ",
          length(p.stats), " stats)")

"""
    MCResult

Result of [`run_mc`](@ref): `points` (one [`TempResult`](@ref) per temperature, in
run order), the chain's `final_config` and `final_disps`, and the run `seed`. Prints
as a summary table.

`final_disps` is the chain's last displacement configuration in the sampler's
**centre-of-mass-free** frame (model length units) — empty on a pure-spin model. It
is the warm start for a continuation run (`run_mc(H; init = r.final_config,
disps = r.final_disps)`), which is why it leaves the sampler at all; it is one
sample, not an average, so it is not a substitute for the `:u2` observable.
"""
struct MCResult
    points::Vector{TempResult}
    final_config::SpinConfig
    final_disps::Vector{SVector{3,Float64}}
    seed::UInt64
end

Base.show(io::IO, r::MCResult) =
    print(io, "MCResult(", length(r.points), " temperatures, ",
          length(r.final_config), " sites)")

function Base.show(io::IO, ::MIME"text/plain", r::MCResult)
    println(io, "MCResult: ", length(r.points), " temperature(s), ",
            length(r.final_config), " sites, seed ", r.seed)
    _print_points_table(io, r.points, length(r.final_config))
    return nothing
end

# The displacement configuration a finished chain hands back: a copy on a joint
# model, empty on a pure-spin one (where the chain's all-zero vector describes
# nothing — the same convention `MCView` applies, so a caller cannot tell the two
# apart by accident and feed a meaningless warm start onward).
_final_disps(H::TiledHamiltonian, st::ChainState)::Vector{SVector{3,Float64}} =
    has_disp(H) ? copy(st.disps) : SVector{3,Float64}[]

# The shared points table of MCResult / PTResult text/plain printing. The
# displacement columns appear only on a joint run (they are NaN otherwise), and an
# escaped point is flagged inline — a run whose displacement statistics are
# meaningless must not print like any other.
function _print_points_table(io::IO, points::Vector{TempResult}, nsites::Int)
    # `acceptance_disp` too, not `disp_rms` alone: the r.m.s. is measured at
    # renormalization points, so a measurement phase shorter than `renorm_interval`
    # leaves it NaN on a run that did sample displacements — and the table would then
    # print as if the model were pure spin.
    joint = any(p -> !isnan(p.disp_rms) || !isnan(p.acceptance_disp), points)
    @printf(io, "  %-11s %-9s %-22s %-9s %-8s %-9s %-7s %-6s%s\n",
            "kT", "T[K]", "E/site", "C/kB", "|m|", "chi", "U", "acc",
            joint ? " u_rms    acc_u" : "")
    for p in points
        e = get(p.stats, :energy, nothing)
        estr = e === nothing ? "-" :
               @sprintf("%.6g ± %.2g", e.mean[1] / nsites, e.err[1] / nsites)
        @printf(io, "  %-11.5g %-9.4g %-22s %-9s %-8s %-9s %-7s %-6.3f",
                p.kT, p.temperature, estr,
                _stat_str(p.stats, :specific_heat), _stat_str(p.stats, :absm),
                _stat_str(p.stats, :susceptibility), _stat_str(p.stats, :binder),
                p.acceptance_metropolis)
        joint && @printf(io, " %-8.3g %-6.3f%s", p.disp_rms, p.acceptance_disp,
                         p.escaped ? "  ESCAPED" : "")
        println(io)
    end
    joint && any(p -> p.escaped, points) &&
        println(io, "  ESCAPED: the r.m.s. displacement grew rather than " *
                    "equilibrating — displacement observables at those points are " *
                    "meaningless (see the warning above)")
    return nothing
end

function _stat_str(stats::Dict{Symbol,ObservableStat}, name::Symbol)::String
    s = get(stats, name, nothing)
    s === nothing && return "-"
    return @sprintf("%.4g", s.mean[1])
end

# One compound sweep: a Metropolis spin sweep (ergodicity in the spins), then
# `or_per_metropolis` overrelaxation sweeps (decorrelation), then
# `disp_per_metropolis` displacement sweeps (ergodicity in the displacements).
#
# Any FIXED composition of π-stationary kernels is π-stationary, so the two ratios are
# efficiency knobs, not physics: each sweep is a Metropolis kernel for the same joint
# `exp(−βE)` (updates-stationarity.md U1/U7). What is NOT free is making the schedule
# depend on the chain's history — that is why they live in the plan and not in an
# adaptive rule. `disp_per_metropolis` is 0 on every pure-spin run, and
# `displacement_sweep!` is never even entered there, so the RNG consumption — and the
# whole trajectory — is bit-identical to the pre-M4 sampler.
function _compound_sweep!(st::ChainState, H::TiledHamiltonian, β::Float64,
                          scs::Vector{SweepScratch}, plan::UpdatePlan)
    metropolis_sweep!(st, H, β, scs)
    for _ = 1:plan.or_per_metropolis
        overrelaxation_sweep!(st, H, β, scs)
    end
    for _ = 1:plan.disp_per_metropolis
        displacement_sweep!(st, H, β, scs)
    end
    return nothing
end

# Run the chain at one temperature: thermalize (with step adaptation), freeze, then
# measure. Returns the TempResult. `phase0`/`sweep0`/`accs0` resume mid-temperature
# from a checkpoint (fresh entry: `:therm`, 0, `nothing`); `ck` writes periodic
# checkpoints with the completed `points` so far.
function _run_temperature!(st::ChainState, H::TiledHamiltonian, kt::Float64,
                           plan::UpdatePlan, observables::Vector{Observable},
                           evaluables::Vector{Evaluable};
                           phase0::Symbol = :therm, sweep0::Int = 0,
                           accs0::Union{Nothing,Vector{ObsAccumulator}} = nothing,
                           ck = nothing, temp_index::Int = 1,
                           points::Vector{TempResult} = TempResult[])::TempResult
    β = 1.0 / kt
    scs = [SweepScratch(H) for _ = 1:plan.sweep_tasks]
    local accs::Vector{ObsAccumulator}
    msweep0 = 0
    if phase0 === :therm
        st.frozen = false
        # Fresh entry (not a mid-therm resume): this is a new phase, at a new
        # temperature, so the escape detector re-anchors with `max_drift`. Its anchors
        # are r.m.s. values, and `rms ∝ √T` — carrying a colder rung's anchor into a
        # hotter one manufactures growth that is pure thermodynamics.
        sweep0 == 0 && (st.max_drift = 0.0; _reset_escape!(st))
        for sweep = (sweep0 + 1):plan.sweeps_therm
            _compound_sweep!(st, H, β, scs, plan)
            sweep % plan.adapt_interval == 0 && _adapt_step!(st, plan.adapt_target)
            sweep % plan.renorm_interval == 0 && _renormalize!(st, H, scs[1])
            _ck_mc!(ck, H, st, points, temp_index, :therm, sweep, nothing)
        end
        _renormalize!(st, H, scs[1])
        _warn_step_u_saturated(st, H, plan)
        _freeze_and_reset!(st)     # measurement-phase acceptances / drift only
        planned = fld(plan.sweeps_measure, plan.measure_interval)
        accs = [ObsAccumulator(o, planned, plan.nbins) for o in observables]
    else
        accs0 === nothing && throw(ArgumentError(
            "resuming a measurement phase requires the checkpointed accumulators"))
        accs = accs0
        msweep0 = sweep0
    end
    for sweep = (msweep0 + 1):plan.sweeps_measure
        _compound_sweep!(st, H, β, scs, plan)
        sweep % plan.renorm_interval == 0 && _renormalize!(st, H, scs[1])
        if sweep % plan.measure_interval == 0
            view = MCView(H, st.config, st.disps, st.energy)
            for acc in accs
                _measure!(acc, view)
            end
        end
        _ck_mc!(ck, H, st, points, temp_index, :measure, sweep, accs)
    end
    s = _chain_summary(st, has_disp(H))
    stats = _finalize_stats(accs, evaluables, kt, H.n_spin_active, H.n_active)
    return TempResult(kt, kt / KB_EV, stats, s.acc_m, s.acc_o, s.acc_d, st.step,
                      s.step_u, st.max_drift, s.disp_rms, s.disp_max, s.disp_checks,
                      s.escaped)
end

# The shared temperature loop of `run_mc` and a "mc"-kind `resume`: run temperatures
# `start_index:end`, resuming the first one mid-flight when the checkpointed
# `phase0`/`sweep0`/`accs0` say so.
function _mc_loop!(points::Vector{TempResult}, st::ChainState, H::TiledHamiltonian,
                   plan::UpdatePlan, observables::Vector{Observable},
                   evaluables::Vector{Evaluable}, start_index::Int, phase0::Symbol,
                   sweep0::Int, accs0::Union{Nothing,Vector{ObsAccumulator}},
                   ck)::MCResult
    for i = start_index:length(plan.kts)
        resuming = i == start_index && (phase0 !== :therm || sweep0 > 0)
        if !resuming && i > 1 && !plan.carryover
            _reset_config!(st, H, _initial_config(H, nothing, st.rng))
            st.step = plan.step0
            st.step_u = plan.step_u0     # both widths restart with the chain
        end
        p = _run_temperature!(st, H, plan.kts[i], plan, observables, evaluables;
                              phase0 = resuming ? phase0 : :therm,
                              sweep0 = resuming ? sweep0 : 0,
                              accs0 = resuming ? accs0 : nothing, ck = ck,
                              temp_index = i, points = points)
        push!(points, p)
        # boundary checkpoint: the next temperature starts fresh from this state
        ck === nothing ||
            _write_ckpt_mc(ck, H, st, points, i + 1, :therm, 0, nothing)
    end
    return MCResult(points, copy(st.config), _final_disps(H, st), plan.seed)
end

"""
    run_mc(H::TiledHamiltonian; temperature = nothing, kT = nothing, kwargs...)
        -> MCResult

Run single-spin Metropolis Monte Carlo on the tiled Hamiltonian at one absolute
temperature or a ladder of them. Provide **exactly one** of `temperature` (kelvin,
converted with [`KB_EV`](@ref) — assumes an eV-fitted model) or `kT` (`k_B·T` in the
model's energy units), scalar or collection. A collection runs **in the given
order** with the chain carried over (fresh thermalization at each value), so
ordering high → low is an annealing run; pass `carryover = false` for an
independent random restart per temperature.

# Keyword arguments
- `sweeps_therm = 2_000`: equilibration sweeps per temperature (one sweep = one
  single-spin attempt per **active** site; inactive, non-magnetic sites are frozen
  — see [`TiledHamiltonian`](@ref)). The proposal step adapts only here.
- `sweeps_measure = 10_000`: measurement sweeps per temperature.
- `measure_interval = 1`: measure every k-th sweep.
- `or_per_metropolis = 0`: overrelaxation sweeps mixed after each Metropolis sweep.
- `disp_per_metropolis = nothing`: displacement sweeps ([`displacement_sweep!`](@ref))
  after each Metropolis sweep. The default resolves to `1` on a joint model and `0`
  on a pure-spin one — a plain `0` would sample a joint model at its clamped-ion
  displacements, a different ensemble, silently. Any fixed value is valid (the
  composition of π-stationary kernels is π-stationary); it trades displacement
  decorrelation against spin decorrelation. Passing a nonzero value to a pure-spin
  model is an error, not a no-op.
- `step = 0.6`: initial proposal rotation scale (radians).
- `step_u = 0.01`: initial displacement proposal width, a **length** in the model's
  units (Å for a DFT-fitted model) — never an angle. Ignored on a pure-spin model.
- `adapt_target = 0.5`, `adapt_interval = 50`: acceptance target and window (in
  sweeps) of the thermalization-only step adaptation; both widths are adapted
  toward the same target on their own acceptance counters, and both are frozen
  during measurement (a history-dependent kernel would bias expectations) and
  reported as `final_step` / `final_step_u`.
- `renorm_interval = 1_000`: sweeps between renormalize + full-energy re-anchoring
  (drift is recorded in `max_drift`).
- `nbins = 32`: jackknife bins per temperature for the evaluables.
- `observables = standard_observables(H)`, `evaluables = standard_evaluables(H)`.
- `init = nothing`: chain start — a `3 × n_sites` matrix or a vector of 3-vectors
  (normalized), else uniform random.
- `disps = nothing`: displacement start, same two shapes (copied verbatim — not
  normalized), else the clamped-ion state `u = 0`. Nonzero displacements on a
  pure-spin model are an error.
- `carryover = true`: carry the chain state across the temperature ladder.
- `sweep_tasks = 1`: concurrent tasks executing each lattice sweep (color-parallel
  updates — sites sharing no cluster instance update simultaneously). The result
  is **bit-identical for any value**; use it to parallelize a single chain when no
  lane-level parallelism (PT) is using the threads. See the parallelism guide.
- `seed = rand(UInt64)`: drawn fresh per call by default, so repeated runs are
  independent samples. Pass a fixed value for a bit-reproducible run; either way
  the seed actually used is recorded in the result (`MCResult.seed`) and in
  checkpoints, so any run can be reproduced after the fact.
- `checkpoint = nothing`: a file path to write restartable checkpoints to (JLD2,
  schema: `docs/specs/checkpoint-schema.md`); continue with [`resume`](@ref). A
  resumed run is bit-identical to an uninterrupted one.
- `checkpoint_interval = 0`: sweeps between periodic checkpoint writes
  (`0` ⇒ write only at temperature boundaries).
"""
function run_mc(H::TiledHamiltonian; temperature = nothing, kT = nothing,
                sweeps_therm::Integer = 2_000, sweeps_measure::Integer = 10_000,
                measure_interval::Integer = 1, or_per_metropolis::Integer = 0,
                disp_per_metropolis::Union{Nothing,Integer} = nothing,
                step::Real = 0.6, step_u::Real = _DEFAULT_STEP_U,
                adapt_target::Real = 0.5,
                adapt_interval::Integer = 50, renorm_interval::Integer = 1_000,
                nbins::Integer = 32,
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(H),
                init = nothing, disps = nothing, carryover::Bool = true,
                sweep_tasks::Integer = 1, seed::Integer = rand(UInt64),
                checkpoint::Union{Nothing,AbstractString} = nothing,
                checkpoint_interval::Integer = 0)::MCResult
    ndisp = _resolve_disp_passes(H, disp_per_metropolis)
    plan = UpdatePlan(resolve_kt(temperature, kT); sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = or_per_metropolis,
                      disp_per_metropolis = ndisp, step = step, step_u = step_u,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = carryover, sweep_tasks = sweep_tasks,
                      seed = seed)
    _check_observables(observables)
    _warn_escape_cadence(H, plan)
    sweep_tasks > Threads.nthreads() && @warn(
        "sweep_tasks = $sweep_tasks exceeds the $(Threads.nthreads()) available " *
        "threads; the run stays correct and bit-identical but oversubscribed",
        maxlog = 1)
    ck = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "mc", 0)
    rng = Xoshiro(plan.seed)
    st = ChainState(H, _initial_config(H, init, rng), rng, plan.step0;
                    disps = disps, step_u = plan.step_u0)
    return _mc_loop!(TempResult[], st, H, plan, observables, evaluables, 1, :therm,
                     0, nothing, ck)
end

function _check_observables(observables::Vector{Observable})
    isempty(observables) && throw(ArgumentError("the observable list is empty"))
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    return nothing
end

# --- joint-run cadence diagnostics ---------------------------------------------------
#
# Both of these report a SILENCE: a configuration in which the escape detector or the
# displacement channel cannot say anything. The failure they prevent is the one U8 is
# about — a chain whose displacement numbers are meaningless while every diagnostic
# reads normal — so an unusable detector must itself be reported.

# How many renormalization checks a phase needs before `_ESCAPE_STRIKES` consecutive
# block strikes are even reachable. Derived from the ladder rather than hard-coded: the
# blocks double (1, 2, 4, …, capped at `_ESCAPE_WINDOW`), the first block has no
# predecessor to compare against, so strikes can only start at the second completed
# block.
function _escape_min_checks()::Int
    total = 0
    cap = 1
    for blk = 1:(_ESCAPE_STRIKES + 1)
        total += cap
        cap = min(2 * cap, _ESCAPE_WINDOW)
    end
    return total
end

# Warn once, up front, when the planned check count cannot support the block test —
# early enough that the user can lower `renorm_interval` before spending the run.
function _warn_escape_cadence(H::TiledHamiltonian, plan::UpdatePlan)
    (has_disp(H) && plan.disp_per_metropolis > 0) || return nothing
    need = _escape_min_checks()
    checks = fld(plan.sweeps_measure, plan.renorm_interval)
    checks >= need && return nothing
    @warn "this joint run gets $(checks) renormalization checks per measurement " *
          "phase (sweeps_measure = $(plan.sweeps_measure), renorm_interval = " *
          "$(plan.renorm_interval)), but the escape detector's block test needs " *
          "$(need) before it can raise $(_ESCAPE_STRIKES) consecutive strikes. Only " *
          "the absolute $(_ESCAPE_ABSOLUTE)× guard is live, which catches a fast " *
          "escape but not a slow one; `TempResult.escaped == false` will therefore " *
          "mean *not screened* rather than *clean*. Lower renorm_interval to " *
          "$(max(1, fld(plan.sweeps_measure, need))) or below to arm it." maxlog = 1
    return nothing
end

# Warn once when the displacement proposal width finished thermalization pinned at a
# clamp. The acceptance is the discriminator: at the ceiling with a HIGH acceptance the
# chain is running downhill (the clamp is hiding a runaway, which is exactly what U8
# says must never be silent); at the floor with a LOW one the displacement channel is
# effectively frozen and its marginal is unsampled.
function _warn_step_u_saturated(st::ChainState, H::TiledHamiltonian, plan::UpdatePlan)
    (has_disp(H) && plan.disp_per_metropolis > 0 && st.att_disp > 0) || return nothing
    a = st.acc_disp / st.att_disp
    if st.step_u >= _STEP_U_MAX
        @warn "the displacement proposal width finished thermalization pinned at its " *
              "ceiling $(_STEP_U_MAX) (model length units) with acceptance " *
              "$(round(a; digits = 3)). An acceptance this far above the target " *
              "$(plan.adapt_target) means the proposals are downhill, i.e. the chain " *
              "is not equilibrating but descending — check the model's dynamical " *
              "stability before using any displacement number from this run." maxlog = 1
    elseif st.step_u <= _STEP_U_MIN
        @warn "the displacement proposal width finished thermalization pinned at its " *
              "floor $(_STEP_U_MIN) (model length units) with acceptance " *
              "$(round(a; digits = 3)); the displacement channel is effectively " *
              "frozen and its marginal is not being sampled. The ΔE is also being " *
              "read off a near-cancellation at this width." maxlog = 1
    end
    return nothing
end

# Resolve the driver's `disp_per_metropolis` default. `nothing` means "whatever this
# Hamiltonian needs": one pass on a joint model, none on a pure-spin one. A plain
# integer default of 0 would be the silent-wrong-ensemble trap — a joint model sampled
# at frozen `u`; and a nonzero count on a pure-spin model is a no-op sweep over an empty
# site list, which almost certainly means the caller believes they are sampling
# displacements that this Hamiltonian does not have.
function _resolve_disp_passes(H::TiledHamiltonian,
                              disp_per_metropolis::Union{Nothing,Integer})::Int
    disp_per_metropolis === nothing && return has_disp(H) ? 1 : 0
    n = Int(disp_per_metropolis)
    n > 0 && !has_disp(H) && throw(ArgumentError(
        "disp_per_metropolis = $n, but this Hamiltonian has no displacement rows: it " *
        "describes the clamped-ion (u = 0) energy only, so the displacement sweeps " *
        "would have no site to attempt"))
    return n
end
