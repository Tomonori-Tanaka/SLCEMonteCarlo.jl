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
    # The NPT strain channel (all inert when `strain_interval == 0`, the fixed-cell
    # default): the outer move fires every `strain_interval`-th compound sweep, with a
    # symmetric step of width `strain_step` in `strain_proposal`'s variable, against a
    # hydrostatic `pressure` in MODEL units (eV/Å³ — the GPa conversion happens once,
    # at keyword resolution, never here).
    strain_interval::Int
    strain_proposal::Symbol
    strain_step::Float64
    pressure::Float64

    function UpdatePlan(kts::Vector{Float64}; sweeps_therm::Integer,
                        sweeps_measure::Integer, measure_interval::Integer,
                        or_per_metropolis::Integer, disp_per_metropolis::Integer,
                        step::Real, step_u::Real, adapt_target::Real,
                        adapt_interval::Integer, renorm_interval::Integer,
                        nbins::Integer, carryover::Bool, seed::Integer,
                        sweep_tasks::Integer = 1, strain_interval::Integer = 0,
                        strain_proposal::Symbol = :logvolume,
                        strain_step::Real = 0.0, pressure::Real = 0.0)
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
        strain_interval >= 0 || throw(ArgumentError(
            "strain_interval must be ≥ 0; got $strain_interval"))
        if strain_interval > 0
            _strain_check_proposal(strain_proposal)
            strain_step > 0 || throw(ArgumentError(
                "strain_step must be > 0 on a strained run; got $strain_step"))
            isfinite(pressure) || throw(ArgumentError(
                "pressure must be finite; got $pressure"))
        end
        return new(kts, sweeps_therm, sweeps_measure, measure_interval,
                   or_per_metropolis, disp_per_metropolis, Float64(step),
                   Float64(step_u), Float64(adapt_target),
                   adapt_interval, renorm_interval, nbins, carryover,
                   Int(sweep_tasks), UInt64(seed), Int(strain_interval),
                   strain_proposal, Float64(strain_step), Float64(pressure))
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

On a strained (NPT) run, `strain_min` / `strain_max` bracket the linear cell scales the
phase actually visited and `strain_outside` is the fraction of strain proposals that
fell outside the volume grid's [`strain_domain`](@ref) and were therefore rejected by
the grid rather than by the Boltzmann factor (all three `NaN` on a fixed-cell run,
where there is no scale to report — the same convention as `acceptance_strain`).

They exist because a chain the pressure pushes past the grid does not fail: a proposal
beyond the domain is rejected, never clamped, so the chain piles up at the edge with a
merely low strain acceptance — and a low acceptance is also what an oversized
`strain_step` gives. Its volume marginal is then truncated rather than sampled, which
is precisely the boundary term `:pressure`'s stationarity identity assumes negligible;
`:enthalpy` and `:npt_specific_heat` average over the truncated distribution. All three
keep returning confident finite numbers. `strain_outside` is the sharp diagnostic (a
rate, so it does not grow with run length the way the sampled extremes do) and the
drivers warn on it; the range is what says which edge.
"""
struct TempResult
    kT::Float64
    temperature::Float64
    stats::Dict{Symbol,ObservableStat}
    acceptance_metropolis::Float64
    acceptance_or::Float64
    acceptance_disp::Float64
    acceptance_strain::Float64       # NaN on a fixed-cell run
    strain_min::Float64              # the phase's sampled cell-scale range, and the
    strain_max::Float64              # fraction of strain proposals the volume grid
    strain_outside::Float64          # refused; all NaN on a fixed-cell run
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
            acc_s = st.att_strain == 0 ? NaN : st.acc_strain / st.att_strain,
            # gated on the same counter as `acc_s`: a chain that never attempted a
            # strain move has a scale, but not a sampled RANGE, and reporting the fixed
            # 1.0 as `[min, max]` would read as a measured — and reassuringly narrow —
            # volume distribution
            s_min = st.att_strain == 0 ? NaN : st.strain_min,
            s_max = st.att_strain == 0 ? NaN : st.strain_max,
            s_out = st.att_strain == 0 ? NaN :
                    st.att_strain_out / st.att_strain,
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
run order), the chain's `final_config`, `final_disps` and `final_strain`, and the
run `seed`. Prints as a summary table.

`final_disps` is the chain's last displacement configuration in the sampler's
**centre-of-mass-free** frame (model length units) — empty on a pure-spin model. It
is the warm start for a continuation run (`run_mc(H; init = r.final_config,
disps = r.final_disps)`), which is why it leaves the sampler at all; it is one
sample, not an average, so it is not a substitute for the `:u2` observable.

`final_strain` is the chain's last cell scale on a strained (NPT) run and `nothing`
on a fixed-cell one — **`nothing` is not `1.0`**, exactly as on [`MCView`](@ref): a
fixed-cell chain has no scale to report, and fabricating the reference would let a
strained consumer read a fixed-cell result as "ended at the reference". On a
strained run it completes the warm-start recipe —
`run_mc(H; strain = sch, strain_init = r.final_strain, init = r.final_config,
disps = r.final_disps, ...)` — and is the scale `final_disps`' absolute lengths are
expressed at. A warm start is a new chain (fresh RNG streams, adaptation restarted),
not a bit-identical continuation; for that, [`resume`](@ref) from a checkpoint.
"""
struct MCResult
    points::Vector{TempResult}
    final_config::SpinConfig
    final_disps::Vector{SVector{3,Float64}}
    final_strain::Union{Nothing,Float64}
    seed::UInt64
end

Base.show(io::IO, r::MCResult) =
    print(io, "MCResult(", length(r.points), " temperatures, ",
          length(r.final_config), " sites)")

function Base.show(io::IO, ::MIME"text/plain", r::MCResult)
    println(io, "MCResult: ", length(r.points), " temperature(s), ",
            length(r.final_config), " sites, seed ", r.seed)
    _print_points_table(io, r.points, length(r.final_config))
    # a strained result says where the cell ended; a fixed-cell one stays silent
    # (`nothing` is not a scale, so there is no line to print it on)
    r.final_strain === nothing ||
        println(io, "  final cell scale s = ", @sprintf("%.6g", r.final_strain))
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
    # Omitted, not refused, on a lattice-only model: `metropolis_sweep!` guards its own
    # entry (a direct call there is a caller mistake), but here the absence of spins is
    # the model's shape, and the displacement passes below are the whole run. The skip
    # consumes no randomness — the sweep's draws come from the per-site streams of the
    # sites it attempts — so every pure-spin trajectory is bit-identical to before.
    H.n_spin_active > 0 && metropolis_sweep!(st, H, β, scs)
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
# from a checkpoint (fresh entry: `:therm`, 0, `nothing`); `checkpointer` writes periodic
# checkpoints with the completed `points` so far.
function _run_temperature!(st::ChainState, H::TiledHamiltonian, kt::Float64,
                           plan::UpdatePlan, observables::Vector{Observable},
                           evaluables::Vector{Evaluable};
                           phase0::Symbol = :therm, sweep0::Int = 0,
                           accs0::Union{Nothing,Vector{ObsAccumulator}} = nothing,
                           checkpointer = nothing, temp_index::Int = 1,
                           points::Vector{TempResult} = TempResult[],
                           sctx::Union{Nothing,
                                       Tuple{StrainSchedule,StrainScratch}} = nothing,
                           )::TempResult
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
        sweep0 == 0 && (st.max_drift = 0.0; _reset_phase_diagnostics!(st))
        for sweep = (sweep0 + 1):plan.sweeps_therm
            _compound_sweep!(st, H, β, scs, plan)
            # The strain move runs during thermalization too — volume equilibration is
            # exactly what a fresh NPT chain needs — with a FIXED width (no
            # adaptation: the outer move fires too rarely to adapt on, and a wrong
            # width shows in `acceptance_strain` rather than a wrong ensemble).
            sctx !== nothing && sweep % plan.strain_interval == 0 &&
                strain_move!(st, H, sctx[1], sctx[2], kt; pressure = plan.pressure,
                             step = plan.strain_step, proposal = plan.strain_proposal,
                             check_pairing = false)   # checked once, at driver entry
            sweep % plan.adapt_interval == 0 && _adapt_step!(st, plan.adapt_target)
            sweep % plan.renorm_interval == 0 && _renormalize!(st, H, scs[1])
            _checkpoint_mc!(checkpointer, H, st, points, temp_index, :therm, sweep, nothing)
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
        sctx !== nothing && sweep % plan.strain_interval == 0 &&
            strain_move!(st, H, sctx[1], sctx[2], kt; pressure = plan.pressure,
                         step = plan.strain_step, proposal = plan.strain_proposal,
                         check_pairing = false)       # checked once, at driver entry
        sweep % plan.renorm_interval == 0 && _renormalize!(st, H, scs[1])
        if sweep % plan.measure_interval == 0
            # a strained run's view carries the cell scale; a fixed cell has no
            # strain DoF and the view says so with `nothing`, never a confident 1.0
            view = sctx === nothing ? MCView(H, st.config, st.disps, st.energy) :
                   MCView(H, st.config, st.disps, st.energy, st.strain)
            for acc in accs
                _measure!(acc, view)
            end
        end
        _checkpoint_mc!(checkpointer, H, st, points, temp_index, :measure, sweep, accs)
    end
    s = _chain_summary(st, has_disp(H))
    stats = _finalize_stats(accs, evaluables, kt, H.n_spin_active, H.n_active)
    return TempResult(kt, kt / KB_EV, stats, s.acc_m, s.acc_o, s.acc_d, s.acc_s,
                      s.s_min, s.s_max, s.s_out,
                      st.step, s.step_u, st.max_drift, s.disp_rms, s.disp_max,
                      s.disp_checks, s.escaped)
end

# The shared temperature loop of `run_mc` and a "mc"-kind `resume`: run temperatures
# `start_index:end`, resuming the first one mid-flight when the checkpointed
# `phase0`/`sweep0`/`accs0` say so.
function _mc_loop!(points::Vector{TempResult}, st::ChainState, H::TiledHamiltonian,
                   plan::UpdatePlan, observables::Vector{Observable},
                   evaluables::Vector{Evaluable}, start_index::Int, phase0::Symbol,
                   sweep0::Int, accs0::Union{Nothing,Vector{ObsAccumulator}},
                   checkpointer,
                   sctx::Union{Nothing,Tuple{StrainSchedule,StrainScratch}} = nothing,
                   )::MCResult
    for i = start_index:length(plan.kts)
        resuming = i == start_index && (phase0 !== :therm || sweep0 > 0)
        if !resuming && i > 1 && !plan.carryover
            # an independent restart is independent in EVERY channel: the cell returns
            # to the reference scale (coefficients first, so the energy recompute
            # below sees them), then the state resets. The float `!= 1.0` skip is
            # sound because `1.0` is only ever written EXACTLY (ctor, this reset)
            # and `st.strain == 1.0 ⟺ H at the reference` holds by construction —
            # keep that invariant if you touch either side.
            if sctx !== nothing && st.strain != 1.0
                set_coefficients!(H, strain_coefficients!(sctx[2].coef, sctx[1], 1.0);
                                  recheck_translation = false)
                st.strain = 1.0
            end
            _reset_config!(st, H, _initial_config(H, nothing, st.rng))
            st.step = plan.step0
            st.step_u = plan.step_u0     # both widths restart with the chain
        end
        p = _run_temperature!(st, H, plan.kts[i], plan, observables, evaluables;
                              phase0 = resuming ? phase0 : :therm,
                              sweep0 = resuming ? sweep0 : 0,
                              accs0 = resuming ? accs0 : nothing, checkpointer = checkpointer,
                              temp_index = i, points = points, sctx = sctx)
        push!(points, p)
        # boundary checkpoint: the next temperature starts fresh from this state
        checkpointer === nothing ||
            _write_ckpt_mc(checkpointer, H, st, points, i + 1, :therm, 0, nothing)
    end
    _warn_strain_boundary(points, sctx === nothing ? nothing : sctx[1], plan)
    return MCResult(points, copy(st.config), _final_disps(H, st),
                    sctx === nothing ? nothing : st.strain, plan.seed)
end

"""
    run_mc(H::TiledHamiltonian; temperature = nothing, kT = nothing, kwargs...)
        -> MCResult

Run single-spin Metropolis Monte Carlo on the tiled Hamiltonian at one absolute
temperature or a ladder of them. Provide **exactly one** of `temperature` (kelvin,
converted with `KB_EV` — assumes an eV-fitted model) or `kT` (`k_B·T` in the
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

# NPT (fluctuating cell) keyword arguments

Passing a [`StrainSchedule`](@ref) as `strain` turns the run isothermal–isobaric:
an outer strain move ([`strain_move!`](@ref)) rescales the cell over the schedule's
volume grid. Without it the run samples the **constant-strain (fixed cell)
ensemble** — a different ensemble (`F(T, ε)`, no volume Jacobian, no `P·V`), which
is what a fixed-geometry magnetostriction calculation wants. The chain starts at
the reference scale `s = 1` (or at `strain_init`), which must lie inside the
schedule's domain, and `H` is handed back at the reference when the run returns —
never at the chain's final scale (the chain's own end scale is
`MCResult.final_strain`).

- `strain = nothing`: the sampler-side volume grid, built by
  [`StrainSchedule(sm, H)`](@ref) against **this** Hamiltonian (checked).
- `pressure_GPa` XOR `pressure`: the hydrostatic pressure, in GPa or in the model's
  own units (eV/Å³ for an eV/Å-fitted model). An NPT run requires **exactly one**,
  explicitly — `0.0` is a physical choice, not a default; a fixed-cell run accepts
  neither. Hydrostatic only in v0: `P·V(ε)` is a state function with no
  strain-measure ambiguity, while a general applied stress is work-conjugate to a
  specific strain measure.
- `strain_interval = nothing`: compound sweeps between strain attempts (resolves to
  `1` with a schedule, `0` without; an explicit value contradicting the schedule's
  presence is an error, mirroring `disp_per_metropolis`).
- `strain_proposal = :logvolume`: the symmetric proposal variable — `:logvolume`
  (step in `ln V`) or `:scale` (step in the linear scale `s`).
- `strain_step = nothing`: proposal width in that variable; defaults to a tenth of
  the schedule's domain. Fixed for the whole run (never adapted) — a poor width
  shows up in `TempResult.acceptance_strain`, not in the sampled ensemble.
- `strain_init = nothing`: the cell scale the chain STARTS at (default: the
  reference `s = 1`; must lie inside the schedule's domain; requires `strain`).
  With `init` / `disps` this is the strained warm start — continue from a previous
  run's `final_config` / `final_disps` / `final_strain` without re-thermalizing
  from the reference cell; `disps` are absolute lengths at that scale, exactly as
  `final_disps` left them. A warm start is a NEW chain (fresh RNG streams,
  adaptation restarted — with `sweeps_therm = 0` there is no adaptation at all, so
  forward the parent's tuned widths too: `step = r.points[end].final_step,
  step_u = r.points[end].final_step_u`), not a bit-identical continuation — that
  is [`resume`](@ref)'s job. On a multi-temperature ladder with `carryover = false`,
  the independent restarts between temperatures return to the reference (they
  discard `init` the same way); only the first temperature starts at
  `strain_init`.

On a strained run `:energy` / `:specific_heat` are configurational-only (neither
`C_V` nor the NPT `C_P`); append [`npt_observables`](@ref) — built with this run's
schedule and pressure — for the β-conjugate `:enthalpy` and the isobaric
`:npt_specific_heat`.
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
                checkpoint_interval::Integer = 0,
                strain::Union{Nothing,StrainSchedule} = nothing,
                strain_interval::Union{Nothing,Integer} = nothing,
                strain_proposal::Symbol = :logvolume,
                strain_step::Union{Nothing,Real} = nothing,
                strain_init::Union{Nothing,Real} = nothing,
                pressure_GPa::Union{Nothing,Real} = nothing,
                pressure::Union{Nothing,Real} = nothing)::MCResult
    ndisp = _resolve_disp_passes(H, disp_per_metropolis)
    nor = _resolve_or_passes(H, or_per_metropolis)
    _require_moves(H, ndisp)
    s0 = _resolve_strain_init(strain, strain_init)
    nstrain = _resolve_strain_moves(strain, strain_interval)
    p_model = _resolve_pressure(strain, pressure_GPa, pressure)
    sstep = strain === nothing ? 0.0 :
            strain_step === nothing ? _default_strain_step(strain, strain_proposal) :
            Float64(strain_step)
    plan = UpdatePlan(resolve_kt(temperature, kT); sweeps_therm = sweeps_therm,
                      sweeps_measure = sweeps_measure,
                      measure_interval = measure_interval,
                      or_per_metropolis = nor,
                      disp_per_metropolis = ndisp, step = step, step_u = step_u,
                      adapt_target = adapt_target, adapt_interval = adapt_interval,
                      renorm_interval = renorm_interval, nbins = nbins,
                      carryover = carryover, sweep_tasks = sweep_tasks,
                      seed = seed, strain_interval = nstrain,
                      strain_proposal = strain_proposal, strain_step = sstep,
                      pressure = p_model)
    _check_observables(observables)
    _check_evaluables(observables, evaluables)
    strain === nothing && _refuse_npt_observables(observables)
    _warn_escape_cadence(H, plan)
    sweep_tasks > Threads.nthreads() && @warn(
        "sweep_tasks = $sweep_tasks exceeds the $(Threads.nthreads()) available " *
        "threads; the run stays correct and bit-identical but oversubscribed",
        maxlog = 1)
    sctx = nothing
    if strain !== nothing
        _check_strain_pairing(H, strain)
        sc = StrainScratch(H)
        # establish the (H, s = 1) contract the move maintains — with the one-time
        # translation-flatness recheck, since this install is the run's anchor
        set_coefficients!(H, strain_coefficients!(sc.coef, strain, 1.0))
        sctx = (strain, sc)
    end
    # the checkpointer captures the model fingerprint HERE, while `H` carries the
    # reference coefficients — the identity a strained run's file must store
    checkpointer = _make_checkpointer(checkpoint, checkpoint_interval, H, plan, observables,
                            "mc", 0;
                            grid_fp = strain === nothing ? UInt64(0) :
                                      _grid_fingerprint(strain))
    # the warm-start scale is installed AFTER the checkpointer captured the
    # reference-scale fingerprint (order load-bearing, strain-move.md S7) and
    # BEFORE the chain computes its initial energy below — the (H, chain)
    # contract holds from the first sweep. (`sctx` is never `nothing` when `s0`
    # isn't — `_resolve_strain_init` requires the schedule — the extra guard
    # just proves it to the type layer.)
    if s0 !== nothing && sctx !== nothing
        set_coefficients!(H, strain_coefficients!(sctx[2].coef, sctx[1], s0);
                          recheck_translation = false)
    end
    rng = Xoshiro(plan.seed)
    st = ChainState(H, _initial_config(H, init, rng), rng, plan.step0;
                    disps = disps, step_u = plan.step_u0)
    s0 === nothing || (st.strain = s0)
    r = _mc_loop!(TempResult[], st, H, plan, observables, evaluables, 1, :therm,
                  0, nothing, checkpointer, sctx)
    # hand `H` back at the REFERENCE scale, not wherever the chain ended: the caller's
    # next `model_fingerprint` / `total_energy` / fixed-cell run would otherwise see
    # a silently rescaled model (checkpoint writes are unaffected — the checkpointer
    # captured its fingerprint at construction)
    sctx === nothing ||
        set_coefficients!(H, strain_coefficients!(sctx[2].coef, sctx[1], 1.0);
                          recheck_translation = false)
    return r
end

function _check_observables(observables::Vector{Observable})
    isempty(observables) && throw(ArgumentError("the observable list is empty"))
    allunique(o.name for o in observables) ||
        throw(ArgumentError("observable names must be unique"))
    return nothing
end

# The evaluables' half of the same entry check, and it is worth as much as the
# observables' half: `_finalize_stats` raises exactly these three errors, but it runs
# AFTER the measurement phase, so a mistyped input name costs the whole run's samples
# (a resume re-throws at the same place — the accumulators are checkpointed, the
# finalized result is not). The name collision is the quiet one: `_finalize_stats`
# writes raw stats and evaluables into one `Dict`, so an evaluable named after an
# observable silently REPLACES that observable's binning result, and everything
# downstream — the printed table, the stored `TempResult` — reports the substitute.
function _check_evaluables(observables::Vector{Observable}, evaluables::Vector{Evaluable})
    isempty(evaluables) && return nothing
    byname = Dict(o.name => o for o in observables)
    allunique(e.name for e in evaluables) ||
        throw(ArgumentError("evaluable names must be unique"))
    for ev in evaluables
        haskey(byname, ev.name) && throw(ArgumentError(
            "evaluable :$(ev.name) has the same name as a measured observable; it " *
            "would replace that observable's statistics in the result. Rename one."))
        for name in ev.inputs
            obs = get(byname, name, nothing)
            obs === nothing && throw(ArgumentError(
                "evaluable :$(ev.name) needs observable :$name, which is not measured"))
            obs.ncomp == 1 || throw(ArgumentError(
                "evaluable :$(ev.name) input :$name is not a scalar observable " *
                "(ncomp = $(obs.ncomp))"))
        end
    end
    return nothing
end

# Refuse `npt_observables` on a FIXED-CELL run at entry, by observable name (the
# run_pt/pressure_diagnostics precedent): the per-view `strain(v)` throw is loud but
# fires only at the first measurement — after the whole thermalization phase is spent.
function _refuse_npt_observables(observables::Vector{Observable})
    for o in observables
        o.name in (:enthalpy, :enthalpy2) && throw(ArgumentError(
            "npt_observables (`:$(o.name)`) needs a strained run: a fixed-cell run " *
            "has no volume degree of freedom, so there is no `j0(s)` or `P·V(s)` to " *
            "measure — its `:energy` / `:specific_heat` are already the ensemble's " *
            "conjugate pair. Pass the run's `strain` schedule, or rename a " *
            "same-named observable of your own."))
        # `pressure_diagnostics` reads the cell scale through `strain(v)` too, and it
        # is constructible without a run (it needs only a paired schedule and `H`), so
        # the same late-throw hazard applies: without this it thermalizes first and
        # dies at the first measurement.
        o.name in (:strain_dEdV, :strain_invV) && throw(ArgumentError(
            "pressure_diagnostics (`:$(o.name)`) needs a strained run: a fixed-cell " *
            "run has no volume to differentiate with respect to. Pass the run's " *
            "`strain` schedule, or rename a same-named observable of your own."))
    end
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

# Screen a finished strained run for a chain that piled up against the volume grid.
#
# A proposal outside `strain_domain` is REJECTED, never clamped — correct as a move
# rule (a truncating clamp is an asymmetric proposal), but it means a chain the
# pressure pushes past the grid does not fail: it sits at the edge with a low strain
# acceptance and keeps producing numbers. Every mechanical observable then describes a
# volume-CLAMPED cell rather than the NPT ensemble it names — `:pressure`'s
# integration-by-parts identity drops exactly the boundary term it assumes negligible,
# and `:enthalpy` / `:npt_specific_heat` average `W` over a truncated volume marginal
# — while still returning confident finite values. Nothing else notices: the strain
# acceptance is merely low, not zero, and a low acceptance is also what an oversized
# `strain_step` gives.
#
# TWO conditions, because neither alone separates the pathology from a merely
# inefficient run. Measured on the `_ss_zeta_grid` fixture (domain [0.9, 1.1], (2,1,1),
# kT = 0.05, 5000 measurement sweeps):
#
#   P = 0,    strain_step = 0.05   outside  0.0 %  acc 0.71  range [0.9396, 1.0628]
#   P = 0,    strain_step = 0.25   outside 24.6 %  acc 0.25  range [0.9434, 1.0565]
#   P = -0.5, strain_step = 0.05   outside 49.8 %  acc 0.03  range [1.0967, 1.1000]
#   P = -1.0, strain_step = 0.05   outside 50.8 %  acc 0.02  range [1.0981, 1.1000]
#   P = +1.0, strain_step = 0.05   outside 48.2 %  acc 0.02  range [0.9000, 0.9028]
#
# Row 2 is the trap: an oversized step throws a quarter of its proposals out of the
# domain while the chain's marginal stays entirely INSIDE it. That run is inefficient,
# not wrong — the rejections are ordinary Metropolis rejections and no boundary term is
# being dropped — so a rate-only rule false-fires on it. Conversely a margin-only rule
# fails the other way: `strain_min`/`strain_max` are extreme-value statistics, so a long
# healthy run eventually brushes the edge once.
#
# So: the grid must be absorbing real proposal mass AND the chain must actually be
# sitting at an edge. The thresholds bracket the measurements with room on both sides —
# 5 % against 0.0 % healthy and ~50 % pinned; 5 % of the domain width (0.01 here)
# against ≤ 1.7 % for every pinned row and 21.5 % for the oversized-step row.
const _STRAIN_OUTSIDE_WARN = 0.05
const _STRAIN_EDGE_WARN = 0.05

function _warn_strain_boundary(points::Vector{TempResult},
                               sch::Union{Nothing,StrainSchedule}, plan::UpdatePlan)
    sch === nothing && return nothing
    lo, hi = strain_domain(sch)
    margin = _STRAIN_EDGE_WARN * (hi - lo)
    for p in points
        isnan(p.strain_outside) && continue
        p.strain_outside > _STRAIN_OUTSIDE_WARN || continue
        at_lo = p.strain_min - lo < margin
        at_hi = hi - p.strain_max < margin
        (at_lo || at_hi) || continue
        @warn "at kT = $(p.kT) the chain is pinned against the " *
              "$(at_lo ? "lower" : "upper") edge of the volume grid's domain " *
              "[$lo, $hi]: it sampled cell scales " *
              "[$(p.strain_min), $(p.strain_max)] and the grid refused " *
              "$(round(100 * p.strain_outside; digits = 1)) % of the strain proposals " *
              "for falling outside it (strain acceptance $(p.acceptance_strain)). A " *
              "proposal beyond the grid is rejected rather than clamped, so the cell " *
              "is effectively held there and this point's volume marginal is " *
              "TRUNCATED rather than sampled: `:pressure`'s stationarity identity " *
              "drops exactly the boundary term it assumes negligible, and " *
              "`:enthalpy` / `:npt_specific_heat` average over the truncated " *
              "distribution — all three still return confident finite numbers. Widen " *
              "the volume grid to cover the equilibrium volume at this pressure and " *
              "temperature."
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

# The overrelaxation analogue. Asked for on a model with no `l = 1` channel anywhere,
# the sweep walks every site, skips every one of them and returns 0 — the same silent
# no-op `_resolve_disp_passes` refuses on the displacement side, and the asymmetry was
# real: a lattice-only model (or a purely biquadratic spin model) accepted
# `or_per_metropolis > 0` and reported `acceptance_or = NaN` from a run that had done
# nothing with it. Zero stays legal on every model — that is the default, not a request.
function _resolve_or_passes(H::TiledHamiltonian, or_per_metropolis::Integer)::Int
    n = Int(or_per_metropolis)
    n > 0 && !any(H.site_has_l1) && throw(ArgumentError(
        "or_per_metropolis = $n, but no site of this Hamiltonian carries an l = 1 " *
        "spin channel$(H.lmax < 0 ? " (it is a lattice-only model)" : ""): the " *
        "overrelaxation move reflects a spin about its local field, which is built " *
        "from exactly that channel, so every site would be skipped and the sweep " *
        "would attempt no move at all"))
    return n
end

# A run must be able to move SOMETHING. Mirrors the guard the device driver has
# carried since G8 (`gpu_run_sweeps!`): with no spin-active site and no displacement
# pass, `run_mc` walks its whole sweep budget attempting nothing and returns
# zero-variance "results" — `E = 0.0 ± 0.0`, `NaN` acceptances, the initial
# configuration handed back unchanged — which reads as a converged run, not as a
# refusal. `n_disp_active`, not `has_disp`: displacement rows whose couplings all
# fitted to zero give a sweep with no site to attempt either.
#
# Note this is deliberately NOT `_require_spin_sites`: a lattice-only model with a
# displacement pass is a first-class run here, and `_compound_sweep!` simply omits the
# spin sweep for it.
function _require_moves(H::TiledHamiltonian, ndisp::Int)
    H.n_spin_active > 0 || (ndisp > 0 && H.n_disp_active > 0) || throw(ArgumentError(
        "this run would attempt no move at all: the Hamiltonian has no spin-active " *
        "site and no displacement pass to run. It has " *
        "$(H.n_spin_active) spin-active and $(H.n_disp_active) displacement-active " *
        "sites, with disp_per_metropolis = $ndisp"))
    return nothing
end

# Resolve the strain cadence, mirroring `_resolve_disp_passes`: `nothing` means
# "whatever the schedule's presence implies", and an explicit value contradicting it
# is named rather than silently reconciled.
function _resolve_strain_moves(sch::Union{Nothing,StrainSchedule},
                               strain_interval::Union{Nothing,Integer})::Int
    strain_interval === nothing && return sch === nothing ? 0 : 1
    n = Int(strain_interval)
    n > 0 && sch === nothing && throw(ArgumentError(
        "strain_interval = $n, but no strain schedule was passed: there is no volume " *
        "grid to move the cell on. Pass `strain = StrainSchedule(sm, H)`."))
    n == 0 && sch !== nothing && throw(ArgumentError(
        "a strain schedule was passed but strain_interval = 0 would never move the " *
        "cell — the run would sample the fixed-cell ensemble at s = 1 while looking " *
        "like an NPT run. Drop `strain`, or use strain_interval ≥ 1."))
    return n
end

# Resolve the hydrostatic pressure to MODEL units. The `temperature` XOR `kT`
# discipline, applied to pressure (design record §8 (ε)): an NPT run states its
# pressure in exactly one named unit — `0.0` included, it is a physical choice and
# not a default — and a fixed-cell run has no `P·V` term to accept one for.
function _resolve_pressure(sch::Union{Nothing,StrainSchedule},
                           pressure_GPa::Union{Nothing,Real},
                           pressure::Union{Nothing,Real})::Float64
    if sch === nothing
        pressure_GPa === nothing && pressure === nothing || throw(ArgumentError(
            "a pressure was passed without a strain schedule; a fixed-cell run " *
            "samples the constant-strain ensemble, which has no P·V term"))
        return 0.0
    end
    (pressure_GPa === nothing) == (pressure === nothing) && throw(ArgumentError(
        "an NPT run needs exactly one of `pressure_GPa` (GPa) or `pressure` " *
        "(model units, eV/Å³ for an eV/Å-fitted model) — 0.0 is a physical choice, " *
        "not a default"))
    p = pressure === nothing ? Float64(pressure_GPa) / GPA_PER_EV_A3 : Float64(pressure)
    isfinite(p) || throw(ArgumentError("the pressure must be finite; got $p"))
    return p
end

# Resolve the warm-start scale: requires a schedule and must lie inside its domain.
# The domain check is an ENTRY guard for two silent-then-loud consequences: the
# install would EXTRAPOLATE the Horner interpolant outside the grid (no domain
# check in `strain_coefficients!` — silent), and the first strain move would then
# throw on its out-of-domain current scale mid-run, after the setup is spent.
# `nothing` means the reference, exactly as before the keyword existed.
function _resolve_strain_init(sch::Union{Nothing,StrainSchedule},
                              strain_init::Union{Nothing,Real})::Union{Nothing,
                                                                       Float64}
    strain_init === nothing && return nothing
    sch === nothing && throw(ArgumentError(
        "strain_init was passed without a strain schedule; a fixed-cell run has " *
        "no cell scale to start at"))
    s0 = Float64(strain_init)
    in_strain_domain(sch, s0) || throw(ArgumentError(
        "strain_init = $s0 is outside this schedule's domain " *
        "$(strain_domain(sch)); a warm start must begin inside the grid"))
    return s0
end

# The ladder form of the same resolution: a scalar broadcasts to every rung (one
# common warm-start cell), a vector is per-lane (a previous `PTResult.final_strains`);
# each entry passes the scalar rule. The no-schedule refusal is hoisted so the
# per-entry loop runs with a narrowed `sch::StrainSchedule` (type layer).
function _resolve_strain_init_pt(sch::Union{Nothing,StrainSchedule}, strain_init,
                                 R::Int)::Union{Nothing,Vector{Float64}}
    strain_init === nothing && return nothing
    sch === nothing && throw(ArgumentError(
        "strain_init was passed without a strain schedule; a fixed-cell run has " *
        "no cell scale to start at"))
    strain_init isa AbstractVector || return fill(
        _resolve_strain_init(sch, strain_init)::Float64, R)
    length(strain_init) == R || throw(ArgumentError(
        "strain_init has $(length(strain_init)) entries for a ladder of $R rungs"))
    return Float64[_resolve_strain_init(sch, s) for s in strain_init]
end

# A tenth of the schedule's domain in the proposal's own variable — wide enough to
# cross the grid in a few accepted moves, narrow enough for a usable acceptance.
function _default_strain_step(sch::StrainSchedule, proposal::Symbol)::Float64
    _strain_check_proposal(proposal)
    lo, hi = strain_domain(sch)
    return (_strain_y(proposal, hi) - _strain_y(proposal, lo)) / 10
end

# The schedule stores neither the models nor `H`, so the run driver re-checks the
# pairing: the counts, and the structural term fingerprint the constructor captured —
# the deep per-term check happened at `StrainSchedule(sm, H)` construction against
# SOME Hamiltonian, and counts alone cannot tell two same-shape models apart (that
# silent mis-assignment is the exact hazard the constructor exists to prevent).
function _check_strain_pairing(H::TiledHamiltonian, sch::StrainSchedule)
    size(sch.coefpoly, 2) == H.n_input_terms &&
        sch.n_mobile == H.n_disp_active &&
        sch.d_dim == 3 * H.n_disp_active - count(H.comp_free) &&
        sch.n_cells * H.n_cell_atoms == H.n_sites || throw(ArgumentError(
        "this strain schedule does not describe this Hamiltonian (terms " *
        "$(size(sch.coefpoly, 2)) vs $(H.n_input_terms), displacement-active sites " *
        "$(sch.n_mobile) vs $(H.n_disp_active), D $(sch.d_dim) vs " *
        "$(3 * H.n_disp_active - count(H.comp_free))); build it with " *
        "`StrainSchedule(sm, H)` against the Hamiltonian the run uses"))
    sch.term_fp == _schedule_term_fp(H) || throw(ArgumentError(
        "this strain schedule was converted against a DIFFERENT Hamiltonian with " *
        "the same term counts: the term lists disagree in atoms or images, so " *
        "installing the grid's coefficients here would write each one onto another " *
        "model's cluster. Build the schedule with `StrainSchedule(sm, H)` against " *
        "the Hamiltonian the run uses"))
    in_strain_domain(sch, 1.0) || throw(ArgumentError(
        "the chain starts at the reference scale s = 1, which is outside this " *
        "schedule's domain $(strain_domain(sch)); regrid around the reference"))
    return nothing
end
