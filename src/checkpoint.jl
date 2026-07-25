# Checkpoint / resume (schema: `docs/specs/checkpoint-schema.md`).
#
# Design rules: the file holds ONLY plain data (Bool/Int/Float64/UInt64/String and
# arrays thereof, in named JLD2 groups) — no Julia struct reconstruction, so the
# format cannot silently break with a package refactor (the SpinClusterMC failure
# mode). Everything the trajectory depends on is captured bit-exactly — configs,
# incremental energies (restored verbatim, never recomputed), Xoshiro words,
# counters, accumulator cascades — and every schedule is deterministic in the
# stored counters, so a resumed run is bit-identical to an uninterrupted one.
# Writes go to a temp file, then an atomic `mv`. Checkpoint writing consumes no RNG.

const _CKPT_SCHEMA_VERSION = 2

# The run-side checkpoint writer: the target path, the write cadence, and the
# run-description needed to make the file self-contained.
mutable struct _Checkpointer
    const path::String
    const interval::Int              # sweeps between writes; 0 ⇒ boundaries only
    since::Int
    const fingerprint::UInt64
    const plan::UpdatePlan
    const obs_names::Vector{String}
    const obs_ncomps::Vector{Int}
    const kind::String               # "mc" | "pt"
    const exchange_interval::Int     # pt only (0 for mc)
end

function _make_checkpointer(path::Union{Nothing,AbstractString}, interval::Integer,
                            H::TiledHamiltonian, plan::UpdatePlan,
                            observables::Vector{Observable}, kind::String,
                            exchange_interval::Int)
    path === nothing && return nothing
    interval >= 0 ||
        throw(ArgumentError("checkpoint_interval must be ≥ 0; got $interval"))
    return _Checkpointer(String(path), Int(interval), 0, _fingerprint(H), plan,
                         [String(o.name) for o in observables],
                         [o.ncomp for o in observables], kind, exchange_interval)
end

# --- model fingerprint (stable FNV-1a — deliberately NOT Base.hash, which is
# --- Julia-version-dependent) -------------------------------------------------------

@inline _fp_mix(h::UInt64, x::UInt64)::UInt64 = (h ⊻ x) * 0x00000100000001b3
@inline _fp_mix(h::UInt64, x::Integer)::UInt64 =
    _fp_mix(h, reinterpret(UInt64, Int64(x)))
@inline _fp_mix(h::UInt64, x::Float64)::UInt64 = _fp_mix(h, reinterpret(UInt64, x))

# Fingerprint of the tiled Hamiltonian a checkpoint belongs to: dims + every term's
# payload. A resume against a different model/dims errors instead of silently
# continuing the wrong physics.
function _fingerprint(H::TiledHamiltonian)::UInt64
    h = 0xcbf29ce484222325
    h = _fp_mix(h, H.n_cell_atoms)
    for d in H.dims
        h = _fp_mix(h, d)
    end
    for t in H.terms
        h = _fp_mix(h, t.coef)
        for a in t.atoms
            h = _fp_mix(h, a)
        end
        for s in t.shifts
            h = _fp_mix(h, s[1])
            h = _fp_mix(h, s[2])
            h = _fp_mix(h, s[3])
        end
        for l in t.ls
            h = _fp_mix(h, l)
        end
        for v in t.folded
            h = _fp_mix(h, v)
        end
    end
    return h
end

"""
    model_fingerprint(H::TiledHamiltonian) -> UInt64

Stable FNV-1a fingerprint of the tiled model — `dims`, the cell-atom count, and
every scaled term's payload. This is the identity a checkpoint file carries so a
resume against a different model, supercell, or coefficient set errors instead of
silently continuing the wrong physics. Deliberately **not** `Base.hash` (which is
Julia-version-dependent); the value is part of the checkpoint format. Public for
dependent packages' checkpoint formats (e.g. `SLCEDynamics`).
"""
model_fingerprint(H::TiledHamiltonian)::UInt64 = _fingerprint(H)

# --- plain-data (de)serializers ------------------------------------------------------

_rng_words(rng::Xoshiro)::Vector{UInt64} =
    UInt64[getfield(rng, f) for f in fieldnames(Xoshiro)]

function _rng_from_words(words::Vector{UInt64})::Xoshiro
    length(words) == fieldcount(Xoshiro) || error(
        "checkpoint RNG state has $(length(words)) words; this Julia's Xoshiro " *
        "has $(fieldcount(Xoshiro)) — the checkpoint was written by an " *
        "incompatible Julia version")
    return Xoshiro(words...)
end

_config_matrix(config::SpinConfig)::Matrix{Float64} =
    [config[s][row] for row = 1:3, s = 1:length(config)]

_config_from_matrix(m::Matrix{Float64})::SpinConfig =
    SpinConfig([SVector{3,Float64}(m[1, s], m[2, s], m[3, s])
                for s = 1:size(m, 2)])

function _write_chain(f, g::String, st::ChainState)
    f["$g/config"] = _config_matrix(st.config)
    f["$g/energy"] = st.energy
    f["$g/rng"] = _rng_words(st.rng)
    f["$g/site_rngs"] = reduce(hcat, [_rng_words(r) for r in st.site_rngs])
    f["$g/step"] = st.step
    f["$g/frozen"] = st.frozen
    f["$g/counters"] = Int[st.acc_metro, st.att_metro, st.acc_or, st.att_or]
    f["$g/max_drift"] = st.max_drift
    return nothing
end

function _read_chain(f, g::String, H::TiledHamiltonian)::ChainState
    config = _config_from_matrix(f["$g/config"])
    length(config) == H.n_sites || error(
        "checkpoint config has $(length(config)) sites; the Hamiltonian has " *
        "$(H.n_sites)")
    zrows = _zrows(H, config)         # pure function of config — bit-reproducible
    cnt = f["$g/counters"]
    srw = f["$g/site_rngs"]::Matrix{UInt64}
    size(srw, 2) == H.n_sites || error(
        "checkpoint has $(size(srw, 2)) site RNG streams; the Hamiltonian has " *
        "$(H.n_sites) sites")
    site_rngs = [_rng_from_words(srw[:, s]) for s = 1:H.n_sites]
    return ChainState(config, zrows, f["$g/energy"], _rng_from_words(f["$g/rng"]),
                      site_rngs, f["$g/step"], f["$g/frozen"], cnt[1], cnt[2],
                      cnt[3], cnt[4], f["$g/max_drift"])
end

function _write_accs(f, g::String, accs::Vector{ObsAccumulator})
    for acc in accs
        b, s = acc.binner, acc.store
        ag = "$g/$(acc.obs.name)"
        f["$ag/binner/count"] = b.count
        f["$ag/binner/sums"] = b.sums
        f["$ag/binner/sums2"] = b.sums2
        f["$ag/binner/pending"] = b.pending
        f["$ag/binner/pending_full"] = b.pending_full
        f["$ag/binner/n"] = b.n
        f["$ag/store/bin_size"] = s.bin_size
        f["$ag/store/means"] = s.means
        f["$ag/store/nfull"] = s.nfull
        f["$ag/store/acc"] = s.acc
        f["$ag/store/nacc"] = s.nacc
    end
    return nothing
end

function _read_accs(f, g::String,
                    observables::Vector{Observable})::Vector{ObsAccumulator}
    return [begin
                ag = "$g/$(o.name)"
                binner = LogBinner(o.ncomp, f["$ag/binner/count"],
                                   f["$ag/binner/sums"], f["$ag/binner/sums2"],
                                   f["$ag/binner/pending"],
                                   f["$ag/binner/pending_full"], f["$ag/binner/n"])
                store = BinStore(o.ncomp, f["$ag/store/bin_size"],
                                 f["$ag/store/means"], f["$ag/store/nfull"],
                                 f["$ag/store/acc"], f["$ag/store/nacc"])
                ObsAccumulator(o, binner, store, zeros(o.ncomp))
            end
            for o in observables]
end

function _write_point(f, g::String, p::TempResult)
    f["$g/kT"] = p.kT
    f["$g/acceptance_metropolis"] = p.acceptance_metropolis
    f["$g/acceptance_or"] = p.acceptance_or
    f["$g/final_step"] = p.final_step
    f["$g/max_drift"] = p.max_drift
    f["$g/stat_names"] = String[String(k) for k in keys(p.stats)]
    for (k, s) in p.stats
        f["$g/stats/$k/mean"] = s.mean
        f["$g/stats/$k/err"] = s.err
        f["$g/stats/$k/tau_int"] = s.tau_int
        f["$g/stats/$k/count"] = s.count
    end
    return nothing
end

function _read_point(f, g::String)::TempResult
    stats = Dict{Symbol,ObservableStat}()
    for name in f["$g/stat_names"]
        k = Symbol(name)
        stats[k] = ObservableStat(k, f["$g/stats/$k/mean"], f["$g/stats/$k/err"],
                                  f["$g/stats/$k/tau_int"], f["$g/stats/$k/count"])
    end
    kt = f["$g/kT"]
    return TempResult(kt, kt / KB_EV, stats, f["$g/acceptance_metropolis"],
                      f["$g/acceptance_or"], f["$g/final_step"], f["$g/max_drift"])
end

function _write_header(f, ck::_Checkpointer)
    f["schema_version"] = _CKPT_SCHEMA_VERSION
    f["kind"] = ck.kind
    f["julia_version"] = string(VERSION)
    f["package_version"] = string(pkgversion(SLCEMonteCarlo))
    f["model_fingerprint"] = ck.fingerprint
    f["checkpoint_interval"] = ck.interval
    f["exchange_interval"] = ck.exchange_interval
    p = ck.plan
    f["plan/kts"] = p.kts
    f["plan/sweeps_therm"] = p.sweeps_therm
    f["plan/sweeps_measure"] = p.sweeps_measure
    f["plan/measure_interval"] = p.measure_interval
    f["plan/or_per_metropolis"] = p.or_per_metropolis
    f["plan/step0"] = p.step0
    f["plan/adapt_target"] = p.adapt_target
    f["plan/adapt_interval"] = p.adapt_interval
    f["plan/renorm_interval"] = p.renorm_interval
    f["plan/nbins"] = p.nbins
    f["plan/carryover"] = p.carryover
    f["plan/sweep_tasks"] = p.sweep_tasks
    f["plan/seed"] = p.seed
    f["plan/observable_names"] = ck.obs_names
    f["plan/observable_ncomps"] = ck.obs_ncomps
    return nothing
end

function _read_plan(f)::UpdatePlan
    return UpdatePlan(f["plan/kts"]; sweeps_therm = f["plan/sweeps_therm"],
                      sweeps_measure = f["plan/sweeps_measure"],
                      measure_interval = f["plan/measure_interval"],
                      or_per_metropolis = f["plan/or_per_metropolis"],
                      step = f["plan/step0"], adapt_target = f["plan/adapt_target"],
                      adapt_interval = f["plan/adapt_interval"],
                      renorm_interval = f["plan/renorm_interval"],
                      nbins = f["plan/nbins"], carryover = f["plan/carryover"],
                      sweep_tasks = f["plan/sweep_tasks"],
                      # keep the UInt64 — Int() would InexactError on seeds ≥ 2^63,
                      # i.e. on half of the default rand(UInt64) seeds
                      seed = f["plan/seed"])
end

# --- writers (atomic: temp file + mv) ------------------------------------------------

function _write_ckpt_mc(ck::_Checkpointer, H::TiledHamiltonian, st::ChainState,
                        points::Vector{TempResult}, temp_index::Int, phase::Symbol,
                        sweep::Int, accs::Union{Nothing,Vector{ObsAccumulator}})
    tmp = ck.path * ".tmp." * string(getpid())   # one writer per path assumed
    jldopen(tmp, "w") do f
        _write_header(f, ck)
        f["progress/temp_index"] = temp_index
        f["progress/phase"] = String(phase)
        f["progress/sweep"] = sweep
        f["npoints"] = length(points)
        for (i, p) in enumerate(points)
            _write_point(f, "points/$i", p)
        end
        _write_chain(f, "chain", st)
        f["has_accs"] = accs !== nothing
        accs === nothing || _write_accs(f, "accs", accs)
    end
    mv(tmp, ck.path; force = true)
    return nothing
end

# Periodic-write tick for the MC drivers (one call per sweep; no-op without a
# checkpointer or with the boundary-only interval 0).
function _ck_mc!(ck, H::TiledHamiltonian, st::ChainState,
                 points::Vector{TempResult}, temp_index::Int, phase::Symbol,
                 sweep::Int, accs::Union{Nothing,Vector{ObsAccumulator}})
    ck === nothing && return nothing
    ck.interval > 0 || return nothing
    ck.since += 1
    ck.since >= ck.interval || return nothing
    ck.since = 0
    _write_ckpt_mc(ck, H, st, points, temp_index, phase, sweep, accs)
    return nothing
end

function _write_ckpt_pt(ck::_Checkpointer, H::TiledHamiltonian,
                        lanes::Vector{_PTLane}, phase::Symbol, done::Int,
                        parity::Int, exchange_rng::Xoshiro, swap_att::Vector{Int},
                        swap_acc::Vector{Int})
    tmp = ck.path * ".tmp." * string(getpid())
    measure = phase === :measure
    jldopen(tmp, "w") do f
        _write_header(f, ck)
        f["progress/phase"] = String(phase)
        f["progress/done"] = done
        f["progress/parity"] = parity
        f["exchange_rng"] = _rng_words(exchange_rng)
        f["swap_att"] = swap_att
        f["swap_acc"] = swap_acc
        f["nlanes"] = length(lanes)
        for (r, lane) in enumerate(lanes)
            _write_chain(f, "lane/$r", lane.st)
            measure && _write_accs(f, "lane/$r/accs", lane.accs)
        end
    end
    mv(tmp, ck.path; force = true)
    return nothing
end

# Periodic-write tick for the PT segment driver (one call per segment, `n` = the
# segment's sweep count).
function _ck_pt!(ck, n::Int, H::TiledHamiltonian, lanes::Vector{_PTLane},
                 phase::Symbol, done::Int, parity::Int, exchange_rng::Xoshiro,
                 swap_att::Vector{Int}, swap_acc::Vector{Int})
    ck === nothing && return nothing
    ck.interval > 0 || return nothing
    ck.since += n
    ck.since >= ck.interval || return nothing
    ck.since = 0
    _write_ckpt_pt(ck, H, lanes, phase, done, parity, exchange_rng, swap_att,
                   swap_acc)
    return nothing
end

# --- resume --------------------------------------------------------------------------

"""
    resume(path, H::TiledHamiltonian;
           observables = standard_observables(H),
           evaluables = standard_evaluables(),
           checkpoint = path, checkpoint_interval = nothing)
        -> MCResult | PTResult

Continue a checkpointed [`run_mc`](@ref) / [`run_pt`](@ref) run from the state
saved at `path` and return the **full** run's result — bit-identical to the
uninterrupted run. The caller re-supplies the Hamiltonian and the observable /
evaluable *functions* (closures are not serialized); the checkpoint stores the
model fingerprint and the observable names/component counts and errors on any
mismatch. By default the resumed run keeps checkpointing to the same `path` with
the stored cadence (`checkpoint = nothing` disables; `checkpoint_interval`
overrides).
"""
function resume(path::AbstractString, H::TiledHamiltonian;
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(),
                checkpoint::Union{Nothing,AbstractString} = path,
                checkpoint_interval::Union{Nothing,Integer} = nothing)
    isfile(path) || throw(ArgumentError("no checkpoint file at $path"))
    # Read and validate EVERYTHING eagerly, closing the file before the long
    # computation starts — the resumed run typically overwrites this very path
    # with new checkpoints, and holding it open meanwhile is fragile (and fails
    # outright on platforms without POSIX rename-over-open semantics).
    data = jldopen(String(path), "r") do f
        f["schema_version"] == _CKPT_SCHEMA_VERSION || error(
            "checkpoint schema v$(f["schema_version"]) ≠ " *
            "v$(_CKPT_SCHEMA_VERSION) of this package version")
        f["model_fingerprint"] == _fingerprint(H) || error(
            "checkpoint model fingerprint does not match this TiledHamiltonian " *
            "(different model, dims, or coefficients)")
        _check_observables(observables)
        names = f["plan/observable_names"]
        ncomps = f["plan/observable_ncomps"]
        (names == [String(o.name) for o in observables] &&
         ncomps == [o.ncomp for o in observables]) || error(
            "the resumed observables (names/ncomps) do not match the checkpoint; " *
            "stored: $(names) with $(ncomps)")
        plan = _read_plan(f)
        kind = f["kind"]
        body = if kind == "mc"
            (; points = TempResult[_read_point(f, "points/$i")
                                   for i = 1:f["npoints"]],
             st = _read_chain(f, "chain", H),
             temp_index = f["progress/temp_index"]::Int,
             phase = Symbol(f["progress/phase"]), sweep = f["progress/sweep"]::Int,
             accs = f["has_accs"] ? _read_accs(f, "accs", observables) : nothing)
        elseif kind == "pt"
            R = f["nlanes"]::Int
            R == length(plan.kts) || error("checkpoint lane count $R ≠ ladder " *
                                           "length $(length(plan.kts))")
            phase = Symbol(f["progress/phase"])
            done = f["progress/done"]::Int
            measure = phase === :measure
            (; lanes = [_PTLane(_read_chain(f, "lane/$r", H),
                                [SweepScratch(H) for _ = 1:plan.sweep_tasks],
                                plan.kts[r], 1.0 / plan.kts[r],
                                measure ?
                                _read_accs(f, "lane/$r/accs", observables) :
                                ObsAccumulator[], done)
                        for r = 1:R],
             phase, done, parity = f["progress/parity"]::Int,
             exchange_rng = _rng_from_words(f["exchange_rng"]),
             swap_att = f["swap_att"]::Vector{Int},
             swap_acc = f["swap_acc"]::Vector{Int})
        else
            error("unknown checkpoint kind $(kind)")
        end
        (; kind, plan, stored_interval = Int(f["checkpoint_interval"]),
         exch = Int(f["exchange_interval"]), body)
    end
    interval = checkpoint_interval === nothing ? data.stored_interval :
               Int(checkpoint_interval)
    ck = _make_checkpointer(checkpoint, interval, H, data.plan, observables,
                            data.kind, data.exch)
    b = data.body
    if data.kind == "mc"
        b.temp_index > length(data.plan.kts) &&
            return MCResult(b.points, copy(b.st.config), data.plan.seed)
        return _mc_loop!(b.points, b.st, H, data.plan, observables, evaluables,
                         b.temp_index, b.phase, b.sweep, b.accs, ck)
    end
    nt = min(length(data.plan.kts), Threads.nthreads())
    return _pt_run!(b.lanes, H, data.plan, observables, evaluables, data.exch, nt,
                    b.exchange_rng, b.swap_att, b.swap_acc, b.phase, b.done,
                    b.parity, ck)
end
