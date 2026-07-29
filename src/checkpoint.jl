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

const _CKPT_SCHEMA_VERSION = 5

# The run-side checkpoint writer: the target path, the write cadence, and the
# run-description needed to make the file self-contained.
#
# On a strained (NPT) run, `fingerprint` is the model fingerprint AT THE REFERENCE
# SCALE `s = 1` — `_fingerprint` mixes the coefficient values (deliberately: a
# coefficient hot-swap must refuse to resume), and a strained chain's coefficients
# move with its volume, so the only well-defined identity is the reference's. The
# drivers construct the checkpointer while `H` carries the reference coefficients,
# and `resume` reinstalls them before comparing. `grid_fp` identifies the volume
# grid itself (zero on a fixed-cell run).
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
    const grid_fp::UInt64            # strain-grid identity (0 = fixed cell)
end

function _make_checkpointer(path::Union{Nothing,AbstractString}, interval::Integer,
                            H::TiledHamiltonian, plan::UpdatePlan,
                            observables::Vector{Observable}, kind::String,
                            exchange_interval::Int; grid_fp::UInt64 = UInt64(0))
    path === nothing && return nothing
    interval >= 0 ||
        throw(ArgumentError("checkpoint_interval must be ≥ 0; got $interval"))
    return _Checkpointer(String(path), Int(interval), 0, _fingerprint(H), plan,
                         [String(o.name) for o in observables],
                         [o.ncomp for o in observables], kind, exchange_interval,
                         grid_fp)
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
    # The row layout, mixed only on a joint Hamiltonian (zero change for every
    # pure-spin model, so pre-M4 checkpoints still identify theirs). It is NOT
    # redundant with the slot data below: a `TermSlot` records `row0`, a
    # layout-relative block start, and `(k, l) -> row0` is not injective ACROSS
    # layouts — a `disp = (degree = 3:5,)` sector's `(1,1),(2,1)` blocks start where a
    # `1:3` sector's `(0,1),(1,1)` do, so two models differing only by the radial power
    # `k` (hence by a factor |u|²) would otherwise collide while having different
    # energies. Mixing `disp_factors` also separates a pure-spin model from a joint one
    # whose displacement couplings all fitted to zero.
    if has_disp(H)
        h = _fp_mix(h, H.layout.nrows)
        h = _fp_mix(h, H.layout.spin_lmax)
        for (k, l) in H.layout.disp_factors
            h = _fp_mix(h, k)
            h = _fp_mix(h, l)
        end
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
        for sl in t.slots
            h = _fp_mix(h, sl.l)
        end
        # The slot layout beyond the degrees — which site each axis reads, and its
        # channel — is mixed ONLY when the term is not the pure-spin identity layout
        # (axis i = the spin factor of site i). There it carries no information the
        # degrees above do not already have, so every pure-spin model's fingerprint
        # (and hence every checkpoint written before the displacement channel existed)
        # is unchanged by M4. See `_is_spin_identity` in hamiltonian.jl.
        if !_is_spin_identity(t.slots)
            for sl in t.slots
                h = _fp_mix(h, sl.site)
                h = _fp_mix(h, sl.row0)
                h = _fp_mix(h, sl.spin ? 1 : 0)
            end
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

# The volume grid's identity, for a strained run's checkpoint: everything the
# schedule's conversion produced. Two runs may share a model fingerprint at the
# reference and still disagree off it — same reference fit, different grid — so the
# grid needs its own identity, and it is checked BEFORE the reference coefficients
# are reinstalled from the schedule the caller supplies.
function _grid_fingerprint(sch::StrainSchedule)::UInt64
    h = 0xcbf29ce484222325
    h = _fp_mix(h, length(sch.scales))
    for s in sch.scales
        h = _fp_mix(h, s)
    end
    for c in codeunits(String(sch.abscissa))
        h = _fp_mix(h, c)
    end
    h = _fp_mix(h, sch.x0)
    h = _fp_mix(h, sch.xw)
    # shape prefixes: two grids differing only in interpolation degree must not rely
    # on coefficient VALUES to separate their flattened polynomials
    h = _fp_mix(h, size(sch.coefpoly, 1))
    h = _fp_mix(h, size(sch.coefpoly, 2))
    for v in sch.coefpoly
        h = _fp_mix(h, v)
    end
    h = _fp_mix(h, sch.term_fp)
    for v in sch.j0poly
        h = _fp_mix(h, v)
    end
    h = _fp_mix(h, sch.v_train)
    h = _fp_mix(h, sch.n_cells)
    h = _fp_mix(h, sch.n_mobile)
    h = _fp_mix(h, sch.d_dim)
    return h
end

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

# A `Vector{SVector{3,Float64}}` as a plain 3 × n matrix (and back). Written for
# `n == 0` too — a pure-spin chain has no displacement components, and a 3×0 array is
# the honest encoding of that.
_vec3_matrix(v::Vector{SVector{3,Float64}})::Matrix{Float64} =
    [x[row] for row = 1:3, x in v]

_vec3_from_matrix(m::Matrix{Float64})::Vector{SVector{3,Float64}} =
    [SVector{3,Float64}(m[1, i], m[2, i], m[3, i]) for i = 1:size(m, 2)]

function _write_chain(f, g::String, st::ChainState)
    f["$g/config"] = _config_matrix(st.config)
    f["$g/disps"] = _vec3_matrix(st.disps)
    f["$g/com_removed"] = _vec3_matrix(st.com_removed)
    f["$g/energy"] = st.energy
    f["$g/rng"] = _rng_words(st.rng)
    f["$g/site_rngs"] = reduce(hcat, [_rng_words(r) for r in st.site_rngs])
    f["$g/step"] = st.step
    f["$g/step_u"] = st.step_u
    f["$g/strain"] = st.strain
    f["$g/frozen"] = st.frozen
    f["$g/counters"] = Int[st.acc_metro, st.att_metro, st.acc_or, st.att_or,
                           st.acc_disp, st.att_disp, st.acc_strain, st.att_strain]
    f["$g/max_drift"] = st.max_drift
    # The escape detector's accumulators. They steer no random decision, so a resume
    # is bit-identical with or without them — but dropping them would restart the
    # block ladder at every checkpoint, and a chain checkpointed often enough would
    # never accumulate the consecutive strikes that report an escape.
    f["$g/escape_f"] = Float64[st.disp_rms, st.disp_max, st.disp_rms0,
                               st.disp_ms_sum, st.disp_blk_sum, st.disp_ref_ms]
    f["$g/escape_i"] = Int[st.disp_checks, st.disp_blk_n, st.disp_blk_cap,
                           st.escape_strikes]
    f["$g/escape_warned"] = st.escape_warned
    return nothing
end

function _read_chain(f, g::String, H::TiledHamiltonian)::ChainState
    config = _config_from_matrix(f["$g/config"])
    length(config) == H.n_sites || error(
        "checkpoint config has $(length(config)) sites; the Hamiltonian has " *
        "$(H.n_sites)")
    disps = _vec3_from_matrix(f["$g/disps"])
    length(disps) == H.n_sites || error(
        "checkpoint has $(length(disps)) displacements; the Hamiltonian has " *
        "$(H.n_sites) sites")
    com = _vec3_from_matrix(f["$g/com_removed"])
    length(com) == H.n_disp_comps || error(
        "checkpoint has $(length(com)) re-centring records; the Hamiltonian has " *
        "$(H.n_disp_comps) displacement-coupling components")
    # Pure function of (config, disps) — bit-reproducible, so the rows are rebuilt
    # rather than stored (the energy is NOT: it is the incremental one, restored
    # verbatim, drift and all).
    zrows = has_disp(H) ? _zrows(H, config, disps) : _zrows(H, config)
    cnt = f["$g/counters"]
    srw = f["$g/site_rngs"]::Matrix{UInt64}
    size(srw, 2) == H.n_sites || error(
        "checkpoint has $(size(srw, 2)) site RNG streams; the Hamiltonian has " *
        "$(H.n_sites) sites")
    site_rngs = [_rng_from_words(srw[:, s]) for s = 1:H.n_sites]
    ef = f["$g/escape_f"]
    ei = f["$g/escape_i"]
    return ChainState(config, disps, zrows, f["$g/energy"],
                      _rng_from_words(f["$g/rng"]), site_rngs, f["$g/step"],
                      f["$g/step_u"], f["$g/strain"], f["$g/frozen"], cnt[1], cnt[2],
                      cnt[3], cnt[4], cnt[5], cnt[6], cnt[7], cnt[8],
                      f["$g/max_drift"], com,
                      ef[1], ef[2], ef[3], ei[1], ef[4], ef[5], ei[2], ei[3], ef[6],
                      ei[4], f["$g/escape_warned"])
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
    f["$g/acceptance_disp"] = p.acceptance_disp
    f["$g/acceptance_strain"] = p.acceptance_strain
    f["$g/final_step"] = p.final_step
    f["$g/final_step_u"] = p.final_step_u
    f["$g/max_drift"] = p.max_drift
    f["$g/disp_rms"] = p.disp_rms
    f["$g/disp_max"] = p.disp_max
    f["$g/disp_checks"] = p.disp_checks
    f["$g/escaped"] = p.escaped
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
                      f["$g/acceptance_or"], f["$g/acceptance_disp"],
                      f["$g/acceptance_strain"],
                      f["$g/final_step"], f["$g/final_step_u"], f["$g/max_drift"],
                      f["$g/disp_rms"], f["$g/disp_max"], f["$g/disp_checks"],
                      f["$g/escaped"])
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
    f["plan/disp_per_metropolis"] = p.disp_per_metropolis
    f["plan/step0"] = p.step0
    f["plan/step_u0"] = p.step_u0
    f["plan/adapt_target"] = p.adapt_target
    f["plan/adapt_interval"] = p.adapt_interval
    f["plan/renorm_interval"] = p.renorm_interval
    f["plan/nbins"] = p.nbins
    f["plan/carryover"] = p.carryover
    f["plan/sweep_tasks"] = p.sweep_tasks
    f["plan/seed"] = p.seed
    f["plan/strain_interval"] = p.strain_interval
    f["plan/strain_proposal"] = String(p.strain_proposal)
    f["plan/strain_step"] = p.strain_step
    f["plan/pressure"] = p.pressure
    f["grid_fingerprint"] = ck.grid_fp
    f["plan/observable_names"] = ck.obs_names
    f["plan/observable_ncomps"] = ck.obs_ncomps
    return nothing
end

function _read_plan(f)::UpdatePlan
    return UpdatePlan(f["plan/kts"]; sweeps_therm = f["plan/sweeps_therm"],
                      sweeps_measure = f["plan/sweeps_measure"],
                      measure_interval = f["plan/measure_interval"],
                      or_per_metropolis = f["plan/or_per_metropolis"],
                      disp_per_metropolis = f["plan/disp_per_metropolis"],
                      step = f["plan/step0"], step_u = f["plan/step_u0"],
                      adapt_target = f["plan/adapt_target"],
                      adapt_interval = f["plan/adapt_interval"],
                      renorm_interval = f["plan/renorm_interval"],
                      nbins = f["plan/nbins"], carryover = f["plan/carryover"],
                      sweep_tasks = f["plan/sweep_tasks"],
                      # keep the UInt64 — Int() would InexactError on seeds ≥ 2^63,
                      # i.e. on half of the default rand(UInt64) seeds
                      seed = f["plan/seed"],
                      strain_interval = f["plan/strain_interval"],
                      strain_proposal = Symbol(f["plan/strain_proposal"]),
                      strain_step = f["plan/strain_step"],
                      pressure = f["plan/pressure"])
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

function _write_ckpt_pt(ck::_Checkpointer, lanes::Vector{_PTLane}, phase::Symbol,
                        done::Int, parity::Int, exchange_rng::Xoshiro,
                        swap_att::Vector{Int}, swap_acc::Vector{Int})
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
function _ck_pt!(ck, n::Int, lanes::Vector{_PTLane}, phase::Symbol, done::Int,
                 parity::Int, exchange_rng::Xoshiro, swap_att::Vector{Int},
                 swap_acc::Vector{Int})
    ck === nothing && return nothing
    ck.interval > 0 || return nothing
    ck.since += n
    ck.since >= ck.interval || return nothing
    ck.since = 0
    _write_ckpt_pt(ck, lanes, phase, done, parity, exchange_rng, swap_att,
                   swap_acc)
    return nothing
end

# --- resume --------------------------------------------------------------------------

"""
    resume(path, H::TiledHamiltonian;
           observables = standard_observables(H),
           evaluables = standard_evaluables(H),
           checkpoint = path, checkpoint_interval = nothing,
           strain = nothing)
        -> MCResult | PTResult

Continue a checkpointed [`run_mc`](@ref) / [`run_pt`](@ref) run from the state
saved at `path` and return the **full** run's result — bit-identical to the
uninterrupted run. The caller re-supplies the Hamiltonian and the observable /
evaluable *functions* (closures are not serialized); the checkpoint stores the
model fingerprint and the observable names/component counts and errors on any
mismatch. By default the resumed run keeps checkpointing to the same `path` with
the stored cadence (`checkpoint = nothing` disables; `checkpoint_interval`
overrides).

A strained (NPT) run's checkpoint additionally stores the chain's cell scale and
the volume grid's fingerprint; resuming it requires the same [`StrainSchedule`](@ref)
as `strain` (checked against the stored fingerprint). `resume` then reinstalls the
schedule's **reference** coefficients into `H` before the model-fingerprint check —
the fingerprint mixes coefficient values, and a strained chain's move with its
volume, so the reference is the identity both sides compare — and finally installs
the checkpointed scale's coefficients so the `(H, chain)` contract holds when the
run continues: on the "mc" kind into `H` itself, on the "pt" kind into a fresh
per-lane coefficient clone at each lane's own checkpointed scale (`H` then never
leaves the reference at all). On return `H` is handed back at the reference, as
the run drivers do.
`H`'s current coefficient state on entry is therefore irrelevant (and is
overwritten) — including on a FAILED resume: a fingerprint or observable mismatch
raised after the reinstall leaves `H` at the schedule's reference.
"""
function resume(path::AbstractString, H::TiledHamiltonian;
                observables::Vector{Observable} = standard_observables(H),
                evaluables::Vector{Evaluable} = standard_evaluables(H),
                checkpoint::Union{Nothing,AbstractString} = path,
                checkpoint_interval::Union{Nothing,Integer} = nothing,
                strain::Union{Nothing,StrainSchedule} = nothing)
    isfile(path) || throw(ArgumentError("no checkpoint file at $path"))
    # Read and validate EVERYTHING eagerly, closing the file before the long
    # computation starts — the resumed run typically overwrites this very path
    # with new checkpoints, and holding it open meanwhile is fragile (and fails
    # outright on platforms without POSIX rename-over-open semantics).
    data = jldopen(String(path), "r") do f
        f["schema_version"] == _CKPT_SCHEMA_VERSION || error(
            "checkpoint schema v$(f["schema_version"]) ≠ " *
            "v$(_CKPT_SCHEMA_VERSION) of this package version" *
            (f["schema_version"] < _CKPT_SCHEMA_VERSION ?
             " (v5 marks the strained-PT-capable format — an older reader would " *
             "silently continue a strained PT run as fixed-cell; resume this file " *
             "with the package version that wrote it)" : ""))
        plan = _read_plan(f)
        # The strain handshake comes BEFORE the model fingerprint: on a strained
        # run the stored fingerprint is the model's at the REFERENCE scale, and the
        # only way to compare against it is to reinstall the reference coefficients
        # from the schedule the caller supplies — after checking that schedule IS
        # the stored grid.
        if plan.strain_interval > 0
            strain === nothing && error(
                "this checkpoint belongs to a strained (NPT) run; pass its volume " *
                "grid as `resume(path, H; strain = sch)` — the schedule is not " *
                "serialized, only its fingerprint")
            f["grid_fingerprint"] == _grid_fingerprint(strain) || error(
                "the supplied strain schedule is not the grid this checkpoint was " *
                "written with (grid fingerprint mismatch)")
            _check_strain_pairing(H, strain)
            # the default flatness recheck stays ON: this install is the resumed
            # run's anchor, exactly as in `run_mc`
            set_coefficients!(H, strain_coefficients(strain, 1.0))
        else
            strain === nothing || error(
                "a strain schedule was passed, but this checkpoint belongs to a " *
                "fixed-cell run")
        end
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
            # On a strained run each lane gets its own coefficient clone (built while
            # `H` carries the reference the handshake installed above); the lane's
            # checkpointed scale is installed into it after the file closes.
            (; lanes = [_PTLane(_read_chain(f, "lane/$r", H),
                                [SweepScratch(H) for _ = 1:plan.sweep_tasks],
                                plan.kts[r], 1.0 / plan.kts[r],
                                measure ?
                                _read_accs(f, "lane/$r/accs", observables) :
                                ObsAccumulator[], done,
                                strain === nothing ? H : _coefficient_clone(H),
                                strain === nothing ? nothing :
                                (strain, StrainScratch(H)))
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
    # `H` still carries the reference coefficients here (installed above), so the
    # continued file's fingerprint is the same identity the original run wrote.
    ck = _make_checkpointer(checkpoint, interval, H, data.plan, observables,
                            data.kind, data.exch;
                            grid_fp = strain === nothing ? UInt64(0) :
                                      _grid_fingerprint(strain))
    b = data.body
    if data.kind == "mc"
        # a completed run has nothing to continue — H stays at the reference the
        # validation installed
        b.temp_index > length(data.plan.kts) &&
            return MCResult(b.points, copy(b.st.config), _final_disps(H, b.st),
                            strain === nothing ? nothing : b.st.strain,
                            data.plan.seed)
        sctx = nothing
        if strain !== nothing
            # re-establish the (H, chain) contract at the CHECKPOINTED scale —
            # bit-identical to what the interrupted run held, because the Horner
            # pass is deterministic
            sc = StrainScratch(H)
            set_coefficients!(H, strain_coefficients!(sc.coef, strain, b.st.strain);
                              recheck_translation = false)
            sctx = (strain, sc)
        end
        r = _mc_loop!(b.points, b.st, H, data.plan, observables, evaluables,
                      b.temp_index, b.phase, b.sweep, b.accs, ck, sctx)
        # hand `H` back at the reference, as `run_mc` does
        sctx === nothing ||
            set_coefficients!(H, strain_coefficients!(sctx[2].coef, sctx[1], 1.0);
                              recheck_translation = false)
        return r
    end
    if strain !== nothing
        # re-establish each lane's (H, chain) contract at its CHECKPOINTED scale —
        # bit-identical to what the interrupted run held (deterministic Horner
        # pass); the caller's `H` stays at the reference the handshake installed
        for lane in b.lanes
            set_coefficients!(lane.H,
                              strain_coefficients!(lane.sctx[2].coef, strain,
                                                   lane.st.strain);
                              recheck_translation = false)
        end
    end
    nt = min(length(data.plan.kts), Threads.nthreads())
    return _pt_run!(b.lanes, data.plan, observables, evaluables, data.exch, nt,
                    b.exchange_rng, b.swap_att, b.swap_acc, b.phase, b.done,
                    b.parity, ck)
end
