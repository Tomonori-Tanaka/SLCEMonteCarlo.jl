# Mutable chain state and thread-confined scratch — deliberately separated from the
# immutable `TiledHamiltonian` and from run configuration (no God-struct).

# Initial width of the Gaussian displacement proposal, in the model's LENGTH units
# (Å for DFT-fitted models) — a hundredth of a typical bond length, small enough to
# be accepted from a clamped-ion start; `_adapt_step!` then takes it to the target
# acceptance during thermalization, on its own counters and its own (length) clamp.
# Deliberately not derived from `step` (radians): the two proposals have different
# dimensions and there is no scale in the Hamiltonian that converts one to the other.
const _DEFAULT_STEP_U = 0.01

"""
    ChainState

The mutable state of one Markov chain: the spin `config` and the Cartesian
displacements `disps` (one `SVector{3,Float64}` per site, model length units —
all-zero on a pure-spin model, where they are never read), with their cached basis
rows `zrows` (column `s` = the site's full row table: `Z_lm(e_s)` in the SPIN block
and `|u_s|^{2k} R_lm(u_s)` in each displacement block), the incrementally maintained
total `energy` (model units, `j0` excluded — kept exact by the ΔE bookkeeping and
re-anchored at every renormalization), the chain-owned `rng` (config resets) plus one
independent proposal/accept stream per site (`site_rngs`, derived from `rng` at
construction — what makes the sweeps deterministic regardless of how many tasks
execute a color class), the two Metropolis proposal widths — `step` in radians for
the spin rotation (adapted during thermalization by `_adapt_step!`, then frozen once
`frozen` is set) and `step_u` as a **length** in the model's units for the
displacement shift (the two have different dimensions; never route one through the
other's adaptation or clamp) —, windowed acceptance counters per move type, the
worst incremental-energy
`max_drift` observed at renormalization points, and `com_removed` — the accumulated
centre-of-mass shift `_recenter!` has taken out of each displacement-coupling
component, so the uncentred displacement of a site in component `c` is
`disps[s] + com_removed[c]`.

Since M5, the chain also carries its cell's linear scale `strain` (`s`, so the cell is
`s·A₀`; `1.0` and never moved on a fixed-cell run) with acceptance counters for the
outer strain move ([`strain_move!`](@ref)).

`config`/`disps`/`zrows`/`energy`/`com_removed`/`strain` are the swappable payload of
a replica-exchange move (`_swap_payload!` exchanges the references); the RNG streams
and the proposal widths stay with the lane.
"""
mutable struct ChainState
    config::SpinConfig
    disps::Vector{SVector{3,Float64}}
    zrows::Matrix{Float64}
    energy::Float64
    const rng::Xoshiro
    const site_rngs::Vector{Xoshiro}
    step::Float64
    step_u::Float64
    strain::Float64                  # linear cell scale s (1.0 on a fixed-cell chain)
    frozen::Bool
    acc_metro::Int
    att_metro::Int
    acc_or::Int
    att_or::Int
    acc_disp::Int
    att_disp::Int
    acc_strain::Int
    att_strain::Int
    max_drift::Float64
    com_removed::Vector{SVector{3,Float64}}
    # escape detector (`_check_escape!`), all measured at renormalization points in the
    # centre-of-mass-free frame
    disp_rms::Float64                # latest r.m.s. displacement (a SNAPSHOT)
    disp_max::Float64                # largest single displacement over the phase
    disp_rms0::Float64               # the first one, for the absolute growth guard
    disp_checks::Int                 # renormalizations counted in this phase
    disp_ms_sum::Float64             # Σ rms² over the whole PHASE (⟨|u|²⟩ estimator)
    disp_blk_sum::Float64            # Σ rms² over the block being accumulated
    disp_blk_n::Int
    disp_blk_cap::Int                # its target length (doubles up to _ESCAPE_WINDOW)
    disp_ref_ms::Float64             # the previous block's mean rms²
    escape_strikes::Int
    escape_warned::Bool
end

function ChainState(H::TiledHamiltonian, config::SpinConfig, rng::Xoshiro,
                    step::Real; disps = nothing, step_u::Real = _DEFAULT_STEP_U)
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    step_u > 0 || throw(ArgumentError("step_u must be > 0; got $step_u"))
    u = _initial_disps(H, disps)
    # The pure-spin path keeps the pre-M4 whole-column fill (`_zrows`'s two-argument
    # form); a joint chain takes the displacement-aware one.
    zrows = has_disp(H) ? _zrows(H, config, u) : _zrows(H, config)
    # One-word seeding is sound here: Julia ≥ 1.11 expands an integer seed into
    # the five-word Xoshiro state through a SHA-2-based hash, so sequentially
    # drawn seeds give effectively independent streams (a 64-bit birthday
    # collision — two sites sharing a proposal stream — has P ≈ n²/2⁶⁵).
    site_rngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:H.n_sites]
    return ChainState(config, u, zrows, _total_energy(H, zrows), rng, site_rngs,
                      Float64(step), Float64(step_u), 1.0, false,
                      0, 0, 0, 0, 0, 0, 0, 0, 0.0,
                      zeros(SVector{3,Float64}, H.n_disp_comps),
                      0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 1, 0.0, 0, false)
end

Base.show(io::IO, st::ChainState) =
    print(io, "ChainState(", length(st.config), " sites, E=",
          @sprintf("%.6g", st.energy), ", step=", @sprintf("%.3g", st.step),
          isempty(st.com_removed) ? "" : ", step_u=" * @sprintf("%.3g", st.step_u),
          st.frozen ? ", frozen" : "", ")")

"""
    SweepScratch(H::TiledHamiltonian)

Per-task scratch buffers for the sweep kernels (`c` — leave-one-out coefficients over
the site's full row table, `znew` — the proposed row table of the site under attempt
(only the moved channel's block is written; the ΔE is range-limited to it), `plm` —
the associated-Legendre recursion workspace of the internal `_zlm_row!`, `rbuf` — the
solid-harmonic batch workspace of `_disp_rows!` (empty on a pure-spin model), `dE` —
the per-site accepted-ΔE staging buffer the deterministic energy reduction reads back
in color order; when several tasks execute one sweep, only the **first** scratch's
`dE` is used — its writes are per-site disjoint). One per task, never shared across
tasks.
"""
struct SweepScratch
    c::Vector{Float64}
    znew::Vector{Float64}
    plm::Vector{Float64}
    rbuf::Vector{Float64}
    dE::Vector{Float64}
end

SweepScratch(H::TiledHamiltonian) =
    SweepScratch(zeros(H.nrows), zeros(H.nrows),
                 Vector{Float64}(undef, max(0, H.lmax + 1)),
                 Vector{Float64}(undef, max(0, (H.disp_lmax + 1)^2)),
                 zeros(H.n_sites))

# --- configuration helpers ----------------------------------------------------------

# Uniform random unit vector (Gaussian-normalized). An (astronomically improbable)
# near-zero draw would put a NaN spin into the chain; redraw instead. The extra
# branch never fires in practice, so RNG consumption — and bit-determinism — are
# unchanged on the no-retry path.
function _random_unit(rng::AbstractRNG)::SVector{3,Float64}
    while true
        v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
        n = norm(v)
        n > 1e-12 && return v / n
    end
end

# Resolve a chain start: `nothing` → uniform random from `rng`; a `3 × n_sites`
# matrix or a vector of 3-vectors → normalized copy.
function _initial_config(H::TiledHamiltonian, init, rng::AbstractRNG)::SpinConfig
    init === nothing &&
        return SpinConfig([_random_unit(rng) for _ = 1:H.n_sites])
    if init isa AbstractMatrix
        size(init) == (3, H.n_sites) || throw(DimensionMismatch(
            "init is $(size(init, 1))×$(size(init, 2)); expected 3×$(H.n_sites)"))
        return SpinConfig([_unit_or_throw(SVector{3,Float64}(init[1, s], init[2, s],
                                                             init[3, s]))
                           for s = 1:H.n_sites])
    end
    length(init) == H.n_sites || throw(DimensionMismatch(
        "init has $(length(init)) sites; expected $(H.n_sites)"))
    return SpinConfig([_unit_or_throw(SVector{3,Float64}(e)) for e in init])
end

function _unit_or_throw(e::SVector{3,Float64})::SVector{3,Float64}
    n = norm(e)
    n > 1e-12 || throw(ArgumentError("init contains a (near-)zero spin vector"))
    return e / n
end

# Resolve a chain's initial displacements: `nothing` → the clamped-ion start `u = 0`
# (a legitimate physical state, unlike an omitted `disps` in an *evaluation* call —
# there the reference frame is the caller's and silence would be a wrong answer;
# here the chain owns the frame and will sample away from it). A `3 × n_sites` matrix
# or a vector of 3-vectors is copied verbatim — displacements are not normalized.
function _initial_disps(H::TiledHamiltonian, init)::Vector{SVector{3,Float64}}
    init === nothing && return zeros(SVector{3,Float64}, H.n_sites)
    u = if init isa AbstractMatrix
        size(init) == (3, H.n_sites) || throw(DimensionMismatch(
            "disps is $(size(init, 1))×$(size(init, 2)); expected 3×$(H.n_sites)"))
        [SVector{3,Float64}(init[1, s], init[2, s], init[3, s]) for s = 1:H.n_sites]
    else
        length(init) == H.n_sites || throw(DimensionMismatch(
            "disps has $(length(init)) sites; expected $(H.n_sites)"))
        [SVector{3,Float64}(x) for x in init]
    end
    # Same footgun as in `_zrows`: a pure-spin Hamiltonian has nowhere to put
    # displacements, so a chain started at nonzero `u` would silently sample `u = 0`.
    has_disp(H) || all(iszero, u) || throw(ArgumentError(
        "this Hamiltonian has no displacement rows, but `disps` contains nonzero " *
        "displacements; it describes the clamped-ion (u = 0) energy only, so a " *
        "chain started there would silently ignore them"))
    return u
end

# Replace the chain's configuration in place (fresh restart): rebuild the basis rows
# and recompute the energy from scratch (no drift bookkeeping — this is not a
# renormalization of an evolved chain). `disps === nothing` restarts from the
# clamped-ion state; the accumulated re-centring record restarts with it.
function _reset_config!(st::ChainState, H::TiledHamiltonian, config::SpinConfig,
                        disps::Union{Nothing,Vector{SVector{3,Float64}}} = nothing)
    copyto!(st.config, config)
    if disps === nothing
        fill!(st.disps, zero(SVector{3,Float64}))
    else
        copyto!(st.disps, _initial_disps(H, disps))
    end
    fill!(st.com_removed, zero(SVector{3,Float64}))
    _reset_escape!(st)     # a fresh chain is a fresh phase for the detector
    plm = Vector{Float64}(undef, max(0, H.lmax + 1))
    rbuf = Vector{Float64}(undef, max(0, (H.disp_lmax + 1)^2))
    joint = has_disp(H)
    for s = 1:H.n_sites
        col = view(st.zrows, :, s)
        if joint
            _zlm_row!(view(col, 1:H.nlm), st.config[s], H.lmax, plm)
            _disp_rows!(col, H, st.disps[s], rbuf)
        else
            _zlm_row!(col, st.config[s], H.lmax, plm)
        end
    end
    st.energy = _total_energy(H, st.zrows)
    return st
end

# Thermalization → measurement boundary: freeze the transition kernel (a step that
# keeps responding to history is a bias source and breaks bit-reproducible resume) and
# clear every reported window, so the acceptance fractions and `max_drift` describe the
# measurement phase alone. ONE helper for both drivers on purpose: a new move type's
# counters must be reset here too, and a forgotten one surfaces as a quietly diluted
# acceptance rate rather than as a failure.
function _freeze_and_reset!(st::ChainState)::ChainState
    st.frozen = true
    st.acc_metro = 0
    st.att_metro = 0
    st.acc_or = 0
    st.att_or = 0
    st.acc_disp = 0
    st.att_disp = 0
    st.max_drift = 0.0
    _reset_escape!(st)
    return st
end

# Re-anchor the escape detector on the phase that is about to start.
#
# EVERY phase boundary must call this, not just the thermalization→measurement one:
# the test is "is the r.m.s. displacement of THIS phase flat", and its anchors
# (`disp_rms0`, `disp_ref_ms`) are r.m.s. values at the phase's own temperature. A
# stale anchor from a colder temperature makes a bounded model false-alarm — `rms ∝
# √T`, so a ladder spanning a factor of 200 in `T` produces a 14× growth that is pure
# thermodynamics and trips the 10× absolute guard, with a warning text that asserts the
# model is unnormalizable. The three call sites are `_freeze_and_reset!` (the
# measurement boundary), the start of each temperature's thermalization, and
# `_reset_config!` (the `carryover = false` restart); they are one function so they
# cannot drift apart.
function _reset_escape!(st::ChainState)::ChainState
    st.disp_rms = 0.0
    st.disp_max = 0.0
    st.disp_rms0 = 0.0
    st.disp_checks = 0
    st.disp_ms_sum = 0.0
    st.disp_blk_sum = 0.0
    st.disp_blk_n = 0
    st.disp_blk_cap = 1
    st.disp_ref_ms = 0.0
    st.escape_strikes = 0
    st.escape_warned = false
    return st
end

# Remove the centre-of-mass displacement of every displacement-coupling component.
#
# Where a component's rigid shift IS a symmetry, the sampler performs an unbiased
# random walk along it: after M sweeps the component's mean displacement has grown to
# |ū| ≈ σ√(3pM/N). Note the acoustic sum rule alone does NOT give that symmetry: with
# `E = Σ_c E_c(u_c)` over disjoint components, global translational invariance only
# forces `Σ_c v_c = 0` for the per-component shift responses `v_c`. Per-component (and
# per-direction) flatness is strictly stronger, which is exactly why it is MEASURED
# (`_translation_residuals`) rather than assumed.
#
# WHY THIS IS EXACT, and not merely cheap. Energy neutrality is the weaker statement;
# what is needed is that re-centring preserves the sampled measure. Split the state as
# (gauge coordinate ū) × (quotient state w with Σw = 0). `E` does not depend on ū; the
# single-site proposal projected onto the quotient is symmetric with a law that does not
# depend on ū either; so the chain is a skew product — the w-marginal is itself Markov
# and reversible for the intended measure, and ū is a decoupled passenger that nothing
# measures. `_recenter!` transforms only that passenger, hence is exactly
# stationarity-preserving. **The load-bearing hypothesis is that every restriction of
# the state space is a gauge-invariant function of the displacements.** Add a
# gauge-DEPENDENT restriction downstream (an absolute radius bound, say) and this
# argument fails — that is a standing constraint on anything added later, not a detail.
#
# What re-centring buys, given that it is free:
#   1. the reporting convention — ⟨|u|²⟩ measured against a drifting frame never reaches
#      a plateau, and any odd-in-u spin–lattice correlator picks up a first-order
#      contamination. Measured relative to one atom instead, it would be 2σ²(1−c(r)),
#      an EXAFS-style relative MSD, comparable neither with a Debye–Waller factor nor
#      with the upstream force constants;
#   2. numerical conditioning — `_disp_rows!` evaluates the solid harmonics at the
#      ABSOLUTE u, so under drift a slot's row grows like |ū|^{2k+l} (its total degree)
#      while the instance sum stays invariant: the energy becomes a difference of large
#      near-cancelling terms (measured ~30× loss of significance at |ū| ≈ 0.27 against a
#      relative amplitude of 0.05 — a degree-2 factor and (0.27/0.05)² ≈ 29 — and the
#      drift grows like √M). The centre-of-mass-free representative is the ℓ²-minimal
#      point of the gauge orbit;
#   3. the `com_removed` growth is a free null-model check that the direction really is
#      flat.
#
# Only over `site_has_disp` sites, and only per component: two components share no
# instance, so their rigid shifts are independent symmetries and a global mean would
# mix them — and only along the directions the construction gate measured as flat
# (`H.comp_free[d, c]`). A pinned direction's absolute frame is physical (a
# substrate-clamped slab is pinned along its normal and free in the plane), so
# re-centring there would change the energy; projecting the mean onto the flat subspace
# keeps the free directions from drifting without touching the pinned one.
#
# NEVER call this inside the color loop of a sweep: a mean over sites is an
# order-dependent floating-point reduction, and P6 (bit-determinism for any
# `sweep_tasks`) rests on every cross-site reduction being single-threaded and
# fixed-order. `_renormalize!` is that place.
function _recenter!(st::ChainState, H::TiledHamiltonian)::ChainState
    @inbounds for c = 1:H.n_disp_comps
        # PER COMPONENT **AND PER DIRECTION**: a pinned component next door is no reason
        # to let a flat one's frame drift, and a component pinned along one axis (a
        # substrate-clamped slab) is still free in the other two — projecting the mean
        # out of exactly the flat directions is what keeps those from random-walking.
        any(view(H.comp_free, :, c)) || continue
        lo = Int(H.disp_comp_ptr[c])
        hi = Int(H.disp_comp_ptr[c + 1]) - 1
        n = hi - lo + 1
        n == 0 && continue
        acc = zero(SVector{3,Float64})
        for q = lo:hi
            acc += st.disps[Int(H.disp_comp_sites[q])]
        end
        shift = (acc / n) .* SVector{3,Float64}(H.comp_free[1, c], H.comp_free[2, c],
                                                H.comp_free[3, c])
        iszero(shift) && continue              # exact at a clamped-ion start
        for q = lo:hi
            s = Int(H.disp_comp_sites[q])
            st.disps[s] -= shift
        end
        st.com_removed[c] += shift
    end
    return st
end

# Growth of the r.m.s. displacement over a DOUBLING of the observation window that is
# taken as evidence of escape rather than of equilibration. Calibrated on measured
# behaviour: a stationary chain gives 1.0, free diffusion of a flat direction gives
# √2 ≈ 1.41, and the escape reproduced on the joint test fixture grows r.m.s. ∝ M^1.6,
# i.e. 2^1.6 ≈ 3.0 per doubling. The threshold sits above free diffusion and far below
# the escape. Two consecutive strikes are required, so the one-off growth of a chain
# equilibrating from the clamped-ion start does not trip it.
# Escape-detector calibration. The test compares the mean square displacement of one
# observation block against the previous block's, so a stationary chain gives a ratio of
# 1.0, free diffusion of a surviving flat direction gives √2 ≈ 1.41, and the escape
# reproduced on the joint test fixture (r.m.s. ∝ M^1.6) gives ≈ 3.0.
#
# Blocks DOUBLE until they reach `_ESCAPE_WINDOW`, then stay that length. The cap is what
# makes a late-onset escape detectable at all: with an unbounded doubling ladder the
# comparisons fall at checks 1, 2, 4, 8, …, so an escape that starts after a quarter of
# the run never gets two consecutive comparisons and is never reported — and a
# spin-driven escape (the model only becomes unbounded once the spins order) is exactly
# a late-onset escape. Averaging over the block also cuts the sampling noise of the
# ratio, which is set by the effective number of degrees of freedom carrying ⟨|u|²⟩ —
# NOT by the site count. A single dominant soft mode has ν ≈ 1 whatever the system size,
# and the unbounded-ladder scheme false-warns on ~23 % of such runs; blocks of 16 with
# three strikes bring that to ~0.1 % while still firing on the measured escape.
const _ESCAPE_GROWTH = 1.7
const _ESCAPE_STRIKES = 3
const _ESCAPE_WINDOW = 16
# Growth relative to the FIRST measurement of the phase. The block test only compares
# neighbouring blocks, so growth slower than 2^0.766 per block accumulates for ever
# without ever raising a strike; this catches that, and it is a statement no chain with
# a stationary distribution can violate.
const _ESCAPE_ABSOLUTE = 10.0

# Escape detector: the truncated cluster expansion is a finite polynomial in `u`, so
# `exp(−βE)` is a probability measure only when the leading even form is positive
# definite — and nothing upstream guarantees that. When it is not, the chain has no
# stationary distribution and simply runs downhill. NOTHING already in the sampler
# notices: the ΔE bookkeeping stays exact to 1e-14 (so the drift warning is silent by
# construction) and the acceptance sits at 0.97–0.99 (every proposal is downhill). That
# is the classic improper-target pathology — every single-site conditional is proper
# while the joint distribution is not — and the only thing that separates it from
# equilibrium is RECURRENCE, which no existing diagnostic measures.
#
# So measure it here, at the one place that is single-threaded, fixed-order and
# deterministically scheduled (P6): after re-centring, compare block-averaged mean square
# displacements. Cheap — one pass over the sites `_recenter!` has just walked — and it
# detects the degree ≥ 3 unboundedness that a harmonic-stability screen structurally
# cannot (deciding global non-negativity of a quartic form is NP-hard).
#
# Detection only: this warns, it does not constrain. A displacement bound would turn a
# loud failure into a quiet one — a finite number set by the bound rather than by the
# Hamiltonian — and where the boundary genuinely carries thermodynamic weight the right
# instrument is umbrella sampling / Hamiltonian replica exchange over a collective
# displacement coordinate, not a wall (see `docs/specs/updates-stationarity.md` U8).
function _check_escape!(st::ChainState, H::TiledHamiltonian)::ChainState
    n = 0
    q = 0.0
    m2 = 0.0
    @inbounds for c = 1:H.n_disp_comps
        for k = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)
            # Already centre-of-mass-free along every flat direction (`_recenter!` just
            # ran); a pinned direction's absolute frame IS the physical one.
            r2 = let u = st.disps[Int(H.disp_comp_sites[k])]
                dot(u, u)
            end
            q += r2
            m2 = max(m2, r2)
            n += 1
        end
    end
    n == 0 && return st
    rms = sqrt(q / n)
    st.disp_rms = rms
    st.disp_max = max(st.disp_max, sqrt(m2))
    st.disp_checks += 1
    st.disp_ms_sum += rms * rms      # phase total, never reset by the block ladder
    # Anchored on the first NONZERO r.m.s., not on the first check: a joint chain
    # started at the clamped-ion `u = 0` and sampled with `disp_per_metropolis = 0`
    # stays at exactly zero for ever, and a zero anchor would make the ratio test
    # divide by zero rather than simply never fire.
    st.disp_rms0 <= 0.0 && rms > 0.0 && (st.disp_rms0 = rms)
    st.disp_rms0 > 0.0 && rms > _ESCAPE_ABSOLUTE * st.disp_rms0 &&
        _escape_warn!(st, "has grown by more than $(_ESCAPE_ABSOLUTE)× since the start " *
                          "of this phase")
    st.disp_blk_sum += rms * rms
    st.disp_blk_n += 1
    st.disp_blk_n < st.disp_blk_cap && return st
    ms = st.disp_blk_sum / st.disp_blk_n
    if st.disp_ref_ms > 0.0
        if sqrt(ms / st.disp_ref_ms) > _ESCAPE_GROWTH
            st.escape_strikes += 1
            st.escape_strikes >= _ESCAPE_STRIKES &&
                _escape_warn!(st, "has grown by more than $(_ESCAPE_GROWTH)× in each of " *
                                  "$(_ESCAPE_STRIKES) consecutive observation blocks")
        else
            st.escape_strikes = 0
        end
    end
    st.disp_ref_ms = ms
    st.disp_blk_sum = 0.0
    st.disp_blk_n = 0
    st.disp_blk_cap = min(2 * st.disp_blk_cap, _ESCAPE_WINDOW)
    return st
end

# Once per chain phase, not once per session: `maxlog = 1` is per callsite for the whole
# logger, so a single spurious strike anywhere would silence every genuine report from
# every other temperature and every other replica-exchange lane for the rest of the run.
# `_freeze_and_reset!` clears the flag at the thermalization boundary, so the measurement
# phase — the one that produces the published numbers — always gets its own say.
function _escape_warn!(st::ChainState, why::AbstractString)::Nothing
    st.escape_warned && return nothing
    st.escape_warned = true
    @warn "the r.m.s. displacement $(why) (now $(st.disp_rms), max so far " *
          "$(st.disp_max), after $(st.disp_checks) renormalizations). A chain in " *
          "equilibrium holds it flat, so this is evidence that the model's " *
          "displacement energy is unbounded below and the chain has no stationary " *
          "distribution — in which case no displacement observable from this run means " *
          "anything. Check the dynamical stability of the model at the sampled spin " *
          "configurations; a purely harmonic (max displacement degree 2) model with an " *
          "imaginary branch cannot be sampled at any temperature."
    return nothing
end

# Renormalize every spin-active spin, re-centre the displacements, rebuild the
# affected basis rows, and re-anchor the incremental energy on a full recomputation.
# Records the observed drift; returns it. Sites inactive in a channel stay bitwise
# frozen in it (never updated, so no drift to fix; those rows are never read).
function _renormalize!(st::ChainState, H::TiledHamiltonian,
                       sc::SweepScratch)::Float64
    for s = 1:H.n_sites
        H.site_has_spin[s] || continue
        e = normalize(st.config[s])
        st.config[s] = e
        _zlm_row!(view(st.zrows, 1:H.nlm, s), e, H.lmax, sc.plm)
    end
    if has_disp(H)
        _recenter!(st, H)
        for s = 1:H.n_sites
            H.site_has_disp[s] || continue
            _disp_rows!(view(st.zrows, :, s), H, st.disps[s], sc.rbuf)
        end
        _check_escape!(st, H)
    end
    E = _total_energy(H, st.zrows)
    drift = abs(st.energy - E)
    st.max_drift = max(st.max_drift, drift)
    if drift > 1e-8 * max(1.0, abs(E))
        @warn "incremental-energy drift $(drift) at renormalization (E = $E); " *
              "consider a smaller renorm_interval" *
              (has_disp(H) ? " — unless the model's uniform-shift direction is only " *
                             "marginally flat (translation_residual = " *
                             "$(H.translation_residual)), in which case re-centring " *
                             "is the source and a smaller interval makes it worse" :
                             "") maxlog = 1
    end
    st.energy = E
    return drift
end

# Convenience form (tests / scratch-less callers): allocates the workspace.
_renormalize!(st::ChainState, H::TiledHamiltonian)::Float64 =
    _renormalize!(st, H, SweepScratch(H))
