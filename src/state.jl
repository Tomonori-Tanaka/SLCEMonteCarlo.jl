# Mutable chain state and thread-confined scratch — deliberately separated from the
# immutable `TiledHamiltonian` and from run configuration (no God-struct).

"""
    ChainState

The mutable state of one Markov chain: the spin `config` with its cached tesseral
rows `zrows` (column `s` = `Z_lm(e_s)`), the incrementally maintained total `energy`
(model units, `j0` excluded — kept exact by the ΔE bookkeeping and re-anchored at
every renormalization), the chain-owned `rng` (config resets) plus one independent
proposal/accept stream per site (`site_rngs`, derived from `rng` at construction —
what makes the sweeps deterministic regardless of how many tasks execute a color
class), the Metropolis proposal `step` (radians; adapted during thermalization,
frozen once `frozen` is set), windowed acceptance counters, and the worst
incremental-energy `max_drift` observed at renormalization points.
"""
mutable struct ChainState
    # config/zrows/energy are the swappable "payload" of a replica-exchange move
    # (`_swap_payload!` exchanges the references) — hence not `const`. The RNG
    # streams stay with the lane, like `rng`.
    config::SpinConfig
    zrows::Matrix{Float64}
    energy::Float64
    const rng::Xoshiro
    const site_rngs::Vector{Xoshiro}
    step::Float64
    frozen::Bool
    acc_metro::Int
    att_metro::Int
    acc_or::Int
    att_or::Int
    max_drift::Float64
end

function ChainState(H::TiledHamiltonian, config::SpinConfig, rng::Xoshiro,
                    step::Real)
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    _require_spin_only(H, "ChainState (and hence run_mc / run_pt)")
    zrows = _zrows(H, config)
    # One-word seeding is sound here: Julia ≥ 1.11 expands an integer seed into
    # the five-word Xoshiro state through a SHA-2-based hash, so sequentially
    # drawn seeds give effectively independent streams (a 64-bit birthday
    # collision — two sites sharing a proposal stream — has P ≈ n²/2⁶⁵).
    site_rngs = [Xoshiro(rand(rng, UInt64)) for _ = 1:H.n_sites]
    return ChainState(config, zrows, _total_energy(H, zrows), rng, site_rngs,
                      Float64(step), false, 0, 0, 0, 0, 0.0)
end

Base.show(io::IO, st::ChainState) =
    print(io, "ChainState(", length(st.config), " sites, E=",
          @sprintf("%.6g", st.energy), ", step=", @sprintf("%.3g", st.step),
          st.frozen ? ", frozen" : "", ")")

"""
    SweepScratch(H::TiledHamiltonian)

Per-task scratch buffers for the sweep kernels (`c` — leave-one-out coefficients,
`znew` — the proposed spin's tesseral row, `plm` — the associated-Legendre
recursion workspace of the internal `_zlm_row!`, `dE` — the per-site accepted-ΔE
staging buffer the deterministic energy reduction reads back in color order; when
several tasks execute one sweep, only the **first** scratch's `dE` is used — its
writes are per-site disjoint). One per task, never shared across tasks.
"""
struct SweepScratch
    c::Vector{Float64}
    znew::Vector{Float64}
    plm::Vector{Float64}
    dE::Vector{Float64}
end

SweepScratch(H::TiledHamiltonian) =
    SweepScratch(zeros(H.nlm), zeros(H.nlm), Vector{Float64}(undef, H.lmax + 1),
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

# Replace the chain's configuration in place (fresh restart): rebuild the tesseral
# rows and recompute the energy from scratch (no drift bookkeeping — this is not a
# renormalization of an evolved chain).
function _reset_config!(st::ChainState, H::TiledHamiltonian, config::SpinConfig)
    copyto!(st.config, config)
    plm = Vector{Float64}(undef, H.lmax + 1)
    for s = 1:H.n_sites
        _zlm_row!(view(st.zrows, :, s), st.config[s], H.lmax, plm)
    end
    st.energy = _total_energy(H, st.zrows)
    return st
end

# Renormalize every active spin, rebuild its tesseral rows, and re-anchor the
# incremental energy on a full recomputation. Records the observed drift; returns
# it. Inactive sites stay bitwise frozen (never updated, so no drift to fix; their
# zrows columns are never read).
function _renormalize!(st::ChainState, H::TiledHamiltonian,
                       plm::Vector{Float64})::Float64
    for s = 1:H.n_sites
        H.site_has_spin[s] || continue
        e = normalize(st.config[s])
        st.config[s] = e
        _zlm_row!(view(st.zrows, :, s), e, H.lmax, plm)
    end
    E = _total_energy(H, st.zrows)
    drift = abs(st.energy - E)
    st.max_drift = max(st.max_drift, drift)
    if drift > 1e-8 * max(1.0, abs(E))
        @warn "incremental-energy drift $(drift) at renormalization (E = $E); " *
              "consider a smaller renorm_interval" maxlog = 1
    end
    st.energy = E
    return drift
end

# Convenience form (tests / scratch-less callers): allocates the workspace.
_renormalize!(st::ChainState, H::TiledHamiltonian)::Float64 =
    _renormalize!(st, H, Vector{Float64}(undef, H.lmax + 1))
