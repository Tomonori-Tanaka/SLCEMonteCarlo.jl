# Ground-state search (design + correctness arguments:
# `docs/specs/ground-state-search.md`).
#
# Two entry points share one result type: `minimize_energy` — deterministic
# Riemannian projected-gradient descent on the product of unit spheres (BB1 step +
# nonmonotone Armijo safeguard, no RNG) — and `find_ground_state` — bit-reproducible
# multi-start simulated annealing (with optional thermal cycling) whose every start
# is polished by the same descent. The descent is the only new numerical kernel;
# annealing reuses the sweep kernels of `updates.jl` unchanged.

const _NM_WINDOW = 10        # GLL nonmonotone reference window (accepted energies)
const _ARMIJO_SIGMA = 1e-4   # sufficient-decrease fraction of the Armijo model
const _BACKTRACK = 0.5       # step-halving factor
const _MAX_BACKTRACKS = 30   # give up (stagnation) after this many halvings

# Max over sites of Σ_{adjacent instances} |coef|·sum(abs, folded) — a cheap
# deterministic overestimate of the per-site energy scale (the coef carries the
# (4π)^(body/2) factor that roughly cancels the tesseral normalization). Anchors the
# scale-aware default `gtol` and the default annealing ladder; a too-large scale only
# wastes a few hot sweeps / loosens gtol proportionally.
function _site_energy_scale(H::TiledHamiltonian)::Float64
    t_scale = [abs(t.coef) * sum(abs, t.folded) for t in H.terms]
    scale = 0.0
    @inbounds for s = 1:H.n_sites
        acc = 0.0
        for j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
            acc += t_scale[H.inst_term[H.site_inst[j]]]
        end
        scale = max(scale, acc)
    end
    return scale > 0.0 ? scale : 1.0   # all-zero folded is representable, not useful
end

# Default annealing ladder: geometric, 20 rungs over three decades below the
# per-site energy scale (hot enough to decorrelate, cold enough that the gradient
# polish takes over).
_default_anneal_kts(H::TiledHamiltonian)::Vector{Float64} =
    _site_energy_scale(H) .* exp10.(range(0.0, -3.0; length = 20))

function _resolve_gtol(H::TiledHamiltonian, gtol::Union{Nothing,Real})::Float64
    gtol === nothing && return 1e-8 * _site_energy_scale(H)
    g = Float64(gtol)
    g > 0 || throw(ArgumentError("gtol must be > 0; got $gtol"))
    return g
end

# All-site tangent-projected gradient of +E into `G` (the descent direction is −G),
# from precomputed tesseral rows; returns `max_s |G_s|`. Per site this is the shared
# `_site_grad` kernel of energy.jl — the same arithmetic as the public
# `site_gradient` / `energy_gradient!` (the `==` consistency gates in
# test_minimize.jl / test_gradient.jl pin them together) — but sharing one `zrows`
# matrix and one coefficient buffer: one pass costs one Metropolis sweep, not
# `n_sites` full-row rebuilds. `c` is scratch (overwritten).
function _gradient!(G::Vector{SVector{3,Float64}}, H::TiledHamiltonian,
                    config::SpinConfig, zrows::Matrix{Float64},
                    c::Vector{Float64}, plm::Vector{Float64})::Float64
    gsup = 0.0
    for s = 1:H.n_sites
        if !H.site_has_spin[s]           # spin-independent site: exactly zero, as
            G[s] = zero(SVector{3,Float64})  # site_gradient returns (the == gate)
            continue
        end
        g = _site_grad(H, s, config[s], zrows, c, plm)
        G[s] = g
        gsup = max(gsup, norm(g))
    end
    return gsup
end

# Buffers of one minimization (one per start, never shared across tasks).
mutable struct _MinimizeScratch
    const G::Vector{SVector{3,Float64}}        # gradient at the current iterate
    const Gprev::Vector{SVector{3,Float64}}    # … at the previous iterate (BB pair)
    const xprev::SpinConfig
    const trial::SpinConfig
    zrows::Matrix{Float64}     # swapped with ztrial on an accepted step — not const
    ztrial::Matrix{Float64}
    const c::Vector{Float64}   # leave-one-out coefficient scratch
    const plm::Vector{Float64} # associated-Legendre recursion workspace
    const fwin::Vector{Float64}   # circular GLL window of accepted energies
end

_MinimizeScratch(H::TiledHamiltonian) = _MinimizeScratch(
    Vector{SVector{3,Float64}}(undef, H.n_sites),
    Vector{SVector{3,Float64}}(undef, H.n_sites),
    SpinConfig(undef, H.n_sites), SpinConfig(undef, H.n_sites),
    Matrix{Float64}(undef, H.nlm, H.n_sites),
    Matrix{Float64}(undef, H.nlm, H.n_sites),
    zeros(H.nlm), Vector{Float64}(undef, H.lmax + 1), fill(-Inf, _NM_WINDOW))

# Riemannian BB1 projected-gradient descent with a GLL nonmonotone Armijo safeguard
# on the product of unit spheres; retraction = per-site normalize (division-safe:
# G ⊥ e ⇒ |e − αG| ≥ 1). Mutates `config` to the final iterate and returns
# (energy, gradnorm, iterations, converged). Consumes no RNG — this is what makes
# the threaded multi-start trivially deterministic.
function _minimize!(config::SpinConfig, H::TiledHamiltonian, ms::_MinimizeScratch,
                    gtol::Float64, maxiter::Int)
    n = H.n_sites
    for s = 1:n
        _zlm_row!(view(ms.zrows, :, s), config[s], H.lmax, ms.plm)
    end
    E = _total_energy(H, ms.zrows)
    gsup = _gradient!(ms.G, H, config, ms.zrows, ms.c, ms.plm)
    gsup <= gtol && return (E, gsup, 0, true)
    fill!(ms.fwin, -Inf)
    ms.fwin[1] = E
    widx = 1
    α = 0.1 / gsup                       # first step: max rotation ≈ 0.1 rad
    for k = 1:maxiter
        g2 = 0.0
        for s = 1:n
            g2 += dot(ms.G[s], ms.G[s])
        end
        # The exact first-order model on the manifold: d/dα E(R(x − αG))|₀ = −‖G‖²
        # (the differential of normalize at a unit point is the tangent projector
        # and G is already tangent), so `fref − σ·α·g2` is a genuine Armijo test.
        fref = maximum(ms.fwin)
        Etrial = E
        accepted = false
        for _ = 0:_MAX_BACKTRACKS
            for s = 1:n
                # Inactive sites stay bitwise frozen (G ≡ 0 there, and normalize of
                # an already-unit spin could still drift the last bits).
                ms.trial[s] = H.site_has_spin[s] ?
                              normalize(config[s] - α * ms.G[s]) : config[s]
            end
            for s = 1:n
                _zlm_row!(view(ms.ztrial, :, s), ms.trial[s], H.lmax, ms.plm)
            end
            Etrial = _total_energy(H, ms.ztrial)
            if Etrial <= fref - _ARMIJO_SIGMA * α * g2
                accepted = true
                break
            end
            α *= _BACKTRACK
        end
        # Exhausted backtracking = stagnation at the energy-resolution floor:
        # report the current iterate honestly instead of throwing.
        accepted || return (E, gsup, k - 1, false)
        copyto!(ms.xprev, config)
        copyto!(ms.Gprev, ms.G)
        copyto!(config, ms.trial)
        ms.zrows, ms.ztrial = ms.ztrial, ms.zrows   # reference swap
        E = Etrial
        widx = mod1(widx + 1, _NM_WINDOW)
        ms.fwin[widx] = E
        gsup = _gradient!(ms.G, H, config, ms.zrows, ms.c, ms.plm)
        gsup <= gtol && return (E, gsup, k, true)
        # BB1 step from the ambient-coordinate secant pair — a scaling heuristic
        # only (the Armijo safeguard owns correctness); nonconvex curvature
        # (sy ≤ 0) falls back to the largest admissible step.
        ss = 0.0
        sy = 0.0
        for s = 1:n
            Δx = config[s] - ms.xprev[s]
            ss += dot(Δx, Δx)
            sy += dot(Δx, ms.G[s] - ms.Gprev[s])
        end
        # (the retraction angle is atan(α|G_s|), so the π/gsup cap bounds every
        # per-site rotation by atan(π) ≈ 1.26 rad)
        α = sy > 0 ? clamp(ss / sy, 1e-10 / gsup, π / gsup) : π / gsup
    end
    return (E, gsup, maxiter, false)
end

"""
    GroundStateResult

Result of [`minimize_energy`](@ref) / [`find_ground_state`](@ref). The winner:
`config` (the lowest-energy polished configuration found), its `energy` (model
units, `j0` excluded, recomputed from scratch — no incremental drift), `gradnorm`
(`max_s |∇E|` on the sphere), the polish `iterations`, and `converged` (the winning
start reached `gtol` within `maxiter`). The full per-start table, **in start
order**: `configs` / `energies` / `gradnorms` / `converged_starts` — the spread of
`energies` is a cheap degeneracy/landscape diagnostic (identical values ⇒ the same
basin or symmetry copies; a wide spread ⇒ a rugged landscape, consider more starts,
`cycles`, or the PT-polish recipe). `best` is the winning start index (ties break
to the lowest index) and `seed` the seed actually used. `config` **aliases**
`configs[best]` (same array — copy before mutating).
"""
struct GroundStateResult
    config::SpinConfig
    energy::Float64
    gradnorm::Float64
    iterations::Int
    converged::Bool
    best::Int
    configs::Vector{SpinConfig}
    energies::Vector{Float64}
    gradnorms::Vector{Float64}
    converged_starts::Vector{Bool}
    seed::UInt64
end

Base.show(io::IO, r::GroundStateResult) =
    print(io, "GroundStateResult(E=", @sprintf("%.6g", r.energy), ", ",
          length(r.energies), " start(s), ", length(r.config), " sites",
          r.converged ? "" : ", NOT converged", ")")

function Base.show(io::IO, ::MIME"text/plain", r::GroundStateResult)
    n = length(r.energies)
    println(io, "GroundStateResult: ", length(r.config), " sites, ", n,
            " start(s), ", count(r.converged_starts), " converged, seed ", r.seed)
    @printf(io, "  E = %.10g   |grad| = %.3g   %s (%d iterations, start %d)\n",
            r.energy, r.gradnorm, r.converged ? "converged" : "NOT converged",
            r.iterations, r.best)
    @printf(io, "  %-6s %-22s %-12s %-10s %-4s\n",
            "start", "E", "E - E_best", "|grad|", "conv")
    order = sortperm(r.energies)
    shown = min(n, 16)
    for i = 1:shown
        s = order[i]
        @printf(io, "  %-6d %-22.10g %-12.4g %-10.3g %-4s\n", s, r.energies[s],
                r.energies[s] - r.energy, r.gradnorms[s],
                r.converged_starts[s] ? "yes" : "no")
    end
    shown < n && println(io, "  … ", n - shown, " more")
    return nothing
end

"""
    minimize_energy(H::TiledHamiltonian; init = nothing, gtol = nothing,
                    maxiter = 1_000, seed = rand(UInt64)) -> GroundStateResult

Deterministic **local** energy minimization on the product of unit spheres:
Barzilai–Borwein projected-gradient descent with a nonmonotone Armijo safeguard
(algorithm and correctness arguments: `docs/specs/ground-state-search.md`), from
`init` — a `3 × n_sites` matrix or a vector of 3-vectors (normalized on ingest);
default: uniform random drawn from `seed`. The descent itself consumes no RNG, so
the result is bit-reproducible given an explicit `init`; either way the seed used
is recorded in the result.

Converges when the largest on-sphere gradient magnitude drops to `gtol` (default
`1e-8 ×` a per-site energy scale of the model; the scale ignores the harmonic
magnitudes, so on high-`l` models the default flag can be optimistic — pass an
explicit `gtol` where the stationarity certificate matters). Exhausting `maxiter`,
or stagnating
at the energy-resolution floor, returns `converged = false` — never throws. Every
accepted iterate has `E ≤ E(init)` (the nonmonotone window never rises above the
start).

This polishes the nearest stationary point only — for a search across basins use
[`find_ground_state`](@ref) (or its PT-polish recipe).
"""
function minimize_energy(H::TiledHamiltonian; init = nothing,
                         gtol::Union{Nothing,Real} = nothing,
                         maxiter::Integer = 1_000,
                         seed::Integer = rand(UInt64))::GroundStateResult
    _require_spin_only(H, "minimize_energy")
    gt = _resolve_gtol(H, gtol)
    maxiter >= 0 || throw(ArgumentError("maxiter must be ≥ 0; got $maxiter"))
    seed >= 0 || throw(ArgumentError("seed must be ≥ 0; got $seed"))
    seed_u = UInt64(seed)
    config = _initial_config(H, init, Xoshiro(seed_u))
    E, gn, it, cv = _minimize!(config, H, _MinimizeScratch(H), gt, Int(maxiter))
    return GroundStateResult(config, E, gn, it, cv, 1, [config], [E], [gn], [cv],
                             seed_u)
end

# One start of `find_ground_state`: resolve the start's initial configuration from
# its own RNG, anneal down the ladder (`cycles`-fold thermal cycling, keeping the
# best cold-end configuration), then polish deterministically (no RNG). Writes only
# slot `r` of the result vectors — starts never share mutable state.
function _gs_start!(configs::Vector{SpinConfig}, energies::Vector{Float64},
                    gradnorms::Vector{Float64}, iters::Vector{Int},
                    convs::Vector{Bool}, r::Int, H::TiledHamiltonian, init,
                    rng::Xoshiro, kts::Vector{Float64}, anneal_sweeps::Int,
                    cycles::Int, reheat::Float64, orpm::Int, step::Float64,
                    adapt_target::Float64, adapt_interval::Int, sweep_tasks::Int,
                    gtol::Float64, maxiter::Int)
    config = _initial_config(H, init, rng)
    if anneal_sweeps > 0
        st = ChainState(H, config, rng, step)
        scs = [SweepScratch(H) for _ = 1:sweep_tasks]
        nr = length(kts)
        rentry = clamp(ceil(Int, reheat * nr), 1, nr)  # re-entry rung of cycles ≥ 2
        best_E = Inf
        bestcfg = config
        for cycle = 1:cycles
            for k = (cycle == 1 ? 1 : rentry):nr
                β = 1.0 / kts[k]
                for sweep = 1:anneal_sweeps
                    metropolis_sweep!(st, H, β, scs)
                    for _ = 1:orpm
                        overrelaxation_sweep!(st, H, β, scs)
                    end
                    sweep % adapt_interval == 0 && _adapt_step!(st, adapt_target)
                end
                _renormalize!(st, H, scs[1])   # exact rung-boundary energy
            end
            # cycle 1 always snapshots (also the NaN-safe fallback: a non-finite
            # chain energy can never leave `bestcfg` aliased to the live config)
            if cycle == 1 || st.energy < best_E
                best_E = st.energy
                bestcfg = copy(st.config)
            end
        end
        config = bestcfg
    end
    E, gn, it, cv = _minimize!(config, H, _MinimizeScratch(H), gtol, maxiter)
    configs[r] = config
    energies[r] = E
    gradnorms[r] = gn
    iters[r] = it
    convs[r] = cv
    return nothing
end

"""
    find_ground_state(H::TiledHamiltonian; temperature = nothing, kT = nothing,
                      nstarts = nothing, inits = nothing, kwargs...)
        -> GroundStateResult

Stochastic **global** ground-state search: independent multi-start simulated
annealing — each start anneals down a strictly decreasing temperature ladder with
the package's Metropolis (+ optional overrelaxation) sweeps, optionally with
**thermal cycling**, and is then polished by the deterministic descent of
[`minimize_energy`](@ref). Starts run concurrently over threads; results are
**bit-identical for a fixed seed regardless of `ntasks` and the thread count**
(per-start RNGs are split from the master in start order, the polish consumes no
RNG, and each start writes only its own result slot). The winner is the lowest
per-start energy (ties break to the lowest start index); the full per-start energy
table in the result is a degeneracy/landscape diagnostic.

This is a heuristic — no finite search certifies a global minimum. Cross-check
rugged landscapes with more starts, `cycles`, or the PT-polish recipe below.

# Keyword arguments
- `temperature` / `kT`: the annealing ladder (kelvin / model energy units, exactly
  one; strictly decreasing when more than one value). Default: a geometric
  20-rung ladder over three decades below a per-site energy scale of the model —
  pass an explicit ladder for production work.
- `nstarts = 8`: number of independent random starts. Mutually exclusive with
  `inits` — a vector of explicit starting configurations (each a `3 × n_sites`
  matrix or a vector of 3-vectors), e.g. `PTResult.final_configs`.
- `anneal_sweeps = 100`: sweeps per ladder rung; `0` skips annealing entirely
  (pure parallel multi-start descent). The recipe
  `inits = pt.final_configs, anneal_sweeps = 0` polishes a finished
  [`run_pt`](@ref) — parallel tempering is the principled "temperature up-down"
  and the strongest basin escape this package offers.
- `cycles = 1`, `reheat = 0.5`: thermal cycling (Möbius et al., PRL 79, 4297
  (1997)). Cycles after the first re-enter the ladder at rung
  `ceil(reheat · n_rungs)` — partial re-heating keeps the found basin's
  information while allowing barrier crossings (re-heating to the top would be a
  plain restart, which is what `nstarts` is for). The best cold-end configuration
  across cycles is the one polished.
- `or_per_metropolis = 0`, `step = 0.6`, `adapt_target = 0.5`,
  `adapt_interval = 10`: sweep mixing and (never-frozen) step adaptation during
  annealing, as in [`run_mc`](@ref) thermalization.
- `ntasks = min(number of starts, nthreads())`: concurrent start tasks.
- `sweep_tasks = 1`: concurrent tasks inside each annealing sweep (color-parallel
  updates; bit-identical for any value — keep `ntasks · sweep_tasks` within the
  thread count).
- `gtol`, `maxiter`, `seed`: as in [`minimize_energy`](@ref).
"""
function find_ground_state(H::TiledHamiltonian; temperature = nothing,
                           kT = nothing, nstarts::Union{Nothing,Integer} = nothing,
                           inits::Union{Nothing,AbstractVector} = nothing,
                           anneal_sweeps::Integer = 100, cycles::Integer = 1,
                           reheat::Real = 0.5, or_per_metropolis::Integer = 0,
                           step::Real = 0.6, adapt_target::Real = 0.5,
                           adapt_interval::Integer = 10,
                           sweep_tasks::Integer = 1,
                           ntasks::Union{Nothing,Integer} = nothing,
                           gtol::Union{Nothing,Real} = nothing,
                           maxiter::Integer = 1_000,
                           seed::Integer = rand(UInt64))::GroundStateResult
    _require_spin_only(H, "find_ground_state")
    kts = temperature === nothing && kT === nothing ? _default_anneal_kts(H) :
          resolve_kt(temperature, kT)
    length(kts) == 1 || all(diff(kts) .< 0) || throw(ArgumentError(
        "the annealing ladder must be strictly decreasing (hot → cold); got $kts"))
    if inits === nothing
        n = nstarts === nothing ? 8 : Int(nstarts)
        n >= 1 || throw(ArgumentError("nstarts must be ≥ 1; got $n"))
        ins = nothing
    else
        nstarts === nothing ||
            throw(ArgumentError("pass either nstarts or inits, not both"))
        ins = collect(inits)
        n = length(ins)
        n >= 1 || throw(ArgumentError("inits is empty"))
    end
    anneal_sweeps >= 0 ||
        throw(ArgumentError("anneal_sweeps must be ≥ 0; got $anneal_sweeps"))
    cycles >= 1 || throw(ArgumentError("cycles must be ≥ 1; got $cycles"))
    0 < reheat < 1 || throw(ArgumentError("reheat must be in (0, 1); got $reheat"))
    or_per_metropolis >= 0 || throw(ArgumentError(
        "or_per_metropolis must be ≥ 0; got $or_per_metropolis"))
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    0 < adapt_target < 1 || throw(ArgumentError(
        "adapt_target must be in (0, 1); got $adapt_target"))
    adapt_interval >= 1 ||
        throw(ArgumentError("adapt_interval must be ≥ 1; got $adapt_interval"))
    nt = ntasks === nothing ? min(n, Threads.nthreads()) : Int(ntasks)
    nt >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $nt"))
    sweep_tasks >= 1 ||
        throw(ArgumentError("sweep_tasks must be ≥ 1; got $sweep_tasks"))
    gt = _resolve_gtol(H, gtol)
    maxiter >= 0 || throw(ArgumentError("maxiter must be ≥ 0; got $maxiter"))
    seed >= 0 || throw(ArgumentError("seed must be ≥ 0; got $seed"))
    seed_u = UInt64(seed)

    # RNG discipline (as run_pt): master → one Xoshiro per start, in start order,
    # before any spawning; each stream is consumed only inside its own start.
    master = Xoshiro(seed_u)
    start_rngs = [Xoshiro(rand(master, UInt64), rand(master, UInt64),
                          rand(master, UInt64), rand(master, UInt64)) for _ = 1:n]
    configs = Vector{SpinConfig}(undef, n)
    energies = Vector{Float64}(undef, n)
    gradnorms = Vector{Float64}(undef, n)
    iters = Vector{Int}(undef, n)
    convs = Vector{Bool}(undef, n)
    run_start!(r) = _gs_start!(configs, energies, gradnorms, iters, convs, r, H,
                               ins === nothing ? nothing : ins[r], start_rngs[r],
                               kts, Int(anneal_sweeps), Int(cycles),
                               Float64(reheat), Int(or_per_metropolis),
                               Float64(step), Float64(adapt_target),
                               Int(adapt_interval), Int(sweep_tasks), gt,
                               Int(maxiter))
    if nt <= 1
        for r = 1:n
            run_start!(r)
        end
    else
        chunk = cld(n, nt)
        @sync for lo = 1:chunk:n
            hi = min(lo + chunk - 1, n)
            Threads.@spawn for r = lo:hi
                run_start!(r)
            end
        end
    end
    best = argmin(energies)     # first minimum — deterministic tie-break
    # `isless` sorts NaN last, so argmin avoids non-finite energies unless every
    # start diverged — never hand back a NaN "winner" silently
    nbad = count(!isfinite, energies)
    nbad == 0 ||
        @warn "find_ground_state: $nbad of $n starts produced a non-finite energy"
    isfinite(energies[best]) ||
        error("find_ground_state: every start produced a non-finite energy")
    return GroundStateResult(configs[best], energies[best], gradnorms[best],
                             iters[best], convs[best], best, configs, energies,
                             gradnorms, convs, seed_u)
end
