# Shared fixtures and timing helpers for the SLCEMonteCarlo benchmark scripts.
#
# `include`d (not `import`ed) by each bench/bench_*.jl. Mirrors the SLCE.jl
# bench convention (standalone scripts, own environment, `bench_one` harness).
#
# Two fixture models span the kernel-cost regimes:
#
# - `bcc_fe_model()` — 2-atom bcc Fe training cell, isotropic l = 1 Heisenberg
#   (nlm = 4, one pair orbit): the LIGHT-kernel / large-lattice regime, where sweep
#   throughput is dominated by bookkeeping, not the tensor contraction.
# - `nd2fe14b_model()` — the 68-atom Nd₂Fe₁₄B cell (assets/nd2fe14b.toml), 9
#   sublattice species, l ≤ 2, every resolvable pair: the HEAVY-kernel regime
#   (thousands of terms, cf. the real l02 production model with 4692 terms), where
#   `site_coeffs!` dominates.
#
# Coefficients are synthetic (seeded — timing does not depend on values); energies
# are meaningless physically but the acceptance rates they produce are realistic at
# the default kT below.

using SLCEMonteCarlo
using SLCEMonteCarlo: ChainState, SweepScratch, SpinConfig, site_coeffs!, delta_energy,
                     site_gradient, metropolis_sweep!, overrelaxation_sweep!
using SLCE
using Spglib: Spglib                    # activates SLCE's SpglibBackend extension
using LinearAlgebra: norm, normalize, I
using StaticArrays: SVector
using Statistics: median, mean
using Printf: @printf, @sprintf
using Random: MersenneTwister, Xoshiro
using BenchmarkTools: @belapsed, @benchmark

const MC = SLCEMonteCarlo

# ---------------------------------------------------------------------------
# Fixture models.
# ---------------------------------------------------------------------------

"""
    bcc_fe_model(; lmax = 1, cutoff = 2.6, seed = 11) -> SLCEModel

The light-kernel fixture: a 2-atom conventional bcc Fe training cell (a = 2.87 Å),
isotropic pair basis up to `lmax`, first shell by default (the 8 NN bonds — WS
boundary ties — enter as shift-carrying pair terms, so the tiling path is
exercised). Coefficients: seeded `0.02·randn` eV per SALC.
"""
function bcc_fe_model(; lmax::Integer = 1, cutoff::Real = 2.6, seed::Integer = 11)
    a = 2.87
    lat = Lattice([a 0 0; 0 a 0; 0 0 a])
    cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = cutoff, lmax = [lmax],
                               isotropy = true); backend = SpglibBackend())
    jphi = 0.02 .* randn(MersenneTwister(seed), n_salcs(b))
    return SLCEModel(b, 0.0, jphi)
end

"""
    nd2fe14b_model(; nbody = 2, cutoff = Inf, lmax_nd = 2, lmax_fe = 2,
                   isotropy = true, seed = 13) -> SLCEModel

The heavy-kernel fixture: the 68-atom Nd₂Fe₁₄B cell (`assets/nd2fe14b.toml`
structure), anisotropic multi-species basis (`lmax = [lmax_nd ×2, lmax_fe ×6, 0]` —
B non-magnetic), every resolvable pair by default. The `l ≤ 2` / all-pairs default
mirrors the real l02 production model. Coefficients: seeded `0.01·randn` eV.
Builds in ~1 s (basis) — reuse the returned model across a script.
"""
function nd2fe14b_model(; nbody::Integer = 2, cutoff::Real = Inf, lmax_nd::Integer = 2,
                        lmax_fe::Integer = 2, isotropy::Bool = true, seed::Integer = 13)
    inp = read_setup(joinpath(@__DIR__, "assets", "nd2fe14b.toml"))
    lmax = vcat(fill(Int(lmax_nd), 2), fill(Int(lmax_fe), 6), [0])
    spec = BasisSpec(; nbody = nbody, cutoff = cutoff, lmax = lmax,
                     isotropy = isotropy)
    b = SLCEBasis(inp.crystal, spec; backend = SpglibBackend(), tol = inp.tol)
    jphi = 0.01 .* randn(MersenneTwister(seed), n_salcs(b))
    return SLCEModel(b, 0.0, jphi)
end

"""
    nd2fe14b3_model(; cutoff = 3.5, seed = 13) -> SLCEModel

The TRIPLET-heavy fixture: the Nd₂Fe₁₄B cell with `nbody = 3` at a 3.5 Å cutoff
(pairs *and* triplet cliques — every Fe/Nd site active, ~17k terms, ~98 % of the
walked site-program entries are body-3). Mirrors the production l044/l064/l066
regime, where the 3-body contraction dominates `site_coeffs!`.
"""
nd2fe14b3_model(; cutoff::Real = 3.5, seed::Integer = 13) =
    nd2fe14b_model(; nbody = 3, cutoff = cutoff, seed = seed)

# kT [eV] used by the sweep/run benches (≈ 300 K — moderate acceptance for the
# coefficient scales above, so accepted-move bookkeeping is realistically mixed in).
const BENCH_KT = 0.025

"""
    rand_config(H; seed = 1) -> SpinConfig

Seeded uniform-random spin configuration on the sites of `H`.
"""
function rand_config(H::MC.TiledHamiltonian; seed::Integer = 1)
    rng = MersenneTwister(seed)
    unit() = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
    return SpinConfig([unit() for _ = 1:n_sites(H)])
end

"""
    chain_state(H; seed = 2, step = 0.6) -> (ChainState, SweepScratch)

A ready-to-sweep chain on a seeded random configuration (frozen step — the bench
measures a fixed kernel, not the adaptation).
"""
function chain_state(H::MC.TiledHamiltonian; seed::Integer = 2, step::Real = 0.6)
    st = ChainState(H, rand_config(H; seed = seed), Xoshiro(UInt64(seed)), step)
    st.frozen = true
    return st, SweepScratch(H)
end

# One-line structural summary of a tiled Hamiltonian (printed by every script).
function describe(H::MC.TiledHamiltonian)
    adj = length(H.site_inst) / H.n_active
    act = H.n_active < n_sites(H) ? " ($(H.n_active) active)" : ""
    return "sites=$(n_sites(H))$act  terms=$(length(H.terms))  " *
           "instances=$(length(H.inst_term))  nlm=$(H.nlm)  " *
           "mean active-site adjacency=$(@sprintf("%.1f", adj))"
end

# ---------------------------------------------------------------------------
# Timing harness (same shape as SLCE/bench).
# ---------------------------------------------------------------------------

"""
    bench_header(title)

Print a banner with the machine/Julia context for a benchmark script.
"""
function bench_header(title::AbstractString)
    println("=" ^ 72)
    println(title)
    println("julia $(VERSION)   threads=$(Threads.nthreads())")
    println("=" ^ 72)
end

"""
    bench_one(label, f; ntrials = 3) -> NamedTuple

Warm up `f()` once (compile), then run `ntrials` timed passes; report median/min
wall time plus the per-call allocation count and median bytes. For paths too heavy
for `@benchmark` sampling.
"""
function bench_one(label::AbstractString, f; ntrials::Integer = 3)
    f()                                              # warm-up / compile
    times = Float64[]
    bytes = Float64[]
    for _ = 1:ntrials
        s = @timed f()
        push!(times, s.time)
        push!(bytes, Float64(s.bytes))
    end
    allocs = @allocations f()
    @printf("%-34s  min=%9.3f ms  med=%9.3f ms  allocs=%-11d mem=%8.2f MiB\n",
            label, 1e3 * minimum(times), 1e3 * median(times), allocs,
            median(bytes) / 2^20)
    return (; label = String(label), min_s = minimum(times), med_s = median(times),
            allocs = allocs)
end

# Parse positional ARGS with defaults: argn(2, 30) → 2nd arg or 30 (Int);
# argf(3, 6.0) → 3rd arg or 6.0 (Float64, accepts `inf`).
argn(i::Integer, default::Integer) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default
argf(i::Integer, default::Real) =
    length(ARGS) >= i ? parse(Float64, ARGS[i]) : Float64(default)
