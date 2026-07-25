# The 4-function energy contract between `TiledHamiltonian` and everything else:
# `total_energy` / `site_coeffs!` / `delta_energy` / `site_gradient`. Updates and
# observables touch the energy only through these.
#
# The hot kernels walk the precompiled sparse contraction programs of
# `TiledHamiltonian.progs` (`_ContractionPrograms`, hamiltonian.jl): per instance
# they only index plain arrays — no dynamic dispatch on the rank-erased
# `ScaledTerm.folded` (2–3 heap allocations per instance in the rank-generic form —
# the pre-optimization sweep bottleneck), no zero-entry scanning, no `lm_index`
# recomputation; body-2/3 site programs additionally take fully precomputed
# pair/triplet fast paths (`site_col`(2)/`pent_row`(2)). The rank-generic
# *reference kernels* at the bottom of this file are the readable spec — the
# site-generalized siblings of SLCETools' `mc/metropolis.jl`
# (`_accumulate_site_term!` / `_term_energy`): the same `μ = idx − l − 1` index
# mapping, rank-specialized function barriers over `folded`, contracted against
# concrete tesseral rows — here columns of a dense `nlm × n_sites` matrix, with the
# member slot precomputed in the CSR adjacency instead of a `findfirst`. The program
# kernels must reproduce the reference kernels **bitwise** (same entry, factor, and
# accumulation order — gate: test_energy.jl "program kernels ≡ reference kernels").

# Tabulate the full tesseral row Z_lm(e), l = 0:lmax, into `z` (ordered by
# `Harmonics.lm_index`, which is sequential in this loop order). `e` must be unit.
# `plm` (length ≥ lmax + 1, contents irrelevant) is the associated-Legendre
# recursion workspace — threading it through makes the call allocation-free with
# bit-identical values (see `Zlm_unsafe`); hot callers hold one in their scratch.
function _zlm_row!(z::AbstractVector{Float64}, e::SVector{3,Float64}, lmax::Int,
                   plm::Vector{Float64})::AbstractVector{Float64}
    i = 0
    @inbounds for l = 0:lmax, m = -l:l
        i += 1
        z[i] = Harmonics.Zlm_unsafe(l, m, e, plm)
    end
    return z
end

# Convenience form for cold paths and tests: allocates the workspace per call.
_zlm_row!(z::AbstractVector{Float64}, e::SVector{3,Float64}, lmax::Int) =
    _zlm_row!(z, e, lmax, Vector{Float64}(undef, lmax + 1))

# Fresh `nlm × n_sites` tesseral-row matrix of a configuration (column s = Z_lm(e_s)).
function _zrows(H::TiledHamiltonian, config::SpinConfig)::Matrix{Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    zrows = Matrix{Float64}(undef, H.nlm, H.n_sites)
    plm = Vector{Float64}(undef, H.lmax + 1)
    for s = 1:H.n_sites
        _zlm_row!(view(zrows, :, s), config[s], H.lmax, plm)
    end
    return zrows
end

# Total energy from precomputed tesseral rows (the incremental-state entry point):
# per instance, its template's energy program — nonzero entries, member slot k as
# factor k — bitwise identical to `_total_energy_ref` (gate in test_energy.jl).
function _total_energy(H::TiledHamiltonian, zrows::Matrix{Float64})::Float64
    pr = H.progs
    E = 0.0
    @inbounds for i in eachindex(H.inst_term)
        k = Int(H.inst_term[i])
        off = Int(H.inst_ptr[i]) - 1
        Ei = 0.0
        for e = Int(pr.eprog_ptr[k]):(Int(pr.eprog_ptr[k + 1]) - 1)
            p = 1.0
            f0 = Int(pr.efac_ptr[e]) - 1
            # factor position m IS the member slot: energy factors are pushed for
            # k = 1:body in order, none skipped (`_push_term_programs!`)
            for m = 1:(Int(pr.efac_ptr[e + 1]) - 1 - f0)
                p *= zrows[pr.efac_row[f0 + m], H.inst_sites[off + m]]
            end
            Ei += pr.eent_w[e] * p
        end
        E += pr.term_coef[k] * Ei
    end
    return E
end

"""
    total_energy(H::TiledHamiltonian, config::SpinConfig) -> Float64

The SCE energy of `config` on the tiled supercell, in the model's energy units with
the intercept `j0` excluded: the sum of every cluster instance's contraction
`coef · Σ_μ folded[μ] ∏ᵢ Z_{lᵢμᵢ}(e_{siteᵢ})`. On the training cell
(`dims = (1,1,1)`) this equals `predict_energy(model, config) − intercept(model)`.
"""
total_energy(H::TiledHamiltonian, config::SpinConfig)::Float64 =
    _total_energy(H, _zrows(H, config))

"""
    site_coeffs!(c, H::TiledHamiltonian, s::Integer, zrows) -> c

Leave-one-out coefficient vector of site `s`: accumulate into `c` (length `H.nlm`,
**not** zeroed here) the coefficient of `Z_lm(e_s)` from every instance touching `s`,
contracting each template's `folded` against the concrete tesseral columns of the
*other* member sites. Because every cluster's sites are distinct (constructor
invariant), `c` is independent of `e_s`, so the site energy is exactly `c · Z(e_s)`
and a single-spin move has the exact energy change
[`delta_energy`](@ref)`(c, Z(e_s), Z(e_s′))` — any body order, no linearization.
β never enters; `c` is in the model's energy units.
"""
function site_coeffs!(c::Vector{Float64}, H::TiledHamiltonian, s::Integer,
                      zrows::Matrix{Float64})::Vector{Float64}
    pr = H.progs
    @inbounds for j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
        pid = Int(pr.site_prog[j])
        col = Int(pr.site_col[j])
        # pair/triplet fast paths (body-2/3 — the bulk of every adjacency): one/two
        # factors per entry, always the other member slots (ascending), so the
        # neighbor columns and the factor rows (`pent_row`/`pent_row2`) are
        # precomputed and the general path's per-entry sfac_ptr/sfac_slot/inst_sites
        # indirections vanish. The sign of `col` tags the path (pair +, triplet −,
        # general 0) so the pair path never loads `site_col2`. Bitwise identical:
        # `(1.0 · z₁) · z₂… ≡ z₁ · z₂…`, same skip, same accumulation order.
        if col > 0
            for e = Int(pr.sprog_ptr[pid]):(Int(pr.sprog_ptr[pid + 1]) - 1)
                z = zrows[pr.pent_row[e], col]
                z == 0.0 && continue
                c[pr.sent_tgt[e]] += pr.sent_w[e] * z
            end
        elseif col < 0
            col2 = Int(pr.site_col2[j])
            for e = Int(pr.sprog_ptr[pid]):(Int(pr.sprog_ptr[pid + 1]) - 1)
                p = zrows[pr.pent_row[e], -col] * zrows[pr.pent_row2[e], col2]
                p == 0.0 && continue
                c[pr.sent_tgt[e]] += pr.sent_w[e] * p
            end
        else
            off = Int(H.inst_ptr[H.site_inst[j]]) - 1
            for e = Int(pr.sprog_ptr[pid]):(Int(pr.sprog_ptr[pid + 1]) - 1)
                p = 1.0
                for f = Int(pr.sfac_ptr[e]):(Int(pr.sfac_ptr[e + 1]) - 1)
                    p *= zrows[pr.sfac_row[f], H.inst_sites[off + pr.sfac_slot[f]]]
                end
                p == 0.0 && continue
                c[pr.sent_tgt[e]] += pr.sent_w[e] * p
            end
        end
    end
    return c
end

"""
    delta_energy(c, zold, znew) -> Float64

Exact energy change of a single-spin move whose leave-one-out coefficients are `c`
(from [`site_coeffs!`](@ref)) and whose old/new tesseral rows are `zold`/`znew`:
`Σ_k c_k (znew_k − zold_k)`. β-free — the caller applies the Boltzmann factor.
"""
function delta_energy(c::Vector{Float64}, zold::AbstractVector{Float64},
                      znew::AbstractVector{Float64})::Float64
    ΔE = 0.0
    @inbounds for k in eachindex(c)
        ck = c[k]
        ck == 0.0 && continue
        ΔE += ck * (znew[k] - zold[k])
    end
    return ΔE
end

"""
    site_gradient(H::TiledHamiltonian, s::Integer, config::SpinConfig)
        -> SVector{3,Float64}

On-sphere (tangent-projected) gradient of the total energy with respect to the spin
direction of site `s`: `∇E = Σ_k c_k ∇Z_k(e_s)` with the leave-one-out coefficients
of [`site_coeffs!`](@ref) and `SLCE.Harmonics.grad_Zlm_unsafe` (so
`e_s · ∇E = 0`). Diagnostics/tests — not on the sweep hot path.
"""
function site_gradient(H::TiledHamiltonian, s::Integer,
                       config::SpinConfig)::SVector{3,Float64}
    zrows = _zrows(H, config)
    c = site_coeffs!(zeros(H.nlm), H, s, zrows)
    e = config[s]
    g = zero(SVector{3,Float64})
    i = 0
    for l = 0:H.lmax, m = -l:l
        i += 1
        ck = c[i]
        ck == 0.0 && continue
        g += ck * Harmonics.grad_Zlm_unsafe(l, m, e)
    end
    return g
end

# Single-site gradient from precomputed rows and caller-owned scratch: the ONE
# per-site kernel behind `energy_gradient!` and `minimize.jl`'s `_gradient!`, and
# arithmetically identical to the public `site_gradient` (same `(l, m)` loop over
# `lm_index` order, same `ck == 0` skip — the bitwise `==` gates in
# test_gradient.jl / test_minimize.jl pin all three together). `c` is scratch
# (zeroed here); `plm` is the associated-Legendre recursion workspace.
function _site_grad(H::TiledHamiltonian, s::Int, e::SVector{3,Float64},
                    zrows::Matrix{Float64}, c::Vector{Float64},
                    plm::Vector{Float64})::SVector{3,Float64}
    fill!(c, 0.0)
    site_coeffs!(c, H, s, zrows)
    g = zero(SVector{3,Float64})
    i = 0
    for l = 0:H.lmax, m = -l:l
        i += 1
        ck = c[i]
        ck == 0.0 && continue
        # cache-threaded variant — bit-identical to the plain call (the == gate)
        g += ck * Harmonics.grad_Zlm_unsafe(l, m, e, plm)
    end
    return g
end

"""
    energy_gradient!(G, H::TiledHamiltonian, config::SpinConfig; ntasks = 1) -> G
    energy_gradient(H::TiledHamiltonian, config::SpinConfig; ntasks = 1)
        -> Vector{SVector{3,Float64}}

All-site, tangent-projected gradient of the total SCE energy: `G[s] = ∂E/∂e_s`
with `e_s · G[s] == 0` exactly (the radial part is removed by
`SLCE.Harmonics.grad_Zlm_unsafe`), exact at any body order — the
leave-one-out coefficients of [`site_coeffs!`](@ref) are independent of `e_s`, so
no linearization is involved. Inactive sites receive exactly zero. Units: model
energy per unit direction. The physical (Landau–Lifshitz) torque is
`τ_s = G[s] × e_s` — matching `SLCE.predict_torque` on the training cell —
and the effective field is `B_s = −G[s]/(magmom_s·μ_B)`; moment magnitudes are
the caller's (this package holds unit directions only).

One call costs about one Metropolis sweep. `ntasks > 1` splits the site loop
across that many tasks: the pass is read-only (no coloring needed, unlike
sweeps), each site writes only its own entry, and every task owns its scratch,
so the result is **bit-identical for any task count**.

Note: the overrelaxation axis of `updates.jl` is *not* this gradient — it is the
`l = 1` channel of `c` only. Use this function wherever the full field is meant
(e.g. spin dynamics).
"""
function energy_gradient!(G::Vector{SVector{3,Float64}}, H::TiledHamiltonian,
                          config::SpinConfig; ntasks::Integer = 1)
    length(G) == H.n_sites || throw(DimensionMismatch(
        "G has $(length(G)) entries but the Hamiltonian has $(H.n_sites) sites"))
    ntasks >= 1 || throw(ArgumentError("ntasks must be ≥ 1; got $ntasks"))
    zrows = _zrows(H, config)                       # also validates config length
    nt = min(Int(ntasks), H.n_sites)
    if nt <= 1
        _gradient_chunk!(G, H, config, zrows, 1, H.n_sites)
    else
        chunk = cld(H.n_sites, nt)
        @sync for lo = 1:chunk:H.n_sites
            hi = min(lo + chunk - 1, H.n_sites)
            Threads.@spawn _gradient_chunk!(G, H, config, zrows, lo, hi)
        end
    end
    return G
end

@doc (@doc energy_gradient!)
energy_gradient(H::TiledHamiltonian, config::SpinConfig; ntasks::Integer = 1) =
    energy_gradient!(Vector{SVector{3,Float64}}(undef, H.n_sites), H, config;
                     ntasks = ntasks)

# One task's block of the site loop. Scratch is task-local by construction — the
# bit-identity of `energy_gradient!` for any `ntasks` rests on it (`G` writes are
# per-site disjoint, `zrows`/`config` are read-only).
function _gradient_chunk!(G::Vector{SVector{3,Float64}}, H::TiledHamiltonian,
                          config::SpinConfig, zrows::Matrix{Float64},
                          lo::Int, hi::Int)::Nothing
    c = zeros(H.nlm)
    plm = Vector{Float64}(undef, H.lmax + 1)
    for s = lo:hi
        G[s] = H.site_active[s] ? _site_grad(H, s, config[s], zrows, c, plm) :
               zero(SVector{3,Float64})     # spin-independent site: exactly zero
    end
    return nothing
end

# --- reference kernels ----------------------------------------------------------
#
# The rank-generic contraction the programs were flattened from — the readable spec
# of the μ-mapping and of the entry/factor/accumulation order the program kernels
# must reproduce bitwise. Tests only (per-instance dynamic dispatch on the
# rank-erased `folded` keeps these off every hot path).

# One instance's full contraction against the concrete site columns of `zrows`
# (rank-specialized barrier; `sites` is that instance's slice of `inst_sites`).
@inline function _instance_energy(coef::Float64, sites::AbstractVector{Int32},
                                  ls::Vector{Int}, folded::Array{Float64,D},
                                  zrows::Matrix{Float64})::Float64 where {D}
    E = 0.0
    @inbounds for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            μk = idx[k] - ls[k] - 1
            p *= zrows[Harmonics.lm_index(ls[k], μk), sites[k]]
        end
        E += w * p
    end
    return coef * E
end

# Reference form of `_total_energy`.
function _total_energy_ref(H::TiledHamiltonian, zrows::Matrix{Float64})::Float64
    E = 0.0
    @inbounds for i in eachindex(H.inst_term)
        term = H.terms[H.inst_term[i]]
        sites = view(H.inst_sites, H.inst_ptr[i]:(H.inst_ptr[i + 1] - 1))
        E += _instance_energy(term.coef, sites, term.ls, term.folded, zrows)
    end
    return E
end

# One instance's contribution to the coefficient-of-Z_lm at member position `slot`
# (rank-specialized barrier; `off` is the instance's offset into `inst_sites`).
@inline function _accumulate_instance!(c::Vector{Float64}, slot::Int, coef::Float64,
                                       off::Int, inst_sites::Vector{Int32},
                                       ls::Vector{Int}, folded::Array{Float64,D},
                                       zrows::Matrix{Float64}) where {D}
    @inbounds for idx in CartesianIndices(folded)
        w = coef * folded[idx]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            k == slot && continue
            μk = idx[k] - ls[k] - 1
            p *= zrows[Harmonics.lm_index(ls[k], μk), inst_sites[off + k]]
        end
        p == 0.0 && continue
        μi = idx[slot] - ls[slot] - 1
        c[Harmonics.lm_index(ls[slot], μi)] += w * p
    end
    return c
end

# Reference form of `site_coeffs!` (same contract: accumulates into `c`, not zeroed).
function _site_coeffs_ref!(c::Vector{Float64}, H::TiledHamiltonian, s::Integer,
                           zrows::Matrix{Float64})::Vector{Float64}
    @inbounds for j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
        i = H.site_inst[j]
        term = H.terms[H.inst_term[i]]
        _accumulate_instance!(c, Int(H.site_slot[j]), term.coef,
                              Int(H.inst_ptr[i]) - 1, H.inst_sites, term.ls,
                              term.folded, zrows)
    end
    return c
end
