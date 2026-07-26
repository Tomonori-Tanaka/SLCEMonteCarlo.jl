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

# Tabulate one site's DISPLACEMENT rows into `rows` (a full `H.nrows` column):
# `|u|^{2k} R_lm(u)` for every `(k, l)` block of the layout. `rbuf` (length ≥
# `(disp_lmax + 1)²`) is the solid-harmonic batch workspace — one batch per site serves
# every block, whereas SLCE's reference filler calls the convenience `Rlm` accessor per
# `(k, l, m)` and re-batches each time. The values must be identical either way (the
# recurrences fill bottom-up, so a higher `lmax` cannot change a lower entry); the
# bitwise gate against `SLCE.site_rows!` in test_joint.jl is what holds that.
function _disp_rows!(rows::AbstractVector{Float64}, H::TiledHamiltonian,
                     u::SVector{3,Float64}, rbuf::Vector{Float64})
    SolidHarmonics.solid_harmonics!(rbuf, H.disp_lmax, u)
    r2 = dot(u, u)
    L = H.layout
    @inbounds for (i, (k, l)) in pairs(L.disp_factors)
        r2k = r2^k
        base = L.disp_starts[i]
        for m = -l:l
            rows[base + m + l + 1] =
                r2k * rbuf[SolidHarmonics.solid_harmonic_index(l, m)]
        end
    end
    return rows
end

# Fresh `nrows × n_sites` basis-row matrix of a state: the SPIN block (rows
# `1:nlm`, column s = Z_lm(e_s)) and, on a joint Hamiltonian, the displacement blocks.
# `disps === nothing` is legal only when `H` has no displacement rows — sampling a
# joint model at an unstated `u` would silently mean `u = 0`.
function _zrows(H::TiledHamiltonian, config::SpinConfig,
                disps::Union{Nothing,Vector{SVector{3,Float64}}} = nothing)::Matrix{Float64}
    length(config) == H.n_sites || throw(DimensionMismatch(
        "config has $(length(config)) sites but the Hamiltonian has $(H.n_sites)"))
    if disps === nothing
        has_disp(H) && throw(ArgumentError(
            "this Hamiltonian carries displacement rows $(H.layout.disp_factors); " *
            "pass the displacements explicitly (a missing `disps` would silently " *
            "mean u = 0)"))
    else
        length(disps) == H.n_sites || throw(DimensionMismatch(
            "disps has $(length(disps)) sites but the Hamiltonian has $(H.n_sites)"))
        # The same footgun from the other side: a pure-spin Hamiltonian has nowhere to
        # put displacements, so accepting nonzero ones would silently evaluate at u = 0.
        # (All-zero is fine — that IS the state such a model describes, and it lets a
        # driver pass `disps` uniformly across a joint model and its `restrict`ion.)
        has_disp(H) || all(iszero, disps) || throw(ArgumentError(
            "this Hamiltonian has no displacement rows, but `disps` contains nonzero " *
            "displacements; it describes the clamped-ion (u = 0) energy only, so " *
            "evaluating it here would silently ignore them"))
    end
    zrows = Matrix{Float64}(undef, H.nrows, H.n_sites)
    plm = Vector{Float64}(undef, H.lmax + 1)
    # Two loops rather than one with a per-site branch: `disps === nothing` narrows the
    # union away (JET has no other way to know `disps[s]` is reachable only when the
    # displacements are there), and the pure-spin branch stays the pre-M4 whole-column
    # fill with no nested view.
    if disps === nothing
        for s = 1:H.n_sites
            _zlm_row!(view(zrows, :, s), config[s], H.lmax, plm)
        end
        return zrows
    end
    rbuf = Vector{Float64}(undef, max(0, (H.disp_lmax + 1)^2))
    joint = has_disp(H)
    for s = 1:H.n_sites
        col = view(zrows, :, s)
        _zlm_row!(view(col, 1:H.nlm), config[s], H.lmax, plm)
        joint && _disp_rows!(col, H, disps[s], rbuf)
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
            # one factor per tensor axis, pushed in axis order with none skipped
            # (`_push_term_programs!`); `efac_site` maps the axis to its member site
            # position, which for a pure-spin term is the axis index itself
            for m = 1:(Int(pr.efac_ptr[e + 1]) - 1 - f0)
                f = f0 + m
                p *= zrows[pr.efac_row[f], H.inst_sites[off + pr.efac_site[f]]]
            end
            Ei += pr.eent_w[e] * p
        end
        E += pr.term_coef[k] * Ei
    end
    return E
end

"""
    total_energy(H::TiledHamiltonian, config::SpinConfig) -> Float64
    total_energy(H::TiledHamiltonian, config::SpinConfig, disps) -> Float64

The SCE energy of a state on the tiled supercell, in the model's energy units with
the intercept `j0` excluded: the sum of every cluster instance's contraction
`coef · Σ_μ folded[μ] ∏ᵢ fᵢ(μᵢ)` over its tensor axes, with `fᵢ` the axis's own site
factor — `Z_{l,m}(ê_a)` on a spin axis, `|u_a|^{2k} R_{l,m}(u_a)` on a displacement
one. On the training cell (`dims = (1,1,1)`) this equals
`predict_energy(model, config[, disps]) − intercept(model)`.

`disps` is a `Vector{SVector{3,Float64}}` of Cartesian displacements, one per site, in
the model's length units. It is **required** on a Hamiltonian with displacement rows
([`has_disp`](@ref)) — omitting it there throws rather than silently meaning `u = 0`.
On a Hamiltonian *without* displacement rows only an all-zero `disps` is accepted (that
is the clamped-ion state such a model describes); nonzero displacements throw rather
than being ignored.
"""
total_energy(H::TiledHamiltonian, config::SpinConfig)::Float64 =
    _total_energy(H, _zrows(H, config))

total_energy(H::TiledHamiltonian, config::SpinConfig,
            disps::Vector{SVector{3,Float64}})::Float64 =
    _total_energy(H, _zrows(H, config, disps))

"""
    site_coeffs!(c, H::TiledHamiltonian, s::Integer, zrows) -> c

Leave-one-out coefficient vector of site `s`: accumulate into `c` (length `H.nrows`,
**not** zeroed here) the coefficient of each of site `s`'s basis rows, from every
instance touching `s`, contracting each template's `folded` against the concrete
columns of the *other* axes. Because every cluster's member sites are distinct
(constructor invariant), a move at `s` has the exact energy change

    delta_energy(c, rows_old, rows_new)

for any body order, with no linearization. β never enters; `c` is in the model's energy
units.

**On a joint Hamiltonian that ΔE identity is the whole contract** — in particular
`c · rows(s)` is *not* the site energy there. A site's spin and displacement axes
multiply each other, so each two-axis instance contributes to the coefficient of *both*
of the site's rows and `dot(c, rows(s))` counts it twice (measured: ~0.5–0.75 of the
true per-site sum on a mixed fixture). The `c · rows(s)` reading is valid only when
every instance puts exactly one axis on `s`, i.e. on a pure-spin model.

The ΔE identity holds exactly for a move that changes **one channel** of the site: the
other channel's row is then a constant factor already folded into `c`, and the rows that
did not move contribute `c_k · 0`. A *simultaneous* spin+displacement move on one site
misses exactly the cross term `Δzᵀ M Δr` (there is no `Δz·Δz` or `Δr·Δr` remainder — the
constructor enforces at most one axis per `(site, channel)`). Every update this package
proposes is single-channel.
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
    delta_energy(c, zold, znew, lo, hi) -> Float64

Exact energy change of a single-site move whose leave-one-out coefficients are `c`
(from [`site_coeffs!`](@ref)) and whose old/new basis rows are `zold`/`znew`:
`Σ_k c_k (znew_k − zold_k)`. β-free — the caller applies the Boltzmann factor.

`lo`/`hi` restrict the sum to one row range, which is what makes a **single-channel**
move exact from a partially-written `znew`: a spin move rewrites only the SPIN block
(`1:nlm`) and a displacement move only the displacement blocks (`nlm+1:nrows`), and
the rows the move did not touch contribute `c_k · 0` — so summing over the moved
block alone is not an approximation, it is the same number without the cancelling
terms (and without reading the stale half of the buffer). The three-argument form is
the whole table, `1:length(c)`.

Keep the **row-difference** form `Σ c_k (znew_k − zold_k)`: the algebraically equal
`c·znew − c·zold` cancels two large sums against each other and loses two to three
orders of magnitude of accuracy in the ΔE that drives the accept step.
"""
delta_energy(c::Vector{Float64}, zold::AbstractVector{Float64},
             znew::AbstractVector{Float64})::Float64 =
    delta_energy(c, zold, znew, 1, length(c))

function delta_energy(c::Vector{Float64}, zold::AbstractVector{Float64},
                      znew::AbstractVector{Float64}, lo::Integer,
                      hi::Integer)::Float64
    ΔE = 0.0
    @inbounds for k = Int(lo):Int(hi)
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
    _require_spin_only(H, "site_gradient")
    zrows = _zrows(H, config)
    c = site_coeffs!(zeros(H.nrows), H, s, zrows)
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
    _require_spin_only(H, "energy_gradient!")
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
        G[s] = H.site_has_spin[s] ? _site_grad(H, s, config[s], zrows, c, plm) :
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
                                  slots::Vector{TermSlot}, folded::Array{Float64,D},
                                  zrows::Matrix{Float64})::Float64 where {D}
    E = 0.0
    @inbounds for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        p = 1.0
        for k = 1:D
            # row of component μ = idx − l − 1 is row0 + μ + l + 1 = row0 + idx
            p *= zrows[slots[k].row0 + idx[k], sites[slots[k].site]]
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
        E += _instance_energy(term.coef, sites, term.slots, term.folded, zrows)
    end
    return E
end

# One instance's contribution to the row coefficients of member site position `q` —
# summed over EVERY tensor axis sitting on that site, in ascending axis order (the
# concatenation order of the merged site program). A pure-spin term has exactly one
# axis per site, so the `v` loop runs once and this is the pre-M4 kernel verbatim.
# (Rank-specialized barrier; `off` is the instance's offset into `inst_sites`.)
@inline function _accumulate_instance!(c::Vector{Float64}, q::Int, coef::Float64,
                                       off::Int, inst_sites::Vector{Int32},
                                       slots::Vector{TermSlot}, folded::Array{Float64,D},
                                       zrows::Matrix{Float64}) where {D}
    @inbounds for v = 1:D
        slots[v].site == q || continue
        for idx in CartesianIndices(folded)
            w = coef * folded[idx]
            w == 0.0 && continue
            p = 1.0
            for k = 1:D
                k == v && continue
                p *= zrows[slots[k].row0 + idx[k], inst_sites[off + slots[k].site]]
            end
            p == 0.0 && continue
            c[slots[v].row0 + idx[v]] += w * p
        end
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
                              Int(H.inst_ptr[i]) - 1, H.inst_sites, term.slots,
                              term.folded, zrows)
    end
    return c
end
