# Cell reduction: a model fitted on a supercell, re-expressed in a user-chosen
# smaller cell so MC supercells are built from *that* cell's integer multiples instead
# of the (possibly large) training cell's — decoupling the finite-size-scaling grid
# from the fitting cell. Validity is **verified**, never assumed: the structure and
# every fitted term must actually respect the smaller cell's translations, and any
# violation is a hard error (see `docs/specs/cell-reduction.md`).
#
# The whole reduction is exact integer bookkeeping. With `A_train = A_sub · M`
# (`M` integer), training atom `a` decomposes as (sub-cell atom `b_a`, integer
# sub-lattice offset `o_a`), and member `i` of a training term carries the sub-lattice
# shift `σᵢ = o_{aᵢ} + M·sᵢ`. The anchored form `σᵢ − σ₁` is invariant under the
# |det M| coset translations, so grouping terms by it and tallying the groups IS the
# Hamiltonian-periodicity verification: each orbit must contribute the same number of
# training terms, with equal `coef`/`scale`/`folded`, to EVERY ONE of the |det M|
# cosets — the coset being read off the absolute anchor σ₁ with the integer adjugate.
# Pure translations rotate no spin and move no displacement vector, which is what makes
# the tensors comparable across copies in both channels.

"""
    ReducedCell{T}

A fitted training-cell Hamiltonian re-expressed in a smaller (or re-based) unit cell
by [`reduce_cell`](@ref). Feed it to [`TiledHamiltonian`](@ref) to tile MC supercells
in multiples of the *reduced* cell.

`T` is the term type the reduction was fed: `SLCE.SpinMultipoleTerm` for a pure-spin
model, `SLCE.DecoratedTerm` for a general (joint spin–lattice) one. The two carry
different amounts of information — a decorated term labels every tensor axis with its
own `(channel, k, l)` and its own site — so the reduced list is emitted in whichever
form it came in, never converted.

Fields: `n_atoms` (atoms of the reduced cell), `terms` (one representative term per
translation orbit, **raw** coefficients, sub-cell atom indices and sub-lattice integer
shifts), `crystal` (the reduced cell, for geometry I/O such as
[`supercell_crystal`](@ref)), `M` (the integer matrix with `A_train = A_sub · M`),
`parent_atoms` (reduced atom `b` → its representative training atom), `atom_map`
(training atom `a` → `(b, offset)` with `offset ∈ ℤ³` in reduced-lattice units), and
`layout` (the model's `SLCE.RowLayout` for a decorated reduction, `nothing` for a
pure-spin one, whose row numbering the `SpinMultipoleTerm` constructor derives).
"""
struct ReducedCell{T}
    n_atoms::Int
    terms::Vector{T}
    crystal::Crystal
    M::SMatrix{3,3,Int,9}
    parent_atoms::Vector{Int}
    atom_map::Vector{Tuple{Int,SVector{3,Int}}}
    layout::Union{Nothing,RowLayout}

    function ReducedCell(n_atoms::Int, terms::Vector{T}, crystal::Crystal,
                         M::SMatrix{3,3,Int,9}, parent_atoms::Vector{Int},
                         atom_map::Vector{Tuple{Int,SVector{3,Int}}},
                         layout::Union{Nothing,RowLayout}) where {T}
        # `ReducedCell` is exported, so a hand-built one must fail here rather than at
        # the `TiledHamiltonian` method table with a bare MethodError.
        (T === SpinMultipoleTerm || T === DecoratedTerm) || throw(ArgumentError(
            "a ReducedCell holds SLCE.SpinMultipoleTerm or SLCE.DecoratedTerm, not $T"))
        # The invariant every consumer leans on: a decorated reduction carries the
        # model's own row numbering (its slots name `(channel, k, l)` factors that only
        # a layout can place), a pure-spin one does not (the tesseral block is derived
        # from the terms' `ls`, exactly as before M4).
        (T === DecoratedTerm) == (layout !== nothing) || throw(ArgumentError(
            "a $T reduction must carry " *
            (T === DecoratedTerm ? "the model's RowLayout" : "no RowLayout")))
        return new{T}(n_atoms, terms, crystal, M, parent_atoms, atom_map, layout)
    end
end

Base.show(io::IO, red::ReducedCell) =
    print(io, "ReducedCell(", red.n_atoms, " atoms, ", length(red.terms),
          " terms, |det M| = ", abs(_det3(red.M)), ")")

"""
    n_atoms(red::ReducedCell) -> Int

Number of atoms in the reduced cell (`n_atoms(training crystal) / |det M|`).
"""
n_atoms(red::ReducedCell)::Int = red.n_atoms

_det3(m::SMatrix{3,3,Int})::Int =
    m[1, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) -
    m[1, 2] * (m[2, 1] * m[3, 3] - m[2, 3] * m[3, 1]) +
    m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])

# The integer adjugate, `adj(M) · M = det(M) · I` — the exact-arithmetic stand-in for
# `M⁻¹` used to decide coset membership without ever leaving ℤ.
_adj3(m::SMatrix{3,3,Int})::SMatrix{3,3,Int,9} =
    SMatrix{3,3,Int}(m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2],
                     m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3],
                     m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1],
                     m[1, 3] * m[3, 2] - m[1, 2] * m[3, 3],
                     m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1],
                     m[1, 2] * m[3, 1] - m[1, 1] * m[3, 2],
                     m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2],
                     m[1, 3] * m[2, 1] - m[1, 1] * m[2, 3],
                     m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])

# Which of the `nc = |det M|` cosets of `M ℤ³` in `ℤ³` the integer vector `σ` lies in,
# as a canonical label. `σ₁ ≡ σ₂ (mod M ℤ³) ⟺ adj·(σ₁ − σ₂) ≡ 0 (mod nc)`: forward
# because `adj·M = det·I`, backwards because `adj·v = nc·w ⟹ det·v = ±det·M·w`. So the
# residue is a COMPLETE invariant, not merely a necessary condition.
_coset_label(adj::SMatrix{3,3,Int,9}, nc::Int, σ::SVector{3,Int})::SVector{3,Int} =
    mod.(adj * σ, nc)

# Fractional residuals equal modulo the reduced lattice, within `tol` per component.
_same_frac(r1::SVector{3,Float64}, r2::SVector{3,Float64}, tol::Float64)::Bool =
    all(abs(x - round(x)) <= tol for x in r1 - r2)

"""
    reduce_cell(model::SLCEModel, crystal::Crystal, sub_lattice;
                pos_tol = 1e-6, coef_rtol = 1e-10) -> ReducedCell
    reduce_cell(crystal::Crystal, terms::Vector{SpinMultipoleTerm}, sub_lattice;
                pos_tol = 1e-6, coef_rtol = 1e-10) -> ReducedCell
    reduce_cell(crystal::Crystal, terms::Vector{DecoratedTerm}, layout::RowLayout,
                sub_lattice; pos_tol = 1e-6, coef_rtol = 1e-10) -> ReducedCell

Re-express a Hamiltonian fitted on `crystal` (the training cell — passed explicitly,
this package never reads geometry off the model) in the smaller unit cell whose
lattice vectors are the **columns** of the 3 × 3 matrix `sub_lattice`, after
verifying that the choice is legitimate:

1. `A_train = sub_lattice · M` for an integer matrix `M` (any integer `M`, not just
   diagonal — a bcc conventional cell under a primitive-fitted model, or a mere
   re-basing with `|det M| = 1`, both work);
2. the atomic basis maps onto itself under all `|det M|` coset translations
   (positions within `pos_tol`, in fractional units, and matching species);
3. every fitted term has, **in each of the `|det M|` cosets**, the same number of
   translation copies with equal coefficient and coupling tensor (relative tolerance
   `coef_rtol`), compared in the canonical site order (sorted `(reduced atom, shift)`,
   tensor axes aligned) — so copies anchored at different member sites match. That
   common count `q` is the number of representatives emitted: a term list may legally
   carry `q` identical summands per instance (e.g. hand-built directed pairs), and
   canonical model terms always have `q = 1`. The check is per coset rather than on the
   total, because a total that is merely divisible by `|det M|` is satisfied by a term
   that lives in one coset only — a Hamiltonian without the requested periodicity. The
   price of accepting `q > 1` is that an *accidental* exact duplication of a model term
   in every coset would pass this census too.

Any violation throws an `ArgumentError` — a fit that does not actually have the
requested periodicity (e.g. a distorted structure, or couplings that break it) is
never silently symmetrized.

**Channels.** A joint spin–lattice model reduces through the general
`SLCE.DecoratedTerm` surface, which needs the model's own row numbering — the
one-argument-per-model form reads `SLCE.row_layout(model)` for you, hand-built lists
pass it explicitly. There the site sort also relabels each slot's site and re-sorts the
slots into canonical `(channel, site, k, l)` order, carrying `folded`'s axes with them.
Lattice translations move no displacement vector and rotate no spin, so orbit members
share `coef`, `scale` and the aligned `folded` exactly — the same argument that makes
the pure-spin census exact.

`scale` is taken from the representative **verbatim** and never re-derived from the
cluster shape (`(4π)^(body/2)` is the general `(4π)^(n_spin_slots/2)` rule only when
every site carries exactly one spin factor). All copies on one key must declare the
identical value, which is checked and reported as its own error. What is *not* checked
is whether that value follows the rule at all: a hand-built list is free to use `scale`
as a plain multiplier as long as it does so consistently, exactly as
`TiledHamiltonian` — the surface that applies it — does.

The returned [`ReducedCell`](@ref) plugs into `TiledHamiltonian(red; dims)` with
`dims` now counted in **reduced-cell** units, so finite-size checks are no longer
restricted to integer multiples of the training cell. Example — model fitted on a
4×4×4 bcc *conventional* supercell (128 atoms), reduced to the 2-atom conventional
cube:

```julia
red = reduce_cell(model, crystal_train, Matrix(crystal_train.lattice.vectors) / 4)
H   = TiledHamiltonian(red; dims = (6, 6, 6))          # 432 sites — not a ×4 multiple
out = supercell_crystal(red.crystal, (6, 6, 6))        # matching geometry for I/O
```

Sublattice observables (`:sublattice_m`) of the reduced Hamiltonian index the
*reduced*-cell atoms; `red.parent_atoms` / `red.atom_map` translate back to
training-cell atom indices.
"""
function reduce_cell(model::SLCEModel, crystal::Crystal,
                     sub_lattice::AbstractMatrix{<:Real};
                     pos_tol::Real = 1e-6, coef_rtol::Real = 1e-10)
    n_atoms(model) == n_atoms(crystal) || throw(ArgumentError(
        "crystal has $(n_atoms(crystal)) atoms but the model was fitted on " *
        "$(n_atoms(model)) — pass the training-cell Crystal"))
    layout = row_layout(model)
    # The same branch `TiledHamiltonian(model)` takes, for the same reason: a model
    # with no displacement rows goes down the frozen pure-spin path byte for byte,
    # and `restrict` is what makes the pathological "declared a displacement sector,
    # built no displacement SALC" model reduce instead of being refused.
    isempty(layout.disp_factors) &&
        return reduce_cell(crystal, spin_multipole_terms(restrict(model, :spin)),
                           sub_lattice; pos_tol = pos_tol, coef_rtol = coef_rtol)
    return reduce_cell(crystal, decorated_terms(model), layout, sub_lattice;
                       pos_tol = pos_tol, coef_rtol = coef_rtol)
end

reduce_cell(crystal::Crystal, mterms::Vector{SpinMultipoleTerm},
            sub_lattice::AbstractMatrix{<:Real};
            pos_tol::Real = 1e-6, coef_rtol::Real = 1e-10) =
    _reduce_cell(crystal, mterms, nothing, sub_lattice, pos_tol, coef_rtol)

reduce_cell(crystal::Crystal, dterms::Vector{DecoratedTerm}, layout::RowLayout,
            sub_lattice::AbstractMatrix{<:Real};
            pos_tol::Real = 1e-6, coef_rtol::Real = 1e-10) =
    _reduce_cell(crystal, dterms, layout, sub_lattice, pos_tol, coef_rtol)

function _reduce_cell(crystal::Crystal, mterms::Vector{T},
                      layout::Union{Nothing,RowLayout},
                      sub_lattice::AbstractMatrix{<:Real},
                      pos_tol::Real, coef_rtol::Real)::ReducedCell{T} where {T}
    size(sub_lattice) == (3, 3) || throw(ArgumentError(
        "sub_lattice must be a 3 × 3 matrix (columns = lattice vectors); " *
        "got size $(size(sub_lattice))"))
    isempty(mterms) && throw(ArgumentError("the term list is empty"))

    a_train = Matrix(crystal.lattice.vectors)
    a_sub = Matrix{Float64}(sub_lattice)
    mf = a_sub \ a_train
    mi = round.(Int, mf)
    maximum(abs, a_sub * mi - a_train) <= pos_tol * maximum(abs, a_train) ||
        throw(ArgumentError(
            "the training lattice is not an integer combination of the given " *
            "cell's vectors: A_sub \\ A_train = $mf"))
    m = SMatrix{3,3,Int}(mi)
    nc = abs(_det3(m))
    # defensive — a singular integer mi cannot pass the full-rank residual check above
    nc >= 1 || throw(ArgumentError(
        "the given cell is singular relative to the training cell (det M = 0)"))
    nat = n_atoms(crystal)
    nat % nc == 0 || throw(ArgumentError(
        "n_atoms = $nat is not divisible by the cell ratio |det M| = $nc — the " *
        "structure cannot have the periodicity of the given cell"))

    # --- atoms: decompose each as (reduced atom, integer sub-lattice offset) -------
    # f_sub = M f_train; snap the floor by pos_tol so residuals near 1 wrap to ~0.
    offs = Vector{SVector{3,Int}}(undef, nat)
    resid = Vector{SVector{3,Float64}}(undef, nat)
    for a = 1:nat
        f = m * SVector{3,Float64}(view(crystal.frac_positions, :, a))
        o = floor.(Int, f .+ pos_tol)
        offs[a] = o
        resid[a] = f - o
    end
    groups = Vector{Vector{Int}}()          # ordered by first occurrence
    for a = 1:nat
        g = findfirst(grp -> crystal.species[grp[1]] == crystal.species[a] &&
                             _same_frac(resid[grp[1]], resid[a], 2 * Float64(pos_tol)),
                      groups)
        g === nothing ? push!(groups, [a]) : push!(groups[g], a)
    end
    for grp in groups
        length(grp) == nc || throw(ArgumentError(
            "training atom $(grp[1]) has $(length(grp)) translation images under " *
            "the given cell, expected $nc: the structure does not have that " *
            "periodicity (or loosen pos_tol)"))
        allunique(offs[a] for a in grp) || throw(ArgumentError(
            "training atoms $grp fold onto one reduced-cell site — coincident " *
            "positions?"))
    end
    parent_atoms = [grp[1] for grp in groups]
    atom_map = Vector{Tuple{Int,SVector{3,Int}}}(undef, nat)
    for (g, grp) in enumerate(groups), a in grp
        atom_map[a] = (g, offs[a])
    end

    red_terms = _reduce_terms(mterms, atom_map, m, nc, nat, Float64(coef_rtol))

    # --- the reduced crystal (geometry I/O; ordering = reduced atom index) ---------
    nsub = length(groups)
    frac = Matrix{Float64}(undef, 3, nsub)
    species = Vector{Int}(undef, nsub)
    for (g, grp) in enumerate(groups)
        r = resid[grp[1]]
        for row = 1:3
            frac[row, g] = abs(r[row]) <= pos_tol ? 0.0 : r[row]
        end
        species[g] = crystal.species[grp[1]]
    end
    lat = Lattice(a_sub; pbc = Tuple(crystal.lattice.pbc))
    sub_crystal = Crystal(lat, frac, species, copy(crystal.species_labels))

    return ReducedCell(nsub, red_terms, sub_crystal, m, parent_atoms, atom_map, layout)
end

# --- terms: canonical anchored reduced form; count each translation orbit -----------
# Canonical members arrive one per physical instance (SLCE's `_canonicalize_members`),
# so two translation copies of the same instance are generally anchored at different
# member sites — in reduced coordinates they differ by a joint site permutation (with
# the `folded` axes permuted the same way) on top of the coset translation. Align each
# term to the sorted `(reduced atom, shift)` order, re-anchor, and carry the axis labels
# and `folded` through the permutation before grouping; the aligned copies then match
# exactly. The two term surfaces differ only in what an axis is (`_align_reduced`).

_reduced_key_type(::Type{SpinMultipoleTerm}) =
    Tuple{Vector{Int},Vector{SVector{3,Int}},Vector{Int}}
_reduced_key_type(::Type{DecoratedTerm}) =
    Tuple{Vector{Int},Vector{SVector{3,Int}},Vector{SLCE.Slot}}

function _reduce_terms(mterms::Vector{T},
                       atom_map::Vector{Tuple{Int,SVector{3,Int}}},
                       m::SMatrix{3,3,Int,9}, nc::Int, nat::Int,
                       coef_rtol::Float64)::Vector{T} where {T}
    Key = _reduced_key_type(T)
    adj = _adj3(m)
    keys_order = Key[]                       # deterministic output ordering
    # per key: (term idx, aligned folded, the anchor's coset label)
    Entry = Tuple{Int,Array{Float64},SVector{3,Int}}
    bucket = Dict{Key,Vector{Entry}}()
    for (k, mt) in enumerate(mterms)
        key, pf, anchor = _align_reduced(mt, atom_map, m, nat, k)
        entries = get!(bucket, key) do
            push!(keys_order, key)
            Entry[]
        end
        push!(entries, (k, pf, _coset_label(adj, nc, anchor)))
    end

    red_terms = T[]
    for key in keys_order
        # Same canonical anchored structure; split by (coef, aligned folded) —
        # distinct SALCs on the same cluster stay distinct, translation copies of
        # one SALC merge. `scale` is settled first, for the whole key at once.
        _check_scales(mterms, bucket[key])
        reps = Entry[]
        tallies = Dict{SVector{3,Int},Int}[]
        for (k, pf, coset) in bucket[key]
            j = findfirst(rep -> isapprox(mterms[k].coef, mterms[rep[1]].coef;
                                          rtol = coef_rtol) &&
                                 isapprox(pf, rep[2]; rtol = coef_rtol), reps)
            if j === nothing
                push!(reps, (k, pf, coset))
                push!(tallies, Dict(coset => 1))
            else
                tallies[j][coset] = get(tallies[j], coset, 0) + 1
            end
        end
        for ((r, rf, _), tally) in zip(reps, tallies)
            # The invariant is PER COSET, not in total: a periodic Hamiltonian puts the
            # same number `q` of summands of this class in every one of the `nc` cosets.
            # A global `count % nc == 0` would accept (2, 0) for nc = 2 — a term living
            # in one coset only, i.e. exactly the non-periodicity this check exists to
            # catch — and emit it as if it sat in every reduced cell. `q > 1` is the
            # legal case of a raw list carrying several identical summands per instance
            # (hand-built directed pairs, which the canonical alignment folds onto one
            # key); the price of accepting it is that an *accidental* exact duplication
            # of a model term passes too (canonical model terms always have `q = 1`).
            q = first(values(tally))
            (length(tally) == nc && all(==(q), values(tally))) || throw(ArgumentError(
                "term $r (atoms = $(mterms[r].atoms), shifts = $(mterms[r].shifts))" *
                " has $(sum(values(tally))) translation copies spread over " *
                "$(length(tally)) of the $nc cosets with per-coset counts " *
                "$(sort(collect(values(tally)))): a term of a Hamiltonian with that " *
                "periodicity contributes equally to every coset"))
            for _ = 1:q
                push!(red_terms, _emit_reduced(mterms[r], key, rf))
            end
        end
    end
    return red_terms
end

# The member sites in canonical reduced order: `(reduced atom, σ)` sorted, re-anchored
# at `σ₁ = 0`. Returns the permutation (new position `j` holds old member `perm[j]`) and
# the ABSOLUTE anchor `σ₁`, whose coset decides which translation copy this is.
function _reduced_sites(t, atom_map::Vector{Tuple{Int,SVector{3,Int}}},
                        m::SMatrix{3,3,Int,9}, nat::Int, k::Int)
    body = length(t.atoms)
    body >= 1 || throw(ArgumentError("term $k: a term needs at least one member site"))
    length(t.shifts) == body ||
        throw(ArgumentError("term $k: atoms/shifts lengths disagree"))
    all(a -> 1 <= a <= nat, t.atoms) ||
        throw(ArgumentError("term $k: atoms $(t.atoms) outside 1:$nat"))
    bs = Vector{Int}(undef, body)
    sh = Vector{SVector{3,Int}}(undef, body)
    for i = 1:body
        b, o = atom_map[t.atoms[i]]
        bs[i] = b
        sh[i] = o + m * t.shifts[i]
    end
    perm = sortperm(1:body; by = i -> (bs[i], Tuple(sh[i])))
    bs = bs[perm]
    sh = sh[perm]
    anchor = sh[1]
    for i = 1:body
        sh[i] -= anchor
    end
    return bs, sh, perm, anchor
end

# `folded`'s axis `v` contracts a `2l + 1`-component factor; a list whose tensor does
# not say so would otherwise surface as a `DimensionMismatch` out of the census.
function _check_extents(folded::Array{Float64}, ls, k::Int)
    ndims(folded) == length(ls) || throw(ArgumentError(
        "term $k: $(length(ls)) axes but a rank-$(ndims(folded)) folded tensor"))
    for v in eachindex(ls)
        size(folded, v) == 2 * ls[v] + 1 || throw(ArgumentError(
            "term $k axis $v: folded extent $(size(folded, v)) but l = $(ls[v]) " *
            "needs $(2 * ls[v] + 1)"))
    end
    return nothing
end

function _align_reduced(mt::SpinMultipoleTerm, atom_map, m::SMatrix{3,3,Int,9}, nat::Int,
                        k::Int)
    length(mt.ls) == length(mt.atoms) ||
        throw(ArgumentError("term $k: atoms/ls lengths disagree"))
    _check_extents(mt.folded, mt.ls, k)
    bs, sh, perm, anchor = _reduced_sites(mt, atom_map, m, nat, k)
    pf = perm == 1:length(perm) ? mt.folded : permutedims(mt.folded, perm)
    return (bs, sh, mt.ls[perm]), pf, anchor
end

function _align_reduced(dt::DecoratedTerm, atom_map, m::SMatrix{3,3,Int,9}, nat::Int,
                        k::Int)
    body = length(dt.atoms)
    dt.body == body || throw(ArgumentError(
        "term $k: body = $(dt.body) but $body atoms"))
    _check_extents(dt.folded, [s.factor.l for s in dt.slots], k)
    bad = findfirst(s -> !(1 <= s.site <= body), dt.slots)
    bad === nothing || throw(ArgumentError(
        "term $k slot $bad: member site $(dt.slots[bad].site) is outside 1:$body"))
    bs, sh, perm, anchor = _reduced_sites(dt, atom_map, m, nat, k)
    # A slot names its site by MEMBER POSITION, so the site sort relabels it; the slots
    # then have to be re-sorted into SLCE's canonical `(channel, site, k, l)` axis
    # order, with `folded`'s axes carried along. The trailing `v` makes the order
    # independent of the sort algorithm when a term repeats one factor on one site.
    back = invperm(perm)
    moved = [SLCE.Slot(back[s.site], s.factor) for s in dt.slots]
    sperm = sortperm(1:length(moved);
                     by = v -> (moved[v].factor.channel, moved[v].site,
                                moved[v].factor.k, moved[v].factor.l, v))
    pf = sperm == 1:length(sperm) ? dt.folded : permutedims(dt.folded, sperm)
    return (bs, sh, moved[sperm]), pf, anchor
end

# `scale` is `(4π)^(n_spin_slots/2)` — a function of the slots, which the anchored key
# already pins — so every term on one key must carry the identical value. It is
# compared EXACTLY and never recomputed (SLCE design record §13 risk 3: a consumer that
# re-derives the scale from the cluster shape mis-scales mixed terms silently). Note
# what this does NOT check: a hand-built list is free to declare a scale that is not the
# rule at all, as long as it says so consistently — `TiledHamiltonian`, the site that
# applies it, treats the field as declared data too, and the package's kernel-arithmetic
# fixtures rely on that.
_check_scales(::Vector{SpinMultipoleTerm}, entries) = nothing
function _check_scales(dterms::Vector{DecoratedTerm}, entries)
    k0 = first(entries)[1]
    s0 = dterms[k0].scale
    for (k, _, _) in entries
        dterms[k].scale == s0 || throw(ArgumentError(
            "terms $k0 and $k reduce to the same cluster and slots but declare " *
            "different `scale`s ($s0 vs $(dterms[k].scale)) — the scale is a function " *
            "of the slots, so translation copies cannot disagree about it (two " *
            "spellings of one value look like this: `sqrt(4π)` and `(4π)^0.5` differ " *
            "by 1 ulp)"))
    end
    return nothing
end

_emit_reduced(mt::SpinMultipoleTerm, key, folded::Array{Float64}) =
    SpinMultipoleTerm(mt.coef, length(key[1]), copy(key[1]), copy(key[2]), copy(key[3]),
                  copy(folded))
_emit_reduced(dt::DecoratedTerm, key, folded::Array{Float64}) =
    DecoratedTerm(dt.coef, dt.scale, length(key[1]), copy(key[1]), copy(key[2]),
                  copy(key[3]), copy(folded))

"""
    TiledHamiltonian(red::ReducedCell; dims = (1, 1, 1), fixed_reference = false)

Tile a [`reduce_cell`](@ref) result: `dims` counts multiples of the **reduced** cell.
Equivalent to `TiledHamiltonian(red.n_atoms, red.terms[, red.layout]; dims)` — the
`(4π)` scale is applied there (reduction keeps coefficients raw), from the term's own
`scale` field on the decorated path.
"""
TiledHamiltonian(red::ReducedCell{SpinMultipoleTerm}; dims::NTuple{3,Integer} = (1, 1, 1),
                 fixed_reference::Bool = false) =
    TiledHamiltonian(red.n_atoms, red.terms; dims = dims,
                     fixed_reference = fixed_reference)

function TiledHamiltonian(red::ReducedCell{DecoratedTerm};
                          dims::NTuple{3,Integer} = (1, 1, 1),
                          fixed_reference::Bool = false)
    layout = red.layout
    layout === nothing && throw(ArgumentError(   # the ctor invariant rules this out
        "a decorated ReducedCell carries no RowLayout"))
    return TiledHamiltonian(red.n_atoms, red.terms, layout; dims = dims,
                            fixed_reference = fixed_reference)
end
