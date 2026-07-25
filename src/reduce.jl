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
# |det M| coset translations, so grouping terms by it and counting group sizes IS the
# Hamiltonian-periodicity verification: each orbit must contribute exactly |det M|
# training terms with equal `coef`/`folded` (pure translations do not rotate spins).

"""
    ReducedCell

A fitted training-cell Hamiltonian re-expressed in a smaller (or re-based) unit cell
by [`reduce_cell`](@ref). Feed it to [`TiledHamiltonian`](@ref) to tile MC supercells
in multiples of the *reduced* cell.

Fields: `n_atoms` (atoms of the reduced cell), `terms` (one representative
`MultipoleTerm` per translation orbit, **raw** coefficients, sub-cell atom indices and
sub-lattice integer shifts), `crystal` (the reduced cell, for geometry I/O such as
[`supercell_crystal`](@ref)), `M` (the integer matrix with `A_train = A_sub · M`),
`parent_atoms` (reduced atom `b` → its representative training atom), and `atom_map`
(training atom `a` → `(b, offset)` with `offset ∈ ℤ³` in reduced-lattice units).
"""
struct ReducedCell
    n_atoms::Int
    terms::Vector{MultipoleTerm}
    crystal::Crystal
    M::SMatrix{3,3,Int,9}
    parent_atoms::Vector{Int}
    atom_map::Vector{Tuple{Int,SVector{3,Int}}}
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

# Fractional residuals equal modulo the reduced lattice, within `tol` per component.
_same_frac(r1::SVector{3,Float64}, r2::SVector{3,Float64}, tol::Float64)::Bool =
    all(abs(x - round(x)) <= tol for x in r1 - r2)

"""
    reduce_cell(model::SLCEModel, crystal::Crystal, sub_lattice;
                pos_tol = 1e-6, coef_rtol = 1e-10) -> ReducedCell
    reduce_cell(crystal::Crystal, terms::Vector{MultipoleTerm}, sub_lattice;
                pos_tol = 1e-6, coef_rtol = 1e-10) -> ReducedCell

Re-express a Hamiltonian fitted on `crystal` (the training cell — passed explicitly,
this package never reads geometry off the model) in the smaller unit cell whose
lattice vectors are the **columns** of the 3 × 3 matrix `sub_lattice`, after
verifying that the choice is legitimate:

1. `A_train = sub_lattice · M` for an integer matrix `M` (any integer `M`, not just
   diagonal — a bcc conventional cell under a primitive-fitted model, or a mere
   re-basing with `|det M| = 1`, both work);
2. the atomic basis maps onto itself under all `|det M|` coset translations
   (positions within `pos_tol`, in fractional units, and matching species);
3. every fitted term has exactly `|det M|` translation copies with equal
   coefficient and coupling tensor (relative tolerance `coef_rtol`), compared in
   the canonical site order (sorted `(reduced atom, shift)`, tensor axes aligned) —
   so copies anchored at different member sites match. A term list carrying `q`
   identical summands per instance (e.g. hand-built directed pairs) is accepted
   and reduces to `q` copies of the representative; the price of that acceptance
   is that an *accidental* exact integer-multiple duplication of a model term
   would pass this census too (canonical model terms always have `q = 1`).

Any violation throws an `ArgumentError` — a fit that does not actually have the
requested periodicity (e.g. a distorted structure, or couplings that break it) is
never silently symmetrized.

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
                     pos_tol::Real = 1e-6, coef_rtol::Real = 1e-10)::ReducedCell
    n_atoms(model) == n_atoms(crystal) || throw(ArgumentError(
        "crystal has $(n_atoms(crystal)) atoms but the model was fitted on " *
        "$(n_atoms(model)) — pass the training-cell Crystal"))
    return reduce_cell(crystal, multipole_terms(model), sub_lattice;
                       pos_tol = pos_tol, coef_rtol = coef_rtol)
end

function reduce_cell(crystal::Crystal, mterms::Vector{MultipoleTerm},
                     sub_lattice::AbstractMatrix{<:Real};
                     pos_tol::Real = 1e-6, coef_rtol::Real = 1e-10)::ReducedCell
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

    # --- terms: canonical anchored reduced form; count each translation orbit ------
    # Canonical members arrive one per physical instance (SLCE's
    # `_canonicalize_members`), so two translation copies of the same instance are
    # generally anchored at different member sites — in reduced coordinates they
    # differ by a joint site permutation (with the `folded` axes permuted the same
    # way) on top of the coset translation. Align each term to the sorted
    # `(reduced atom, shift)` order, re-anchor, and carry `ls`/`folded` through the
    # permutation before grouping; the aligned copies then match exactly.
    Key = Tuple{Vector{Int},Vector{SVector{3,Int}},Vector{Int}}
    keys_order = Key[]                       # deterministic output ordering
    bucket = Dict{Key,Vector{Tuple{Int,Array{Float64}}}}()   # (term idx, aligned folded)
    for (k, mt) in enumerate(mterms)
        body = length(mt.atoms)
        (length(mt.shifts) == body && length(mt.ls) == body) ||
            throw(ArgumentError("term $k: atoms/shifts/ls lengths disagree"))
        all(a -> 1 <= a <= nat, mt.atoms) ||
            throw(ArgumentError("term $k: atoms $(mt.atoms) outside 1:$nat"))
        bs = Vector{Int}(undef, body)
        sh = Vector{SVector{3,Int}}(undef, body)
        for i = 1:body
            b, o = atom_map[mt.atoms[i]]
            bs[i] = b
            sh[i] = o + m * mt.shifts[i]
        end
        perm = sortperm(1:body; by = i -> (bs[i], Tuple(sh[i])))
        bs = bs[perm]
        sh = sh[perm]
        anchor = sh[1]
        for i = 1:body
            sh[i] -= anchor
        end
        pf = perm == 1:body ? mt.folded : permutedims(mt.folded, perm)
        key = (bs, sh, mt.ls[perm])
        entries = get!(bucket, key) do
            push!(keys_order, key)
            Tuple{Int,Array{Float64}}[]
        end
        push!(entries, (k, pf))
    end

    red_terms = MultipoleTerm[]
    for key in keys_order
        # Same canonical anchored structure; split by (coef, aligned folded) —
        # distinct SALCs on the same cluster stay distinct, translation copies of
        # one SALC merge.
        reps = Tuple{Int,Array{Float64}}[]
        counts = Int[]
        for (k, pf) in bucket[key]
            j = findfirst(rep -> isapprox(mterms[k].coef, mterms[rep[1]].coef;
                                          rtol = coef_rtol) &&
                                 isapprox(pf, rep[2]; rtol = coef_rtol), reps)
            if j === nothing
                push!(reps, (k, pf))
                push!(counts, 1)
            else
                counts[j] += 1
            end
        end
        for ((r, rf), cnt) in zip(reps, counts)
            # A raw term list may legally carry `q` identical summands per physical
            # instance (e.g. hand-built directed pairs, which the canonical
            # alignment folds onto one key); they reduce to `q` output copies.
            cnt % nc == 0 || throw(ArgumentError(
                "term $r (atoms = $(mterms[r].atoms), shifts = $(mterms[r].shifts))" *
                " has $cnt translation copies under the given cell, expected a " *
                "multiple of $nc: the fitted Hamiltonian does not have that " *
                "periodicity"))
            for _ = 1:(cnt ÷ nc)
                push!(red_terms, MultipoleTerm(mterms[r].coef, length(key[1]),
                                               copy(key[1]), copy(key[2]),
                                               copy(key[3]), copy(rf)))
            end
        end
    end

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

    return ReducedCell(nsub, red_terms, sub_crystal, m, parent_atoms, atom_map)
end

"""
    TiledHamiltonian(red::ReducedCell; dims = (1, 1, 1))

Tile a [`reduce_cell`](@ref) result: `dims` counts multiples of the **reduced** cell.
Equivalent to `TiledHamiltonian(red.n_atoms, red.terms; dims)` — the `(4π)^(body/2)`
scale is applied there (reduction keeps coefficients raw).
"""
TiledHamiltonian(red::ReducedCell; dims::NTuple{3,Integer} = (1, 1, 1)) =
    TiledHamiltonian(red.n_atoms, red.terms; dims = dims)
