# The tiled Hamiltonian: the fitted training-cell SCE unfolded onto an N₁×N₂×N₃
# supercell (see `docs/specs/hamiltonian-tiling.md`).
#
# `MultipoleTerm.shifts` are per-site integer lattice translations of the *training*
# cell (`shifts[1] = 0` anchored), so tiling is pure integer bookkeeping: for every
# supercell cell `t` and every fitted term, one instance places member `i` at
# `site_index(atoms[i], mod.(t + shifts[i], dims))`. Each directed cluster member is a
# plain summand of the energy (the introspection contract — no ½ or 1/N factors), so
# the tiled sum on a periodically replicated configuration is exactly
# `prod(dims) × (predict_energy − intercept)` of the training cell — the M1 gate.
#
# Memory: the `folded` coefficient tensors are stored ONCE per fitted term
# (`ScaledTerm` templates); instances are compact integer CSR index lists. This is the
# SpinClusterMC lesson — per-instance payload duplication is what blew that package to
# multi-GB caches. The constructor additionally flattens the templates' nonzero
# `folded` entries into sparse contraction programs (`_ContractionPrograms` below) —
# the dispatch-free form the hot kernels in energy.jl actually walk.

"""
    SpinConfig

Alias `Vector{SVector{3,Float64}}`: one unit spin direction per supercell site (site
indexing per [`site_index`](@ref)). The 3×n matrix layout of the sibling packages
appears only at the I/O boundary (`to_matrix` / `from_matrix`).
"""
const SpinConfig = Vector{SVector{3,Float64}}

"""
    ScaledTerm

One fitted SCE term template in consumer form: `coef` is the raw fitted `jϕ` times
`(4π)^(body/2)` — the scale is applied here, **exactly once** in the package — with
the member `atoms` (training-cell indices), per-site integer lattice `shifts`
(`shifts[1] = 0`), per-site angular momenta `ls`, and the rank-`body` real coefficient
tensor `folded`. Copied out of `SLCE.MultipoleTerm` (value semantics — never an
alias of the model's arrays).
"""
struct ScaledTerm
    coef::Float64
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    ls::Vector{Int}
    folded::Array{Float64}
end

# Precompiled sparse contraction programs — the hot-kernel view of the templates.
# Rank-generic iteration over the rank-erased `ScaledTerm.folded` costs a dynamic
# dispatch and ~2 heap allocations per *instance*, per site visit — the dominant
# cost of `site_coeffs!` and hence of every sweep (bench_log baseline, 2026-07-14).
# Instead, the constructor flattens each template's nonzero `folded` entries once
# into plain integer/float arrays, in the exact loop order of the reference kernels
# (`_total_energy_ref` / `_site_coeffs_ref!` in energy.jl: `CartesianIndices`
# column-major entries, ascending member slots), so the program kernels reproduce
# them **bitwise** (gate: test_energy.jl "program kernels ≡ reference kernels")
# with no dispatch, no allocation, and no zero-entry scanning.
struct _ContractionPrograms
    # site programs, one per (template, member slot) — leave-one-out accumulation:
    site_prog::Vector{Int32}   # adjacency entry j → program id (parallel to site_inst)
    sprog_ptr::Vector{Int32}   # program p's entries: sprog_ptr[p]:sprog_ptr[p+1]-1
    sent_w::Vector{Float64}    # coef · folded[idx], nonzero entries only
    sent_tgt::Vector{Int32}    # target row lm_index(ls[slot], μ_slot) in `c`
    sfac_ptr::Vector{Int32}    # entry e's factors: sfac_ptr[e]:sfac_ptr[e+1]-1
    sfac_row::Vector{Int32}    # factor row lm_index(ls[k], μ_k) in `zrows`
    sfac_slot::Vector{Int8}    # factor member slot k (site = inst_sites[off + k])
    # pair/triplet fast paths (body-2/3 templates — the bulk of every adjacency):
    # a body-2 (body-3) entry has exactly one (two) factors and they always
    # reference the same member slots (the others, ascending), so the neighbor
    # columns are constant across the program. All indirections are precomputed —
    # `site_coeffs!` then walks purely sequential streams plus the zrows gathers,
    # with `p = (1.0·z₁)·z₂… ≡ z₁·z₂…` bitwise:
    site_col::Vector{Int32}    # adjacency entry j → hoisted neighbor column:
                               #   > 0 pair; < 0 −(first triplet column) — the sign
                               #   is the path tag, so the pair path never touches
                               #   site_col2; 0 → general factor loop
    site_col2::Vector{Int32}   # adjacency entry j → second triplet column (else 0)
    pent_row::Vector{Int32}    # entry e → its first factor row (0 → 0 or ≥3 factors)
    pent_row2::Vector{Int32}   # entry e → its second factor row (0 → ≠ 2 factors)
    # energy programs, one per template — the full contraction:
    eprog_ptr::Vector{Int32}   # template k's entries: eprog_ptr[k]:eprog_ptr[k+1]-1
    eent_w::Vector{Float64}    # raw folded[idx] (the coef multiplies the per-instance
                               #   entry sum — the reference kernel's operation order)
    efac_ptr::Vector{Int32}    # entry e's factors: efac_ptr[e]:efac_ptr[e+1]-1
    efac_row::Vector{Int32}    # factor rows; member slot = position within the range
    term_coef::Vector{Float64} # scaled template coef (== terms[k].coef)
end

# One template flattened into the program arrays (rank-specialized barrier —
# construction-time only). Entry order is the `CartesianIndices` column-major order
# of `folded`; factor order is ascending member slot; the skip predicates
# (`coef·folded == 0` for the site programs, `folded == 0` for the energy program)
# are the reference kernels' own — all verbatim, which is what makes the program
# kernels bitwise-identical to them.
function _push_term_programs!(pr::_ContractionPrograms, coef::Float64,
                              ls::Vector{Int}, folded::Array{Float64,D}) where {D}
    for v = 1:D                          # site program of member slot v
        for idx in CartesianIndices(folded)
            w = coef * folded[idx]
            w == 0.0 && continue
            push!(pr.sent_w, w)
            push!(pr.sent_tgt, Int32(Harmonics.lm_index(ls[v], idx[v] - ls[v] - 1)))
            for k = 1:D
                k == v && continue
                push!(pr.sfac_row, Int32(Harmonics.lm_index(ls[k], idx[k] - ls[k] - 1)))
                push!(pr.sfac_slot, Int8(k))
            end
            push!(pr.sfac_ptr, Int32(length(pr.sfac_row) + 1))
            # body-2/3 ⇒ the loop above pushed exactly one/two factors (ascending
            # slots): their rows, entry-indexed, feed the fast paths of
            # `site_coeffs!`
            push!(pr.pent_row, D == 2 ? pr.sfac_row[end] :
                               D == 3 ? pr.sfac_row[end - 1] : Int32(0))
            push!(pr.pent_row2, D == 3 ? pr.sfac_row[end] : Int32(0))
        end
        push!(pr.sprog_ptr, Int32(length(pr.sent_w) + 1))
    end
    for idx in CartesianIndices(folded)  # energy program (every slot is a factor)
        w = folded[idx]
        w == 0.0 && continue
        push!(pr.eent_w, w)
        for k = 1:D
            push!(pr.efac_row, Int32(Harmonics.lm_index(ls[k], idx[k] - ls[k] - 1)))
        end
        push!(pr.efac_ptr, Int32(length(pr.efac_row) + 1))
    end
    push!(pr.eprog_ptr, Int32(length(pr.eent_w) + 1))
    return pr
end

# Index widths (Int32 ids/pointers, Int8 slots) use checked conversions throughout,
# so a model overflowing them fails loudly at construction (InexactError), never by
# silent wraparound.
function _build_programs(terms::Vector{ScaledTerm}, inst_term::Vector{Int32},
                         inst_ptr::Vector{Int32}, inst_sites::Vector{Int32},
                         site_inst::Vector{Int32},
                         site_slot::Vector{Int8})::_ContractionPrograms
    pr = _ContractionPrograms(Vector{Int32}(undef, length(site_inst)), Int32[1],
                              Float64[], Int32[], Int32[1], Int32[], Int8[],
                              Vector{Int32}(undef, length(site_inst)),
                              Vector{Int32}(undef, length(site_inst)),
                              Int32[], Int32[],
                              Int32[1], Float64[], Int32[1], Int32[],
                              [t.coef for t in terms])
    # program id of (template k, member slot v) = pbase[k] + v;
    # pslot1/pslot2[pid] = the fast paths' hoisted factor slots — the member slots
    # other than v, ascending (body-2: (3 − v, 0); body-3: the remaining two);
    # (0, 0) for any other body order
    pbase = Vector{Int}(undef, length(terms))
    pslot1 = Int8[]
    pslot2 = Int8[]
    np = 0
    for (k, t) in enumerate(terms)
        pbase[k] = np
        D = length(t.ls)
        np += D
        for v = 1:D
            push!(pslot1, D == 2 ? Int8(3 - v) : D == 3 ? Int8(v == 1 ? 2 : 1) :
                          Int8(0))
            push!(pslot2, D == 3 ? Int8(v == 3 ? 2 : 3) : Int8(0))
        end
        _push_term_programs!(pr, t.coef, t.ls, t.folded)
    end
    for j in eachindex(site_inst)
        pid = pbase[inst_term[site_inst[j]]] + site_slot[j]
        pr.site_prog[j] = Int32(pid)
        off = Int(inst_ptr[site_inst[j]]) - 1
        sl1 = pslot1[pid]
        sl2 = pslot2[pid]
        # sign of site_col tags the path (see the struct comment): pair +col,
        # triplet −col₁ (with col₂ in site_col2), general 0
        pr.site_col[j] = sl1 == 0 ? Int32(0) :
                         sl2 == 0 ? inst_sites[off + sl1] : -inst_sites[off + sl1]
        pr.site_col2[j] = sl2 == 0 ? Int32(0) : inst_sites[off + sl2]
    end
    return pr
end

"""
    TiledHamiltonian(model::SLCEModel; dims = (1, 1, 1))
    TiledHamiltonian(n_cell_atoms, terms::Vector{MultipoleTerm}; dims = (1, 1, 1))

The fitted SCE Hamiltonian tiled onto an `dims = (N₁, N₂, N₃)` supercell of the
training cell: `n_sites = n_cell_atoms · N₁N₂N₃` spin sites, with one cluster
*instance* per fitted term and supercell cell (member `i` of a term anchored in cell
`t` sits at `site_index(atoms[i], mod.(t .+ shifts[i], dims))` — toroidal boundary
conditions). Energies are in the model's energy units with the intercept `j0`
excluded; on the training cell (`dims = (1,1,1)`) the total energy equals
`predict_energy(model, config) − intercept(model)`.

The second form consumes a hand-built `MultipoleTerm` list with **raw** (unscaled)
coefficients; the `(4π)^(body/2)` scale is applied here, exactly once. Terms with
`coef == 0` are dropped up front (they contribute nothing anywhere; `multipole_terms`
already filters fitted zeros, so this only affects hand-built lists). Every term's
member sites must be **distinct after the toroidal wrap** — `(atomᵢ, mod.(shiftsᵢ,
dims))` pairwise different — which is what makes the single-site coefficient vector
of [`site_coeffs!`](@ref) independent of that site's own spin (exact single-spin ΔE).
Minimum-image fitted models satisfy this for any `dims` (distinct atoms per cluster);
an `AllImages`-fitted model may reuse an atom across images and then needs `dims`
large enough that the images stay distinct sites. `shifts[1] == 0` (the anchoring
convention of the introspection contract) is required.

**Inactive (non-magnetic) sites.** A site no instance touches — e.g. every site of a
species with `lmax = 0` (boron in Nd₂Fe₁₄B), or one whose SALC coefficients all
fitted to zero — has a spin-independent energy. Such sites are flagged
`site_active[s] == false` (`n_active` counts the rest) and are **skipped by the
update sweeps and excluded from the standard observables and their per-site
normalizations**: they keep whatever direction the initial configuration gave them,
verbatim (under `run_pt`, per config payload — replica-exchange swaps move whole
configurations between lanes, frozen spins included). They remain part of the state
(`n_sites`, `config`, checkpoints, the `3 × n_atoms` I/O layout) so site indexing
stays aligned with the crystal.

Immutable; all mutable chain state lives elsewhere (`ChainState`).
"""
struct TiledHamiltonian
    n_cell_atoms::Int
    dims::SVector{3,Int}
    n_sites::Int
    lmax::Int
    nlm::Int
    terms::Vector{ScaledTerm}        # templates — `folded` payloads stored once
    # enumerated instances, CSR over member sites (body orders vary):
    inst_term::Vector{Int32}         # instance → template index
    inst_ptr::Vector{Int32}          # instance i's sites: inst_sites[ptr[i]:ptr[i+1]-1]
    inst_sites::Vector{Int32}        # global site ids, concatenated
    # per-site adjacency, CSR:
    site_ptr::Vector{Int32}          # site s touches site_inst[ptr[s]:ptr[s+1]-1]
    site_inst::Vector{Int32}         # instance ids
    site_slot::Vector{Int8}          # this site's member slot within that instance
    site_has_l1::Vector{Bool}        # any adjacent instance carries l = 1 at this site
    site_active::Vector{Bool}        # any adjacent instance at all (else non-magnetic)
    n_active::Int                    # number of active sites
    progs::_ContractionPrograms      # precompiled sparse contraction programs
    # proper coloring of the site-conflict graph (conflict = shares an instance):
    # the sweeps scan color classes in order; sites within one class never co-occur
    # in an instance, so their single-spin kernels are exactly independent and a
    # class may be updated concurrently (updates-stationarity.md U1).
    n_colors::Int
    color_ptr::Vector{Int32}         # class c: color_sites[ptr[c]:ptr[c+1]-1]
    color_sites::Vector{Int32}       # active sites, class-major, ascending in class

    function TiledHamiltonian(n_cell_atoms::Integer, mterms::Vector{MultipoleTerm};
                              dims::NTuple{3,Integer} = (1, 1, 1))
        n_cell_atoms >= 1 ||
            throw(ArgumentError("n_cell_atoms must be ≥ 1; got $n_cell_atoms"))
        all(d -> d >= 1, dims) || throw(ArgumentError("dims must be ≥ 1; got $dims"))
        # A coef == 0 term contributes nothing to any energy, coefficient vector, or
        # gradient — drop it so "no adjacent instance" means "spin-independent site".
        mterms = filter(t -> t.coef != 0.0, mterms)
        isempty(mterms) && throw(ArgumentError(
            "the term list is empty (no spin-dependent SALCs with nonzero coefficients)"))

        d = SVector{3,Int}(dims)
        terms = Vector{ScaledTerm}(undef, length(mterms))
        lmax = 0
        for (k, mt) in enumerate(mterms)
            body = length(mt.atoms)
            (length(mt.shifts) == body && length(mt.ls) == body) ||
                throw(ArgumentError("term $k: atoms/shifts/ls lengths disagree"))
            all(a -> 1 <= a <= n_cell_atoms, mt.atoms) || throw(ArgumentError(
                "term $k: atoms $(mt.atoms) outside 1:$n_cell_atoms"))
            # Distinct member *sites* per instance ⇒ the leave-one-out coefficients of
            # `site_coeffs!` are independent of the site's own spin (exact ΔE). The
            # wrapped relative pattern is the same for every cell, so checking the
            # cell-0 instance covers all of them. Minimum-image models have distinct
            # atoms outright; AllImages models may reuse an atom across images and
            # then need `dims` large enough to keep the images distinct sites.
            allunique(zip(mt.atoms, (mod.(sh, d) for sh in mt.shifts))) ||
                throw(ArgumentError(
                    "term $k (atoms = $(mt.atoms), shifts = $(mt.shifts)) folds two " *
                    "member sites onto one supercell site under dims = $dims; the " *
                    "single-spin update assumes distinct sites per cluster — " *
                    "enlarge dims"))
            iszero(mt.shifts[1]) || throw(ArgumentError(
                "term $k: shifts[1] = $(mt.shifts[1]) breaks the home-cell anchoring " *
                "convention (shifts[1] == 0) of the introspection contract"))
            all(l -> l >= 0, mt.ls) ||
                throw(ArgumentError("term $k: negative angular momentum in $(mt.ls)"))
            size(mt.folded) == Tuple(2l + 1 for l in mt.ls) || throw(ArgumentError(
                "term $k: size(folded) = $(size(mt.folded)) does not match " *
                "ls = $(mt.ls)"))
            lmax = max(lmax, maximum(mt.ls))
            # The package's single (4π)^(body/2) application site.
            terms[k] = ScaledTerm(mt.coef * (4π)^(body / 2), copy(mt.atoms),
                                  copy(mt.shifts), copy(mt.ls), copy(mt.folded))
        end

        ncells = prod(d)
        n_sites = n_cell_atoms * ncells
        nterms = length(terms)
        n_inst = nterms * ncells

        # Instances: cell-outer (cell 0 first), term-inner — deterministic ordering.
        inst_term = Vector{Int32}(undef, n_inst)
        inst_ptr = Vector{Int32}(undef, n_inst + 1)
        total_sites = ncells * sum(t -> length(t.atoms), terms)
        inst_sites = Vector{Int32}(undef, total_sites)
        inst_ptr[1] = 1
        i = 0
        p = 0
        for cell3 = 0:(d[3] - 1), cell2 = 0:(d[2] - 1), cell1 = 0:(d[1] - 1)
            t = SVector(cell1, cell2, cell3)
            for (k, term) in enumerate(terms)
                i += 1
                inst_term[i] = k
                for (a, sh) in zip(term.atoms, term.shifts)
                    cw = mod.(t + sh, d)
                    p += 1
                    inst_sites[p] = _site_index(n_cell_atoms, d, a, cw)
                end
                inst_ptr[i + 1] = p + 1
            end
        end

        # Per-site adjacency (CSR): count, prefix-sum, fill.
        counts = zeros(Int32, n_sites)
        for s in inst_sites
            counts[s] += 1
        end
        site_ptr = Vector{Int32}(undef, n_sites + 1)
        site_ptr[1] = 1
        for s = 1:n_sites
            site_ptr[s + 1] = site_ptr[s] + counts[s]
        end
        site_inst = Vector{Int32}(undef, total_sites)
        site_slot = Vector{Int8}(undef, total_sites)
        cursor = copy(@view site_ptr[1:n_sites])
        for inst = 1:n_inst
            for (slot, q) in enumerate(inst_ptr[inst]:(inst_ptr[inst + 1] - 1))
                s = inst_sites[q]
                site_inst[cursor[s]] = inst
                site_slot[cursor[s]] = slot
                cursor[s] += 1
            end
        end

        site_has_l1 = zeros(Bool, n_sites)
        for s = 1:n_sites, j = site_ptr[s]:(site_ptr[s + 1] - 1)
            ls = terms[inst_term[site_inst[j]]].ls
            site_has_l1[s] |= ls[site_slot[j]] == 1
        end
        site_active = [site_ptr[s + 1] > site_ptr[s] for s = 1:n_sites]
        n_active = count(site_active)
        progs = _build_programs(terms, inst_term, inst_ptr, inst_sites, site_inst,
                                site_slot)
        n_colors, color_ptr, color_sites = _color_sites(
            n_sites, site_ptr, site_inst, inst_ptr, inst_sites, site_active)

        return new(n_cell_atoms, d, n_sites, lmax, Harmonics.num_lm(lmax), terms,
                   inst_term, inst_ptr, inst_sites, site_ptr, site_inst, site_slot,
                   site_has_l1, site_active, n_active, progs, n_colors, color_ptr,
                   color_sites)
    end
end

TiledHamiltonian(model::SLCEModel; dims::NTuple{3,Integer} = (1, 1, 1)) =
    TiledHamiltonian(n_atoms(model), multipole_terms(model); dims = dims)

# Greedy proper coloring of the site-conflict graph, in site order (deterministic —
# a function of the Hamiltonian alone). Two sites conflict when some instance
# touches both, i.e. exactly when one's leave-one-out coefficients depend on the
# other's spin. Inactive sites are left uncolored (the sweeps skip them). The
# greedy bound is Δ+1 colors (Δ = conflict degree); the class layout is CSR,
# class-major with sites ascending within a class.
function _color_sites(n_sites::Int, site_ptr::Vector{Int32}, site_inst::Vector{Int32},
                      inst_ptr::Vector{Int32}, inst_sites::Vector{Int32},
                      site_active::Vector{Bool})
    colors = zeros(Int32, n_sites)
    stamp = Int[]                # stamp[c] == s ⇒ color c is taken by a conflictor
    ncol = 0
    for s = 1:n_sites
        site_active[s] || continue
        @inbounds for j = site_ptr[s]:(site_ptr[s + 1] - 1)
            i = site_inst[j]
            for q = inst_ptr[i]:(inst_ptr[i + 1] - 1)
                c = Int(colors[inst_sites[q]])
                c > 0 && (stamp[c] = s)
            end
        end
        c = 1
        while c <= ncol && stamp[c] == s
            c += 1
        end
        if c > ncol
            ncol = c
            push!(stamp, 0)
        end
        colors[s] = Int32(c)
    end
    counts = zeros(Int32, ncol)
    for s = 1:n_sites
        colors[s] > 0 && (counts[colors[s]] += 1)
    end
    color_ptr = Vector{Int32}(undef, ncol + 1)
    color_ptr[1] = 1
    for c = 1:ncol
        color_ptr[c + 1] = color_ptr[c] + counts[c]
    end
    color_sites = Vector{Int32}(undef, color_ptr[ncol + 1] - 1)
    cursor = copy(@view color_ptr[1:ncol])
    for s = 1:n_sites               # site order ⇒ ascending within each class
        c = colors[s]
        c > 0 || continue
        color_sites[cursor[c]] = Int32(s)
        cursor[c] += 1
    end
    return ncol, color_ptr, color_sites
end

Base.show(io::IO, H::TiledHamiltonian) =
    print(io, "TiledHamiltonian(", H.n_cell_atoms, " atoms × ", H.dims[1], "×",
          H.dims[2], "×", H.dims[3], " = ", H.n_sites, " sites",
          H.n_active < H.n_sites ? " ($(H.n_sites - H.n_active) inactive)" : "",
          ", lmax=", H.lmax, ", ", length(H.terms), " terms, ",
          length(H.inst_term), " instances)")

"""
    n_sites(H::TiledHamiltonian) -> Int

Number of spin sites of the tiled supercell (`n_cell_atoms · N₁N₂N₃`).
"""
n_sites(H::TiledHamiltonian)::Int = H.n_sites

@inline function _site_index(n_cell_atoms::Int, dims::SVector{3,Int}, atom::Int,
                             cell::SVector{3,Int})::Int32
    Int32(atom + n_cell_atoms *
          (cell[1] + dims[1] * (cell[2] + dims[2] * cell[3])))
end

"""
    site_index(H::TiledHamiltonian, atom::Integer, cell) -> Int

Global site index of training-cell atom `atom` in supercell cell
`cell = (c₁, c₂, c₃)` (0-based, `0 ≤ cᵢ < dims[i]`):
`atom + n_cell_atoms · (c₁ + N₁·(c₂ + N₂·c₃))` — atom-fastest, then cells in
column-major cell order.
"""
function site_index(H::TiledHamiltonian, atom::Integer, cell)::Int
    1 <= atom <= H.n_cell_atoms ||
        throw(ArgumentError("atom $atom outside 1:$(H.n_cell_atoms)"))
    c = SVector{3,Int}(cell)
    all(i -> 0 <= c[i] < H.dims[i], 1:3) ||
        throw(ArgumentError("cell $c outside 0:dims-1 = $(H.dims .- 1)"))
    return Int(_site_index(H.n_cell_atoms, H.dims, Int(atom), c))
end

"""
    site_atom(H::TiledHamiltonian, s::Integer) -> Int

The training-cell atom index (= sublattice id) of global site `s` — the inverse of
the atom component of [`site_index`](@ref).
"""
site_atom(H::TiledHamiltonian, s::Integer)::Int = mod1(Int(s), H.n_cell_atoms)
