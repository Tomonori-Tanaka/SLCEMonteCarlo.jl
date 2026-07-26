# The tiled Hamiltonian: the fitted training-cell SCE unfolded onto an N₁×N₂×N₃
# supercell (see `docs/specs/hamiltonian-tiling.md`).
#
# A term's `shifts` are per-site integer lattice translations of the *training* cell
# (`shifts[1] = 0` anchored), so tiling is pure integer bookkeeping: for every
# supercell cell `t` and every fitted term, one instance places member `i` at
# `site_index(atoms[i], mod.(t + shifts[i], dims))`. Each directed cluster member is a
# plain summand of the energy (the introspection contract — no ½ or 1/N factors), so
# the tiled sum on a periodically replicated configuration is exactly
# `prod(dims) × (predict_energy − intercept)` of the training cell — the M1 gate.
#
# CHANNELS (M4 slice 3b). A term's tensor axes are *slots*, not sites: a joint
# spin–lattice term reads `Z_{l,m}(ê_a)` on a spin axis and `|u_a|^{2k} R_{l,m}(u_a)` on
# a displacement one, and one site may carry one of each. So `ScaledTerm` carries
# `slots::Vector{TermSlot}` (one per axis) instead of a per-site `ls`, and each slot
# names the row block it gathers from. The row numbering is NOT invented here — it is
# `SLCE.row_layout(model)`, the upstream sampler-row contract, whose SPIN block is
# `Harmonics.lm_index` at offset 0. That is what makes a pure-spin model's row tables —
# and hence every number this package produces for one — bit-for-bit what they were
# before the displacement channel existed.
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
    TermSlot

One tensor axis of a [`ScaledTerm`](@ref): which member it reads (`site`, a **position
in the term's `atoms`**, not a global site id), the per-site row block it gathers from
(`row0`, the row just *before* the `m = −l` component, so component `m` sits at
`row0 + m + l + 1`), the degree `l`, and whether it is a spin axis (`spin`; `false` is
a displacement axis).

Several slots may share a `site` — a site is not an axis. `row0` is derived once, at
construction, from `SLCE.SlotRef` + the model's `SLCE.RowLayout`, so no hot kernel ever
re-derives a row index; that both channels number their `2l + 1` components
contiguously is checked per slot against `SLCE.row_index` when the term is ingested.
"""
struct TermSlot
    site::Int
    row0::Int
    l::Int
    spin::Bool
end

"""
    ScaledTerm

One fitted SCE term template in consumer form: `coef` is the raw fitted `jϕ` times the
term's scale `(4π)^(n_spin_slots/2)` — applied here, **exactly once** in the package —
with the member `atoms` (training-cell indices), per-site integer lattice `shifts`
(`shifts[1] = 0`), one [`TermSlot`](@ref) per tensor axis, and the
rank-`length(slots)` real coefficient tensor `folded`. Copied out of
`SLCE.DecoratedTerm` (or its frozen pure-spin predecessor `SLCE.MultipoleTerm`) with
value semantics — never an alias of the model's arrays.

On a pure-spin term the slots are the identity layout (axis `i` = the spin factor of
site `i`) and the scale reduces to `(4π)^(body/2)`, exactly the pre-M4 form.
"""
struct ScaledTerm
    coef::Float64
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    slots::Vector{TermSlot}
    folded::Array{Float64}
end

# Is this term's slot layout the pure-spin identity — axis i = the spin factor of site
# i? Then `slots` carries no information beyond the old per-site `ls`, which is what
# lets the checkpoint fingerprint stay unchanged for every pure-spin model
# (checkpoint.jl `_fingerprint`).
_is_spin_identity(slots::Vector{TermSlot})::Bool =
    all(v -> slots[v].spin && slots[v].site == v, eachindex(slots))

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
    sfac_row::Vector{Int32}    # factor row (slot k's row0 + μ_k + l_k + 1) in `zrows`
    sfac_slot::Vector{Int8}    # factor's member SITE POSITION (site = inst_sites[off+·])
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
    efac_row::Vector{Int32}    # factor rows, one per axis in slot order
    efac_site::Vector{Int8}    # each factor's member site position — for a pure-spin
                               #   term this is 1, 2, … (axis ≡ site) and the kernel
                               #   reduces to the pre-M4 `off + m` indexing
    term_coef::Vector{Float64} # scaled template coef (== terms[k].coef)
end

# One template flattened into the program arrays (rank-specialized barrier —
# construction-time only). Entry order is the `CartesianIndices` column-major order
# of `folded`; factor order is ascending axis; the skip predicates
# (`coef·folded == 0` for the site programs, `folded == 0` for the energy program)
# are the reference kernels' own — all verbatim, which is what makes the program
# kernels bitwise-identical to them.
#
# There is one site program per member **site position** `q`, not per axis: the
# leave-one-out quantity `site_coeffs!` builds is "the coefficient of each row of site
# q", and when a site carries two axes (a spin factor and a displacement one) both
# contribute, into their own target rows. Their entry lists are concatenated in
# ascending axis order, so a pure-spin term — exactly one axis per site — yields the
# pre-M4 program byte for byte.
#
# `slot_rows[v][j]` is axis `v`'s row for tensor index `j`, and `slot_site[v]` its
# member site position; `q_axes[q]` lists the axes sitting on site position `q`.
function _push_term_programs!(pr::_ContractionPrograms, coef::Float64,
                              slot_site::Vector{Int8}, slot_rows::Vector{Vector{Int32}},
                              q_axes::Vector{Vector{Int}},
                              folded::Array{Float64,D}) where {D}
    for axes in q_axes                   # site program of member site position q
        for v in axes
            for idx in CartesianIndices(folded)
                w = coef * folded[idx]
                w == 0.0 && continue
                push!(pr.sent_w, w)
                push!(pr.sent_tgt, slot_rows[v][idx[v]])
                for k = 1:D
                    k == v && continue
                    push!(pr.sfac_row, slot_rows[k][idx[k]])
                    push!(pr.sfac_slot, slot_site[k])
                end
                push!(pr.sfac_ptr, Int32(length(pr.sfac_row) + 1))
                # rank 2/3 ⇒ the loop above pushed exactly one/two factors (ascending
                # axes): their rows, entry-indexed, feed the fast paths of
                # `site_coeffs!`
                push!(pr.pent_row, D == 2 ? pr.sfac_row[end] :
                                   D == 3 ? pr.sfac_row[end - 1] : Int32(0))
                push!(pr.pent_row2, D == 3 ? pr.sfac_row[end] : Int32(0))
            end
        end
        push!(pr.sprog_ptr, Int32(length(pr.sent_w) + 1))
    end
    for idx in CartesianIndices(folded)  # energy program (every axis is a factor)
        w = folded[idx]
        w == 0.0 && continue
        push!(pr.eent_w, w)
        for k = 1:D
            push!(pr.efac_row, slot_rows[k][idx[k]])
            push!(pr.efac_site, slot_site[k])
        end
        push!(pr.efac_ptr, Int32(length(pr.efac_row) + 1))
    end
    push!(pr.eprog_ptr, Int32(length(pr.eent_w) + 1))
    return pr
end

# The pair/triplet fast paths of `site_coeffs!` hoist the neighbour *columns* out of the
# entry loop, which is sound only when every entry of a site's program reads the same
# columns in the same order. With one axis per site (every pure-spin term) that is
# automatic. When a site carries two axes their entry lists are concatenated and each
# drops a different axis from its factor list, so the ascending-order column tuples
# agree only sometimes — they do when the dropped axes are adjacent in axis order (both
# sit on the same site), and need not when a third axis separates them. Compute the
# tuples and compare rather than reasoning about the canonical axis order: the general
# path is always available, and a disagreement simply selects it.
function _hoisted_columns(slots::Vector{TermSlot}, slot_site::Vector{Int8},
                          axes::Vector{Int})::Tuple{Int8,Int8}
    D = length(slots)
    (D == 2 || D == 3) || return (Int8(0), Int8(0))
    ref = Int8[]
    for v in axes
        cols = Int8[slot_site[k] for k = 1:D if k != v]
        if isempty(ref)
            ref = cols
        elseif cols != ref
            return (Int8(0), Int8(0))
        end
    end
    isempty(ref) && return (Int8(0), Int8(0))
    return (ref[1], D == 3 ? ref[2] : Int8(0))
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
                              Int32[1], Float64[], Int32[1], Int32[], Int8[],
                              [t.coef for t in terms])
    # program id of (template k, member site position q) = pbase[k] + q;
    # pslot1/pslot2[pid] = the fast paths' hoisted factor SITE POSITIONS — for a rank-2
    # or rank-3 template, the positions of the axes other than the one being targeted,
    # ascending — or (0, 0) when no hoisting applies (any other rank, or a site whose
    # axes disagree about the column tuple; see `_hoisted_columns`).
    pbase = Vector{Int}(undef, length(terms))
    pslot1 = Int8[]
    pslot2 = Int8[]
    np = 0
    for (k, t) in enumerate(terms)
        pbase[k] = np
        np += length(t.atoms)
        slot_site = Int8[Int8(s.site) for s in t.slots]
        slot_rows = [Int32[Int32(s.row0 + m + s.l + 1) for m = (-s.l):(s.l)]
                     for s in t.slots]
        q_axes = [findall(s -> s.site == q, t.slots) for q = 1:length(t.atoms)]
        for axes in q_axes
            f1, f2 = _hoisted_columns(t.slots, slot_site, axes)
            push!(pslot1, f1)
            push!(pslot2, f2)
        end
        _push_term_programs!(pr, t.coef, slot_site, slot_rows, q_axes, t.folded)
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
    TiledHamiltonian(n_cell_atoms, terms::Vector{DecoratedTerm}, layout::RowLayout;
                     dims = (1, 1, 1))

The fitted SCE Hamiltonian tiled onto an `dims = (N₁, N₂, N₃)` supercell of the
training cell: `n_sites = n_cell_atoms · N₁N₂N₃` sites, with one cluster
*instance* per fitted term and supercell cell (member `i` of a term anchored in cell
`t` sits at `site_index(atoms[i], mod.(t .+ shifts[i], dims))` — toroidal boundary
conditions). Energies are in the model's energy units with the intercept `j0`
excluded; on the training cell (`dims = (1,1,1)`) the total energy equals
`predict_energy(model, config[, disps]) − intercept(model)`.

The second and third forms consume hand-built term lists with **raw** (unscaled)
coefficients; the scale — `(4π)^(body/2)` for a `MultipoleTerm`, the general
`(4π)^(n_spin_slots/2)` carried by a `DecoratedTerm` — is applied here, exactly once.
Terms with `coef == 0` are dropped up front (they contribute nothing anywhere;
the introspection surfaces already filter fitted zeros, so this only affects hand-built
lists). Every term's member sites must be **distinct after the toroidal wrap** —
`(atomᵢ, mod.(shiftsᵢ, dims))` pairwise different — which is what makes the single-site
coefficient vector of [`site_coeffs!`](@ref) independent of that site's own state (exact
single-site ΔE). Minimum-image fitted models satisfy this for any `dims` (distinct atoms
per cluster); an `AllImages`-fitted model may reuse an atom across images and then needs
`dims` large enough that the images stay distinct sites. `shifts[1] == 0` (the anchoring
convention of the introspection contract) is required.

**Channels.** A term's tensor axes are [`TermSlot`](@ref)s, and one site may carry both
a spin and a displacement axis, so a `DecoratedTerm` model needs the model's own row
numbering — pass `SLCE.row_layout(model)`, which the one-argument form does for you.
`H.lmax`/`H.nlm` describe the SPIN row block alone (so every pure-spin consumer is
unaffected) while `H.nrows` counts all rows; [`total_energy`](@ref) then takes the
displacements as a third argument. Because a site's two axes multiply each other, the
leave-one-out vector of [`site_coeffs!`](@ref) is exact for a move that changes **one
channel** of a site — which is all the updates ever propose.

**Inactive (non-magnetic) sites.** A site no instance touches — e.g. every site of a
species with `lmax = 0` (boron in Nd₂Fe₁₄B), or one whose SALC coefficients all
fitted to zero — has a state-independent energy. Such sites are flagged
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
    lmax::Int                        # SPIN block only (−1 = no spin content)
    nlm::Int                         # SPIN block width == layout.disp_offset
    nrows::Int                       # all per-site rows == layout.nrows
    disp_lmax::Int                   # max displacement l (−1 = pure spin)
    layout::RowLayout                # the upstream sampler-row contract
    terms::Vector{ScaledTerm}        # templates — `folded` payloads stored once
    # enumerated instances, CSR over member sites (body orders vary):
    inst_term::Vector{Int32}         # instance → template index
    inst_ptr::Vector{Int32}          # instance i's sites: inst_sites[ptr[i]:ptr[i+1]-1]
    inst_sites::Vector{Int32}        # global site ids, concatenated
    # per-site adjacency, CSR:
    site_ptr::Vector{Int32}          # site s touches site_inst[ptr[s]:ptr[s+1]-1]
    site_inst::Vector{Int32}         # instance ids
    site_slot::Vector{Int8}          # this site's member SITE POSITION in that
                                     #   instance (1:body — not an axis index)
    site_has_l1::Vector{Bool}        # any adjacent instance carries l = 1 at this site
    site_has_spin::Vector{Bool}      # any adjacent instance carries a SPIN axis here
    site_has_disp::Vector{Bool}      # any adjacent instance carries a DISP axis here
    site_active::Vector{Bool}        # any adjacent instance at all (either channel)
    n_active::Int                    # number of active sites (scheduling / coloring)
    n_spin_active::Int               # number of spin-active sites (spin observables)
    n_disp_active::Int               # number of displacement-active sites
    # Displacement-coupling components (CSR, component-major, sites ascending). Two
    # disp-active sites are joined when some instance carries a DISP axis on BOTH: the
    # energy then splits into groups of displacement variables that never meet, and a
    # uniform shift of EACH group separately is an exact symmetry under the ASR. The
    # flat space is therefore 3 × n_disp_comps, not 3 — the "exactly three zero
    # eigenvalues of D(0)" statement is about a CONNECTED model.
    disp_comp_ptr::Vector{Int32}
    disp_comp_sites::Vector{Int32}
    n_disp_comps::Int
    # How flat each component's rigid-shift directions actually are, measured on this
    # Hamiltonian (see `_translation_residuals`): 0 means exactly flat, ~1 means not flat
    # at all. Kept per (Cartesian direction, component) because BOTH axes matter — the
    # components are independent symmetries, so one pinned component must not disable
    # re-centring for a flat one; and a single component's flat directions can be a
    # proper subspace (a substrate-clamped slab is pinned along the normal, free in the
    # plane), so re-centring must project onto exactly that subspace.
    comp_residual::Matrix{Float64}   # 3 × n_disp_comps, one per Cartesian direction
    comp_free::Matrix{Bool}          # true = that (direction, component) is a symmetry
    translation_residual::Float64    # the worst entry (summary/show)
    translation_invariant::Bool      # `all(comp_free)`
    progs::_ContractionPrograms      # precompiled sparse contraction programs
    # proper coloring of the site-conflict graph (conflict = shares an instance):
    # the sweeps scan color classes in order; sites within one class never co-occur
    # in an instance, so their single-spin kernels are exactly independent and a
    # class may be updated concurrently (updates-stationarity.md U1).
    n_colors::Int
    color_ptr::Vector{Int32}         # class c: color_sites[ptr[c]:ptr[c+1]-1]
    color_sites::Vector{Int32}       # active sites, class-major, ascending in class

    function TiledHamiltonian(n_cell_atoms::Integer, terms::Vector{ScaledTerm},
                              layout::RowLayout; dims::NTuple{3,Integer} = (1, 1, 1),
                              fixed_reference::Bool = false)
        n_cell_atoms >= 1 ||
            throw(ArgumentError("n_cell_atoms must be ≥ 1; got $n_cell_atoms"))
        all(d -> d >= 1, dims) || throw(ArgumentError("dims must be ≥ 1; got $dims"))
        isempty(terms) && throw(ArgumentError(
            "the term list is empty (no state-dependent SALCs with nonzero coefficients)"))

        d = SVector{3,Int}(dims)
        for (k, t) in enumerate(terms)
            body = length(t.atoms)
            body >= 1 || throw(ArgumentError("term $k: no member sites"))
            length(t.shifts) == body ||
                throw(ArgumentError("term $k: atoms/shifts lengths disagree"))
            all(a -> 1 <= a <= n_cell_atoms, t.atoms) || throw(ArgumentError(
                "term $k: atoms $(t.atoms) outside 1:$n_cell_atoms"))
            # Distinct member *sites* per instance ⇒ the leave-one-out coefficients of
            # `site_coeffs!` are independent of the site's own state (exact ΔE). The
            # wrapped relative pattern is the same for every cell, so checking the
            # cell-0 instance covers all of them. Minimum-image models have distinct
            # atoms outright; AllImages models may reuse an atom across images and
            # then need `dims` large enough to keep the images distinct sites.
            allunique(zip(t.atoms, (mod.(sh, d) for sh in t.shifts))) ||
                throw(ArgumentError(
                    "term $k (atoms = $(t.atoms), shifts = $(t.shifts)) folds two " *
                    "member sites onto one supercell site under dims = $dims; the " *
                    "single-site update assumes distinct sites per cluster — " *
                    "enlarge dims"))
            iszero(t.shifts[1]) || throw(ArgumentError(
                "term $k: shifts[1] = $(t.shifts[1]) breaks the home-cell anchoring " *
                "convention (shifts[1] == 0) of the introspection contract"))
            size(t.folded) == Tuple(2s.l + 1 for s in t.slots) || throw(ArgumentError(
                "term $k: size(folded) = $(size(t.folded)) does not match its slot " *
                "degrees $([s.l for s in t.slots])"))
            for s in t.slots
                1 <= s.site <= body || throw(ArgumentError(
                    "term $k: slot site $(s.site) outside 1:$body"))
                s.l >= 0 || throw(ArgumentError("term $k: negative slot degree $(s.l)"))
                (s.row0 >= 0 && s.row0 + 2s.l + 1 <= layout.nrows) ||
                    throw(ArgumentError(
                        "term $k: slot rows $(s.row0 + 1):$(s.row0 + 2s.l + 1) fall " *
                        "outside the layout's 1:$(layout.nrows) — the layout does not " *
                        "belong to these terms"))
                # The channel FLAG and the row BLOCK must say the same thing. `spin` and
                # `row0` are independent fields of the public `TermSlot`, and the
                # single-channel ΔE partitions the row table by BLOCK
                # (`1:disp_offset` vs the rest) while every activity predicate and the
                # one-axis-per-(site, channel) rule below partition it by FLAG. Let the
                # two disagree and a displacement move moves a row the range-limited
                # `delta_energy` treats as a spin row — dropping a genuine cross term
                # while passing every other check (measured 0.25 on |E| ≈ 1).
                (s.spin ? s.row0 + 2s.l + 1 <= layout.disp_offset :
                          s.row0 >= layout.disp_offset) || throw(ArgumentError(
                    "term $k: slot flagged $(s.spin ? "spin" : "displacement") occupies " *
                    "rows $(s.row0 + 1):$(s.row0 + 2s.l + 1), which is not inside the " *
                    "$(s.spin ? "SPIN block 1:$(layout.disp_offset)" :
                       "displacement blocks $(layout.disp_offset + 1):$(layout.nrows)") " *
                    "— the channel flag and the row block must agree, since the exact " *
                    "single-channel ΔE splits the row table by block"))
            end
            for q = 1:body
                # A member site with no axis at all would be flagged active (it appears
                # in `inst_sites`) while contributing nothing — the sweeps would visit it
                # and the observables would count it. That is not a term this package can
                # honour, so refuse it rather than let the inactive-site convention lie.
                nq = count(s -> s.site == q, t.slots)
                nq >= 1 || throw(ArgumentError(
                    "term $k: member site position $q carries no tensor axis"))
                # AT MOST ONE AXIS PER (site, channel) — upstream's `SiteDecor` rule, and
                # the invariant the exact single-channel ΔE rests on. With two axes of the
                # SAME channel on one site, a single-channel move moves both of their rows
                # and `delta_energy` silently drops the `Δz·Δz′` cross term (measured 21 %
                # on a two-spin-axis onsite term) — and such a term has a pure-spin
                # layout, so `_require_spin_only` would not catch it either: it would go
                # straight into the sweeps with wrong Boltzmann weights.
                nq <= 2 && length(unique(s.spin for s in t.slots if s.site == q)) == nq ||
                    throw(ArgumentError(
                        "term $k: member site position $q carries $nq axes of which two " *
                        "share a channel; at most one axis per (site, channel) is " *
                        "allowed — the exact single-channel ΔE of site_coeffs! " *
                        "depends on it"))
            end
        end

        lmax = layout.spin_lmax
        nlm = layout.disp_offset
        disp_lmax = isempty(layout.disp_factors) ? -1 :
                    maximum(l for (_, l) in layout.disp_factors)
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

        # Two per-site activity notions, and on a joint model they DIFFER: a site
        # referenced only by displacement axes (a force-constant-only ligand — boron in
        # Nd₂Fe₁₄B with `lmax = 0` plus a displacement sector is a realistic production
        # case) is active for scheduling/coloring but carries no magnetic moment at all.
        # `site_active` drives the coloring and the sweep schedule (both channels need
        # it); `site_has_spin`/`n_spin_active` is what the SPIN sweeps, `_renormalize!`,
        # the spin observables and their per-site normalizations read. Conflating them
        # would divide `m = Σ_s e_s / n` by a count including sites whose direction is a
        # frozen random vector — the constant observable bias the inactive-site
        # convention exists to avoid. They coincide exactly on a pure-spin model.
        site_has_l1 = zeros(Bool, n_sites)
        site_has_spin = zeros(Bool, n_sites)
        site_has_disp = zeros(Bool, n_sites)
        for s = 1:n_sites, j = site_ptr[s]:(site_ptr[s + 1] - 1)
            slots = terms[inst_term[site_inst[j]]].slots
            q = site_slot[j]
            site_has_l1[s] |= any(sl -> sl.site == q && sl.spin && sl.l == 1, slots)
            site_has_spin[s] |= any(sl -> sl.site == q && sl.spin, slots)
            site_has_disp[s] |= any(sl -> sl.site == q && !sl.spin, slots)
        end
        site_active = [site_ptr[s + 1] > site_ptr[s] for s = 1:n_sites]
        n_active = count(site_active)
        n_spin_active = count(site_has_spin)
        n_disp_active = count(site_has_disp)
        ncomp, dcomp_ptr, dcomp_sites = _disp_components(
            n_sites, terms, inst_term, inst_ptr, inst_sites, site_has_disp)
        progs = _build_programs(terms, inst_term, inst_ptr, inst_sites, site_inst,
                                site_slot)
        n_colors, color_ptr, color_sites = _color_sites(
            n_sites, site_ptr, site_inst, inst_ptr, inst_sites, site_active)

        # `_translation_residuals` needs a finished Hamiltonian to evaluate energies on,
        # and the residual is a field — so build once with a placeholder, measure, and
        # build again with the verdict. `new` may be called more than once; the two
        # objects share every array by reference, so the second is nearly free.
        H0 = new(n_cell_atoms, d, n_sites, lmax, nlm, layout.nrows, disp_lmax,
                 layout, terms,
                 inst_term, inst_ptr, inst_sites, site_ptr, site_inst, site_slot,
                 site_has_l1, site_has_spin, site_has_disp, site_active,
                 n_active, n_spin_active, n_disp_active,
                 dcomp_ptr, dcomp_sites, ncomp, zeros(3, ncomp), trues(3, ncomp),
                 0.0, true,
                 progs, n_colors, color_ptr, color_sites)
        # PER COMPONENT, and kept per component. Two components share no instance, so
        # their rigid shifts are independent symmetries and a single global verdict is
        # the wrong shape: on a mixed model (one flat component, one genuinely pinned)
        # a global `false` would disable re-centring for the flat component too, and its
        # frame would then drift without bound — which costs the reporting convention
        # AND numerical conditioning (see `_recenter!`).
        resid_c = ncomp == 0 ? zeros(3, 0) : _translation_residuals(H0)
        comp_free = resid_c .<= _TRANSLATION_TOL
        resid = isempty(resid_c) ? 0.0 : maximum(resid_c)
        invariant = all(comp_free)
        # A one-site component that IS flat is a pure gauge coordinate: the energy is
        # exactly independent of that site's displacement (a "rigid shift of the
        # component" is just that one site moving). Sampling it would spend randomness
        # on always-accepted moves, dilute the acceptance statistics, and write a
        # meaningless nonzero displacement into every reported configuration — the
        # pathology the inactive-site convention exists to prevent. It also cannot arise
        # from a fitted model: a lone displacement axis is never translation-invariant,
        # so the ASR drives its coefficients to zero and it is pruned. Its presence is
        # therefore information, and the right thing is to say so.
        for c = 1:ncomp
            all(view(comp_free, :, c)) && dcomp_ptr[c + 1] - dcomp_ptr[c] == 1 &&
                throw(ArgumentError(
                    "displacement component $c is the single site " *
                    "$(dcomp_sites[dcomp_ptr[c]]) and its rigid shift is flat along " *
                    "ALL THREE Cartesian directions, so the energy does not depend on " *
                    "that site's displacement: it is a pure gauge coordinate, not a " *
                    "degree of freedom. A fitted model cannot produce this (upstream " *
                    "requires 2k + l ≥ 1 on a displacement factor, and the ASR zeroes a " *
                    "lone displacement axis), so the term list is describing something " *
                    "else — check it, or drop the axis"))
        end
        if !invariant && !fixed_reference
            throw(ArgumentError(
                "this Hamiltonian's uniform-displacement direction is not flat " *
                "(pinned direction(s) $(Tuple.(findall(!, comp_free))) as " *
                "(direction, component), worst relative residual " *
                "$(resid) > $(_TRANSLATION_TOL)): a rigid shift of " *
                "the whole crystal changes the energy as much as a generic distortion " *
                "of the same size. The displacement sampler needs that direction to be " *
                "a symmetry (it re-centres along it, and the fit's trust region is " *
                "centre-of-mass-free), so it refuses this model.\n" *
                "  * If the model is meant to be translation-invariant, refit under " *
                "the acoustic sum rule: `fit(...; asr = true)` (the default).\n" *
                "  * If the absolute position IS physical — a substrate-clamped slab, " *
                "a pinning defect, anything with a fixed external reference — pass " *
                "`fixed_reference = true`. Re-centring is then disabled and the " *
                "displacement guard works in the absolute frame."))
        end
        return new(n_cell_atoms, d, n_sites, lmax, nlm, layout.nrows, disp_lmax,
                   layout, terms,
                   inst_term, inst_ptr, inst_sites, site_ptr, site_inst, site_slot,
                   site_has_l1, site_has_spin, site_has_disp, site_active,
                   n_active, n_spin_active, n_disp_active,
                   dcomp_ptr, dcomp_sites, ncomp, resid_c, comp_free,
                   resid, invariant,
                   progs, n_colors, color_ptr, color_sites)
    end
end

# Connected components of the displacement-coupling graph: disp-active sites joined
# when some instance carries a DISP axis on both. Union-find with path halving, then a
# CSR relabel in site order so the component layout is a deterministic function of `H`
# (the re-centring loop walks it, and P6 needs a fixed order).
function _disp_components(n_sites::Int, terms::Vector{ScaledTerm},
                          inst_term::Vector{Int32}, inst_ptr::Vector{Int32},
                          inst_sites::Vector{Int32}, site_has_disp::Vector{Bool})
    parent = collect(1:n_sites)
    find(x) = (while parent[x] != x
                   parent[x] = parent[parent[x]]
                   x = parent[x]
               end; x)
    for i in eachindex(inst_term)
        t = terms[inst_term[i]]
        off = Int(inst_ptr[i]) - 1
        anchor = 0
        for sl in t.slots
            sl.spin && continue
            s = Int(inst_sites[off + sl.site])
            if anchor == 0
                anchor = s
            else
                ra, rs = find(anchor), find(s)
                ra == rs || (parent[rs] = ra)
            end
        end
    end
    label = zeros(Int32, n_sites)
    ncomp = 0
    for s = 1:n_sites                       # site order ⇒ deterministic labelling
        site_has_disp[s] || continue
        r = find(s)
        label[r] == 0 && (ncomp += 1; label[r] = Int32(ncomp))
        label[s] = label[r]
    end
    counts = zeros(Int32, ncomp)
    for s = 1:n_sites
        site_has_disp[s] && (counts[label[s]] += 1)
    end
    ptr = Vector{Int32}(undef, ncomp + 1)
    ptr[1] = 1
    for c = 1:ncomp
        ptr[c + 1] = ptr[c] + counts[c]
    end
    sites = Vector{Int32}(undef, ptr[ncomp + 1] - 1)
    cursor = copy(@view ptr[1:ncomp])
    for s = 1:n_sites                       # ascending within each component
        site_has_disp[s] || continue
        c = label[s]
        sites[cursor[c]] = Int32(s)
        cursor[c] += 1
    end
    return ncomp, ptr, sites
end

# Relative flatness of the uniform-shift direction, measured on the Hamiltonian itself
# rather than inferred from the fit: shift one displacement component rigidly by `t` and
# compare the energy change against the change a GENERIC distortion of the same size
# makes. Under the ASR the ratio is ~1e-16; without it, ~1. Dimensionless and
# scale-free, so no energy unit or displacement scale has to be assumed — and it tests
# the property the sampler actually depends on, on every construction path (a hand-built
# term list has no `asr_residual` to consult).
const _TRANSLATION_TOL = 1e-10

function _translation_residuals(H::TiledHamiltonian)::Matrix{Float64}
    cfg, u = _probe_state(H)
    E0 = _total_energy(H, _zrows(H, cfg, u))
    resid = zeros(3, H.n_disp_comps)
    ushift = similar(u)
    for t in (0.02, 0.2)
        # Normalize PER COMPONENT, not against the largest component's scale.
        # Per-component invariance is strictly stronger than the ASR's global
        # statement: shifting one component alone changes E by `a_c·t` with only
        # `Σ_c a_c = 0` guaranteed, so a small, weakly-coupled component can carry a
        # real net force and still look flat against a dominant component's
        # denominator — and `_recenter!` would then apply a genuinely biasing
        # projection to exactly that component.
        #
        # The denominator is `max(den, |E0|, tiny)`: `den` is the physical scale, and the
        # `|E0|` floor keeps a genuinely flat component's ratio at roundoff (`num ~
        # eps·|E0|`) instead of 0/0. Note that floor is a GLOBAL energy, so on a large
        # supercell a small component's residual is scaled down by it — conservative for
        # the flat case, and the reason the tolerance is 1e-10 rather than something tight.
        for c = 1:H.n_disp_comps
            copyto!(ushift, u)
            for q = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)
                s = Int(H.disp_comp_sites[q])
                ushift[s] += t * SVector(0.6 * sin(1.7s), -0.8 * cos(2.3s),
                                         0.5 * sin(0.9s + 1.1))   # generic
            end
            scale = max(abs(_total_energy(H, _zrows(H, cfg, ushift)) - E0),
                        abs(E0), 1e-300)
            # ONE RIGID PROBE PER CARTESIAN DIRECTION. A component's flat directions can
            # be a proper subspace of the three: a substrate-clamped slab is pinned along
            # the surface normal and free in the plane. A single probe direction would
            # report the whole component as pinned, `_recenter!` would skip it entirely,
            # and the in-plane centre of mass would then random-walk without bound — with
            # nothing to catch it, since free diffusion sits below the escape detector's
            # threshold by construction.
            for d = 1:3
                copyto!(ushift, u)
                ê = SVector{3,Float64}(ntuple(i -> i == d ? 1.0 : 0.0, 3))
                for q = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)
                    ushift[Int(H.disp_comp_sites[q])] += t * ê
                end
                num = abs(_total_energy(H, _zrows(H, cfg, ushift)) - E0)
                resid[d, c] = max(resid[d, c], num / scale)
            end
        end
    end
    return resid
end

# A deterministic probe state — no RNG, so construction stays a pure function of the
# term list. Spins are a golden-angle spiral (unit by construction); displacements are a
# small incommensurate pattern.
function _probe_state(H::TiledHamiltonian)
    cfg = SpinConfig(undef, H.n_sites)
    u = Vector{SVector{3,Float64}}(undef, H.n_sites)
    for s = 1:H.n_sites
        φ = 2.399963229728653 * s
        z = 1 - 2 * (s - 0.5) / H.n_sites
        r = sqrt(max(0.0, 1 - z * z))
        cfg[s] = SVector(r * cos(φ), r * sin(φ), z)
        u[s] = 0.05 * SVector(sin(0.7s), cos(1.3s), sin(2.1s + 0.4))
    end
    return cfg, u
end

# --- term ingest: the two upstream introspection surfaces -> ScaledTerm --------------

# The pure-spin row layout of a `MultipoleTerm` list: the SPIN block alone, with `lmax`
# read off the surviving terms exactly as it was before the displacement channel
# existed. (`row_layout(model)` would instead cover every factor the BASIS can build,
# which for a pure-spin model is a superset — correct, but it would widen the row
# tables of every existing production run. The frozen surface keeps the frozen layout.)
_spin_row_layout(lmax::Int)::RowLayout =
    RowLayout(Harmonics.num_lm(lmax), lmax, Harmonics.num_lm(lmax),
              Tuple{Int,Int}[], Int[])

function TiledHamiltonian(n_cell_atoms::Integer, mterms::Vector{MultipoleTerm};
                          dims::NTuple{3,Integer} = (1, 1, 1),
                          fixed_reference::Bool = false)
    # A coef == 0 term contributes nothing to any energy, coefficient vector, or
    # gradient — drop it so "no adjacent instance" means "state-independent site".
    mterms = filter(t -> t.coef != 0.0, mterms)
    lmax = 0
    terms = Vector{ScaledTerm}(undef, length(mterms))
    for (k, mt) in enumerate(mterms)
        body = length(mt.atoms)
        (length(mt.shifts) == body && length(mt.ls) == body) ||
            throw(ArgumentError("term $k: atoms/shifts/ls lengths disagree"))
        all(l -> l >= 0, mt.ls) ||
            throw(ArgumentError("term $k: negative angular momentum in $(mt.ls)"))
        lmax = max(lmax, maximum(mt.ls))
        # axis i IS site i, and the SPIN block is `lm_index` at offset 0, so the row of
        # component m is lm_index(l, m) = row0 + m + l + 1 with row0 = lm_index(l, −l) − 1
        slots = [TermSlot(i, Harmonics.lm_index(mt.ls[i], -mt.ls[i]) - 1, mt.ls[i], true)
                 for i in eachindex(mt.ls)]
        # The package's single scale-application site for this surface.
        terms[k] = ScaledTerm(mt.coef * (4π)^(body / 2), copy(mt.atoms), copy(mt.shifts),
                              slots, copy(mt.folded))
    end
    return TiledHamiltonian(n_cell_atoms, terms, _spin_row_layout(lmax); dims = dims,
                            fixed_reference = fixed_reference)
end

function TiledHamiltonian(n_cell_atoms::Integer, dterms::Vector{DecoratedTerm},
                          layout::RowLayout; dims::NTuple{3,Integer} = (1, 1, 1),
                          fixed_reference::Bool = false)
    dterms = filter(t -> t.coef != 0.0, dterms)
    terms = Vector{ScaledTerm}(undef, length(dterms))
    for (k, dt) in enumerate(dterms)
        length(dt.slots) == ndims(dt.folded) || throw(ArgumentError(
            "term $k: $(length(dt.slots)) slots but a rank-$(ndims(dt.folded)) folded " *
            "tensor"))
        slots = Vector{TermSlot}(undef, length(dt.slots))
        for (v, sl) in enumerate(dt.slots)
            l = sl.factor.l
            row0 = row_index(layout, sl.factor, -l) - 1
            # both channels number their 2l+1 components contiguously and ascending in
            # m — the assumption every `row0 + m + l + 1` in this package rests on
            row_index(layout, sl.factor, l) == row0 + 2l + 1 || throw(ArgumentError(
                "term $k slot $v: the layout does not number $(sl.factor) contiguously"))
            slots[v] = TermSlot(sl.site, row0, l, sl.factor.channel == SLCE.SPIN)
        end
        # The package's single scale-application site for this surface — `scale` is the
        # general (4π)^(n_spin_slots/2), read off the field and NEVER re-derived from
        # the cluster shape (SLCE's DecoratedTerm docstring, design record §7).
        terms[k] = ScaledTerm(dt.coef * dt.scale, copy(dt.atoms), copy(dt.shifts),
                              slots, copy(dt.folded))
    end
    return TiledHamiltonian(n_cell_atoms, terms, layout; dims = dims,
                            fixed_reference = fixed_reference)
end

function TiledHamiltonian(model::SLCEModel; dims::NTuple{3,Integer} = (1, 1, 1),
                          fixed_reference::Bool = false)
    layout = row_layout(model)
    # A model with no displacement rows goes down the frozen pure-spin path, byte for
    # byte as before M4. `restrict` is a no-op on a genuinely pure-spin basis and the
    # exact u = 0 sub-model otherwise, which is what makes `multipole_terms` — whose
    # refusal is triggered by the SPEC, not by the surviving coefficients — accept the
    # pathological "declared a displacement sector, built no displacement SALC" case.
    isempty(layout.disp_factors) &&
        return TiledHamiltonian(n_atoms(model), multipole_terms(restrict(model, :spin));
                                dims = dims, fixed_reference = fixed_reference)
    return TiledHamiltonian(n_atoms(model), decorated_terms(model), layout; dims = dims,
                            fixed_reference = fixed_reference)
end

"""
    has_disp(H::TiledHamiltonian) -> Bool

Whether `H` carries displacement rows — i.e. whether its energy is a function of the
displacements as well as the spins. `false` for every pure-spin model, in which case
`H.nrows == H.nlm` and the row tables are exactly the pre-M4 tesseral ones.
"""
has_disp(H::TiledHamiltonian)::Bool = !isempty(H.layout.disp_factors)

# Guard for the surfaces that are still spin-only (chain state, sweeps, descent, GPU):
# refuse a joint Hamiltonian loudly instead of silently sampling at u = 0.
function _require_spin_only(H::TiledHamiltonian, what::AbstractString)
    has_disp(H) && throw(ArgumentError(
        "$what does not support displacement-decorated Hamiltonians yet; this model " *
        "carries displacement rows $(H.layout.disp_factors). Use " *
        "`total_energy(H, config, disps)` for energies, or restrict the model to its " *
        "clamped-ion sub-model with `SLCE.restrict(model, :spin)` to sample spins " *
        "at the ideal lattice"))
    return nothing
end

# The mirror guard, for the surfaces that are only meaningful WITH displacements: a
# pure-spin model has no displacement Hessian, and returning an empty matrix would let
# `eigmin` of nothing read as a clean stability verdict.
function _require_disp(H::TiledHamiltonian, what::AbstractString)
    # `n_disp_active`, not `has_disp`: the latter is a property of the row LAYOUT, and
    # a joint basis whose displacement couplings all fitted to zero has displacement
    # rows and not one displacement-active site. There the honest answer is a 0×0
    # matrix — which is exactly what would let `eigmin` of nothing read as a clean
    # stability verdict.
    H.n_disp_active > 0 || throw(ArgumentError(
        "$what needs a Hamiltonian with at least one displacement-active site; this " *
        "one has $(has_disp(H) ? "displacement rows but no site whose energy depends " *
        "on its displacement (every displacement coupling fitted to zero)" :
        "no displacement rows at all and describes the clamped-ion (u = 0) energy " *
        "only"), so there is no displacement curvature to report"))
    return nothing
end

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
          ", lmax=", H.lmax,
          has_disp(H) ? ", disp $(H.layout.disp_factors)" : "",
          ", ", length(H.terms), " terms, ",
          length(H.inst_term), " instances",
          H.translation_invariant ? "" : ", fixed reference", ")")

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
