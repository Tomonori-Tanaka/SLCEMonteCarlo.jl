# The strain schedule — the sampler's view of a K(ε) volume grid (SLCE design record
# §8's outer strain move, §9a's grid).
#
# WHAT THIS TYPE IS FOR, AND WHY IT IS NOT A `StrainedModels`. A volume grid upstream is a
# set of fitted models plus an interpolation. What a strain move needs, once per proposal,
# is one vector of raw coefficients indexed exactly the way `set_coefficients!` indexes
# them, one `j0`, and a domain to reject outside of. Reaching that through
# `SLCE.model_at(sm, s)` per proposal would rebuild a `Crystal`, a `BasisSpec` and an
# `SLCEBasis`, re-solve a dense Vandermonde over every SALC, and re-enumerate every cluster
# member — orders of magnitude above the 0.033 ms in-place rewrite this whole path exists to
# exploit. So the grid is converted ONCE, into polynomial coefficients over the term list
# `H` was actually built from, and evaluated by a Horner pass per proposal.
#
# WHAT THE CONVERSION VALIDATES, AND WHY IT MUST. `decorated_terms` prunes exactly-zero
# coefficients unless asked not to, which makes its index → SALC map a function of the
# coefficient VALUES (SLCE `introspect.jl`; `keep_zero` exists for this). Two grid points
# whose sparse fits zero DIFFERENT keys then emit equal-length lists with shifted maps, and
# a hot-swap writes each coefficient onto a neighbouring cluster with every length check
# passing. This constructor therefore builds every point's list with `keep_zero = true` and
# checks, term by term, that the atoms and images agree across the grid AND agree with what
# `H` holds. That check is the reason this type exists as a value rather than as a closure.
#
# WHAT IT DELIBERATELY DOES NOT HOLD. Not the models, not `H`, and not a scale of its own:
# the current scale is chain state (it moves with the replica), the coefficients are `H`'s,
# and the two must be written together or not at all. Keeping the schedule immutable and
# scale-free is what stops a caller from installing coefficients for one volume while the
# displacements still belong to another.

"""
    StrainSchedule

The sampler-side form of a `SLCE.StrainedModels` volume grid: polynomial coefficients over
the term list a [`TiledHamiltonian`](@ref) was built from, plus everything the NPT weight
needs (`j0(s)`, the cell volume, the mobile-atom count and the dimension of the sampled
displacement space).

Immutable and scale-free by design — the current scale is chain state, not a property of
the schedule. Build it with [`StrainSchedule(sm, H)`](@ref).
"""
struct StrainSchedule
    scales::Vector{Float64}          # grid nodes, increasing; the closed domain
    abscissa::Symbol                 # :linear | :volume | :logvolume (upstream's choice)
    x0::Float64                      # abscissa centring, for conditioning
    xw::Float64                      # abscissa scaling
    coefpoly::Matrix{Float64}        # (degree+1) × n_terms, in the centred abscissa
    j0poly::Vector{Float64}          # (degree+1), per TRAINING cell
    v_train::Float64                 # training-cell volume at s = 1
    n_cells::Int                     # supercell / training cell
    n_mobile::Int                    # atoms with displacement DoF (H.n_disp_active)
    d_dim::Int                       # 3·n_mobile − count(H.comp_free): the sampled dim

    function StrainSchedule(scales::Vector{Float64}, abscissa::Symbol, x0::Float64,
                            xw::Float64, coefpoly::Matrix{Float64},
                            j0poly::Vector{Float64}, v_train::Float64, n_cells::Int,
                            n_mobile::Int, d_dim::Int)
        length(scales) >= 2 || throw(ArgumentError(
            "a strain schedule needs at least 2 grid points; got $(length(scales))"))
        issorted(scales) && allunique(scales) && all(>(0), scales) || throw(ArgumentError(
            "grid scales must be sorted, distinct and positive; got $scales"))
        abscissa in (:linear, :volume, :logvolume) ||
            throw(ArgumentError("unknown abscissa $abscissa"))
        size(coefpoly, 1) == length(j0poly) || throw(DimensionMismatch(
            "coefficient polynomials have degree $(size(coefpoly, 1) - 1) but j0's has " *
            "$(length(j0poly) - 1)"))
        xw > 0 || throw(ArgumentError("the abscissa scaling must be positive; got $xw"))
        v_train > 0 || throw(ArgumentError("v_train must be positive; got $v_train"))
        n_cells >= 1 || throw(ArgumentError("n_cells must be ≥ 1; got $n_cells"))
        n_mobile >= 0 || throw(ArgumentError("n_mobile must be ≥ 0; got $n_mobile"))
        0 <= d_dim <= 3 * n_mobile || throw(ArgumentError(
            "d_dim = $d_dim is outside [0, 3·n_mobile] = [0, $(3 * n_mobile)]"))
        return new(scales, abscissa, x0, xw, coefpoly, j0poly, v_train, n_cells,
                   n_mobile, d_dim)
    end
end

Base.length(sch::StrainSchedule) = length(sch.scales)

function Base.show(io::IO, sch::StrainSchedule)
    print(io, "StrainSchedule(", length(sch), " points, s ∈ [",
          round(first(sch.scales); sigdigits = 5), ", ",
          round(last(sch.scales); sigdigits = 5), "], ", size(sch.coefpoly, 2),
          " terms, D = ", sch.d_dim, ")")
end

"""
    strain_domain(sch::StrainSchedule) -> (smin, smax)

The closed interval of linear scales the schedule interpolates over. A proposal outside it
is **rejected**, never clamped: a truncating clamp is an asymmetric proposal and biases the
chain toward the boundary, and the upstream `model_at` extrapolates silently rather than
refusing.
"""
strain_domain(sch::StrainSchedule) = (first(sch.scales), last(sch.scales))

in_strain_domain(sch::StrainSchedule, s::Real) =
    first(sch.scales) <= s <= last(sch.scales)

# The abscissa, centred and scaled exactly as the upstream interpolation does — a raw
# Vandermonde in volumes of a few hundred Å³ is unusable by degree 3.
function _sch_z(sch::StrainSchedule, s::Real)::Float64
    x = sch.abscissa === :linear ? Float64(s) :
        sch.abscissa === :volume ? sch.v_train * s^3 :
        log(sch.v_train * s^3)
    return (x - sch.x0) / sch.xw
end

"""
    strain_coefficients!(dst, sch::StrainSchedule, s) -> dst

Evaluate the interpolated **raw** coefficients at linear scale `s` into `dst`, in the
indexing [`set_coefficients!`](@ref) expects (one entry per input term, including terms
whose coefficient is zero). One Horner pass per term; no allocation.
"""
function strain_coefficients!(dst::AbstractVector{Float64}, sch::StrainSchedule, s::Real)
    n = size(sch.coefpoly, 2)
    length(dst) == n || throw(DimensionMismatch(
        "dst has $(length(dst)) entries for $n terms"))
    z = _sch_z(sch, s)
    d = size(sch.coefpoly, 1)
    @inbounds for j = 1:n
        acc = sch.coefpoly[d, j]
        for p = (d - 1):-1:1
            acc = acc * z + sch.coefpoly[p, j]
        end
        dst[j] = acc
    end
    return dst
end

"""
    strain_coefficients(sch::StrainSchedule, s) -> Vector{Float64}

Allocating form of [`strain_coefficients!`](@ref) — convenient for a test or a one-off
inspection. A strain move runs once per proposal and should reuse a buffer.
"""
strain_coefficients(sch::StrainSchedule, s::Real) =
    strain_coefficients!(Vector{Float64}(undef, size(sch.coefpoly, 2)), sch, s)

"""
    strain_j0(sch::StrainSchedule, s) -> Float64

The interpolated intercept at scale `s`, **per training cell**. `j0` is normally dropped
from the sweep energy as a constant — the strain move is precisely where it stops being
one, so a strain `ΔE` must carry `n_cells · Δj0` (design record §8).
"""
function strain_j0(sch::StrainSchedule, s::Real)::Float64
    z = _sch_z(sch, s)
    acc = last(sch.j0poly)
    @inbounds for p = (length(sch.j0poly) - 1):-1:1
        acc = acc * z + sch.j0poly[p]
    end
    return acc
end

"""
    strain_volume(sch::StrainSchedule, s) -> Float64

The **supercell** volume at scale `s`. `P·V` and the Jacobian's mobile-atom count are
supercell quantities while `j0` is per training cell — mixing the two is a factor
`n_cells` error in the applied pressure (design record §8).
"""
strain_volume(sch::StrainSchedule, s::Real)::Float64 =
    sch.n_cells * sch.v_train * s^3

"""
    StrainSchedule(sm::SLCE.StrainedModels, H::TiledHamiltonian) -> StrainSchedule

Convert a volume grid into the sampler's form, once, against the Hamiltonian that will
consume it.

Every grid point's term list is built with `keep_zero = true`, because the default prune
makes the index → SALC map depend on the coefficient values and two points whose sparse
fits zero different keys would otherwise produce equal-length lists with shifted maps
(SLCE `decorated_terms`). The constructor then checks, term by term, that the atoms and
lattice images agree across the whole grid *and* agree with what `H` holds — so a
Hamiltonian built from a different model, a different supercell, or without
`keep_zero_terms = true` is refused here rather than silently mis-assigned later.

`H.n_disp_active` and `H.comp_free` supply the Jacobian's `D = 3·N_mob − count(comp_free)`:
the sampler re-centres per (Cartesian direction, displacement component), so each free
direction removes ONE dimension and there are `n_disp_comps` independent centres of mass —
`D` is not `3(N − 1)` and need not be a multiple of 3.
"""
function StrainSchedule(sm, H::TiledHamiltonian)
    ms = sm.models
    npt = length(ms)
    lists = [SLCE.decorated_terms(m; keep_zero = true) for m in ms]
    nterm = length(lists[1])
    nterm == H.n_input_terms || throw(ArgumentError(
        "this Hamiltonian was built from $(H.n_input_terms) input terms, but the grid's " *
        "models have $nterm. A strain schedule needs the Hamiltonian built from the same " *
        "grid, with `TiledHamiltonian(model; keep_zero_terms = true)` and a model whose " *
        "term list came from `decorated_terms(model; keep_zero = true)` — the default " *
        "prune drops exactly-zero coefficients and makes the index map a function of the " *
        "fit rather than of the basis."))
    for i = 2:npt
        length(lists[i]) == nterm || throw(ArgumentError(
            "grid point $i has $(length(lists[i])) terms against point 1's $nterm"))
        for j = 1:nterm
            (lists[i][j].atoms == lists[1][j].atoms &&
             lists[i][j].shifts == lists[1][j].shifts) || throw(ArgumentError(
                "grid point $i's term $j sits on different atoms/images than point 1's. " *
                "The grid must be built by SCALING one reference cell; independently " *
                "standardized cells can permute the term order while every count matches."))
        end
    end
    # ...and that `H` is this grid's Hamiltonian, not another model's with the same count.
    @inbounds for k in eachindex(H.terms)
        src = H.term_source[k]
        t, d = H.terms[k], lists[1][H.term_source[k]]
        (t.atoms == d.atoms && t.shifts == d.shifts) || throw(ArgumentError(
            "the Hamiltonian's term $k (input index $src) sits on atoms $(t.atoms) at " *
            "$(t.shifts), while the grid's term $src is on $(d.atoms) at $(d.shifts). " *
            "This Hamiltonian was not built from this grid."))
    end

    scales = collect(Float64, SLCE.scales(sm))
    # ASR hard-error at conversion (design record §8): the sampler re-centres along
    # H's flat directions and the NPT weight carries `D` — properties the WHOLE family
    # must share, not just the node `H` was built from. Flatness is linear in the
    # coefficients, so flat at every node ⇒ flat for every interpolated model; a fully
    # invariant `H` therefore requires every node to build cleanly (the 1×1×1
    # constructor measures and refuses). A pinned `H` (`fixed_reference`) declared an
    # absolute frame, and its family's flatness pattern is the caller's contract.
    if H.n_disp_comps > 0 && H.translation_invariant
        for i = 1:npt
            try
                TiledHamiltonian(ms[i]; dims = (1, 1, 1), keep_zero_terms = true)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError(
                    "grid point $i (scale $(scales[i])) is not translation-flat " *
                    "while the Hamiltonian is: the strain move would sample a " *
                    "re-centred, COM-free ensemble at a volume where a rigid shift " *
                    "is not a symmetry. Refit that grid point under the ASR " *
                    "(`fit(...; asr = true)`), or build the Hamiltonian with " *
                    "`fixed_reference = true` if the absolute frame is physical.\n" *
                    "  underlying: " * err.msg))
            end
        end
    end
    v_train = abs(det(ms[1].basis.crystal.lattice.vectors)) / scales[1]^3
    natom = SLCE.n_atoms(ms[1].basis.crystal)
    ncell_f = H.n_sites / natom
    isinteger(ncell_f) || throw(ArgumentError(
        "the Hamiltonian has $(H.n_sites) sites, which is not a whole number of the " *
        "grid's $(natom)-atom cells. `j0` is per TRAINING cell, so this ratio has to be " *
        "an integer — note `prod(dims)` is NOT it when `reduce_cell` was used."))
    x = [_sch_abscissa(sm.abscissa, v_train, s) for s in scales]
    x0 = sum(x) / npt
    xw = maximum(abs, x .- x0)
    xw = xw == 0 ? 1.0 : xw
    z = (x .- x0) ./ xw
    deg = sm.degree
    V = [z[i]^(j - 1) for i = 1:npt, j = 1:(deg + 1)]
    B = Matrix{Float64}(undef, npt, nterm + 1)
    for i = 1:npt
        B[i, 1] = ms[i].j0
        for j = 1:nterm
            B[i, j + 1] = lists[i][j].coef
        end
    end
    C = V \ B
    return StrainSchedule(scales, sm.abscissa, x0, xw, Matrix(C[:, 2:end]), Vector(C[:, 1]),
                          v_train, Int(ncell_f), H.n_disp_active,
                          3 * H.n_disp_active - count(H.comp_free))
end

_sch_abscissa(abscissa::Symbol, v_train::Float64, s::Real)::Float64 =
    abscissa === :linear ? Float64(s) :
    abscissa === :volume ? v_train * s^3 :
    abscissa === :logvolume ? log(v_train * s^3) :
    throw(ArgumentError("unknown abscissa $abscissa"))

# --------------------------------------------------------------------------------------
# The energy contract and the NPT weight — pure arithmetic, no chain state. The strain
# move composes these; the gates compare them against from-scratch total energies and a
# hand-derived closed form (design record §8, §12 gate (l)).

"""
    strain_delta_energy(sch::StrainSchedule, e_old, e_new, s_old, s_new;
                        pressure::Float64) -> Float64

The full energy difference entering the NPT accept step for a strain proposal
`s_old → s_new`:

    ΔE = (e_new − e_old) + n_cells·[j0(s_new) − j0(s_old)] + P·[V(s_new) − V(s_old)]

`e_old` / `e_new` are the **configurational** energies ([`total_energy`](@ref)) of the
same Hamiltonian with the schedule's coefficients installed at each scale and the
displacements rescaled **affinely** by `s_new/s_old` — at fixed scaled coordinates the
configurational measure is unchanged, which is what makes the acceptance Jacobian a
pure volume power rather than `(V′/V)^N`.

The lattice's elastic energy has exactly ONE source: the grid's `j0(s)`, interpolated
per training cell and multiplied up by `n_cells` here. There is no separate elastic
term anywhere in this package, by design — each grid point's fit already contains the
lattice energy of its own cell, and an explicit `(V/2)·εᵀCε` on top would count it
twice with every gate green.

`pressure` is in the model's units (eV/Å³ for a DFT-fitted model) and multiplies the
**supercell** volume. Driver keywords accept GPa and convert once, at resolution —
this function never converts.
"""
function strain_delta_energy(sch::StrainSchedule, e_old::Real, e_new::Real,
                             s_old::Real, s_new::Real; pressure::Float64)::Float64
    isfinite(pressure) || throw(ArgumentError("pressure must be finite; got $pressure"))
    dj0 = sch.n_cells * (strain_j0(sch, s_new) - strain_j0(sch, s_old))
    pdv = pressure * (strain_volume(sch, s_new) - strain_volume(sch, s_old))
    return (e_new - e_old) + dj0 + pdv
end

# The two proposal arms. A strain proposal is a symmetric step in a variable `y`; the
# accept weight then carries the volume power `D/3` from the configurational measure
# PLUS `ln|dV/dy|` from the proposal variable itself:
#
#     :logvolume   y = ln V   power D/3 + 1
#     :scale       y = s      power D/3 + 2/3
#
# so `ln A = 3·power·ln(s′/s) − ΔE/kT` with ΔE from `strain_delta_energy`. The three
# functions below branch on the SAME symbol and sit adjacent on purpose: drawing in one
# arm's `y` while weighting with the other arm's power is §8(β)'s live trap — the grid
# is labelled by `s`, so "uniform in s with the volume-uniform exponent" is off by
# `(V′/V)^{2/3}`, invisible on a production cell and percent-level on the fixtures the
# gates run on.

const _STRAIN_PROPOSALS = (:logvolume, :scale)

function _strain_check_proposal(proposal::Symbol)
    proposal in _STRAIN_PROPOSALS || throw(ArgumentError(
        "unknown strain proposal $proposal; use :logvolume (symmetric step in ln V) " *
        "or :scale (symmetric step in the linear scale s)"))
    return proposal
end

# `y(s)` and its inverse — the move draws `y′ = y(s) + step·ξ` and maps back. For
# :logvolume the additive constant `ln(n_cells·v_train)` is dropped: a symmetric random
# walk is invariant under it.
_strain_y(proposal::Symbol, s::Real)::Float64 =
    proposal === :logvolume ? 3 * log(s) : Float64(s)

_strain_s_of_y(proposal::Symbol, y::Real)::Float64 =
    proposal === :logvolume ? exp(y / 3) : Float64(y)

_strain_power(sch::StrainSchedule, proposal::Symbol)::Float64 =
    sch.d_dim / 3 + (proposal === :logvolume ? 1.0 : 2 / 3)

"""
    _strain_log_weight(sch, proposal, s_old, s_new, delta_e, kt) -> Float64

The log of the Metropolis acceptance ratio for a strain proposal drawn symmetrically in
`proposal`'s variable: `3·power·ln(s_new/s_old) − delta_e/kt`, where `delta_e` is
[`strain_delta_energy`](@ref)'s full ΔE and the power is `D/3 + 1` (`:logvolume`) or
`D/3 + 2/3` (`:scale`) with `D = 3·n_mobile − count(comp_free)`. Accept with
probability `min(1, exp(·))`.
"""
function _strain_log_weight(sch::StrainSchedule, proposal::Symbol, s_old::Real,
                            s_new::Real, delta_e::Real, kt::Real)::Float64
    _strain_check_proposal(proposal)
    return 3 * _strain_power(sch, proposal) * log(s_new / s_old) - delta_e / kt
end

# --------------------------------------------------------------------------------------
# The outer strain move.

# Mutable: an accepted move SWAPS the proposal buffers with the chain's state arrays
# (reference swaps, no copy), so the scratch's fields are rebound per acceptance.
"""
    StrainScratch(H::TiledHamiltonian)

Preallocated buffers for [`strain_move!`](@ref): the raw-coefficient vector, the
proposed displacements, a full proposed row table, and the solid-harmonic batch
workspace. One per chain — the strain move is an outer, serial step, never run inside
the color-parallel sweep layer.
"""
mutable struct StrainScratch
    unew::Vector{SVector{3,Float64}}
    zrows::Matrix{Float64}
    const coef::Vector{Float64}
    const rbuf::Vector{Float64}
end

StrainScratch(H::TiledHamiltonian) =
    StrainScratch(Vector{SVector{3,Float64}}(undef, H.n_sites),
                  Matrix{Float64}(undef, H.nrows, H.n_sites),
                  Vector{Float64}(undef, H.n_input_terms),
                  Vector{Float64}(undef, max(0, (H.disp_lmax + 1)^2)))

"""
    strain_move!(st::ChainState, H::TiledHamiltonian, sch::StrainSchedule,
                 sc::StrainScratch, kt::Real, pressure::Real, step::Real;
                 proposal::Symbol = :logvolume) -> Bool

One isothermal–isobaric (NPT) strain move: propose a new linear cell scale by a
symmetric step of width `step` in `proposal`'s variable, rescale the displacements
**affinely** (`u → (s′/s)·u`, fixed scaled coordinates), install the schedule's
coefficients for the proposed scale, and accept by `_strain_log_weight`'s Metropolis
rule. Returns whether the move was accepted.

The contract with the caller: on entry `H`'s coefficients are the schedule's at
`st.strain`, and on exit they are the schedule's at the NEW `st.strain` — accepted or
not, the pair `(H, st)` stays consistent, so the sweep layer can run in between with
no knowledge that the cell moves. A proposal outside [`strain_domain`](@ref) is
**rejected**, never clamped (a truncating clamp is an asymmetric proposal and biases
the chain toward the boundary). Draws come from the chain-level `st.rng` only — one
normal per attempt, plus one uniform when the proposal lands inside the domain.

`pressure` is hydrostatic, in the model's units (eV/Å³): `P·V(ε)` is a state function
with no strain-measure ambiguity, which is why v0 is hydrostatic-only — a general
applied stress is work-conjugate to a specific strain measure and would reopen the
measure choice. A run WITHOUT this move samples the constant-strain (fixed-cell)
ensemble — a different ensemble giving `F(T, ε)` with neither the volume Jacobian nor
the `P·V` term, which is what magnetostriction under fixed geometry wants; the two
specific heats differ.

An accepted move is a phase boundary for the escape detector (`_reset_escape!`): its
radius statistics are absolute lengths, and the cell they are measured against has
just changed.
"""
function strain_move!(st::ChainState, H::TiledHamiltonian, sch::StrainSchedule,
                      sc::StrainScratch, kt::Real, pressure::Real, step::Real;
                      proposal::Symbol = :logvolume)::Bool
    _strain_check_proposal(proposal)
    kt > 0 || throw(ArgumentError("kt must be > 0; got $kt"))
    step > 0 || throw(ArgumentError("the strain proposal width must be > 0; got $step"))
    s = st.strain
    in_strain_domain(sch, s) || throw(ArgumentError(
        "the chain sits at scale $s, outside the schedule's domain " *
        "$(strain_domain(sch)): the Hamiltonian and the schedule disagree about " *
        "which grid this chain samples"))
    st.att_strain += 1
    sp = _strain_s_of_y(proposal, _strain_y(proposal, s) + step * randn(st.rng))
    in_strain_domain(sch, sp) || return false
    lam = sp / s

    # The proposed state: affinely rescaled displacements, their rows, the proposed
    # scale's coefficients. Spin rows are untouched by a cell rescale, so the copy
    # carries them and only the displacement blocks are refilled — the same fill
    # `_zrows` runs, so the proposed table is bit-identical to a fresh build.
    @inbounds for i in eachindex(st.disps)
        sc.unew[i] = lam * st.disps[i]
    end
    copyto!(sc.zrows, st.zrows)
    if has_disp(H)
        for i = 1:H.n_sites
            _disp_rows!(view(sc.zrows, :, i), H, sc.unew[i], sc.rbuf)
        end
    end
    strain_coefficients!(sc.coef, sch, sp)
    # Family flatness is established once, at `StrainSchedule` construction — the
    # per-proposal recheck would re-measure a property linear in the coefficients.
    set_coefficients!(H, sc.coef; recheck_translation = false)
    e_new = _total_energy(H, sc.zrows)
    de = strain_delta_energy(sch, st.energy, e_new, s, sp;
                             pressure = Float64(pressure))
    if log(rand(st.rng)) < _strain_log_weight(sch, proposal, s, sp, de, kt)
        st.strain = sp
        st.energy = e_new
        st.disps, sc.unew = sc.unew, st.disps
        st.zrows, sc.zrows = sc.zrows, st.zrows
        # the re-centring record is a set of absolute lengths in the same frame
        @inbounds for c in eachindex(st.com_removed)
            st.com_removed[c] = lam * st.com_removed[c]
        end
        st.acc_strain += 1
        _reset_escape!(st)
        return true
    end
    # Reject: reinstall the CURRENT scale's coefficients. The Horner pass is
    # deterministic, so the restore is bit-identical — no rollback buffer needed.
    strain_coefficients!(sc.coef, sch, s)
    set_coefficients!(H, sc.coef; recheck_translation = false)
    return false
end
