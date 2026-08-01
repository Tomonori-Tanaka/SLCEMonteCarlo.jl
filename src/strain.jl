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
needs (`j0(s)`, the cell volume, and the displacement-active site count `n_mobile`
that sets the volume power).

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
    n_mobile::Int                    # displacement-active SITE count (H.n_disp_active)
    d_dim::Int                       # 3·n_mobile − count(H.comp_free): the sampled dim
    term_fp::UInt64                  # structural identity of H's term list

    function StrainSchedule(scales::Vector{Float64}, abscissa::Symbol, x0::Float64,
                            xw::Float64, coefpoly::Matrix{Float64},
                            j0poly::Vector{Float64}, v_train::Float64, n_cells::Int,
                            n_mobile::Int, d_dim::Int, term_fp::UInt64)
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
                   n_mobile, d_dim, term_fp)
    end
end

# The structural identity of the term list a Hamiltonian was built from: input index,
# atoms and lattice images per template term (FNV-1a, same mixer as the checkpoint
# fingerprints). `_check_strain_pairing` compares it, because the constructor's deep
# per-term check runs against SOME Hamiltonian — not necessarily the one a later run
# passes — and counts alone cannot tell two same-shape models apart.
function _schedule_term_fp(H::TiledHamiltonian)::UInt64
    h = 0xcbf29ce484222325
    @inbounds for k in eachindex(H.terms)
        h = _fingerprint_mix(h, Int(H.term_source[k]))
        t = H.terms[k]
        for a in t.atoms
            h = _fingerprint_mix(h, a)
        end
        for sv in t.shifts
            h = _fingerprint_mix(h, sv[1])
            h = _fingerprint_mix(h, sv[2])
            h = _fingerprint_mix(h, sv[3])
        end
    end
    return h
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

"""
    in_strain_domain(sch::StrainSchedule, s) -> Bool

Whether the linear scale `s` lies inside [`strain_domain`](@ref)'s closed interval.
"""
in_strain_domain(sch::StrainSchedule, s::Real) =
    first(sch.scales) <= s <= last(sch.scales)

# The abscissa, centred and scaled exactly as the upstream interpolation does — a raw
# Vandermonde in volumes of a few hundred Å³ is unusable by degree 3.
function _schedule_z(sch::StrainSchedule, s::Real)::Float64
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
    z = _schedule_z(sch, s)
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
    z = _schedule_z(sch, s)
    acc = last(sch.j0poly)
    @inbounds for p = (length(sch.j0poly) - 1):-1:1
        acc = acc * z + sch.j0poly[p]
    end
    return acc
end

"""
    strain_volume(sch::StrainSchedule, s) -> Float64

The **supercell** volume at scale `s`. `P·V` and the volume power's site count are
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

`H.n_disp_active` supplies `n_mobile`, the displacement-active site count whose volume
power `V^{N_mob}` the NPT weight carries — NOT reduced by the re-centred COM
directions: those are gauge directions whose range is the cell, so quotienting them
out hides their measure factor without removing it (the ideal-gas COM factor). The
sampled dimension `D = 3·N_mob − count(comp_free)` is stored as `d_dim` for
diagnostics only.
"""
function StrainSchedule(sm::SLCE.StrainedModels, H::TiledHamiltonian)
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
    # ASR hard-error at conversion (design record §8): `_recenter!` projects along
    # H's free (direction, component) pairs at every interpolated scale, so every
    # grid node must be flat along AT LEAST those pairs — flatness is linear in the
    # coefficients, so flat at every node ⇒ flat for the whole interpolated family.
    # Three details are load-bearing. (i) The check runs at `H`'s OWN dims: the
    # component partition depends on the supercell (a chain of cross-cell pairs can
    # be one component at 1×1×1 and two at 2×1×1), and per-component flatness is
    # strictly stronger than the global sum rule. (ii) The gate is `any(comp_free)`,
    # not `translation_invariant`: a partially pinned `fixed_reference` Hamiltonian
    # (a slab — free in-plane, pinned along the normal) is still re-centred along its
    # free directions. (iii) The comparison is ONE-SIDED — a node may be flatter than
    # `H` (sampling a flat direction is merely diffusive), but a node that PINS a
    # direction `H` re-centres would get a biasing projection at that volume.
    if H.n_disp_comps > 0 && any(H.comp_free)
        for i = 1:npt
            Hn = try
                TiledHamiltonian(ms[i]; dims = Tuple(H.dims), keep_zero_terms = true,
                                 fixed_reference = true)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError(
                    "grid point $i (scale $(scales[i])) fails to build at the " *
                    "Hamiltonian's supercell:\n  " * err.msg))
            end
            all(Hn.comp_free .| .!H.comp_free) || throw(ArgumentError(
                "grid point $i (scale $(scales[i])) is not translation-flat along " *
                "direction(s) $(Tuple.(findall(H.comp_free .& .!Hn.comp_free))) " *
                "(as (direction, component)) that the Hamiltonian re-centres: the " *
                "strain move would sample a COM-projected ensemble at a volume " *
                "where that rigid shift is not a symmetry. Refit that grid point " *
                "under the ASR (`fit(...; asr = true)`), or build the Hamiltonian " *
                "with `fixed_reference = true` if the absolute frame is physical."))
        end
    end
    v_train = abs(det(ms[1].basis.crystal.lattice.vectors)) / scales[1]^3
    natom = SLCE.n_atoms(ms[1].basis.crystal)
    ncell_f = H.n_sites / natom
    isinteger(ncell_f) || throw(ArgumentError(
        "the Hamiltonian has $(H.n_sites) sites, which is not a whole number of the " *
        "grid's $(natom)-atom cells. `j0` is per TRAINING cell, so this ratio has to be " *
        "an integer. (A `reduce_cell` Hamiltonian never reaches this point — its term "*
        "count differs from the grid's, so the term check above refuses it first; "*
        "strain + reduced cells is unsupported.)"))
    x = [_schedule_abscissa(sm.abscissa, v_train, s) for s in scales]
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
                          3 * H.n_disp_active - count(H.comp_free),
                          _schedule_term_fp(H))
end

_schedule_abscissa(abscissa::Symbol, v_train::Float64, s::Real)::Float64 =
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
# accept weight carries the volume power `N_mob` of the configurational measure
# (`d^{3N}r = V^N d^{3N}x` in scaled coordinates — Frenkel–Smit's `N ln(V′/V)`, with
# `N` the displacement-ACTIVE site count) PLUS `ln|dV/dy|` from the proposal variable:
#
#     :logvolume   y = ln V   ln A carries (3·N_mob + 3)·ln(s′/s)
#     :scale       y = s      ln A carries (3·N_mob + 2)·ln(s′/s)
#
# NOT `D = 3·N_mob − count(comp_free)`: the flat COM directions are gauge directions
# whose RANGE is the cell, so quotienting them out (as `_recenter!` does) hides their
# measure factor without removing it — each flat direction still contributes one `s`
# to the target (the ideal-gas COM factor; corrected 2026-07-29, see
# `docs/specs/strain-move.md` S3). `D` survives only as the dimension of the sampled
# subspace, which the weight never needs.
#
# The functions below branch on the SAME symbol and sit adjacent on purpose: drawing
# in one arm's `y` while weighting with the other arm's exponent is §8(β)'s live trap —
# the grid is labelled by `s`, so "uniform in s with the volume-uniform exponent" is
# off by `(V′/V)^{2/3}`, invisible on a production cell and percent-level on the
# fixtures the gates run on.

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

# The coefficient of ln(s′/s) in the log weight, exact in integers.
_strain_s_exponent(sch::StrainSchedule, proposal::Symbol)::Int =
    3 * sch.n_mobile + (proposal === :logvolume ? 3 : 2)

"""
    _strain_log_weight(sch, proposal, s_old, s_new, delta_e, kt) -> Float64

The log of the Metropolis acceptance ratio for a strain proposal drawn symmetrically in
`proposal`'s variable: `(3·N_mob + 3)·ln(s_new/s_old) − delta_e/kt` for `:logvolume`
and `(3·N_mob + 2)·ln(s_new/s_old) − delta_e/kt` for `:scale`, where `delta_e` is
[`strain_delta_energy`](@ref)'s full ΔE and `N_mob = n_mobile` counts the
displacement-active sites. Accept with probability `min(1, exp(·))`.
"""
function _strain_log_weight(sch::StrainSchedule, proposal::Symbol, s_old::Real,
                            s_new::Real, delta_e::Real, kt::Real)::Float64
    _strain_check_proposal(proposal)
    return _strain_s_exponent(sch, proposal) * log(s_new / s_old) - delta_e / kt
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
                 sc::StrainScratch, kt::Real;
                 pressure::Real, step::Real, proposal::Symbol = :logvolume) -> Bool

One isothermal–isobaric (NPT) strain move: propose a new linear cell scale by a
symmetric step of width `step` in `proposal`'s variable, rescale the displacements
**affinely** (`u → (s′/s)·u`, fixed scaled coordinates), install the schedule's
coefficients for the proposed scale, and accept by `_strain_log_weight`'s Metropolis
rule. Returns whether the move was accepted. `pressure` and `step` are required
keywords — two bare positional scalars of the same type would make a transposition
undetectable.

The contract with the caller: on entry `H`'s coefficients are the schedule's at
`st.strain`, and on exit they are the schedule's at the NEW `st.strain` — accepted or
not, the pair `(H, st)` stays consistent, so the sweep layer can run in between with
no knowledge that the cell moves. All argument validation happens BEFORE the first
write to `H`, so a thrown error also leaves the pair consistent. A proposal outside
[`strain_domain`](@ref) is **rejected**, never clamped (a truncating clamp is an
asymmetric proposal and biases the chain toward the boundary). Draws come from the
chain-level `st.rng` only — one normal per attempt, plus one uniform when the
proposal lands inside the domain.

`pressure` is hydrostatic, in the model's units (eV/Å³, **never GPa** — the driver
keyword `pressure_GPa` converts once, at resolution): `P·V(ε)` is a state function
with no strain-measure ambiguity, which is why v0 is hydrostatic-only — a general
applied stress is work-conjugate to a specific strain measure and would reopen the
measure choice. A run WITHOUT this move samples the constant-strain (fixed-cell)
ensemble — a different ensemble giving `F(T, ε)` with neither the volume Jacobian nor
the `P·V` term, which is what magnetostriction under fixed geometry wants; the two
specific heats differ.

An accepted move rescales the escape **detector's** anchors together with the state:
the affine map is exactly known, so they are covariant (`rms → λ·rms`) rather than a
phase boundary to reset at — a reset would disarm the detector's block ladder on every
accepted move, i.e. permanently at the default cadence. The phase's REPORTING
accumulators (`TempResult`'s `disp_rms` and `disp_max`) are deliberately left alone, so
they stay absolute time-averages comparable with the `:u2` observable; see
`_rescale_escape!`.
"""
function strain_move!(st::ChainState, H::TiledHamiltonian, sch::StrainSchedule,
                      sc::StrainScratch, kt::Real; pressure::Real, step::Real,
                      proposal::Symbol = :logvolume, check_pairing::Bool = true)::Bool
    # The schedule↔Hamiltonian pairing, checked here because this is a PUBLIC entry
    # point: `set_coefficients!` below validates only the term COUNT, so a schedule
    # converted against a different model of the same shape would write each
    # interpolated coefficient onto another model's cluster — silently, with every
    # other diagnostic green. That is exactly what `_schedule_term_fp` exists to catch,
    # and every other public consumer of a schedule (the drivers, `resume`,
    # `energy_volume_derivative`, `pressure_diagnostics`, `npt_observables`) already
    # asks for it. `check_pairing = false` is for the in-package drivers, which check
    # once at entry — the fingerprint walks every tiled term, so it does not belong in
    # a per-move path.
    check_pairing && _check_strain_pairing(H, sch)
    _strain_check_proposal(proposal)
    kt > 0 || throw(ArgumentError("kt must be > 0; got $kt"))
    step > 0 || throw(ArgumentError("the strain proposal width must be > 0; got $step"))
    # validated HERE, not inside `strain_delta_energy`: by that point the proposed
    # scale's coefficients are already installed, and a throw would leave (H, chain)
    # inconsistent
    isfinite(pressure) || throw(ArgumentError("pressure must be finite; got $pressure"))
    s = st.strain
    in_strain_domain(sch, s) || throw(ArgumentError(
        "the chain sits at scale $s, outside the schedule's domain " *
        "$(strain_domain(sch)): the Hamiltonian and the schedule disagree about " *
        "which grid this chain samples"))
    st.att_strain += 1
    _note_strain!(st)
    sp = _strain_s_of_y(proposal, _strain_y(proposal, s) + step * randn(st.rng))
    if !in_strain_domain(sch, sp)
        st.att_strain_out += 1
        return false
    end
    lam = sp / s

    # Both sides of ΔE from the same estimator: `st.energy` is the incrementally
    # accumulated value, and mixing it with a from-scratch `e_new` would put the
    # accumulated drift into the acceptance ratio asymmetrically (an O(drift)
    # detailed-balance violation). So `e_old` is recomputed from scratch here, while
    # `H` still carries the current scale, and the drift itself is carried across an
    # accepted move unchanged — `_renormalize!`'s `max_drift` accounting keeps
    # seeing it.
    e_old = _total_energy(H, st.zrows)
    drift = st.energy - e_old

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
    de = strain_delta_energy(sch, e_old, e_new, s, sp;
                             pressure = Float64(pressure))
    if log(rand(st.rng)) < _strain_log_weight(sch, proposal, s, sp, de, kt)
        st.strain = sp
        _note_strain!(st)
        st.energy = e_new + drift
        st.disps, sc.unew = sc.unew, st.disps
        st.zrows, sc.zrows = sc.zrows, st.zrows
        # the re-centring record is a set of absolute lengths in the same frame
        @inbounds for c in eachindex(st.com_removed)
            st.com_removed[c] = lam * st.com_removed[c]
        end
        st.acc_strain += 1
        _rescale_escape!(st, lam)
        return true
    end
    # Reject: reinstall the CURRENT scale's coefficients. The Horner pass is
    # deterministic, so the restore is bit-identical — no rollback buffer needed.
    strain_coefficients!(sc.coef, sch, s)
    set_coefficients!(H, sc.coef; recheck_translation = false)
    return false
end

# --------------------------------------------------------------------------------------
# The mechanical-equilibrium identity (design record §8 (ζ)): `dE_total/dV` at a sampled
# state, exactly, and the observable pair + evaluable that turn it into the sampled
# pressure. Stationarity of the NPT marginal `p(V) ∝ V^{N_mob}·e^{−β(E_total + P·V)}`
# gives, by integration by parts in V at fixed (spins, scaled displacements),
#
#     N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩ = P_applied
#
# up to boundary terms of the schedule's bounded domain — negligible exactly when the
# chain's volume distribution is confined well inside the grid, which is the only regime
# a production NPT run is meaningful in anyway. This is the one production-scale check of
# the whole strain channel: it exercises the volume power, the elastic j0, the coefficient
# interpolation and the P·V term against each other in a single number.
#
# The derivative is EXACT, not a finite difference, via two facts. (1) The coefficients
# are polynomials in the centred abscissa — differentiate the same Horner pass
# `strain_coefficients!` runs. (2) At fixed scaled coordinates `w = u/s` the displacement
# content's explicit `s`-dependence is `Φ(s·w)`, and every displacement factor
# `|u|^{2k} R_{l,m}(u)` is homogeneous of degree `2k + l` — so by Euler's theorem the
# virial `Σ_sites u·∂Φ/∂u` of a template's contraction is `deg·Φ` with
# `deg = Σ_disp-slots (2k + l)`, a per-template integer. No displacement gradient is
# ever formed.

# d(centred abscissa)/ds — the chain-rule factor between the Horner variable `z` and the
# linear scale. Differentiating in `z` alone would be silently wrong by `xw` (and by
# `3·v·s²` on a :volume grid).
_schedule_dz_ds(sch::StrainSchedule, s::Real)::Float64 =
    (sch.abscissa === :linear ? 1.0 :
     sch.abscissa === :volume ? 3 * sch.v_train * Float64(s)^2 :
     3.0 / Float64(s)) / sch.xw          # :logvolume — d ln(v·s³)/ds = 3/s

# dJ_j/ds of every input term's interpolated raw coefficient — `strain_coefficients!`'s
# Horner pass, differentiated. A degree-0 polynomial comes out exactly 0.0 (the leading
# `(d − 1)·c_d` seed is zero).
function _strain_dcoefficients!(dst::AbstractVector{Float64}, sch::StrainSchedule,
                                s::Real)
    n = size(sch.coefpoly, 2)
    length(dst) == n || throw(DimensionMismatch(
        "dst has $(length(dst)) entries for $n terms"))
    z = _schedule_z(sch, s)
    dzds = _schedule_dz_ds(sch, s)
    d = size(sch.coefpoly, 1)
    @inbounds for j = 1:n
        acc = (d - 1) * sch.coefpoly[d, j]
        for p = (d - 1):-1:2
            acc = acc * z + (p - 1) * sch.coefpoly[p, j]
        end
        dst[j] = acc * dzds
    end
    return dst
end

# dj0/ds, per TRAINING cell — the derivative of `strain_j0`'s Horner pass.
function _strain_dj0(sch::StrainSchedule, s::Real)::Float64
    z = _schedule_z(sch, s)
    d = length(sch.j0poly)
    acc = (d - 1) * sch.j0poly[d]
    @inbounds for p = (d - 1):-1:2
        acc = acc * z + (p - 1) * sch.j0poly[p]
    end
    return acc * _schedule_dz_ds(sch, s)
end

# Per-template displacement homogeneity degree `Σ_disp-slots (2k + l)`. The `k` is
# recovered from the slot's row block: `row0` IS the layout's `disp_starts` entry for
# its `(k, l)` (`SLCE.row_index`), which holds on every model-built Hamiltonian. The
# raw public constructor (`TiledHamiltonian(n, ::Vector{ScaledTerm}, layout)`) only
# range-checks `row0`, so a hand-built slot pointing mid-block CAN reach the throws
# below — but such a slot already reads wrong `(k, l)` rows in `_disp_rows!`, so the
# loud error here is strictly better than the silent misread it accompanies.
function _term_disp_degrees(H::TiledHamiltonian)::Vector{Int}
    L = H.layout
    deg = zeros(Int, length(H.terms))
    for (t, term) in enumerate(H.terms)
        d = 0
        for sl in term.slots
            sl.spin && continue
            i = findfirst(==(sl.row0), L.disp_starts)
            i === nothing && error("slot row block at $(sl.row0) is not a layout " *
                                   "displacement block start; a model-built " *
                                   "Hamiltonian cannot produce this — check any " *
                                   "hand-built ScaledTerm slots")
            k, l = L.disp_factors[i]
            l == sl.l || error("slot has l = $(sl.l) but its layout block carries " *
                               "l = $l; a model-built Hamiltonian cannot produce " *
                               "this — check any hand-built ScaledTerm slots")
            d += 2k + l
        end
        deg[t] = d
    end
    return deg
end

# `_total_energy` with an external per-template coefficient vector `g` in place of
# `pr.term_coef` — same instances, same entries, same loop order. Kept as a separate
# mirror rather than a parameterized `_total_energy` so the hot path stays untouched
# (its trajectory is pinned byte-identical); keep the two in lockstep.
function _energy_with_coefs(H::TiledHamiltonian, zrows::Matrix{Float64},
                            g::Vector{Float64})::Float64
    pr = H.progs
    E = 0.0
    @inbounds for i in eachindex(H.inst_term)
        k = Int(H.inst_term[i])
        off = Int(H.inst_ptr[i]) - 1
        Ei = 0.0
        for e = Int(pr.eprog_ptr[k]):(Int(pr.eprog_ptr[k + 1]) - 1)
            p = 1.0
            f0 = Int(pr.efac_ptr[e]) - 1
            for m = 1:(Int(pr.efac_ptr[e + 1]) - 1 - f0)
                f = f0 + m
                p *= zrows[pr.efac_row[f], H.inst_sites[off + pr.efac_site[f]]]
            end
            Ei += pr.eent_w[e] * p
        end
        E += g[k] * Ei
    end
    return E
end

# The buffer-level kernel: `dE_total/dV` at scale `s` from a filled row table. Reads
# NOTHING off `H`'s currently installed coefficients — `J(s)` comes from the schedule,
# so the result is a pure function of (schedule, state, s) whatever the caller last
# installed.
function _energy_volume_derivative(sch::StrainSchedule, H::TiledHamiltonian,
                                   zrows::Matrix{Float64}, s::Float64,
                                   coefs::Vector{Float64}, dcoef::Vector{Float64},
                                   g::Vector{Float64}, deg::Vector{Int})::Float64
    in_strain_domain(sch, s) || throw(ArgumentError(
        "scale $s is outside the schedule's domain $(strain_domain(sch)); the " *
        "interpolant does not extrapolate"))
    strain_coefficients!(coefs, sch, s)
    _strain_dcoefficients!(dcoef, sch, s)
    @inbounds for k in eachindex(g)
        src = Int(H.term_source[k])
        # d/ds of the SCALED coefficient, plus the Euler virial `deg·J/s` — both carry
        # the same `(4π)` scale the constructor applied, so it factors out front
        g[k] = H.term_scale[k] * (dcoef[src] + deg[k] * coefs[src] / s)
    end
    de_ds = _energy_with_coefs(H, zrows, g) + sch.n_cells * _strain_dj0(sch, s)
    dv_ds = 3 * sch.n_cells * sch.v_train * s^2
    return de_ds / dv_ds
end

"""
    energy_volume_derivative(sch::StrainSchedule, H::TiledHamiltonian,
                             config::SpinConfig, [disps,] s) -> Float64

`dE_total/dV` of the NPT energy at a state and linear scale `s`, in the model's units
(eV/Å³ for a DFT-fitted model): the exact derivative, at fixed spins and fixed **scaled**
displacements `w = u/s`, of `E_config + n_cells·j0(s)` with respect to the supercell
volume `V(s)`. `disps` are the Cartesian displacements **at scale `s`** (the state as
sampled); they are required exactly when `H` carries displacement rows, like
[`total_energy`](@ref).

Exact, not a finite difference: the coefficient drift is the schedule's differentiated
Horner pass, and the displacement content's affine response is Euler's theorem on each
factor's homogeneity degree `2k + l` — no displacement gradient is formed. The result
does not depend on which scale's coefficients `H` currently holds.

This is the estimator half of the §8(ζ) mechanical-equilibrium identity
`N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩ = P` — see [`pressure_diagnostics`](@ref) for the
packaged observable form. Multiply by [`GPA_PER_EV_A3`](@ref) for GPa.
"""
function energy_volume_derivative(sch::StrainSchedule, H::TiledHamiltonian,
                                  config::SpinConfig, s::Real)::Float64
    _check_strain_pairing(H, sch)      # BEFORE the O(nrows·n_sites) row-table build,
                                       # so a mispaired (sch, H) gets the pairing
                                       # message, not a zrows shape error
    return _evd_alloc(sch, H, _zrows(H, config), Float64(s))
end

function energy_volume_derivative(sch::StrainSchedule, H::TiledHamiltonian,
                                  config::SpinConfig,
                                  disps::Vector{SVector{3,Float64}}, s::Real)::Float64
    _check_strain_pairing(H, sch)
    return _evd_alloc(sch, H, _zrows(H, config, disps), Float64(s))
end

function _evd_alloc(sch::StrainSchedule, H::TiledHamiltonian, zrows::Matrix{Float64},
                    s::Float64)::Float64
    n = size(sch.coefpoly, 2)
    return _energy_volume_derivative(sch, H, zrows, s,
                                     Vector{Float64}(undef, n),
                                     Vector{Float64}(undef, n),
                                     Vector{Float64}(undef, length(H.terms)),
                                     _term_disp_degrees(H))
end

"""
    pressure_diagnostics(sch::StrainSchedule, H::TiledHamiltonian)
        -> (; observables::Vector{Observable}, evaluables::Vector{Evaluable})

The §8(ζ) mechanical-equilibrium check, packaged for a strained [`run_mc`](@ref): two
raw observables — `:strain_dEdV` ([`energy_volume_derivative`](@ref) at the sampled
state) and `:strain_invV` (`1/V`) — plus the jackknifed evaluable

    :pressure  =  N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩

in the model's units (eV/Å³; multiply by [`GPA_PER_EV_A3`](@ref) for GPa). On an
equilibrated NPT chain its mean equals the applied pressure — a disagreement beyond its
error bar means the elastic `j0` bookkeeping, the virial, the coefficient interpolation
or the `P·V` term is wrong, which is what makes this the production-scale gate of the
strain channel. (Its sensitivity to the volume-power convention itself is only
`kT·⟨1/V⟩` per unit of `N_mob` — often within an error bar; that convention is pinned
by the §8(γ) marginal gate, not here.) Append both vectors to the run's
`observables` / `evaluables`.

Two validity conditions, both properties of the run rather than of this code. The
identity holds up to **boundary terms of the schedule's bounded domain** — trust it only
when the sampled volume distribution is confined well inside [`strain_domain`](@ref)
(a chain pressed against a grid edge is answering a different question). And the virial
half reads the view's **gauge-reduced** displacements; it is gauge-invariant exactly
because the flat directions certified at schedule construction have zero force sum, so
on a healthy schedule the reduction is invisible here.

Build the diagnostic from the **same `sch` the run uses**: the view carries no
schedule, so a second, pairing-compatible schedule would silently supply its own
`J(s)` against a chain sampling the other — only the Hamiltonian identity is
checkable at measurement time.

The observables capture preallocated scratch sized for **this** `H` (a view from any
other Hamiltonian is refused) and are therefore serial — one chain at a time, i.e. a
strained [`run_mc`](@ref). They are **not usable under a strained [`run_pt`](@ref)**:
its lanes measure concurrently (a shared-scratch race) and sweep per-lane coefficient
clones, so `run_pt` refuses these observables by name at entry — and should anything
bypass that, the per-view identity check still throws on a clone's view rather than
racing silently. Each `:strain_dEdV` measurement rebuilds the state's full row
table, an `O(nrows·n_sites)` cost comparable
to one fresh [`total_energy`](@ref); at the default measurement cadence this is noise.
"""
function pressure_diagnostics(sch::StrainSchedule, H::TiledHamiltonian)
    _check_strain_pairing(H, sch)
    deg = _term_disp_degrees(H)
    n = size(sch.coefpoly, 2)
    coefs = Vector{Float64}(undef, n)
    dcoef = Vector{Float64}(undef, n)
    g = Vector{Float64}(undef, length(H.terms))
    dedv = Observable(:strain_dEdV, 1, v -> begin
        v.H === H || throw(ArgumentError(
            "this pressure diagnostic was built against a different Hamiltonian; " *
            "build it with the `H` the run uses"))
        s = strain(v)                      # throws on a fixed-cell view, by design
        zr = has_disp(H) ? _zrows(H, v.config, v.disps) : _zrows(H, v.config)
        _energy_volume_derivative(sch, H, zr, s, coefs, dcoef, g, deg)
    end)
    invv = Observable(:strain_invV, 1, v -> 1.0 / strain_volume(sch, strain(v)))
    nmob = sch.n_mobile
    # The captured `nmob` is deliberate and the injected `n` is deliberately UNUSED:
    # the identity's count is the volume power's `N_mob = n_disp_active`, while the
    # Evaluable-scope `n` is `n_active` (`:energy`) — the two coincide on an all-mobile
    # model and diverge on any model with spin-only sites. Do not "simplify" to `n`.
    press = Evaluable(:pressure, [:strain_dEdV, :strain_invV],
                      (m, kT, n) -> nmob * kT * m.strain_invV - m.strain_dEdV;
                      scope = :energy)
    return (; observables = [dedv, invv], evaluables = [press])
end

"""
    npt_observables(sch::StrainSchedule, H::TiledHamiltonian;
                    pressure_GPa = nothing, pressure = nothing)
        -> (; observables::Vector{Observable}, evaluables::Vector{Evaluable})

The β-conjugate state energy of a strained (NPT) run, packaged as observables. Two raw
observables —

    :enthalpy   =  W  =  E_config + n_cells·j0(s) + P·V(s)
    :enthalpy2  =  W²

(model units) — where `W` is the **configurational enthalpy**: exactly the state energy
whose Boltzmann factor the strain move accepts against and the strained PT exchange rule
weighs (`docs/specs/strain-move.md` S10/S11). Plus the jackknifed evaluable

    :npt_specific_heat  =  (⟨W²⟩ − ⟨W⟩²) / (n_active (k_BT)²)

the **isobaric specific heat** per active site, in units of ``k_B``. The sampled NPT
measure is `p ∝ V^{N_mob}·e^{−βW}` with a temperature-independent volume-Jacobian
factor, so `C/k_B = var(W)/(k_BT)² = d⟨W⟩/d(k_BT)` holds exactly in this ensemble —
including on a volume domain truncated by the schedule's grid, since the domain is
β-independent too. On a strained run this is the quantity `:specific_heat` is **not**:
that evaluable sees `var(E_config)` alone, which is neither `C_V` nor `C_P` (see the
observables guide). Append both vectors to the run's `observables` / `evaluables`.

Two interpretation caveats, both properties of the run. The identity stays exact on
the truncated measure, but the truncated measure is a **constrained system**: read the
number as the physical `C_P` only while the sampled volume distribution is confined
well inside [`strain_domain`](@ref) — a chain pressed against a grid edge measures the
heat capacity of a volume-clamped cell, and a broad `p(V)` (near a transition, where
`C_P` peaks) is exactly when that bites. And v0 samples the isotropic scale only, so
this is the isobaric `C` of the sampled **hydrostatic, fixed-shape ensemble** — the
five homogeneous shear degrees of freedom are frozen (their classical contribution is
`O(1) k_B` per supercell).

Pass the **same pressure the run uses** (`pressure_GPa` XOR `pressure`, resolved exactly
like [`run_mc`](@ref)'s keywords) and build from the **same `sch`**: the view carries
neither, so a mismatched pressure would silently weigh another run's target. The
Hamiltonian itself IS checked at measurement time, by identity on a structural array —
a view from any other `H` is refused, while a strained [`run_pt`](@ref)'s per-lane
coefficient clones share the array by reference and pass; after [`resume`](@ref),
rebuild the observables against the `H` handed to it (closures are not serialized).

Unlike [`pressure_diagnostics`](@ref), these observables capture no per-measurement
scratch — the closures are pure functions of the view, the immutable schedule and the
resolved pressure — so they are usable under a strained `run_pt` too (each measurement
costs two `j0` Horner passes — `:enthalpy2` re-evaluates `W` — not a row table). On a
**fixed-cell** run both drivers refuse `:enthalpy` / `:enthalpy2` by name at entry
(there is no volume degree of freedom, and the [`strain`](@ref) accessor's per-view
throw would otherwise fire only after a spent thermalization phase).

Configurational, like every energy of this sampler: no momenta are carried. For an
absolute heat capacity add the classical kinetic term analytically — `(3/2) k_B` per
mobile atom, identical at constant `V` and constant `P`, i.e. add
`(3/2)·n_disp_active/n_active` to the reported per-active-site value (the two counts
differ exactly on a model with spin-only sites).
"""
function npt_observables(sch::StrainSchedule, H::TiledHamiltonian;
                         pressure_GPa::Union{Nothing,Real} = nothing,
                         pressure::Union{Nothing,Real} = nothing)
    _check_strain_pairing(H, sch)
    p = _resolve_pressure(sch, pressure_GPa, pressure)
    # Identity on a shared STRUCTURAL array, not `===` on `H` (pressure_diagnostics'
    # guard) and not counts: a strained `run_pt` measures through per-lane
    # coefficient clones, which share `inst_term` by reference — while a same-shape
    # Hamiltonian built from another grid point's model has its own copy, and counts
    # alone cannot tell the two apart (the `_check_strain_pairing` hazard, reopened
    # at measurement time). `W` reads only the view's energy and strain against the
    # immutable schedule, so a clone's view is exactly as good as the parent's.
    guard = H.inst_term
    w = function (v::MCView)
        v.H.inst_term === guard || throw(ArgumentError(
            "this npt_observables was built against a different Hamiltonian; " *
            "build it with the `H` the run uses (a run_pt lane's coefficient " *
            "clone shares its structure and passes; after `resume`, rebuild the " *
            "observables against the `H` handed to it)"))
        s = strain(v)                  # throws on a fixed-cell view, by design
        in_strain_domain(sch, s) || throw(ArgumentError(
            "scale s = $s is outside this schedule's domain " *
            "$(strain_domain(sch)); evaluating the interpolant there would " *
            "silently extrapolate"))
        return v.energy + sch.n_cells * strain_j0(sch, s) + p * strain_volume(sch, s)
    end
    obs = [Observable(:enthalpy, 1, w), Observable(:enthalpy2, 1, v -> w(v)^2)]
    cp = Evaluable(:npt_specific_heat, [:enthalpy, :enthalpy2],
                   (m, kT, n) -> (m.enthalpy2 - m.enthalpy^2) / (n * kT^2);
                   scope = :energy)
    return (; observables = obs, evaluables = [cp])
end
