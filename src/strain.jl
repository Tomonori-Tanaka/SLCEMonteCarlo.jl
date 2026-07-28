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
