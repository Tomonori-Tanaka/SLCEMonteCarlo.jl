# Harmonic stability of the sampled well — the screen that runs BEFORE a joint run,
# complementing the escape detector (`docs/specs/updates-stationarity.md` U8), which
# only reports the case that has already diverged.
#
# WHAT THIS CAN AND CANNOT DO. A negative eigenvalue of the displacement Hessian is a
# proof that the chain has no stationary distribution: the energy decreases without
# bound along that eigenvector for a purely harmonic model, and for any model it means
# the expansion point is not a minimum. A clean spectrum proves NOTHING about
# boundedness — deciding global non-negativity of a quartic form in 3N variables is
# NP-hard (Hilbert's 1888 non-SOS non-negative quartics; Murty–Kabadi 1987), so a
# harmonic screen structurally cannot be completed for max displacement degree ≥ 4.
# The three statements the package can make about an unbounded target are therefore:
# this screen (necessary, not sufficient), the `:u_moment_ratio` observable
# (continuous, moves before anything breaks), and `_check_escape!` (after the fact).
#
# WHY FINITE DIFFERENCES rather than the analytic harmonic part. The exact route is
# upstream: `SLCE.force_constants` / `SLCE.dynamical_matrix` derive the force constants
# from the model's polynomial coefficients, sharing `solid_harmonic_poly` with the ASR
# builder. Re-deriving them here from the TILED programs would duplicate that
# convention — the class of coupled-site drift this package's CLAUDE.md exists to
# prevent — while answering a slightly different question. This one deliberately
# differentiates `total_energy` of the tiled object the SAMPLER actually evaluates, so
# it also covers the tiling, the supercell folding, and any coefficient set installed
# after the fit. `E` is a polynomial in `u`, so the central differences carry no
# model error beyond the `O(h²)` truncation of degrees ≥ 4 and the `O(ε/h²)` roundoff.

# Default displacement increment (model length units). `E` is a low-degree polynomial,
# so this is a roundoff-vs-truncation balance and not a convergence question. Two
# things about the roundoff term are easy to get wrong:
#
#   * it is `ε·|E_total|/h²`, and `E_total` is EXTENSIVE while `Φ` is intensive, so the
#     floor RELATIVE to `max|Φ|` grows with the site count — at `_FC_MAXDIM` that is
#     ~500 sites of accumulated cancellation;
#   * measured on the joint fixture it is `≈ 3e-9·max|Φ|` at `h = 1e-3`, scaling
#     exactly as `1/h²` (3e-11 / 3e-9 / 4.6e-7 / 5.3e-5 at `h = 1e-2 … 1e-5`).
#
# The truncation term is `h²E⁗/12`, identically ZERO for a model of max displacement
# degree 2 — there, larger `h` is strictly better. `1e-3` is tuned for a quartic at
# O(1) total energy; `harmonic_stability` derives its verdict tolerance from the
# measured floor rather than from this constant, so raising `step` on a harmonic model
# is safe and loosens nothing silently.
const _FC_STEP = 1e-3

# Above this many displacement degrees of freedom the O(d²) evaluation count stops
# being a screen and becomes the run. Screens belong on a small cell.
const _FC_MAXDIM = 1_500

"""
    force_constant_matrix(H::TiledHamiltonian, config; disps = nothing,
                          step = 1e-3, maxdim = 1500) -> Symmetric{Float64}

The displacement Hessian `Φ_{(s,i),(t,j)} = ∂²E / ∂u_{s,i} ∂u_{t,j}` of the tiled
Hamiltonian at the given spin `config` and expansion point `disps` (default: the
clamped-ion state `u = 0`), in model energy units per length squared.

Rows and columns run over the **displacement-active** sites in ascending order —
`sites = findall(H.site_has_disp)`, site `sites[k]`'s three Cartesian components at
`3k-2 : 3k`. Sites with no displacement axis are excluded rather than carried as
exact zero rows, which would put a spurious zero eigenvalue in every spectrum and
make `eigmin` meaningless.

Computed by central differences of [`total_energy`](@ref) (`step`, see the source for
the roundoff/truncation balance), so it describes the tiled object the sampler
actually evaluates — tiling, supercell folding, and any installed coefficient set
included.

The cost is `O(d²)` energy evaluations for `d = 3·n_disp_active`, and each of those
walks the **whole** tiled Hamiltonian rather than the two touched sites, so the
wall-clock cost is `O(d²·n_instances)`: enlarging `dims` raises it quadratically
through `d` and linearly again through the instance count. `maxdim` refuses a problem
where that has stopped being a screen. The harmonic part is a property of the model,
so screen on the smallest supercell that resolves the wavevectors you care about
(a `1×1×1` tiling gives the Γ point alone).

Requires a joint model ([`has_disp`](@ref)). See [`harmonic_stability`](@ref) for the
verdict, and `SLCE.dynamical_matrix` for the exact, `q`-resolved force constants of a
*model* (this one is a supercell Γ-point quantity).
"""
force_constant_matrix(H::TiledHamiltonian, config::SpinConfig; kwargs...) =
    first(_hessian(H, config; kwargs...))

# The shared worker: the Hessian AND the base energy, which `harmonic_stability` needs
# to size the finite-difference noise floor its verdict rests on. One function so the
# stencil and the floor can never describe different calculations.
function _hessian(H::TiledHamiltonian, config::SpinConfig;
                  disps::Union{Nothing,AbstractVector} = nothing,
                  step::Real = _FC_STEP, maxdim::Integer = _FC_MAXDIM)
    _require_disp(H, "force_constant_matrix")
    step > 0 || throw(ArgumentError("step must be > 0; got $step"))
    sites = findall(H.site_has_disp)
    d = 3 * length(sites)
    d <= maxdim || throw(ArgumentError(
        "this Hamiltonian has $(length(sites)) displacement-active sites ($d degrees " *
        "of freedom), and the Hessian costs O(d²) ≈ $(2 * d^2) energy evaluations — " *
        "past the point where it screens a run rather than being one. Screen on a " *
        "smaller cell (the harmonic part is a property of the model, not of `dims`), " *
        "or raise `maxdim` deliberately"))
    u0 = disps === nothing ? zeros(SVector{3,Float64}, H.n_sites) :
         _initial_disps(H, disps)
    h = Float64(step)
    Φ = zeros(d, d)
    work = copy(u0)
    E(u) = total_energy(H, config, u)
    # NOT named `e0`: `2e0` is the Float64 literal 2.0, so `ep - 2e0 + em` below would
    # parse as `ep - 2.0 + em` and silently return a Hessian of −2/h².
    ebase = E(u0)
    # `a` is a flat degree of freedom index; `(site, component)` its decomposition.
    site_of(a) = sites[(a - 1) ÷ 3 + 1]
    comp_of(a) = (a - 1) % 3 + 1
    bump(u, a, δ) = (s = site_of(a); u[s] + δ * _unit3(comp_of(a)))
    for a = 1:d
        sa = site_of(a)
        work[sa] = bump(u0, a, h)
        ep = E(work)
        work[sa] = bump(u0, a, -h)
        em = E(work)
        work[sa] = u0[sa]
        Φ[a, a] = (ep - 2 * ebase + em) / h^2
        for b = (a + 1):d
            sb = site_of(b)
            v = 0.0
            for (σa, σb) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                work[sa] = bump(u0, a, σa * h)
                # a and b may sit on the SAME site: apply b's bump on top of a's,
                # never on top of `u0`, or the mixed derivative silently becomes the
                # single-variable one.
                work[sb] = work[sb] + σb * h * _unit3(comp_of(b))
                v += σa * σb * E(work)
                work[sa] = u0[sa]
                work[sb] = u0[sb]
            end
            Φ[a, b] = Φ[b, a] = v / (4h^2)
        end
    end
    return Symmetric(Φ), ebase, h
end

@inline _unit3(i::Int)::SVector{3,Float64} =
    i == 1 ? SVector(1.0, 0.0, 0.0) :
    i == 2 ? SVector(0.0, 1.0, 0.0) : SVector(0.0, 0.0, 1.0)

"""
    harmonic_stability(H::TiledHamiltonian, config; kwargs...) -> NamedTuple

Screen the harmonic part of a joint model at one spin configuration. Returns
`(; min_eigenvalue, n_negative, tol, acoustic_residual, eigenvalues)` from the
[`force_constant_matrix`](@ref) (same keyword arguments).

- `min_eigenvalue < -tol` is a **proof of failure**: the energy decreases without
  bound along that eigenvector for a harmonic model, and for any model the expansion
  point is not a minimum. Such a chain has no stationary distribution at any
  temperature and no displacement observable from it means anything.
- `min_eigenvalue ≥ -tol` proves **nothing**. Deciding global non-negativity of a
  quartic form is NP-hard, so no harmonic screen can be completed for max displacement
  degree ≥ 4; a model can pass this and still be unbounded. Pair it with the
  `:u_moment_ratio` observable and the escape detector
  (`docs/specs/updates-stationarity.md` U8).
- `n_negative = count(< -tol, λ)` — the imaginary branches. **The tolerance is not
  cosmetic.** A translation-invariant model has `3·n_disp_comps` exact zero
  eigenvalues (the acoustic modes, one rigid shift per direction per
  displacement-coupling component), and finite differences scatter each of them across
  zero at the `ε|E|/h²` roundoff floor — measured `2.5e-10 … 3.0e-9` on the joint
  fixture, and *step-dependent*. Counting those as imaginary branches makes a healthy
  model report `min_eigenvalue < 0`, i.e. the documented proof of failure, about half
  the time. `tol` is derived from that measured floor and returned so the verdict is
  auditable.
- `acoustic_residual` is the acoustic sum rule as a **measured residual**:
  `maximum(abs, Φ_c t)` over the three uniform-translation vectors of each
  displacement-coupling component `c`, each normalized by that **component's own**
  scale and the worst reported. Per component, not globally, for the reason
  `_translation_residuals` states: a small, weakly-coupled component can carry a real
  net force and still look flat against a dominant component's denominator — and a
  single pinned component would otherwise hide a broken sum rule in every other one.
  ~0 on a translation-invariant model, O(1) with a pinned frame
  (`fixed_reference = true`).

The eigenvalues are of the **supercell Γ-point** Hessian: they are the `q = 0` folded
spectrum for the tiled `dims`, not a phonon band structure. For `q`-resolved force
constants of a model, use `SLCE.dynamical_matrix`.
"""
function harmonic_stability(H::TiledHamiltonian, config::SpinConfig; kwargs...)
    Φ, ebase, h = _hessian(H, config; kwargs...)
    λ = eigvals(Φ)
    scale = maximum(abs, Φ)
    # A Hessian that is numerically zero is the ONE case where a clean spectrum is
    # maximally misleading: it is what a displacement sector of degree 4 only produces
    # at `u = 0` — a state an ASR-constrained fit can reach — and reporting
    # `(0.0, 0, 0.0)` there would be the most reassuring possible output about the
    # model that carries the least harmonic information.
    scale > 0 || throw(ArgumentError(
        "the displacement Hessian is identically zero at this expansion point, so it " *
        "carries no stability information at all — the leading displacement term is " *
        "of degree ≥ 3 here. Screen at a displaced expansion point (`disps = …`), or " *
        "rely on `:u_moment_ratio` and the escape detector instead"))
    probe = _acoustic_probe(H, Φ)
    # The verdict floor, MEASURED rather than modelled. Along a direction the
    # construction gate certified as flat, `Φ_c t` is exactly zero, so whatever comes
    # out is pure finite-difference error — and it tracks the scatter of the acoustic
    # eigenvalues about zero to ~12 % across three decades of `step` (2.7e-11 / 2.7e-9
    # / 4.4e-7 against 3.0e-11 / 3.0e-9 / 4.6e-7 at h = 1e-2 / 1e-3 / 1e-4). The factor
    # 4 is margin for the configuration dependence of that ratio. The a-priori term
    # `ε|E|/h²` (four energies of size |E| cancelling into a difference divided by h²)
    # is the fallback where there is no flat direction to measure on — a model with a
    # pinned frame — and the `1e-12·scale` term keeps the tolerance meaningful when the
    # base energy happens to vanish at the expansion point.
    tol = max(16 * eps() * abs(ebase) / h^2, 4 * probe.flat_absolute) + 1e-12 * scale
    return (; min_eigenvalue = minimum(λ), n_negative = count(<(-tol), λ), tol,
            acoustic_residual = probe.normalized, eigenvalues = λ)
end

# The acoustic sum rule per displacement-coupling component, in two forms.
#
# The components are independent symmetries and the Hessian is exactly block diagonal
# across them (two components share no instance), so each block gets its own
# rigid-shift probe and its own denominator — the normalization rule
# `_translation_residuals` already applies on the construction side, and for the same
# reason: a small, weakly-coupled component can carry a real net force and still look
# flat against a dominant component's scale.
#
#   `normalized`     — the reported diagnostic: worst over ALL three directions of
#                      every component, each divided by that component's own scale. A
#                      pinned direction contributes its physical O(1) value, which is
#                      the point (it says the frame is not free).
#   `flat_absolute`  — the noise measurement the verdict tolerance rests on: worst over
#                      the directions the construction gate certified FLAT, unnormalized.
#                      There the exact answer is zero, so this is finite-difference
#                      error and nothing else. A pinned direction must never enter it.
function _acoustic_probe(H::TiledHamiltonian, Φ::Symmetric{Float64})
    # `disp_comp_sites` holds GLOBAL site ids; the Hessian is indexed by position in
    # the ascending list of displacement-active sites. Invert once.
    sites = findall(H.site_has_disp)
    pos = zeros(Int, H.n_sites)
    for (k, s) in enumerate(sites)
        pos[s] = k
    end
    worst = 0.0
    flat = 0.0
    for c = 1:H.n_disp_comps
        lo = Int(H.disp_comp_ptr[c])
        hi = Int(H.disp_comp_ptr[c + 1]) - 1
        rows = Int[]
        for q = lo:hi
            k = pos[Int(H.disp_comp_sites[q])]
            append!(rows, (3k - 2, 3k - 1, 3k))
        end
        blk = view(Φ, rows, rows)
        scale = maximum(abs, blk)
        scale > 0 || continue
        for i = 1:3
            t = zeros(length(rows))
            t[i:3:end] .= 1.0
            r = maximum(abs, blk * t)
            worst = max(worst, r / scale)
            H.comp_free[i, c] && (flat = max(flat, r))
        end
    end
    return (; normalized = worst, flat_absolute = flat)
end
