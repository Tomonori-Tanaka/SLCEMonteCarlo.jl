# Shared fixtures for the unit suite. `MC` aliases the package so internal (non-exported)
# names resolve as `MC._name`.

using SLCEMonteCarlo
using SLCE
using Spglib: Spglib          # activates SLCE's SpglibBackend extension
using LinearAlgebra
using Logging: Logging        # `@test_logs min_level = Logging.Warn`
using Random
using StaticArrays
using Statistics: mean, std
using Test

const MC = SLCEMonteCarlo

# Classical Langevin function L(x) = coth(x) − 1/x.
_langevin(x) = coth(x) - 1 / x

# --- fitted-model fixtures (mirroring SLCETools' MC suite) --------------------------

# A clean ferromagnetic Heisenberg dimer: 4 atoms in a column, pair cutoff couples
# only atoms 1–2 (atoms 3–4 free); the single active SALC is the isotropic l=1 pair.
function _dimer_crystal()
    lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
    return Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
end

function _dimer_model()
    b = SLCEBasis(_dimer_crystal(), BasisSpec(; nbody = 2, cutoff = 2.6,
                                             lmax = [1], soc = false))
    return SLCEModel(b, 0.0, vcat([-0.02], zeros(n_salcs(b) - 1)))  # < 0 ⇒ ferro
end

# A fitted model that genuinely IS a 2× stack of a 1-atom cell: two atoms along z,
# every SALC given the same coefficient, so the Hamiltonian keeps the half-cell
# translation symmetry whatever the orbit granularity of the symmetry backend.
function _stacked_chain_model()
    lat = Lattice(Matrix(4.0 * I(3)))
    cr = Crystal(lat, [0 0; 0 0; 0.0 0.5], [1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.1, lmax = [1],
                               soc = false))
    return SLCEModel(b, 0.0, fill(-0.02, n_salcs(b))), cr
end

# The pair coupling J of the dimer (E = J e₁·e₂), read off the tiled energies of the
# aligned / anti-aligned configurations — no dependence on SLCETools' ExchangeModel.
function _dimer_J(H::MC.TiledHamiltonian)
    up = SVector(0.0, 0.0, 1.0)
    aligned = MC.SpinConfig([up for _ = 1:H.n_sites])
    anti = copy(aligned)
    anti[1] = -up
    return (total_energy(H, aligned) - total_energy(H, anti)) / 2
end

# A genuine higher-multipole two-atom model (l ≤ 2, anisotropic, random couplings).
function _biquadratic_model(seed)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                               soc = true))
    return SLCEModel(b, 0.0, 0.05 .* randn(MersenneTwister(seed), n_salcs(b)))
end

# Anisotropic (l ≤ 2) variants of the same 2× z-stack.
# With the spglib backend the SALC orbits are translation-closed, so ANY coefficient
# vector keeps the half-cell periodicity — several channels per cluster then survive
# a genuine fitted reduction (the (coef, folded) sub-partition of `reduce_cell`).
# With `NoSymmetry()` each bond is its own orbit and the SALC tensor bases of
# translation-partner bonds need not align channel-by-channel, so even equal
# coefficients on every SALC genuinely BREAK the periodicity.
function _stacked_anisotropic_model(backend; fill_coefs::Bool = false)
    lat = Lattice(Matrix(4.0 * I(3)))
    cr = Crystal(lat, [0 0; 0 0; 0.0 0.5], [1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.1, lmax = [2],
                               soc = true); backend = backend)
    jphi = fill_coefs ? fill(0.03, n_salcs(b)) :
           0.05 .* randn(MersenneTwister(41), n_salcs(b))
    return SLCEModel(b, 0.0, jphi), cr
end

# A fitted model whose training cell is a NON-diagonal (√2×√2, det M = 2) supercell
# of a 1-atom cell: the checkerboard. NN ±x/±y bonds all bridge the two cosets.
function _checkerboard_model()
    lat = Lattice([1.0 1.0 0; -1.0 1.0 0; 0 0 4.0])
    cr = Crystal(lat, [0 0.5; 0 0.5; 0.0 0.0], [1, 1], ["Fe"])
    b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1],
                               soc = false))
    return SLCEModel(b, 0.0, fill(-0.02, n_salcs(b))), cr
end

# A periodic chain of one atom per cell coupled to its ±x neighbor images: the
# smallest fixture whose physics *requires* nonzero shifts (self-image pair — only
# representable on dims with N₁ ≥ 2). Hand-built raw MultipoleTerms: both directed
# members of the +x bond, E = J e_(0)·e_(+x) summed over cells.
function _chain_terms(J)
    n1 = SLCE.Harmonics.N1                      # Z_1m = N1 * (y, z, x)[m+2]
    folded = zeros(3, 3)
    folded[1, 1] = folded[2, 2] = folded[3, 3] = 1.0  # Σ_m Z_1m(a) Z_1m(b) ∝ a·b
    raw = J / (2 * n1^2 * (4π))                       # both members + (4π)^(2/2) scale
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(raw, 2, [1, 1], [z, x], [1, 1], copy(folded)),
            MultipoleTerm(raw, 2, [1, 1], [z, -x], [1, 1], copy(folded))]
end

# Hand-built 3-body chain cluster (0, +x, +2x) with a few dense folded entries —
# the smallest fixture whose contraction programs take the TRIPLET fast path
# (`site_col < 0`; asserted where used). Not a physically motivated coupling —
# kernel-equivalence gates compare arithmetic, not physics.
function _threebody_terms(J)
    folded = zeros(3, 3, 3)
    folded[2, 2, 2] = 1.0
    folded[1, 3, 1] = 0.7
    folded[3, 1, 2] = -0.4
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(J, 3, [1, 1, 1], [z, x, 2 * x], [1, 1, 1], folded)]
end

# Hand-built 4-body chain cluster (0, +x, +2x, +3x): body ≥ 4 has no fast path,
# so its programs take the GENERAL contraction branch (`site_col == 0` — the
# sfac/inst_sites indirection chain; asserted where used). Needs dims N₁ ≥ 4.
function _fourbody_terms(J)
    folded = zeros(3, 3, 3, 3)
    folded[2, 2, 2, 2] = 1.0
    folded[1, 3, 2, 1] = 0.6
    folded[3, 2, 1, 3] = -0.3
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    return [MultipoleTerm(J, 4, [1, 1, 1, 1], [z, x, 2 * x, 3 * x], [1, 1, 1, 1],
                          folded)]
end

# --- joint (spin + displacement) fixture --------------------------------------------

# A genuine joint spin–lattice model on a 2-atom cell: a pure-spin sector, a coupled
# sector (spin × displacement), and a lattice-only sector. The three together are what
# make the channel gates non-vacuous:
#   * the lattice-only sector has sites carrying NO spin factor, which is exactly where
#     the pure-spin-era `(4π)^(body/2)` shortcut would invent a factor from nothing;
#   * the coupled sector puts two axes (a spin and a displacement one) on ONE site, the
#     case that merges two axis-programs into one site program;
#   * with the disp degree landing on a single site, the coupled rank-3 term has its two
#     same-site axes separated by a third in canonical order — the column tuples then
#     disagree and the fast path must decline (asserted where used).
# No fit is involved: the tiling identity against `predict_energy` holds for any
# coefficient vector. But the displacement sampler requires a flat uniform-shift
# direction, so `asr = true` (the default) projects the random coefficients onto the
# acoustic-sum-rule null space — `SLCE.build_asr(b).Z` — which is exactly what a
# `fit(...; asr = true)` produces and what `TiledHamiltonian` demands. Pass
# `asr = false` for the fixture that must be REFUSED.
function _joint_model(seed = 5; asr::Bool = true)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (nbody = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),
        Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)])
    b = SLCEBasis(cr, spec)
    rng = MersenneTwister(seed)
    jphi = if asr
        # the warning about structurally-zeroed columns is expected on a fixture this
        # small (a few displacement partners fall outside the cutoff)
        Z = (@test_logs (:warn,) match_mode = :any SLCE.build_asr(b)).Z
        0.4 .* (Z * randn(rng, size(Z, 2)))
    else
        0.05 .* randn(rng, n_salcs(b))
    end
    return SLCEModel(b, 0.37, jphi), cr
end

# An Einstein oscillator: one atom per cell in an isotropic harmonic well `E = a|u|²`,
# hand-built as a single rank-1 displacement axis with `(k, l) = (1, 0)` — the factor is
# `|u|^{2} R_{0,0}(u)` and `R_{0,0} ≡ 1`. Deliberately NOT translation-invariant (the
# well pins each atom), hence `fixed_reference = true`; each atom is its own
# displacement-coupling component.
#
# Its whole value is that the answer is known in closed form: `exp(−βa|u|²)` is an
# isotropic Gaussian with per-component variance `kT/(2a)`, so `⟨|u|²⟩ = 3kT/(2a)`
# exactly. Everything else in the displacement suite checks the sampler against itself
# (incremental vs from-scratch, serial vs parallel); this checks it against an external
# truth, and it doubles as the equilibrated control the escape detector is calibrated
# against.
function _einstein_terms(a::Float64 = 2.5)
    L = SLCE.RowLayout(2, 0, 1, [(1, 0)], [1])       # spin row 1, displacement row 2
    return [MC.ScaledTerm(a, [1], [SVector(0, 0, 0)],
                          [MC.TermSlot(1, 1, 0, false)], [1.0])], L
end

# Random displacements on the sites of `H` (small — the polynomial kernel is exact at
# any amplitude, but a physical scale keeps the magnitudes comparable to the spin rows).
_rand_disps(rng, H::MC.TiledHamiltonian; amp = 0.08) =
    [amp .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) for _ = 1:H.n_sites]

# 3×n matrix of a displacement list (for predict_energy cross-checks).
_disp_matrix(disps) = reduce(hcat, [Vector(u) for u in disps])

# Tile a training-cell displacement list periodically onto the supercell of `H`.
_tile_disps(H::MC.TiledHamiltonian, cell_disps) =
    [cell_disps[MC.site_atom(H, s)] for s = 1:H.n_sites]

# Random unit spin from `rng` (Gaussian-normalized — uniform on S²).
_rand_spin(rng) = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))

# Random configuration on the sites of `H`.
_rand_config(rng, H::MC.TiledHamiltonian) =
    MC.SpinConfig([_rand_spin(rng) for _ = 1:H.n_sites])

# 3×n matrix view of a SpinConfig (for predict_energy cross-checks).
_config_matrix(config) = reduce(hcat, [Vector(e) for e in config])

# Tile a training-cell configuration periodically onto the supercell of `H`.
function _tile_config(H::MC.TiledHamiltonian, cell_config::MC.SpinConfig)
    config = MC.SpinConfig(undef, H.n_sites)
    for s = 1:H.n_sites
        config[s] = cell_config[MC.site_atom(H, s)]
    end
    return config
end
