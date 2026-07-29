# The strain schedule (`src/strain.jl`) — the sampler's form of a K(ε) volume grid, and
# the seam the outer strain move is built on.
#
# What is pinned here. (1) The conversion is EXACT at the grid points and between them for
# a family whose coefficients are polynomial in the interpolation abscissa — that is what
# makes "install the coefficients for scale s" a definite operation rather than an
# approximation with an unstated error. (2) `set_coefficients!` + the schedule reproduces a
# Hamiltonian FRESHLY BUILT at that scale, which is the actual claim the move rests on.
# (3) The index map is a property of the basis, not of the fit: a Hamiltonian built without
# `keep_zero_terms` is refused by name, because the failure it would otherwise produce is
# silent (equal-length term lists with shifted maps). (4) `n_cells` counts ATOMS, not
# `prod(dims)`, and `D` comes from `comp_free` — the two quantities the NPT weight is most
# easily wrong about.

using Test
using SLCEMonteCarlo
using SLCE
using LinearAlgebra
using Random
using StaticArrays

const MCs = SLCEMonteCarlo

_ss_crystal(s) = Crystal(Lattice(Matrix(3.0s * I(3))),
                         [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])

# Every cutoff scales with `s` — the key-stability pin, without which the grid's SALC key
# sets diverge and `StrainedModels` refuses the grid upstream.
function _ss_spec(cr, s)
    return BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1s),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1s),
        Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1s)])
end

_ss_basis(s) = (cr = _ss_crystal(s); SLCEBasis(cr, _ss_spec(cr, s)))

# A grid whose coefficients are an exact QUADRATIC in the linear scale — which is not a
# contrivance: the map between two grid points is the re-expansion around the scaled
# geometry, exactly polynomial in `s` of the displacement content's own degree. So an
# exact-interpolation schedule must reproduce it to roundoff, and any error here is the
# schedule's, not the model family's.
function _ss_grid(; scales = [0.98, 1.0, 1.02], zero_key = 0)
    b1 = _ss_basis(1.0)
    Z = (@test_logs (:warn,) match_mode = :any SLCE.build_asr(b1)).Z
    rng = MersenneTwister(0x5511)
    g0 = randn(rng, size(Z, 2))
    g1 = 0.3 .* randn(rng, size(Z, 2))
    g2 = 0.1 .* randn(rng, size(Z, 2))
    models = SLCEModel[]
    for s in scales
        b = _ss_basis(s)
        η = s - 1
        jphi = 0.4 .* (Z * (g0 .+ η .* g1 .+ η^2 .* g2))
        zero_key > 0 && (jphi[zero_key] = 0.0)
        push!(models, SLCEModel(b, 0.37 + 1.5η + 2.0η^2, jphi))
    end
    return SLCE.StrainedModels(models, collect(Float64, scales)), models
end

_ss_cfg(n, seed) = MCs.SpinConfig([SVector{3,Float64}(normalize(randn(
    MersenneTwister(seed + s), 3))) for s = 1:n])
_ss_disps(n, seed) = [SVector{3,Float64}(0.03 .* randn(MersenneTwister(seed + 100s), 3))
                      for s = 1:n]

@testset "StrainSchedule: the sampler's view of a volume grid" begin

    @testset "exact at the nodes, exact between them, and it is the same Hamiltonian" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        @test length(sch) == 3
        @test MCs.strain_domain(sch) == (0.98, 1.02)
        @test occursin("StrainSchedule", sprint(show, sch))

        # the nodes, term by term and intercept by intercept
        for (i, s) in enumerate([0.98, 1.0, 1.02])
            want = [t.coef for t in SLCE.decorated_terms(models[i]; keep_zero = true)]
            @test MCs.strain_coefficients(sch, s) ≈ want rtol = 1e-12
            @test MCs.strain_j0(sch, s) ≈ models[i].j0 rtol = 1e-12
        end

        # ...and between them: the family is quadratic in `s`, the schedule interpolates
        # exactly, so this is a statement about the schedule and not about the fixture
        for s in (0.99, 1.005, 1.013)
            want = [t.coef for t in
                    SLCE.decorated_terms(SLCE.model_at(sm, s); keep_zero = true)]
            @test MCs.strain_coefficients(sch, s) ≈ want rtol = 1e-10
            @test MCs.strain_j0(sch, s) ≈ SLCE.model_at(sm, s).j0 rtol = 1e-10
        end

        # THE CLAIM: installing the schedule's coefficients gives the Hamiltonian a fresh
        # build at that scale would have produced.
        cfg = _ss_cfg(H.n_sites, 11)
        u = _ss_disps(H.n_sites, 11)
        for s in (0.99, 1.02)
            fresh = TiledHamiltonian(SLCE.model_at(sm, s); dims = (2, 1, 1),
                                     keep_zero_terms = true)
            set_coefficients!(H, MCs.strain_coefficients(sch, s))
            @test total_energy(H, cfg, u) ≈ total_energy(fresh, cfg, u) rtol = 1e-10
            @test [t.coef for t in H.terms] ≈ [t.coef for t in fresh.terms] rtol = 1e-10
        end
        # and back to the reference, exactly
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.0))
        ref = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        @test total_energy(H, cfg, u) ≈ total_energy(ref, cfg, u) rtol = 1e-12
    end

    @testset "n_cells counts atoms, D counts free directions" begin
        sm, models = _ss_grid()
        for dims in ((1, 1, 1), (2, 1, 1), (2, 2, 1))
            H = TiledHamiltonian(models[2]; dims = dims, keep_zero_terms = true)
            sch = StrainSchedule(sm, H)
            # `prod(dims)` is the right answer HERE and would be the wrong one after
            # `reduce_cell`; the schedule derives it from the site count either way
            @test sch.n_cells == prod(dims)
            @test sch.n_cells == H.n_sites ÷ 2
            @test MCs.strain_volume(sch, 1.0) ≈ prod(dims) * 27.0
            @test MCs.strain_volume(sch, 1.02) ≈ prod(dims) * 27.0 * 1.02^3
            # D = 3·N_mob − count(comp_free): recentring is per (direction, component),
            # so each free direction removes ONE dimension, not three
            @test sch.n_mobile == H.n_disp_active
            @test sch.d_dim == 3 * H.n_disp_active - count(H.comp_free)
            @test sch.d_dim < 3 * sch.n_mobile        # this fixture is ASR-flat
        end
    end

    @testset "the index map must be a property of the basis, not of the fit" begin
        # A grid whose fits zero one key: built the default way, the Hamiltonian's term
        # list is SHORTER than the schedule's, and the refusal names the flag. Without
        # this check the two lists could coincide in length with shifted maps, and every
        # coefficient would land on a neighbouring cluster silently.
        # `fixed_reference` because zeroing one coefficient takes the model out of the
        # ASR null space, and the displacement sampler's flatness demand is not what this
        # testset is about — the index map is.
        sm, models = _ss_grid(; zero_key = 2)
        pruned = TiledHamiltonian(models[2]; dims = (1, 1, 1), fixed_reference = true)
        kept = TiledHamiltonian(models[2]; dims = (1, 1, 1), fixed_reference = true,
                                keep_zero_terms = true)
        @test kept.n_input_terms > pruned.n_input_terms
        err = try
            StrainSchedule(sm, pruned)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("keep_zero_terms", err.msg)
        @test StrainSchedule(sm, kept) isa StrainSchedule

        # ...and a Hamiltonian built from a different TRUNCATION is refused by name
        # rather than mis-assigned: the count is the first thing that disagrees, and the
        # message points at the flag rather than at the arithmetic
        ob = SLCEBasis(_ss_crystal(1.0),
                       BasisSpec(_ss_crystal(1.0); lmax = 1, pmax = 2,
                                 sectors = [Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
        other = SLCEModel(ob, 0.0, ones(n_salcs(ob)))
        Ho = TiledHamiltonian(other; dims = (1, 1, 1), fixed_reference = true,
                              keep_zero_terms = true)
        @test_throws ArgumentError StrainSchedule(sm, Ho)
    end

    @testset "the default term list is unchanged" begin
        # `keep_zero_terms` reaching the emission must not move anything for a model with
        # no exact zeros — every existing consumer and byte comparison depends on it
        _, models = _ss_grid()
        a = TiledHamiltonian(models[2]; dims = (2, 1, 1))
        b = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        @test a.n_input_terms == b.n_input_terms
        @test [t.coef for t in a.terms] == [t.coef for t in b.terms]
        cfg = _ss_cfg(a.n_sites, 3)
        u = _ss_disps(a.n_sites, 3)
        @test total_energy(a, cfg, u) === total_energy(b, cfg, u)
    end

    @testset "energy contract: ΔE ≡ from-scratch totals, j0 the single elastic source" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        cfg = _ss_cfg(H.n_sites, 21)
        u = _ss_disps(H.n_sites, 21)
        s, sp = 1.0, 1.015
        up = [(sp / s) .* x for x in u]     # the affine rescale the move performs
        P = 0.013                            # model units — the P·ΔV term must fire

        set_coefficients!(H, MCs.strain_coefficients(sch, s))
        e_old = total_energy(H, cfg, u)
        set_coefficients!(H, MCs.strain_coefficients(sch, sp))
        e_new = total_energy(H, cfg, up)
        dE = MCs.strain_delta_energy(sch, e_old, e_new, s, sp; pressure = P)

        # gate (l), strain half: against from-scratch totals at BOTH scales — a fresh
        # Hamiltonian per scale, the interpolated j0, and the PV term, none shared with
        # the hot-swap path being tested
        etot = (scale, disp) -> begin
            Hf = TiledHamiltonian(SLCE.model_at(sm, scale); dims = (2, 1, 1),
                                  keep_zero_terms = true)
            total_energy(Hf, cfg, disp) + sch.n_cells * MCs.strain_j0(sch, scale) +
                P * MCs.strain_volume(sch, scale)
        end
        @test dE ≈ etot(sp, up) - etot(s, u) rtol = 1e-10

        # the pieces, against hand-written arithmetic
        want = (e_new - e_old) +
               sch.n_cells * (MCs.strain_j0(sch, sp) - MCs.strain_j0(sch, s)) +
               P * sch.n_cells * sch.v_train * (sp^3 - s^3)
        @test dE ≈ want rtol = 1e-12
        @test_throws ArgumentError MCs.strain_delta_energy(sch, 0.0, 0.0, s, sp;
                                                           pressure = NaN)

        # the strained reconstruction identity that pins j0 as the ONLY elastic source:
        # configurational + n_cells·j0(s) = the model's full predicted energy, at both
        # scales — an explicit elastic term anywhere would break this at s ≠ 1
        H1 = TiledHamiltonian(models[2]; dims = (1, 1, 1), keep_zero_terms = true)
        sch1 = StrainSchedule(sm, H1)
        cfg1 = _ss_cfg(H1.n_sites, 23)
        u1 = _ss_disps(H1.n_sites, 23)
        for scale in (1.0, 1.015)
            m = SLCE.model_at(sm, scale)
            set_coefficients!(H1, MCs.strain_coefficients(sch1, scale))
            full = total_energy(H1, cfg1, u1) +
                   sch1.n_cells * MCs.strain_j0(sch1, scale)
            @test full ≈ SLCE.predict_energy(m, MCs.to_matrix(cfg1),
                                             _disp_matrix(u1)) rtol = 1e-10
        end
    end

    @testset "NPT log weight: hand-derived closed form in both proposal arms" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        D = 3 * H.n_disp_active - count(H.comp_free)
        @test sch.d_dim == D
        s, sp, dE, kt = 0.99, 1.017, 0.37, 0.025
        lw_logv = MCs._strain_log_weight(sch, :logvolume, s, sp, dE, kt)
        lw_s = MCs._strain_log_weight(sch, :scale, s, sp, dE, kt)
        # hand-derived: (D + 3)·ln(s′/s) − ΔE/kT uniform in ln V, (D + 2)·ln(s′/s) −
        # ΔE/kT uniform in s — written as integer powers of s, independently of the
        # implementation's D/3-power-of-V form
        @test lw_logv ≈ (D + 3) * log(sp / s) - dE / kt rtol = 1e-13
        @test lw_s ≈ (D + 2) * log(sp / s) - dE / kt rtol = 1e-13
        # the arms differ by exactly ln(s′/s) — §8(γ)'s differential check
        @test lw_logv - lw_s ≈ log(sp / s) rtol = 1e-12
        # detailed-balance antisymmetry of the Jacobian half
        @test MCs._strain_log_weight(sch, :logvolume, sp, s, -dE, kt) ≈ -lw_logv rtol =
            1e-12
        @test_throws ArgumentError MCs._strain_log_weight(sch, :volume, s, sp, dE, kt)
        # y(s) ↔ s(y) round-trip in both arms — the draw's map and its inverse
        for arm in (:logvolume, :scale)
            @test MCs._strain_s_of_y(arm, MCs._strain_y(arm, 1.013)) ≈ 1.013 rtol =
                1e-14
        end
    end

    @testset "MCView carries the strain, and refuses to invent one" begin
        _, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (1, 1, 1), keep_zero_terms = true)
        cfg = _ss_cfg(H.n_sites, 5)
        u = _ss_disps(H.n_sites, 5)
        fixed = MCView(H, cfg, u, 1.5)
        @test !MCs.has_strain(fixed)
        @test fixed.strain === nothing
        # `nothing`, not 1.0: a fixed-cell run has no strain DoF, and a confident 1.0
        # would let a magnetostriction observable average a constant and report zero
        err = try
            MCs.strain(fixed)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("no strain", err.msg)
        @test !occursin("strain", sprint(show, fixed))

        strained = MCView(H, cfg, u, 1.5, 1.02)
        @test MCs.has_strain(strained)
        @test MCs.strain(strained) === 1.02
        @test occursin("strain", sprint(show, strained))
        @test_throws ArgumentError MCView(H, cfg, u, 1.5, -1.0)
    end
end
