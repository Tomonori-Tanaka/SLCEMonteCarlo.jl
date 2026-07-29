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

# The NPT-marginal toy (design record §8 (γ)): a decoupled spin sector plus a PURE
# lattice sector whose coefficients are IDENTICAL at every grid point, `j0 ≡ 0`. Held at
# `u ≡ 0` and `P = 0`, the configurational ΔE of a strain move is then exactly zero and
# the chain's stationary volume marginal is the bare Jacobian, `p(V) ∝ V^{D/3}` — an
# analytically known target that measures the implemented exponent rather than asserting
# it. `pinned = false` gives a translation-flat pair model (ASR-null-space
# coefficients); on the (2, 1, 1) supercell the in-cutoff pairs are the cross-cell ones
# only (the in-cell a1–a2 distance is 2.0s > 1.1s), so the four sites split into TWO
# displacement components and `D = 3N − 3·n_comps = 6`. `pinned = true` shrinks the
# cutoff below every pair, leaving on-site `|u|²` content that pins every rigid shift
# (`D = 3N = 12`). The two targets differ by the full `count(comp_free)`, which is the
# only test that isolates the per-(direction, component) COM bookkeeping in `D`.
function _ss_toy_spec(cr, s, pinned)
    return BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1s),
        Sector(disp = (degree = 2,), sites = 1:2, cutoff = (pinned ? 0.5 : 1.1) * s)])
end

_ss_toy_basis(s, pinned) = (cr = _ss_crystal(s); SLCEBasis(cr, _ss_toy_spec(cr, s, pinned)))

function _ss_toy_grid(; pinned::Bool, scales = [0.85, 1.0, 1.15])
    b1 = _ss_toy_basis(1.0, pinned)
    rng = MersenneTwister(pinned ? 0x7001 : 0x7002)
    jphi = if pinned
        randn(rng, n_salcs(b1))
    else
        Z = (@test_logs (:warn,) match_mode = :any SLCE.build_asr(b1)).Z
        Z * randn(rng, size(Z, 2))
    end
    models = [SLCEModel(_ss_toy_basis(s, pinned), 0.0, copy(jphi)) for s in scales]
    return SLCE.StrainedModels(models, collect(Float64, scales))
end

# Run `nmoves` strain moves at `u ≡ 0` and return the empirical mean volume plus the
# acceptance rate. The analytic mean of `p(V) ∝ V^k` on `[Va, Vb]` is `_ss_vk_mean`.
function _ss_toy_chain(sm, pinned, proposal, step, nmoves, seed)
    H = TiledHamiltonian(sm.models[2]; dims = (2, 1, 1), keep_zero_terms = true,
                         fixed_reference = pinned)
    sch = StrainSchedule(sm, H)
    set_coefficients!(H, MCs.strain_coefficients(sch, 1.0))
    st = MCs.ChainState(H, _ss_cfg(H.n_sites, seed), Xoshiro(seed), 0.3;
                        disps = zeros(SVector{3,Float64}, H.n_sites))
    sc = MCs.StrainScratch(H)
    vsum = 0.0
    for _ = 1:nmoves
        MCs.strain_move!(st, H, sch, sc, 0.025, 0.0, step; proposal = proposal)
        vsum += MCs.strain_volume(sch, st.strain)
    end
    return vsum / nmoves, st.acc_strain / st.att_strain, sch
end

_ss_vk_mean(k, va, vb) =
    ((k + 1) / (k + 2)) * (vb^(k + 2) - va^(k + 2)) / (vb^(k + 1) - va^(k + 1))

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

    @testset "strain_move!: invariants, rejection, and the (H, chain) contract" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        H2 = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.0))
        st = MCs.ChainState(H, _ss_cfg(H.n_sites, 31), Xoshiro(0x51), 0.3;
                            disps = _ss_disps(H.n_sites, 31))
        sc = MCs.StrainScratch(H)
        st.com_removed[1] = SVector(1e-3, -2e-3, 5e-4)

        # the contract, after EVERY move: H carries the schedule's coefficients at the
        # chain's scale (bitwise — the reject path restores by re-running the same
        # Horner pass), the cached rows are a fresh build's, the energy is the rows'
        naccept = 0
        for it = 1:40
            u_before = copy(st.disps)
            com_before = copy(st.com_removed)
            s_before = st.strain
            accepted = MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 0.06)
            set_coefficients!(H2, MCs.strain_coefficients(sch, st.strain))
            @test [t.coef for t in H.terms] == [t.coef for t in H2.terms]
            @test st.zrows == MCs._zrows(H, st.config, st.disps)
            @test st.energy == MCs._total_energy(H, st.zrows)
            lam = st.strain / s_before
            if accepted
                naccept += 1
                @test st.strain != s_before
                @test st.disps == [lam * u for u in u_before]
                @test st.com_removed == [lam * c for c in com_before]
                @test st.disp_checks == 0        # accepted rescale resets the detector
            else
                @test st.strain == s_before
                @test st.disps == u_before && st.com_removed == com_before
            end
        end
        @test 0 < naccept < 40                   # both branches actually ran
        @test st.att_strain == 40 && st.acc_strain == naccept

        # out-of-domain proposal: rejected (never clamped), H untouched, and exactly
        # ONE draw consumed — the uniform is only spent on an in-domain proposal
        twin = copy(st.rng)
        s_before = st.strain
        coefs_before = [t.coef for t in H.terms]
        @test !MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 50.0)
        randn(twin)
        @test MCs._rng_words(st.rng) == MCs._rng_words(twin)
        @test st.strain == s_before
        @test [t.coef for t in H.terms] == coefs_before

        # validation: the named errors
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.0, 0.01, 0.05)
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 0.0)
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 0.05;
                                                    proposal = :volume)
        st.strain = 2.0
        err = try
            MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 0.05)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("outside the schedule's domain", err.msg)
    end

    @testset "strain_move!: the NPT marginal is the bare Jacobian on the §8(γ) toy" begin
        # At u ≡ 0, constant coefficients, j0 ≡ 0 and P = 0 the stationary volume
        # marginal is p(V) ∝ V^{D/3} exactly — so the empirical mean volume MEASURES
        # the implemented exponent. Each fixture must match its own D and reject a
        # ±1 power error (the COM bookkeeping: the fixtures' D differ by the full
        # count(comp_free) = 6), and both proposal arms must land on the SAME
        # marginal even though their acceptance powers differ.
        for pinned in (false, true)
            sm = _ss_toy_grid(; pinned = pinned)
            for (proposal, step) in ((:logvolume, 0.75), (:scale, 0.25))
                vmean, arate, sch = _ss_toy_chain(sm, pinned, proposal, step, 150_000,
                                                  pinned ? 0x11 : 0x12)
                @test sch.d_dim == (pinned ? 12 : 6)
                @test 0.15 < arate < 0.95
                va, vb = MCs.strain_volume(sch, 0.85), MCs.strain_volume(sch, 1.15)
                vmid = (va + vb) / 2
                k = sch.d_dim / 3
                # Thresholds calibrated over seeds {0x11, 0x12, 0x21} at this length:
                # the correct power sits ≤ 0.006 of the mid volume, a ±1 power error
                # (the other fixture's D — an off-by-one in the COM bookkeeping)
                # shifts the mean by ≥ 0.024. The ±1/3 mixed-arm trap shifts it by
                # only ~0.011, INSIDE this statistic's noise — it is excluded
                # deterministically instead, by the closed-form arm gate above and
                # the draw/weight pairing testset below.
                @test abs(vmean - _ss_vk_mean(k, va, vb)) / vmid < 0.012
                @test abs(vmean - _ss_vk_mean(k + 1, va, vb)) / vmid > 0.018
                @test abs(vmean - _ss_vk_mean(k - 1, va, vb)) / vmid > 0.018
            end
        end
    end

    @testset "strain_move!: the draw and the weight use the SAME proposal arm" begin
        # White-box replay: for every attempt, reconstruct by hand — from a cloned RNG
        # — the proposal this arm should draw and the log weight this arm should apply,
        # and predict the accept decision. Every decision matching the hand-paired
        # computation is what excludes §8(β)'s mixed-arm trap EXACTLY, where the
        # statistical marginal above cannot resolve its 1/3 power.
        sm, models = _ss_grid()
        for proposal in (:logvolume, :scale)
            H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
            sch = StrainSchedule(sm, H)
            set_coefficients!(H, MCs.strain_coefficients(sch, 1.0))
            st = MCs.ChainState(H, _ss_cfg(H.n_sites, 41), Xoshiro(0x61), 0.3;
                                disps = _ss_disps(H.n_sites, 41))
            sc = MCs.StrainScratch(H)
            nacc = 0
            for it = 1:80
                twin = copy(st.rng)
                s = st.strain
                e_old = st.energy
                u_old = copy(st.disps)
                # the hand side, arm-paired throughout
                sp = MCs._strain_s_of_y(proposal,
                                        MCs._strain_y(proposal, s) + 0.06 * randn(twin))
                want = false
                if MCs.in_strain_domain(sch, sp)
                    Hp = TiledHamiltonian(SLCE.model_at(sm, sp); dims = (2, 1, 1),
                                          keep_zero_terms = true)
                    e_new = total_energy(Hp, st.config, [(sp / s) * u for u in u_old])
                    de = MCs.strain_delta_energy(sch, e_old, e_new, s, sp;
                                                 pressure = 0.01)
                    want = log(rand(twin)) <
                           MCs._strain_log_weight(sch, proposal, s, sp, de, 0.05)
                end
                got = MCs.strain_move!(st, H, sch, sc, 0.05, 0.01, 0.06;
                                       proposal = proposal)
                @test got == want
                @test st.strain == (want ? sp : s)
                nacc += got
            end
            @test 0 < nacc < 80
        end
    end

    @testset "run_mc drives the NPT chain: wiring, resolution, byte-neutrality" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)

        # THE BYTE-NEUTRALITY PIN: this fixed-cell trajectory was captured before the
        # strain wiring landed (at commit d038b86) and must stay bit-identical — the
        # strain channel consumes no randomness and changes no code path when absent.
        # If an intentional sampler change moves it, recapture; anything else moving
        # it is the regression this pin exists to catch.
        r0 = run_mc(H; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100,
                    seed = 0x5150, renorm_interval = 25)
        @test r0.points[1].stats[:energy].mean[1] === -12.866452813199738
        @test sum(sum, r0.final_config) === -0.17069256709356861
        @test sum(x -> sum(abs, x), r0.final_disps) === 1.4251564417024021
        @test isnan(r0.points[1].acceptance_strain)

        # the NPT run: moves fire, the measurement view carries the scale (checked
        # through the observable path, not the internals), the chain stays in domain
        seen = Float64[]
        obs = [standard_observables(H);
               Observable(:scale, 1, v -> (push!(seen, MCs.strain(v)); MCs.strain(v)))]
        r1 = run_mc(H; kT = 0.05, sweeps_therm = 60, sweeps_measure = 120,
                    seed = 0x5151, renorm_interval = 30, strain = sch,
                    pressure = 0.0, observables = obs)
        @test 0.0 < r1.points[1].acceptance_strain < 1.0
        @test !isempty(seen) && all(s -> MCs.in_strain_domain(sch, s), seen)
        @test length(unique(seen)) > 1          # the cell actually moved
        st = r1.points[1].stats[:scale]
        @test MCs.in_strain_domain(sch, st.mean[1])

        # pressure changes the sampled volume the right way: a strong positive
        # pressure pushes the mean scale DOWN relative to a strong negative one
        mean_scale = P -> begin
            empty!(seen)
            run_mc(H; kT = 0.05, sweeps_therm = 60, sweeps_measure = 240,
                   seed = 0x5152, renorm_interval = 60, strain = sch, pressure = P,
                   observables = obs)
            sum(seen) / length(seen)
        end
        @test mean_scale(2.0) < mean_scale(-2.0)

        # resolution: the named contradictions
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain_interval = 2)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = 0.0, strain_interval = 0)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, pressure = 0.0)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = 0.0, pressure_GPa = 0.0)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = 0.0, checkpoint = "x.jld2")
        # ...the schedule of a DIFFERENT Hamiltonian is refused by the pairing check
        H1 = TiledHamiltonian(models[2]; dims = (1, 1, 1), keep_zero_terms = true)
        @test_throws ArgumentError run_mc(H1; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = 0.0)
        # ...and the GPa conversion is the exact constant, applied once
        @test MCs.GPA_PER_EV_A3 === 160.2176634
        @test MCs._resolve_pressure(sch, 160.2176634, nothing) ≈ 1.0 rtol = 1e-15

        # run_pt refuses a strain schedule by name (v0 scope: shared H + NVT swaps)
        err = try
            run_pt(H; kT = [0.05, 0.06], sweeps_therm = 1, sweeps_measure = 2,
                   strain = sch)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("swap rule", err.msg)
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
