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
# silent (equal-length term lists with shifted maps). (4) `n_cells` is derived from the
# ATOM count, not `prod(dims)`, and the NPT volume power is the displacement-active site
# count `n_mobile` — NOT the COM-reduced `D` — the two quantities the weight is most
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

# The Einstein-well grid: displacement energy BOUNDED BELOW (a stiff positive on-site
# |u|² keyed off the basis, dominating small smooth spin/quadrupole/pair couplings,
# everything linear in η), so a long chain has a stationary distribution to test —
# random ASR'd coefficients are a generically indefinite quadratic form and escape
# (U8). The well pins the rigid shift, hence `fixed_reference = true` at every use.
# Shared by the §8(ζ) identity testset and the strained-PT marginal gate.
function _ss_zeta_grid(; scales = [0.9, 1.0, 1.1])
    zeta_model = function (s)
        cr = _ss_crystal(s)
        b = SLCEBasis(cr, _ss_spec(cr, s))
        η = s - 1
        rng = MersenneTwister(0x2ee7)   # the SAME draw at every scale → smooth in η
        jphi = map(b.salc_basis.keys) do k
            r = 2 * rand(rng) - 1
            onsite_u2 = k.body == 1 && length(k.decors) == 1 &&
                        k.decors[1].spin_l == 0 && k.decors[1].disp_k == 1 &&
                        k.decors[1].disp_l == 0
            onsite_u2 ? 2.0 + 0.5 * η : 0.05 * r * (1 + 0.3 * η)
        end
        n_u2 = count(k -> k.body == 1 && length(k.decors) == 1 &&
                          k.decors[1].disp_k == 1 && k.decors[1].disp_l == 0 &&
                          k.decors[1].spin_l == 0, b.salc_basis.keys)
        @test n_u2 > 0                  # the well actually exists in this basis
        return SLCEModel(b, 40.0 * η^2, collect(Float64, jphi))
    end
    zsm = SLCE.StrainedModels([zeta_model(s) for s in scales],
                              collect(Float64, scales))
    return zsm, zsm.models
end

_ss_cfg(n, seed) = MCs.SpinConfig([SVector{3,Float64}(normalize(randn(
    MersenneTwister(seed + s), 3))) for s = 1:n])
_ss_disps(n, seed) = [SVector{3,Float64}(0.03 .* randn(MersenneTwister(seed + 100s), 3))
                      for s = 1:n]

# The NPT-marginal toy (design record §8 (γ)): a decoupled spin sector plus a PURE
# lattice sector whose coefficients are IDENTICAL at every grid point, `j0 ≡ 0`. Held at
# `u ≡ 0` and `P = 0`, the configurational ΔE of a strain move is then exactly zero and
# the chain's stationary volume marginal is the bare Jacobian, `p(V) ∝ V^{N_mob}` — an
# analytically known target that measures the implemented exponent rather than asserting
# it. The power is the displacement-active site count and is INDEPENDENT of
# `comp_free`: the flat COM directions are gauge directions whose range is the cell, so
# they keep their measure factor even though the sampler re-centres them out.
# `pinned = false` gives a translation-flat pair model (ASR-null-space coefficients);
# on the (2, 1, 1) supercell the in-cutoff pairs are the cross-cell ones only (the
# in-cell a1–a2 distance is 2.0s > 1.1s), so the four sites split into TWO displacement
# components (`count(comp_free) = 6`, `d_dim = 6`). `pinned = true` shrinks the cutoff
# below every pair, leaving on-site `|u|²` content that pins every rigid shift
# (`count = 0`, `d_dim = 12`). BOTH must land on the same `V^4` — under the corrected
# convention their exponents agree, and under the rejected `D/3` convention they differ
# by 2, so this pair is exactly the mutation the gate must kill. The `N_mob` dependence
# itself is exercised by running the flat fixture at (1, 1, 1) too (`N_mob = 2`).
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
function _ss_toy_chain(sm, pinned, dims, proposal, step, nmoves, seed)
    H = TiledHamiltonian(sm.models[2]; dims = dims, keep_zero_terms = true,
                         fixed_reference = pinned)
    sch = StrainSchedule(sm, H)
    set_coefficients!(H, MCs.strain_coefficients(sch, 1.0))
    st = MCs.ChainState(H, _ss_cfg(H.n_sites, seed), Xoshiro(seed), 0.3;
                        disps = zeros(SVector{3,Float64}, H.n_sites))
    sc = MCs.StrainScratch(H)
    vsum = 0.0
    for _ = 1:nmoves
        MCs.strain_move!(st, H, sch, sc, 0.025; pressure = 0.0, step = step,
                         proposal = proposal)
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
        # `pmax` omitted, not 2: this sector table is pure spin, so a displacement cap
        # would declare a degree no sector can build — SLCE refuses that now (it always
        # did for the dense spec form), and the basis is identical either way.
        ob = SLCEBasis(_ss_crystal(1.0),
                       BasisSpec(_ss_crystal(1.0); lmax = 1,
                                 sectors = [Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
        other = SLCEModel(ob, 0.0, ones(n_salcs(ob)))
        Ho = TiledHamiltonian(other; dims = (1, 1, 1), fixed_reference = true,
                              keep_zero_terms = true)
        @test_throws ArgumentError StrainSchedule(sm, Ho)
    end

    @testset "the default term list is unchanged" begin
        # `keep_zero_terms` reaching the emission must not move anything beyond the
        # exact zeros themselves — every existing consumer and byte comparison
        # depends on it. The zero count is MEASURED, not assumed zero: the fixture
        # inherits `build_asr`'s SVD null-space BASIS, and LAPACK's choice inside
        # that subspace is platform-dependent — ubuntu x64 CI's rotated basis lands
        # one coefficient on exact 0.0 (218 vs 219 was the failure this wording
        # replaces), and the default prune must drop exactly those terms and
        # nothing else.
        _, models = _ss_grid()
        a = TiledHamiltonian(models[2]; dims = (2, 1, 1))
        b = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        nzero = count(iszero, t.coef for t in b.terms)
        @test a.n_input_terms == b.n_input_terms - nzero
        @test length(a.terms) == length(b.terms) - nzero
        @test [t.coef for t in a.terms] ==
              [t.coef for t in b.terms if !iszero(t.coef)]
        cfg = _ss_cfg(a.n_sites, 3)
        u = _ss_disps(a.n_sites, 3)
        # exact zeros contribute an exact +0.0 per instance, so the accumulation is
        # bit-identical with them present
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
        N = H.n_disp_active
        @test sch.n_mobile == N
        @test sch.d_dim == 3 * N - count(H.comp_free)   # stored for diagnostics only
        s, sp, dE, kt = 0.99, 1.017, 0.37, 0.025
        lw_logv = MCs._strain_log_weight(sch, :logvolume, s, sp, dE, kt)
        lw_s = MCs._strain_log_weight(sch, :scale, s, sp, dE, kt)
        # hand-derived (Frenkel–Smit's N·ln(V′/V) plus the proposal-variable factor):
        # (3N + 3)·ln(s′/s) − ΔE/kT uniform in ln V, (3N + 2)·ln(s′/s) − ΔE/kT
        # uniform in s — N the displacement-active site count, NOT the COM-reduced D
        @test lw_logv ≈ (3 * N + 3) * log(sp / s) - dE / kt rtol = 1e-13
        @test lw_s ≈ (3 * N + 2) * log(sp / s) - dE / kt rtol = 1e-13
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
            accepted = MCs.strain_move!(st, H, sch, sc, 0.05; pressure = 0.01,
                                        step = 0.06)
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
        @test !MCs.strain_move!(st, H, sch, sc, 0.05; pressure = 0.01, step = 50.0)
        randn(twin)
        @test MCs._rng_words(st.rng) == MCs._rng_words(twin)
        @test st.strain == s_before
        @test [t.coef for t in H.terms] == coefs_before

        # validation: the named errors
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.0;
                                                    pressure = 0.01, step = 0.05)
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.05;
                                                    pressure = 0.01, step = 0.0)
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.05;
                                                    pressure = NaN, step = 0.05)
        @test_throws ArgumentError MCs.strain_move!(st, H, sch, sc, 0.05;
                                                    pressure = 0.01, step = 0.05,
                                                    proposal = :volume)
        st.strain = 2.0
        err = try
            MCs.strain_move!(st, H, sch, sc, 0.05; pressure = 0.01,
                             step = 0.05)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("outside the schedule's domain", err.msg)
    end

    @testset "strain_move!: the NPT marginal is the bare Jacobian on the §8(γ) toy" begin
        # At u ≡ 0, constant coefficients, j0 ≡ 0 and P = 0 the stationary volume
        # marginal is p(V) ∝ V^{N_mob} exactly — so the empirical mean volume MEASURES
        # the implemented exponent. Three chains: the flat and pinned (2, 1, 1)
        # fixtures share N_mob = 4 while their count(comp_free) differ by 6, so both
        # landing on V^4 is what kills the rejected COM-reduced `D/3` convention
        # (which would separate them by 2 powers); the flat fixture at (1, 1, 1) has
        # N_mob = 2, which is what shows the exponent moves with the system rather
        # than being asserted. Both proposal arms must land on the same marginal.
        for (pinned, dims, kexp, seed) in ((false, (2, 1, 1), 4, 0x12),
                                           (true, (2, 1, 1), 4, 0x11),
                                           (false, (1, 1, 1), 2, 0x13))
            sm = _ss_toy_grid(; pinned = pinned)
            for (proposal, step) in ((:logvolume, 0.75), (:scale, 0.25))
                vmean, arate, sch = _ss_toy_chain(sm, pinned, dims, proposal, step,
                                                  150_000, seed)
                @test sch.n_mobile == kexp
                @test sch.d_dim ==
                      (pinned ? 3 * kexp : (dims == (2, 1, 1) ? 6 : 3))
                @test 0.15 < arate < 0.95
                va, vb = MCs.strain_volume(sch, 0.85), MCs.strain_volume(sch, 1.15)
                vmid = (va + vb) / 2
                k = Float64(sch.n_mobile)
                # Thresholds calibrated over seeds at this length: the correct power
                # sits within ~0.006 of the mid volume, a ±1 power error shifts the
                # mean by ≥ 0.018. The ±1/3 mixed-arm trap shifts it by ~0.011,
                # INSIDE this statistic's noise — it is excluded deterministically
                # instead, by the closed-form arm gate above and the draw/weight
                # pairing testset below.
                @test abs(vmean - _ss_vk_mean(k, va, vb)) / vmid < 0.012
                @test abs(vmean - _ss_vk_mean(k + 1, va, vb)) / vmid > 0.018
                @test abs(vmean - _ss_vk_mean(k - 1, va, vb)) / vmid > 0.018
                # ...and the rejected convention's exponent for this fixture
                kD = sch.d_dim / 3
                kD == k ||
                    @test abs(vmean - _ss_vk_mean(kD, va, vb)) / vmid > 0.018
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
                got = MCs.strain_move!(st, H, sch, sc, 0.05; pressure = 0.01,
                                       step = 0.06, proposal = proposal)
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
        # The pin's PRECONDITION is asserted, not assumed: the fixture inherits
        # `build_asr`'s SVD null-space basis, and LAPACK's choice inside that
        # subspace is platform-dependent — a rotated basis (ubuntu x64 CI) is a
        # genuinely different model, where the captured values do not apply. The
        # pin fires exactly where the model matches its capture (the fingerprint
        # below, taken together with the pins on macOS aarch64), and the capture
        # platform asserts it DID fire so the detector cannot die silently.
        pin_live = MCs._fingerprint(H) == 0x3020f63b138861f4
        if Sys.isapple() && Sys.ARCH === :aarch64
            @test pin_live
        end
        r0 = run_mc(H; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100,
                    seed = 0x5150, renorm_interval = 25)
        if pin_live
            @test r0.points[1].stats[:energy].mean[1] === -12.866452813199738
            @test sum(sum, r0.final_config) === -0.17069256709356861
            @test sum(x -> sum(abs, x), r0.final_disps) === 1.4251564417024021
        end
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
        # ...the schedule of a DIFFERENT Hamiltonian is refused by the pairing check
        H1 = TiledHamiltonian(models[2]; dims = (1, 1, 1), keep_zero_terms = true)
        @test_throws ArgumentError run_mc(H1; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = 0.0)
        # ...and the GPa conversion is the exact constant, applied once
        @test MCs.GPA_PER_EV_A3 === 160.2176634
        @test MCs._resolve_pressure(sch, 160.2176634, nothing) ≈ 1.0 rtol = 1e-15

        # run_pt takes the same NPT keywords since the PT + strain slice; its
        # wiring, swap rule, determinism, and checkpoints are gated in the
        # "PT + strain" testsets below. Here: the two resolution refusals the
        # later block does not repeat, each pinned to its message
        err = try
            run_pt(H; kT = [0.05, 0.06], sweeps_therm = 1, sweeps_measure = 2,
                   strain = sch)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("exactly one of", err.msg)
        err = try
            run_pt(H; kT = [0.05, 0.06], sweeps_therm = 1, sweeps_measure = 2,
                   pressure = 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError &&
              occursin("without a strain schedule", err.msg)
    end

    @testset "checkpoint v5: a strained run resumes bit-identically" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        dir = mktempdir()
        path = joinpath(dir, "npt.jld2")
        kw = (; kT = [0.06, 0.05], sweeps_therm = 60, sweeps_measure = 120,
              renorm_interval = 30, nbins = 4, seed = 0x71, strain = sch,
              pressure = 0.01)
        a = run_mc(H; kw...)
        # writing consumes no RNG (the completed-file b is also what the
        # idempotency and refusal checks below want)
        b = run_mc(H; kw..., checkpoint = path, checkpoint_interval = 100)
        @test [p.stats[:energy].mean[1] for p in a.points] ==
              [p.stats[:energy].mean[1] for p in b.points]
        @test a.final_config == b.final_config
        @test isfile(path)
        # The resume half runs on an INTERRUPTED writer (the S12 pattern — a
        # completed mc file always ends at the completed marker, so mid-run
        # continuation teeth need a run that actually stops): the poison dies at
        # temp-2 measure sweep 70, the file holds the periodic write at measure
        # sweep 60 — accumulators, chain scale and all — and the resumed run
        # must be the uninterrupted one bit for bit.
        nmeas = Ref(0)
        poison = Observable(:poison, 1,
                            v -> (nmeas[] += 1) >= 190 ?
                                 error("poison interrupt") : 0.0)
        obsR = [standard_observables(H); Observable(:poison, 1, v -> 0.0)]
        aP = run_mc(H; kw..., observables = obsR)
        err = try
            run_mc(H; kw..., observables = [standard_observables(H); poison],
                   checkpoint = path, checkpoint_interval = 100)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("poison", err.msg)
        mid = MCs.jldopen(path, "r") do f
            (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
        end
        @test mid[1] == 2 && mid[2] == "measure" && 0 < mid[3] < 120
        # `run_mc` hands H back at the reference; deliberately knock it OFF the
        # reference before resuming — resume reinstalls the reference before the
        # fingerprint check, so the caller's coefficient state must not matter, and
        # the continued run is the uninterrupted one, bit for bit
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.015);
                          recheck_translation = false)
        c = resume(path, H; observables = obsR, strain = sch)
        a = aP                     # every comparison below is against the
                                   # same-trajectory run carrying :poison stats
        @test [p.stats[:energy].mean[1] for p in a.points] ==
              [p.stats[:energy].mean[1] for p in c.points]
        @test isequal([p.acceptance_strain for p in a.points],
                      [p.acceptance_strain for p in c.points])
        @test a.final_config == c.final_config
        @test a.final_disps == c.final_disps
        # ...and resuming the now-completed file is idempotent (`c`'s own
        # checkpointing completed `path`, which now stores the :poison name —
        # every later reader supplies the matching list)
        d = resume(path, H; observables = obsR, strain = sch)
        @test a.final_config == d.final_config

        # the handshake's refusals, each by name
        err = try
            resume(path, H; observables = obsR)
            nothing
        catch e
            e
        end
        @test err isa Exception && occursin("strained (NPT) run", sprint(showerror, err))
        sm2, _ = _ss_grid(; scales = [0.97, 1.0, 1.03])
        sch2 = StrainSchedule(sm2, H)
        err = try
            resume(path, H; observables = obsR, strain = sch2)
            nothing
        catch e
            e
        end
        @test err isa Exception &&
              occursin("grid fingerprint", sprint(showerror, err))
        # a fixed-cell checkpoint refuses a schedule...
        fpath = joinpath(dir, "fixed.jld2")
        run_mc(H; kT = 0.05, sweeps_therm = 20, sweeps_measure = 40, nbins = 4,
               seed = 0x72, checkpoint = fpath)
        err = try
            resume(fpath, H; strain = sch)
            nothing
        catch e
            e
        end
        @test err isa Exception &&
              occursin("fixed-cell run", sprint(showerror, err))
        # ...and an older-schema file is refused with the hazard named: v4 lane
        # strains were never live, so an older READER given a v5 strained-PT file
        # would silently continue it as fixed-cell — which is why the version moved
        p3 = joinpath(dir, "old.jld2")
        MCs.jldopen(p3, "w") do f
            f["schema_version"] = 4
        end
        err = try
            resume(p3, H)
            nothing
        catch e
            e
        end
        @test err isa Exception && occursin("schema v4", sprint(showerror, err)) &&
              occursin("strained PT", sprint(showerror, err))
    end

    @testset "strain driver corners: pairing, carryover, cadence, domain, payload" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)

        # a same-shape DIFFERENT model is refused by the term fingerprint, not merely
        # by counts: perturb one model's crystal so the images move but every count
        # stays — simplest available: another grid with different scales built
        # against a DIFFERENT H (1×1×1), whose counts differ; and the true same-shape
        # case via a schedule from an H built at the same dims from the zero_key grid
        smz, mz = _ss_grid(; zero_key = 2)
        Hz = TiledHamiltonian(mz[2]; dims = (2, 1, 1), fixed_reference = true,
                              keep_zero_terms = true)
        schz = StrainSchedule(smz, Hz)
        @test schz.term_fp == sch.term_fp   # same basis, same atoms/images — equal
        # ...so the counts-equal, structure-equal case passes the pairing check, and
        # the structural fingerprint's job is the cross-model case, exercised in the
        # run_mc wiring testset via the (1, 1, 1) Hamiltonian (count mismatch) and
        # here via a doctored copy:
        bad = MCs.StrainSchedule(sch.scales, sch.abscissa, sch.x0, sch.xw,
                                 sch.coefpoly, sch.j0poly, sch.v_train, sch.n_cells,
                                 sch.n_mobile, sch.d_dim, sch.term_fp ⊻ 0x1)
        err = try
            run_mc(H; kT = 0.05, sweeps_therm = 1, sweeps_measure = 2, strain = bad,
                   pressure = 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("DIFFERENT Hamiltonian", err.msg)

        # carryover = false: the independent restart returns the cell to the
        # reference, and the run hands H back at the reference
        r = run_mc(H; kT = [0.06, 0.05], sweeps_therm = 30, sweeps_measure = 40,
                   carryover = false, seed = 0x81, strain = sch, pressure = 0.0,
                   renorm_interval = 20)
        @test length(r.points) == 2
        @test all(p -> !isnan(p.acceptance_strain), r.points)
        H2 = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        set_coefficients!(H2, MCs.strain_coefficients(sch, 1.0))
        @test [t.coef for t in H.terms] == [t.coef for t in H2.terms]

        # strain_interval > 1: the cadence fires and the run stays consistent
        r3 = run_mc(H; kT = 0.05, sweeps_therm = 30, sweeps_measure = 60,
                    seed = 0x82, strain = sch, pressure = 0.0, strain_interval = 3,
                    renorm_interval = 30)
        @test !isnan(r3.points[1].acceptance_strain)

        # a grid that excludes the reference scale refuses up front, by name
        smo, mo = _ss_grid(; scales = [1.01, 1.03, 1.05])
        Ho = TiledHamiltonian(mo[2]; dims = (2, 1, 1), keep_zero_terms = true)
        scho = StrainSchedule(smo, Ho)
        err = try
            run_mc(Ho; kT = 0.05, sweeps_therm = 1, sweeps_measure = 2,
                   strain = scho, pressure = 0.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("regrid around the reference", err.msg)

        # the cell scale is replica-exchange payload: it swaps with the state
        stA = MCs.ChainState(H, _ss_cfg(H.n_sites, 61), Xoshiro(0x61), 0.3;
                             disps = _ss_disps(H.n_sites, 61))
        stB = MCs.ChainState(H, _ss_cfg(H.n_sites, 62), Xoshiro(0x62), 0.3;
                             disps = _ss_disps(H.n_sites, 62))
        stA.strain, stB.strain = 1.01, 0.99
        MCs._swap_payload!(stA, stB)
        @test stA.strain == 0.99 && stB.strain == 1.01

        # An accepted rescale keeps the escape DETECTOR armed and covariant while
        # leaving the phase's REPORTING accumulators in absolute lengths. Every one of
        # the six length fields is asserted by name and in the right direction, because
        # the split is the whole content of `_rescale_escape!` and a mutation that
        # rescales one group like the other is exactly the defect this replaces: the
        # detector fields must convert to the new frame (they answer "is the r.m.s.
        # growing BEYOND the affine map?"), and `disp_ms_sum`/`disp_max` must not (they
        # are the phase's time-average and running maximum of |u|, the quantities
        # `TempResult.disp_rms`/`disp_max` report and the `:u2` observable must agree
        # with — rescaling them re-expresses the phase's past in the last accepted
        # move's frame).
        st = MCs.ChainState(H, _ss_cfg(H.n_sites, 63), Xoshiro(0x63), 0.3;
                            disps = _ss_disps(H.n_sites, 63))
        sc = MCs.StrainScratch(H)
        st.disp_rms, st.disp_max, st.disp_rms0 = 0.02, 0.05, 0.019
        st.disp_ms_sum, st.disp_blk_sum, st.disp_ref_ms = 1e-3, 4e-4, 3.9e-4
        st.disp_checks, st.disp_blk_n, st.disp_blk_cap = 7, 2, 4
        naccept = 0
        nlam = 0
        for _ = 1:20
            s0 = st.strain
            rms0, rms00, max0 = st.disp_rms, st.disp_rms0, st.disp_max
            ms0, blk0, ref0 = st.disp_ms_sum, st.disp_blk_sum, st.disp_ref_ms
            if MCs.strain_move!(st, H, sch, sc, 0.05; pressure = 0.0, step = 0.05)
                naccept += 1
                lam = st.strain / s0
                lam == 1.0 || (nlam += 1)        # a λ ≡ 1 move could not tell them apart
                # covariant: the detector's anchors move to the new frame
                @test st.disp_rms == lam * rms0
                @test st.disp_rms0 == lam * rms00
                @test st.disp_blk_sum == lam^2 * blk0
                @test st.disp_ref_ms == lam^2 * ref0
                # absolute: the reported phase statistics are NOT re-framed
                @test st.disp_ms_sum == ms0
                @test st.disp_max == max0
                @test st.disp_checks == 7        # counters untouched — ladder armed
            end
        end
        @test naccept > 0
        @test nlam > 0
    end

    @testset "a pure-spin volume grid: J(V) with no displacement channel" begin
        # A spin-only model on a volume grid is legitimate physics (J(V) plus an
        # elastic j0(s)); the strain machinery must degrade to N_mob = 0 — the log
        # weight is then the bare proposal factor — rather than refuse or crash.
        pspec(cr, s) = BasisSpec(cr; lmax = 1, pmax = 0,
                                 sectors = [Sector(spin = (sites = 1:2,),
                                                   cutoff = 1.1s)])
        function pmk(s)
            cr = _ss_crystal(s)
            b = SLCEBasis(cr, pspec(cr, s))
            return SLCEModel(b, 0.1 * (s - 1), 0.2 .* ones(n_salcs(b)))
        end
        ms = [pmk(s) for s in [0.98, 1.0, 1.02]]
        smp = SLCE.StrainedModels(ms, [0.98, 1.0, 1.02])
        Hp = TiledHamiltonian(ms[2]; dims = (2, 1, 1), keep_zero_terms = true)
        schp = StrainSchedule(smp, Hp)
        @test schp.n_mobile == 0 && schp.d_dim == 0
        set_coefficients!(Hp, MCs.strain_coefficients(schp, 1.0))
        stp = MCs.ChainState(Hp, _ss_cfg(Hp.n_sites, 71), Xoshiro(0x72), 0.3)
        scp = MCs.StrainScratch(Hp)
        for _ = 1:20
            MCs.strain_move!(stp, Hp, schp, scp, 0.05; pressure = 0.0, step = 0.05)
        end
        @test stp.att_strain == 20
        @test MCs.in_strain_domain(schp, stp.strain)
        @test stp.energy == MCs._total_energy(Hp, stp.zrows)
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

    @testset "§8(ζ) dE/dV: exact vs finite differences, upstream, and purity" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)
        cfg = _ss_cfg(H.n_sites, 21)
        w = _ss_disps(H.n_sites, 21)          # scaled coordinates, held fixed below

        # the fixture must exercise the k ≥ 1 half of the Euler degree (the |u|^{2k}
        # prefactor) or the `2k` term of `2k + l` is structurally untested
        @test any(kl -> kl[1] >= 1, H.layout.disp_factors)
        deg = MCs._term_disp_degrees(H)
        @test any(>(0), deg)

        # central finite difference of the SAME interpolated family, at fixed scaled
        # coordinates: coefficients from the schedule, displacements affinely rescaled,
        # the elastic j0 included (dE_total/dV covers config + j0; P·V is separate).
        # Parameterized over ALL THREE interpolation abscissas — `_sch_dz_ds`'s
        # :volume and :logvolume chain-rule branches exist only here, and a fixture
        # family that never leaves the default :linear would leave them ungated (a
        # dropped `/xw` or a wrong dx/ds is a 1e6× FD discriminator on this fixture).
        Ht = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        etot = function (scha, sp)
            set_coefficients!(Ht, MCs.strain_coefficients(scha, sp))
            return total_energy(Ht, cfg, [sp .* x for x in w]) +
                   scha.n_cells * MCs.strain_j0(scha, sp)
        end
        h = 1e-5
        for ab in (:linear, :volume, :logvolume)
            sma = SLCE.StrainedModels(sm.models, [0.98, 1.0, 1.02]; abscissa = ab)
            scha = StrainSchedule(sma, H)
            for s in (0.985, 1.0, 1.013)
                uphys = [s .* x for x in w]    # the state as sampled at scale s
                dedv = MCs.energy_volume_derivative(scha, H, cfg, uphys, s)
                dv_ds = 3 * scha.n_cells * scha.v_train * s^2
                fd = (etot(scha, s + h) - etot(scha, s - h)) / (2h) / dv_ds
                @test dedv ≈ fd rtol = 1e-6
            end
        end

        # a pure function of (schedule, state, s) — H's installed coefficients are
        # never read, so whatever the caller last installed cannot move the answer
        up1 = [1.0 .* x for x in w]
        set_coefficients!(H, MCs.strain_coefficients(sch, 0.99))
        v1 = MCs.energy_volume_derivative(sch, H, cfg, up1, 1.0)
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.02))
        @test MCs.energy_volume_derivative(sch, H, cfg, up1, 1.0) === v1

        # upstream cross-gate at u = 0 on the training cell: the grid derivative is
        # dE/dη = s·dE/ds per CELL, and ours is dE/dV. Both sides build the same
        # centred Vandermonde, so what this pins is the η-convention factor
        # (`dE/dη = s·dE/ds`) and the coefficient-drift path — at u = 0 every
        # displacement row is zero and the Euler half contributes nothing (the FD
        # gate above owns that half)
        H1 = TiledHamiltonian(models[2]; dims = (1, 1, 1), keep_zero_terms = true)
        sch1 = StrainSchedule(sm, H1)
        cfg1 = _ss_cfg(H1.n_sites, 33)
        for s in (0.99, 1.008)
            dedv = MCs.energy_volume_derivative(sch1, H1, cfg1,
                                                zeros(SVector{3,Float64}, H1.n_sites), s)
            dv_ds = 3 * sch1.v_train * s^2
            @test s * dv_ds * dedv ≈
                  SLCE.grid_strain_derivative(sm, s; spins = MCs.to_matrix(cfg1)) rtol =
                  1e-8
        end

        # a pure-spin grid takes the disps-free method and is drift-only (deg ≡ 0)
        pspec(cr, s) = BasisSpec(cr; lmax = 1, pmax = 0,
                                 sectors = [Sector(spin = (sites = 1:2,),
                                                   cutoff = 1.1s)])
        pmk = function (s)
            cr = _ss_crystal(s)
            b = SLCEBasis(cr, pspec(cr, s))
            return SLCEModel(b, 0.1 * (s - 1)^2,
                             (0.2 + 0.05 * (s - 1)) .* ones(n_salcs(b)))
        end
        msp = [pmk(s) for s in [0.95, 1.0, 1.05]]
        smp = SLCE.StrainedModels(msp, [0.95, 1.0, 1.05])
        Hp = TiledHamiltonian(msp[2]; dims = (2, 1, 1), keep_zero_terms = true)
        schp = StrainSchedule(smp, Hp)
        @test all(==(0), MCs._term_disp_degrees(Hp))
        cfgp = _ss_cfg(Hp.n_sites, 47)
        Hpt = TiledHamiltonian(msp[2]; dims = (2, 1, 1), keep_zero_terms = true)
        ep = function (sp)
            set_coefficients!(Hpt, MCs.strain_coefficients(schp, sp))
            return total_energy(Hpt, cfgp) + schp.n_cells * MCs.strain_j0(schp, sp)
        end
        sps = 1.02
        @test MCs.energy_volume_derivative(schp, Hp, cfgp, sps) ≈
              (ep(sps + h) - ep(sps - h)) / (2h) /
              (3 * schp.n_cells * schp.v_train * sps^2) rtol = 1e-6

        # refusals: out-of-domain scale, and a schedule paired with a different H
        @test_throws ArgumentError MCs.energy_volume_derivative(sch, H, cfg, up1, 0.5)
        @test_throws ArgumentError MCs.energy_volume_derivative(sch1, H, cfg, up1, 1.0)

        # gauge invariance under re-centring: on a translation-flat model a rigid
        # shift of one whole displacement component along a free direction must not
        # move dE/dV — certified-flat directions have zero force sum, which is the
        # property the MCView's COM-reduced `disps` rely on, and the one that would
        # break silently if the Euler argument or `_term_disp_degrees` were touched
        smf = _ss_toy_grid(pinned = false)
        Hf = TiledHamiltonian(smf.models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        schf = StrainSchedule(smf, Hf)
        @test Hf.n_disp_comps == 2 && all(Hf.comp_free)
        cfgf = _ss_cfg(Hf.n_sites, 61)
        uf = _ss_disps(Hf.n_sites, 61)
        sg = 1.04
        d0 = MCs.energy_volume_derivative(schf, Hf, cfgf, uf, sg)
        ushift = copy(uf)
        tshift = SVector(0.21, -0.13, 0.08)
        for q = Int(Hf.disp_comp_ptr[1]):(Int(Hf.disp_comp_ptr[2]) - 1)
            sidx = Int(Hf.disp_comp_sites[q])
            ushift[sidx] = ushift[sidx] + tshift
        end
        @test ushift != uf
        @test MCs.energy_volume_derivative(schf, Hf, cfgf, ushift, sg) ≈ d0 rtol = 1e-12
    end

    @testset "§8(ζ) pressure_diagnostics: the NPT mechanical-equilibrium identity" begin
        # A stiff elastic well centred well inside a wide domain: the identity
        #     N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩ = P_applied
        # holds up to boundary terms of the bounded grid, so the fixture confines the
        # volume distribution ≥ 5σ from both edges. Measured decomposition at
        # P = 0.01: ideal +0.0038, coefficient drift −0.0007, virial +0.0039,
        # j0 −(−0.0105), err ≈ 0.0007 — so a lost j0/n_cells factor (≫ 5σ) or a
        # dropped virial (~7σ) bias it past the gate, while the drift term (~1σ) and
        # the volume-power convention (kT·⟨1/V⟩ ≈ 1.3σ per unit of N_mob) are below
        # its resolution here: those are owned by the FD gate above and the §8(γ)
        # marginal toy respectively. (`D/3` vs `N_mob` is indistinguishable on this
        # fully pinned fixture anyway — `count(comp_free) = 0` makes them coincide.)
        #
        # The displacement energy must be bounded below or the chain has no stationary
        # distribution at all — the shared Einstein-well fixture (`_ss_zeta_grid`).
        zsm, zmodels = _ss_zeta_grid()
        # the well pins the rigid shift — that is the point (an absolute reference
        # frame, no re-centring, no ASR machinery in the fixture's way)
        H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                             fixed_reference = true)
        sch = StrainSchedule(zsm, H)
        pd = MCs.pressure_diagnostics(sch, H)
        @test [o.name for o in pd.observables] == [:strain_dEdV, :strain_invV]
        @test [e.name for e in pd.evaluables] == [:pressure]

        P = 0.01                               # eV/Å³ ≈ 1.6 GPa
        r = run_mc(H; kT = 0.05, sweeps_therm = 500, sweeps_measure = 4000,
                   seed = 0x0857, renorm_interval = 50, strain = sch, pressure = P,
                   observables = [standard_observables(H); pd.observables],
                   evaluables = [standard_evaluables(H); pd.evaluables])
        pt = r.points[1]
        @test 0.0 < pt.acceptance_strain < 1.0
        pst = pt.stats[:pressure]
        @test isfinite(pst.mean[1]) && isfinite(pst.err[1])
        # the gate must be able to fail: the error bar is small against P itself.
        # 4σ, not 5: the pinned seed sits at 1.5σ (10 seeds: −0.7σ), and the
        # tighter bound is what gives the dropped-virial mutation (a 5.3σ shift)
        # real headroom instead of a 6% margin
        @test pst.err[1] < P / 3
        @test abs(pst.mean[1] - P) < 4 * pst.err[1]
        # the ideal-gas term genuinely participates (same order as P on this fixture)
        ideal = sch.n_mobile * 0.05 * pt.stats[:strain_invV].mean[1]
        @test ideal > P / 5

        # view-level refusals: a different Hamiltonian, and a fixed-cell view
        cfg = _ss_cfg(H.n_sites, 5)
        u0 = zeros(SVector{3,Float64}, H.n_sites)
        Hother = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                                  fixed_reference = true)
        vother = MCView(Hother, cfg, u0, 0.0, 1.0)
        @test_throws ArgumentError pd.observables[1].f(vother)
        vfixed = MCView(H, cfg, u0, 0.0)
        @test_throws ArgumentError pd.observables[1].f(vfixed)
        @test_throws ArgumentError pd.observables[2].f(vfixed)

        # construction-level pairing refusal
        Hsmall = TiledHamiltonian(zmodels[2]; dims = (1, 1, 1), keep_zero_terms = true,
                                  fixed_reference = true)
        @test_throws ArgumentError MCs.pressure_diagnostics(sch, Hsmall)
    end

    @testset "npt_observables: W = E + n_cells·j0 + P·V and the isobaric C" begin
        zsm, zmodels = _ss_zeta_grid()
        H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                             fixed_reference = true)
        sch = StrainSchedule(zsm, H)
        P = 0.015
        nw = MCs.npt_observables(sch, H; pressure = P)
        @test [o.name for o in nw.observables] == [:enthalpy, :enthalpy2]
        @test [e.name for e in nw.evaluables] == [:npt_specific_heat]

        # The formula, against the FIXTURE's analytics rather than the schedule's
        # helpers: `_ss_zeta_grid` builds its models with j0(s) = 40η² per training
        # cell — a quadratic the 3-node interpolant must reproduce to roundoff at
        # nodes and between them alike — and `_ss_crystal(1)` is the cubic a = 3 Å
        # cell, so V(s) = n_cells·27·s³. This anchor owns the formula structure
        # (the n_cells placement on j0, the supercell V, the s³, the pressure-unit
        # conversion): every dropped/misplaced-term mutation dies here at machine
        # precision, which the statistical FDT gate below cannot resolve.
        cfg = _ss_cfg(H.n_sites, 21)
        us = _ss_disps(H.n_sites, 22)
        E = -0.375
        for s in (0.93, 1.0, 1.045)
            η = s - 1
            # n_cells hardcoded to the fixture's hand-counted 2 (4 sites / 2 cell
            # atoms on dims = (2, 1, 1)) so the anchor is independent of
            # `sch.n_cells` too
            Wref = E + 2 * 40.0 * η^2 + P * 2 * 27.0 * s^3
            v = MCView(H, cfg, us, E, s)
            @test nw.observables[1].f(v) ≈ Wref rtol = 1e-12
            @test nw.observables[2].f(v) ≈ Wref^2 rtol = 1e-12
        end
        # the GPa arm resolves through the same conversion the run keywords use
        nwg = MCs.npt_observables(sch, H; pressure_GPa = P * MCs.GPA_PER_EV_A3)
        vg = MCView(H, cfg, us, E, 0.97)
        @test nwg.observables[1].f(vg) ≈ nw.observables[1].f(vg) rtol = 1e-14
        # a per-lane coefficient clone's view is as good as the parent's (what makes
        # these usable under a strained run_pt, unlike pressure_diagnostics)
        Hc = MCs._coefficient_clone(H)
        @test nw.observables[1].f(MCView(Hc, cfg, us, E, 1.02)) ==
              nw.observables[1].f(MCView(H, cfg, us, E, 1.02))

        # Fluctuation–response: the sampled measure is p ∝ V^{N_mob}·e^{−βW} with a
        # β-independent Jacobian AND a β-independent (truncated) volume domain, so
        #     var(W)/(k_BT)² = d⟨W⟩/d(k_BT)  (= C/k_B)
        # exactly — the ensemble-level identity that makes :npt_specific_heat the
        # enthalpy's temperature derivative. Gate: central FD of ⟨W⟩ across
        # kT = 0.04..0.06 vs the evaluable at the midpoint. Measured over 4 seed
        # sets: deviations −0.2σ, 0.8σ, 2.7σ, 0.0σ (pinned set: −0.2σ); 4σ gate.
        # The central-difference truncation bias (Δ²/6)·d³⟨W⟩/d(kT)³ is
        # unresolvable against σ at this cost (C's own kT-trend is inside the
        # noise) and is absorbed in the 4σ headroom.
        # Scoping: the resolution σ ≈ 0.6–1.0 k_B cannot see the config-only
        # mutation (var(E_config) sits ≈ 0.3–0.6 k_B ≡ 3–6 % below var(W) across
        # the three kT here — the measured caveat the observable exists to fix),
        # so this gate checks the ensemble consistency of the ⟨W⟩/var(W) pair; the
        # formula itself is owned by the exact anchor above and the chain-side
        # weights by the closed-form log-weight and swap-bracket gates. Nor can
        # any gate on THIS fixture fence `scope = :energy`: n_active ==
        # n_spin_active == n_disp_active == 4 (every site carries both channels),
        # so a :spin-scope mutation is invisible here — and the FD comparison
        # re-multiplies by n_active, cancelling the injected count entirely. The
        # scope convention is held by the docs and by the einstein-terms C gate
        # of the standard :specific_heat, not by this testset.
        kts = (0.04, 0.05, 0.06)
        sts = map(enumerate(kts)) do (i, kt)
            r = run_mc(H; kT = kt, sweeps_therm = 500, sweeps_measure = 6000,
                       seed = 0x33c0 + i, renorm_interval = 50, strain = sch,
                       pressure = P,
                       observables = [standard_observables(H); nw.observables],
                       evaluables = [standard_evaluables(H); nw.evaluables])
            r.points[1].stats
        end
        fd = (sts[3][:enthalpy].mean[1] - sts[1][:enthalpy].mean[1]) /
             (kts[3] - kts[1])
        fd_err = sqrt(sts[3][:enthalpy].err[1]^2 + sts[1][:enthalpy].err[1]^2) /
                 (kts[3] - kts[1])
        cst = sts[2][:npt_specific_heat]
        C = H.n_active * cst.mean[1]
        σ = sqrt(fd_err^2 + (H.n_active * cst.err[1])^2)
        @test isfinite(C) && C > 0
        @test σ < C / 8                    # the gate can fail: ≥ 8σ resolution on C
        @test abs(fd - C) < 4σ
        # the strained-run caveat is real on this very fixture: W ≠ E_config as a
        # random variable (⟨W⟩ − ⟨E⟩ = n_cells·⟨j0⟩ + P·⟨V⟩ > 0 — j0 ≥ 0 and V > 0
        # pointwise), so the pair is not redundant
        @test sts[2][:enthalpy].mean[1] > sts[2][:energy].mean[1]

        # refusals: the pressure XOR (as in run_mc), the construction pairing (with
        # a valid pressure, so the pairing check is what throws), the foreign-view
        # guard, the fixed-cell view, and the out-of-domain scale
        @test_throws ArgumentError MCs.npt_observables(sch, H)
        @test_throws ArgumentError MCs.npt_observables(sch, H; pressure = P,
                                                       pressure_GPa = 1.0)
        Hsmall = TiledHamiltonian(zmodels[2]; dims = (1, 1, 1),
                                  keep_zero_terms = true, fixed_reference = true)
        @test_throws ArgumentError MCs.npt_observables(sch, Hsmall; pressure = P)
        vsmall = MCView(Hsmall, _ss_cfg(Hsmall.n_sites, 3),
                        zeros(SVector{3,Float64}, Hsmall.n_sites), 0.0, 1.0)
        @test_throws ArgumentError nw.observables[1].f(vsmall)
        # the guard is identity on a shared structural array, so a SAME-SHAPE
        # Hamiltonian from another grid point's model — which passes every count —
        # is refused too, instead of silently getting H's j0/P·V applied to its
        # own energy (the `_check_strain_pairing` hazard at measurement time)
        Hforeign = TiledHamiltonian(zmodels[1]; dims = (2, 1, 1),
                                    keep_zero_terms = true, fixed_reference = true)
        @test_throws ArgumentError nw.observables[1].f(MCView(Hforeign, cfg, us,
                                                              E, 1.0))
        vfixed = MCView(H, cfg, us, 0.0)
        @test_throws ArgumentError nw.observables[1].f(vfixed)
        @test_throws ArgumentError nw.observables[2].f(vfixed)
        lo, hi = MCs.strain_domain(sch)
        @test_throws ArgumentError nw.observables[1].f(MCView(H, cfg, us, E,
                                                              hi + 0.05))
        # ...and a FIXED-CELL run refuses the observables by name at ENTRY, in both
        # drivers, rather than throwing after a spent thermalization phase
        err = try
            run_mc(H; kT = 0.05, sweeps_therm = 1, sweeps_measure = 2,
                   observables = [standard_observables(H); nw.observables])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("needs a strained run", err.msg)
        err = try
            run_pt(H; kT = [0.05, 0.08], sweeps_therm = 1, sweeps_measure = 2,
                   observables = [standard_observables(H); nw.observables])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("needs a strained run", err.msg)
    end

    @testset "PT + strain: coefficient clones and the NPT swap rule" begin
        sm, models = _ss_grid()
        H = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        sch = StrainSchedule(sm, H)

        # `_coefficient_clone`: the EXACT sharing partition, over every field of
        # both structs — not a hand-picked sample. This is the gate for CLAUDE.md
        # (7)'s hazard: a future coefficient-carrying array added to the programs
        # (and to `set_coefficients!`) but not to the clone would be silently
        # shared by every PT lane, and only an exhaustive partition notices. It
        # doubles as the field-order audit of the raw clone constructor — a
        # transposed same-typed neighbour pair shows up as an extra entry here.
        Hc = MCs._coefficient_clone(H)
        @test [f for f in fieldnames(MCs.TiledHamiltonian)
               if getfield(H, f) !== getfield(Hc, f)] == [:terms, :progs]
        @test [f for f in fieldnames(MCs._ContractionPrograms)
               if getfield(H.progs, f) !== getfield(Hc.progs, f)] ==
              [:sent_w, :term_coef]
        @test Hc.progs.term_coef == H.progs.term_coef &&
              Hc.progs.sent_w == H.progs.sent_w && Hc.terms == H.terms
        # rewriting the clone leaves the parent untouched — checked on BOTH
        # coefficient carriers: `progs.term_coef` (what the energy walks read) and
        # the checkpoint fingerprint (which reads `terms[k].coef` — a clone
        # regression leaking into `H.terms` would be invisible to every energy
        # assertion while breaking every strained-PT resume)
        before = copy(H.progs.term_coef)
        fp0 = MCs._fingerprint(H)
        set_coefficients!(Hc, MCs.strain_coefficients(sch, 1.015);
                          recheck_translation = false)
        @test H.progs.term_coef == before
        @test MCs._fingerprint(H) == fp0
        cfg = _ss_cfg(H.n_sites, 3)
        us = _ss_disps(H.n_sites, 4)
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.015);
                          recheck_translation = false)
        @test total_energy(Hc, cfg, us) === total_energy(H, cfg, us)
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.0);
                          recheck_translation = false)

        # The swap rule, pinned EXACTLY (the statistical marginal gate below cannot
        # resolve it — measured: patching the NVT rule in moves the rung marginals
        # ≤ 0.6σ at its cost, while it shifts this logw by ~2.3): two hand-built
        # lanes at different (kT, s), the hand-derived
        #     logw = (1/kT_a − 1/kT_b)·[(E_a + n_cells·j0(s_a) + P·V_a) − (…_b)]
        # and uniforms bracketing exp(logw) one ulp on each side. The bracket also
        # kills the dropped-j0 (Δlogw ≈ 1.1) and dropped-P·V (≈ 1.2) mutations.
        P = 0.02
        mklane = function (kt, s_target, seed)
            Hl = MCs._coefficient_clone(H)
            set_coefficients!(Hl, MCs.strain_coefficients(sch, s_target);
                              recheck_translation = false)
            rng = Xoshiro(seed)
            lcfg = MCs.SpinConfig([SVector{3,Float64}(normalize(randn(rng, 3)))
                                   for _ = 1:H.n_sites])
            lus = [SVector{3,Float64}(0.02 .* randn(rng, 3)) for _ = 1:H.n_sites]
            st = MCs.ChainState(Hl, lcfg, rng, 0.6; disps = lus, step_u = 0.01)
            st.strain = s_target
            return MCs._PTLane(st, [MCs.SweepScratch(Hl)], kt, 1.0 / kt,
                               MCs.ObsAccumulator[], 0, Hl,
                               (sch, MCs.StrainScratch(Hl)))
        end
        a = mklane(0.05, 0.97, 1)
        b = mklane(0.08, 1.02, 2)
        # the SAME association as `_swap_dweight` — sum of differences, never
        # per-lane totals differenced (the totals form loses `ulp(|W|)` vs
        # `ulp(|ΔW|)`, a conditioning gap growing with n_cells; bitwise pairing
        # with the implementation is the point of this gate).
        # Oracle scoping: `strain_j0`/`strain_volume` here are the
        # implementation's own helpers — what this gate pins is the RULE
        # STRUCTURE (which terms enter, the n_cells/P placement, the β factor,
        # the sign), not their values. Those inputs are trustworthy because they
        # are independently gated above: exact-quadratic interpolation of
        # independently built models (the conversion testset) and the
        # Frenkel–Smit closed-form log-weight gate.
        dW = (a.st.energy - b.st.energy) +
             sch.n_cells * (MCs.strain_j0(sch, a.st.strain) -
                            MCs.strain_j0(sch, b.st.strain)) +
             P * (MCs.strain_volume(sch, a.st.strain) -
                  MCs.strain_volume(sch, b.st.strain))
        logw = (1 / a.kt - 1 / b.kt) * dW
        pacc = exp(min(0.0, logw))
        @test 0.0 < pacc < 1.0            # the bracket must have two sides
        att = zeros(Int, 1)
        acc = zeros(Int, 1)
        ea, eb = a.st.energy, b.st.energy
        sa, sb = a.st.strain, b.st.strain
        Ha, Hb = a.H, b.H
        MCs._attempt_swap!(a, b, 1, prevfloat(pacc), att, acc, P)
        @test acc[1] == 1                 # just below the boundary: accepted
        @test a.st.energy == eb && b.st.energy == ea
        @test a.st.strain == sb && b.st.strain == sa
        # the Hamiltonian reference travels WITH the payload it describes
        @test a.H === Hb && b.H === Ha
        MCs._swap_lanes!(a, b)            # restore for the reject bracket
        MCs._attempt_swap!(a, b, 1, nextfloat(pacc), att, acc, P)
        @test att[1] == 2 && acc[1] == 1  # just above: rejected, payload untouched
        @test a.st.energy == ea && a.st.strain == sa && a.H === Ha

        # THE BYTE-NEUTRALITY PIN (fixed cell): this run_pt trajectory was captured
        # before the PT + strain wiring landed (at commit 67d9363) and must stay
        # bit-identical — without a schedule every lane holds the caller's H, the
        # swap weight reduces to the chain-energy difference, and no code path
        # consumes different randomness. On a FRESHLY BUILT Hamiltonian, so the
        # pin is independent of the `set_coefficients!` installs above (a one-ULP
        # shift in the schedule's Horner restore must not trip a fixed-cell pin).
        # If an intentional sampler change moves it, recapture; anything else
        # moving it is the regression this pin catches.
        # Platform scope, as on the run_mc pin: the fixture inherits LAPACK's ASR
        # null-space basis, so the pin fires exactly where the model matches its
        # capture, and the capture platform (macOS aarch64) asserts that it fired.
        Hpin = TiledHamiltonian(models[2]; dims = (2, 1, 1), keep_zero_terms = true)
        pin_live = MCs._fingerprint(Hpin) == 0x3020f63b138861f4
        if Sys.isapple() && Sys.ARCH === :aarch64
            @test pin_live
        end
        r0 = run_pt(Hpin; kT = [0.05, 0.07], sweeps_therm = 50,
                    sweeps_measure = 100, seed = 0x5150, renorm_interval = 25,
                    exchange_interval = 5, ntasks = 1)
        if pin_live
            @test [p.stats[:energy].mean[1] for p in r0.points] ==
                  [-13.060570421931075, -12.657092511551113]
            @test [sum(sum, c) for c in r0.final_configs] ==
                  [0.7053385500370698, 0.8192785243733386]
            @test [sum(x -> sum(abs, x), d) for d in r0.final_disps] ==
                  [1.7833703780033128, 1.2819181240379507]
            # a trajectory quantity too (it merely happened to agree across
            # platforms at these counts), so it stays under the same scope
            @test r0.swap_acceptance == [0.21428571428571427]
        end
    end

    @testset "PT + strain: run_pt wiring, determinism, and rung marginals" begin
        zsm, zmodels = _ss_zeta_grid()
        H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                             fixed_reference = true)
        sch = StrainSchedule(zsm, H)
        ref = MCs.strain_coefficients(sch, 1.0)
        P = 0.01
        # npt_observables ride along: they are pure closures over the immutable
        # schedule, so concurrent per-lane measurement through coefficient clones
        # is exactly the case they must support (unlike pressure_diagnostics)
        nw = MCs.npt_observables(sch, H; pressure = P)
        obsc = [standard_observables(H); Observable(:scale, 1, v -> MCs.strain(v));
                nw.observables]
        kts = [0.05, 0.08]

        kw = (; kT = kts, sweeps_therm = 300, sweeps_measure = 3000, seed = 0xa7,
              renorm_interval = 50, exchange_interval = 2, strain = sch,
              pressure = P, observables = obsc,
              evaluables = [standard_evaluables(H); nw.evaluables])
        pt = run_pt(H; kw..., ntasks = 2)
        # the wiring is live: strain moves fire on every lane, exchanges happen
        # (in-domain sampling needs no assert — `strain_move!` rejects rather than
        # clamps, so it holds by construction and a mean-in-interval test is
        # tautological)
        @test all(p -> 0.0 < p.acceptance_strain < 1.0, pt.points)
        @test 0.0 < pt.swap_acceptance[1] < 1.0
        # the caller's H is handed back at the reference scale
        @test H.progs.term_coef ==
              [ref[H.term_source[k]] * H.term_scale[k] for k in eachindex(H.terms)]

        # determinism: the serial reference schedule and one-task-per-lane agree
        # bit for bit with the strain channel live (P3, extended)
        pts = run_pt(H; kw..., ntasks = 1)
        @test [p.stats[:energy].mean[1] for p in pt.points] ==
              [p.stats[:energy].mean[1] for p in pts.points]
        @test [p.stats[:scale].mean[1] for p in pt.points] ==
              [p.stats[:scale].mean[1] for p in pts.points]
        @test [p.stats[:enthalpy].mean[1] for p in pt.points] ==
              [p.stats[:enthalpy].mean[1] for p in pts.points]
        @test pt.final_configs == pts.final_configs
        @test pt.final_disps == pts.final_disps
        @test pt.swap_acceptance == pts.swap_acceptance

        # rung marginals ≡ independent NPT run_mc chains at the same (kT, P), within
        # statistics (measured 0.1–1.7σ over rungs × {scale, energy} at these seeds;
        # 4σ gate). This is the end-to-end ensemble check — a lane sweeping another
        # scale's coefficients, a broken affine rescale at a swap, or a corrupted
        # clone all wreck it; the swap RULE itself is pinned by the bracket test
        # above, which is what can actually resolve it.
        for (i, kt) in enumerate(kts)
            mc = run_mc(H; kT = kt, sweeps_therm = 300, sweeps_measure = 3000,
                        seed = 0x91 + i, renorm_interval = 50, strain = sch,
                        pressure = P, observables = obsc)
            for name in (:scale, :energy, :enthalpy)
                m = mc.points[1].stats[name]
                p = pt.points[i].stats[name]
                @test abs(m.mean[1] - p.mean[1]) <
                      4 * sqrt(m.err[1]^2 + p.err[1]^2)
            end
            # the per-rung isobaric C evaluates finite through the lane clones,
            # at the rung's own kT
            c = pt.points[i].stats[:npt_specific_heat]
            @test isfinite(c.mean[1]) && c.mean[1] > 0 && isfinite(c.err[1])
        end

        # keyword resolution mirrors run_mc, each contradiction by name
        @test_throws ArgumentError run_pt(H; kT = kts, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = P, strain_interval = 0)
        @test_throws ArgumentError run_pt(H; kT = kts, sweeps_therm = 1,
                                          sweeps_measure = 2, strain_interval = 2)
        @test_throws ArgumentError run_pt(H; kT = kts, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = P, pressure_GPa = P)
        @test_throws ArgumentError run_pt(H; kT = kts, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch)
        # pressure_diagnostics is refused at ENTRY (by observable name), not after
        # a spent thermalization phase at the first measurement's identity check
        pd = MCs.pressure_diagnostics(sch, H)
        err = try
            run_pt(H; kT = kts, sweeps_therm = 1, sweeps_measure = 2,
                   strain = sch, pressure = P,
                   observables = [standard_observables(H); pd.observables],
                   evaluables = [standard_evaluables(H); pd.evaluables])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("run_mc chain instead", err.msg)
    end

    @testset "PT + strain: checkpoint v5 resumes bit-identically" begin
        zsm, zmodels = _ss_zeta_grid()
        H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                             fixed_reference = true)
        sch = StrainSchedule(zsm, H)
        dir = mktempdir()
        path = joinpath(dir, "pt_npt.jld2")
        kw = (; kT = [0.05, 0.08], sweeps_therm = 60, sweeps_measure = 120,
              renorm_interval = 30, nbins = 4, seed = 0x2b1, exchange_interval = 5,
              strain = sch, pressure = 0.01)
        a = run_pt(H; kw...)
        b = run_pt(H; kw..., checkpoint = path, checkpoint_interval = 50)
        @test a.final_configs == b.final_configs
        @test isfile(path)
        # the resume below must be non-vacuous: the periodic cadence leaves the
        # file mid-measure with at least one lane away from s = 1, so the
        # per-lane checkpointed-scale reinstall is actually exercised
        stored = MCs.jldopen(path, "r") do f
            [f["lane/$r/strain"] for r = 1:f["nlanes"]]
        end
        @test any(s -> s != 1.0, stored)
        # knock H off the reference; resume must reinstall it before comparing
        # fingerprints, rebuild per-lane clones at each lane's checkpointed scale,
        # and continue bit-identically
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.04);
                          recheck_translation = false)
        c = resume(path, H; strain = sch)
        @test [p.stats[:energy].mean[1] for p in a.points] ==
              [p.stats[:energy].mean[1] for p in c.points]
        @test isequal([p.acceptance_strain for p in a.points],
                      [p.acceptance_strain for p in c.points])
        @test a.final_configs == c.final_configs
        @test a.final_disps == c.final_disps
        @test a.swap_acceptance == c.swap_acceptance
        # ...and H is handed back at the reference here too
        ref = MCs.strain_coefficients(sch, 1.0)
        @test H.progs.term_coef ==
              [ref[H.term_source[k]] * H.term_scale[k] for k in eachindex(H.terms)]

        # the handshake's refusals carry over from the mc kind (shared code): the
        # missing schedule and the wrong grid are each named
        err = try
            resume(path, H)
            nothing
        catch e
            e
        end
        @test err isa Exception &&
              occursin("strained (NPT) run", sprint(showerror, err))
        zsm2, _ = _ss_zeta_grid(; scales = [0.92, 1.0, 1.08])
        sch2 = StrainSchedule(zsm2, H)
        err = try
            resume(path, H; strain = sch2)
            nothing
        catch e
            e
        end
        @test err isa Exception &&
              occursin("grid fingerprint", sprint(showerror, err))
    end

    @testset "strained warm start: final scales and strain_init (S12)" begin
        zsm, zmodels = _ss_zeta_grid()
        H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), keep_zero_terms = true,
                             fixed_reference = true)
        sch = StrainSchedule(zsm, H)
        P = 0.01
        ref = MCs.strain_coefficients(sch, 1.0)
        _at_reference(h) = h.progs.term_coef ==
                           [ref[h.term_source[k]] * h.term_scale[k]
                            for k in eachindex(h.terms)]

        # (1) recording. `final_strain` is exactly the strain of the LAST
        # measurement view: with measure_interval = 1 the last measurement sees
        # the state after the final sweep and nothing moves afterwards, so the
        # trace's tail is an independent public witness of the recorded value.
        seen = Float64[]
        trace = Observable(:strace, 1, v -> (push!(seen, MCs.strain(v)); 1.0))
        # ...and the (H, chain) contract is asserted at EVERY measurement, on the
        # live view: the installed coefficients must be the schedule's at the
        # view's own strain. This is the standing-contract gate warm start must
        # not break; what it deliberately cannot see is a wrong ENTRY install
        # alone, because the first strain move re-installs at `st.strain` on
        # accept and reject alike (the Horner restore) — the desync self-heals
        # within one sweep, which is why the entry is additionally pinned by (2).
        contract = Observable(:contract, 1, v -> begin
            expect = MCs.strain_coefficients(sch, MCs.strain(v))
            v.H.progs.term_coef ==
            [expect[v.H.term_source[k]] * v.H.term_scale[k]
             for k in eachindex(v.H.terms)] ? 1.0 : 0.0
        end)
        r = run_mc(H; kT = 0.05, sweeps_therm = 200, sweeps_measure = 600,
                   seed = 0xd51, renorm_interval = 50, strain = sch, pressure = P,
                   observables = [standard_observables(H); trace; contract])
        @test r.final_strain isa Float64
        @test r.final_strain == seen[end]
        @test MCs.in_strain_domain(sch, r.final_strain)
        # non-vacuity, with the mutation size gate (2) must resolve: an ignored
        # `strain_init` reads ≈ 1.0 there, so `|s0 − 1|` must stay well above
        # gate (2)'s 1e-6 walk bound (measured 1.9e-2 at this seed)
        @test abs(r.final_strain - 1.0) > 1e-3
        @test r.points[1].stats[:contract].mean[1] == 1.0
        @test _at_reference(H)               # handed back at the reference
        # fixed-cell: `nothing`, not 1.0 (the MCView discipline)
        rf = run_mc(H; kT = 0.05, sweeps_therm = 2, sweeps_measure = 4, seed = 1)
        @test rf.final_strain === nothing

        # (2) the warm start actually STARTS at s0: with a microscopic proposal
        # width the strain channel is frozen at the entry scale up to a ~1e-8
        # random walk, so the first measured strain pins the start (an ignored
        # `strain_init` reads ≈ 1.0 here, 4 decades above the bound) and the
        # contract observable holds at s0's coefficients throughout.
        s0 = r.final_strain
        empty!(seen)
        rb = run_mc(H; kT = 0.05, sweeps_therm = 0, sweeps_measure = 400,
                    seed = 0xd52, renorm_interval = 50, strain = sch,
                    pressure = P, strain_step = 1e-8, strain_init = s0,
                    init = r.final_config, disps = r.final_disps,
                    observables = [standard_observables(H); trace; contract])
        @test abs(seen[1] - s0) < 1e-6
        @test abs(rb.final_strain - s0) < 1e-4
        @test rb.points[1].stats[:contract].mean[1] == 1.0
        @test _at_reference(H)
        # statistical continuation: a therm-free warm start from an equilibrated
        # parent reproduces the parent's measured marginals (same (kT, P), full
        # strain step). Measured over 3 seed sets: 0.7–1.5σ across
        # {scale, energy} (pinned set 0.8/1.4σ); 4σ gate. This is an ensemble
        # check — entry-install mutations self-heal (above) and are below its
        # resolution by construction.
        rc = run_mc(H; kT = 0.05, sweeps_therm = 0, sweeps_measure = 3000,
                    seed = 0xd53, renorm_interval = 50, strain = sch,
                    pressure = P, strain_init = s0, init = r.final_config,
                    disps = r.final_disps,
                    observables = [standard_observables(H);
                                   Observable(:scale, 1, v -> MCs.strain(v))])
        ra = run_mc(H; kT = 0.05, sweeps_therm = 500, sweeps_measure = 3000,
                    seed = 0xd54, renorm_interval = 50, strain = sch,
                    pressure = P,
                    observables = [standard_observables(H);
                                   Observable(:scale, 1, v -> MCs.strain(v))])
        for name in (:scale, :energy)
            a = ra.points[1].stats[name]
            c = rc.points[1].stats[name]
            @test abs(a.mean[1] - c.mean[1]) < 4 * sqrt(a.err[1]^2 + c.err[1]^2)
        end

        # (3) checkpoint interplay, on a GENUINELY interrupted run: a finished
        # run's file ends at the completed marker (resume then merely returns the
        # stored result), so continuation teeth need a run that actually stops
        # mid-measure — a poison observable throws at measurement 100, leaving
        # the file at the last periodic write (measure sweep 60 of 120), and
        # `resume` must (a) ACCEPT the warm-started file at all — the stored
        # fingerprint is the REFERENCE-scale identity because the checkpointer
        # runs before the s0 install, order load-bearing — and (b) reproduce the
        # uninterrupted run bit for bit from mid-measure, final strain included.
        dir = mktempdir()
        path = joinpath(dir, "warm.jld2")
        nmeas = Ref(0)
        poison = Observable(:poison, 1,
                            v -> (nmeas[] += 1) >= 100 ?
                                 error("poison interrupt") : 0.0)
        benign = Observable(:poison, 1, v -> 0.0)
        obsA = [standard_observables(H); benign]
        kwB = (; kT = 0.05, sweeps_therm = 40, sweeps_measure = 120,
               renorm_interval = 30, nbins = 4, seed = 0xd55, strain = sch,
               pressure = P, strain_init = s0, init = r.final_config,
               disps = r.final_disps)
        a = run_mc(H; kwB..., observables = obsA)
        err = try
            run_mc(H; kwB..., observables = [standard_observables(H); poison],
                   checkpoint = path, checkpoint_interval = 50)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("poison", err.msg)
        @test isfile(path)
        mid = MCs.jldopen(path, "r") do f
            (f["progress/phase"], f["progress/sweep"])
        end
        @test mid[1] == "measure" && 0 < mid[2] < 120   # non-vacuity: mid-run
        # the interrupt skipped run_mc's hand-back; resume revalidates from any
        # coefficient state, so knock H further off the reference first
        set_coefficients!(H, MCs.strain_coefficients(sch, 1.03);
                          recheck_translation = false)
        c = resume(path, H; observables = obsA, strain = sch)
        @test c.final_config == a.final_config
        @test c.final_disps == a.final_disps
        @test c.final_strain == a.final_strain
        @test c.points[1].stats[:energy].mean[1] ==
              a.points[1].stats[:energy].mean[1]
        @test _at_reference(H)

        # (4) run_pt: per-lane strain_init (a vector), broadcast (a scalar), the
        # recorded per-lane finals, determinism, and the fixed-cell `nothing`.
        s0s = [0.97, 1.03]
        # exchanges are deliberately OUT of this run (an accepted swap trades the
        # strain payload between rungs, folding 0.97 ↔ 1.03 into both marginals —
        # measured 6e-3 mean shift); the pin wants each lane's own entry scale.
        # Warm start WITH live exchanges is covered by the resume block below.
        kwP = (; kT = [0.05, 0.08], sweeps_therm = 0, sweeps_measure = 200,
               seed = 0xd56, renorm_interval = 50, exchange_interval = 10_000,
               strain = sch, pressure = P, strain_step = 1e-8,
               strain_init = s0s,
               observables = [standard_observables(H);
                              Observable(:scale, 1, v -> MCs.strain(v))])
        pt = run_pt(H; kwP..., ntasks = 2)
        @test all(isnan, pt.swap_acceptance)   # the exchange-free premise holds
        # each LANE pinned near its own entry scale — a broadcast or permutation
        # bug lands a lane 6e-2 away, 4 decades above the walk bound
        @test pt.final_strains isa Vector{Float64}
        for i = 1:2
            @test abs(pt.points[i].stats[:scale].mean[1] - s0s[i]) < 1e-5
            @test abs(pt.final_strains[i] - s0s[i]) < 1e-4
        end
        @test _at_reference(H)
        # bit-determinism across ntasks with the warm start live (P3, extended)
        pts = run_pt(H; kwP..., ntasks = 1)
        @test pts.final_strains == pt.final_strains
        @test [p.stats[:energy].mean[1] for p in pt.points] ==
              [p.stats[:energy].mean[1] for p in pts.points]
        # scalar broadcast
        ptb = run_pt(H; kT = [0.05, 0.08], sweeps_therm = 0, sweeps_measure = 60,
                     seed = 0xd57, exchange_interval = 5, strain = sch,
                     pressure = P, strain_step = 1e-8, strain_init = 1.02)
        @test all(abs(s - 1.02) < 1e-4 for s in ptb.final_strains)
        # a warm-started PT run, interrupted mid-measure (poison, as in (3);
        # ntasks = 1 in the writer so the interrupt point is deterministic),
        # resumes bit-identically — lane clones rebuilt at the checkpointed
        # per-lane scales, warm-start finals included
        pth = joinpath(dir, "warm_pt.jld2")
        nmeas[] = 0
        kwR = (; kT = [0.05, 0.08], sweeps_therm = 30, sweeps_measure = 90,
               renorm_interval = 30, nbins = 4, seed = 0xd58,
               exchange_interval = 5, strain = sch, pressure = P,
               strain_init = s0s)
        pa = run_pt(H; kwR..., observables = obsA)
        err = try
            run_pt(H; kwR..., observables = [standard_observables(H); poison],
                   checkpoint = pth, checkpoint_interval = 40, ntasks = 1)
            nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin("poison", err.msg)
        pmid = MCs.jldopen(pth, "r") do f
            (f["progress/phase"], f["progress/done"])
        end
        @test pmid[1] == "measure" && 0 < pmid[2] < 90   # non-vacuity: mid-run
        pc = resume(pth, H; observables = obsA, strain = sch)
        @test pc.final_configs == pa.final_configs
        @test pc.final_disps == pa.final_disps
        @test pc.final_strains == pa.final_strains
        @test [p.stats[:energy].mean[1] for p in pc.points] ==
              [p.stats[:energy].mean[1] for p in pa.points]
        @test _at_reference(H)
        # fixed-cell PT: `nothing`
        ptf = run_pt(H; kT = [0.05, 0.08], sweeps_therm = 2, sweeps_measure = 4,
                     seed = 2)
        @test ptf.final_strains === nothing

        # (5) refusals, each by message: no schedule, out-of-domain (scalar and
        # one vector entry), and a wrong-length ladder vector
        err = try
            run_mc(H; kT = 0.05, sweeps_therm = 1, sweeps_measure = 2,
                   strain_init = 1.01)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("without a strain schedule",
                                                err.msg)
        # the run_pt vector path hits the SAME hoisted refusal (the hoist is
        # load-bearing for the type layer — a revert re-breaks JET, so gate it)
        err = try
            run_pt(H; kT = [0.05, 0.08], sweeps_therm = 1, sweeps_measure = 2,
                   strain_init = [1.0, 1.0])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("without a strain schedule",
                                                err.msg)
        lo, hi = MCs.strain_domain(sch)
        @test_throws ArgumentError run_mc(H; kT = 0.05, sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = P,
                                          strain_init = hi + 0.05)
        @test_throws ArgumentError run_pt(H; kT = [0.05, 0.08], sweeps_therm = 1,
                                          sweeps_measure = 2, strain = sch,
                                          pressure = P,
                                          strain_init = [1.0, hi + 0.05])
        err = try
            run_pt(H; kT = [0.05, 0.08], sweeps_therm = 1, sweeps_measure = 2,
                   strain = sch, pressure = P, strain_init = [1.0, 1.0, 1.0])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("ladder of 2 rungs", err.msg)
    end
end

# The volume-grid boundary screen. A proposal outside `strain_domain` is rejected, never
# clamped — correct as a move rule, but it means a chain the pressure pushes past the
# grid does not fail: it piles up at the edge and every mechanical observable goes on
# describing a volume-CLAMPED cell in confident finite numbers. Before this screen
# nothing recorded that at all: no counter, no field, no warning.
#
# The gate is the SEPARATION, not the firing. Two runs must stay silent — a healthy one,
# and (the trap) one whose step is so wide that a quarter of its proposals leave the
# domain while its marginal stays entirely inside — and three pinned ones must speak.
# The oversized-step arm is what makes this more than a smoke test: a rate-only rule
# passes every "does it warn" check and still cries wolf on it.
@testset "the volume-grid boundary is screened, and only when it bites" begin
    zsm, zmodels = _ss_zeta_grid()
    H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), fixed_reference = true)
    sch = MCs.StrainSchedule(zsm, H)
    lo, hi = MCs.strain_domain(sch)
    obs = [Observable(:energy, 1, v -> v.energy)]
    base = (; kT = 0.05, sweeps_therm = 300, sweeps_measure = 2000, nbins = 4,
            renorm_interval = 50, seed = 7, observables = obs,
            evaluables = Evaluable[])

    pinned(p) = occursin("pinned against the", p)
    run_logs(kw) = Test.collect_test_logs() do
        run_mc(H; base..., kw...)
    end

    @testset "a fixed-cell run has no scale range to report" begin
        r = run_mc(H; base...)
        p = r.points[1]
        @test isnan(p.strain_min) && isnan(p.strain_max) && isnan(p.strain_outside)
        @test isnan(p.acceptance_strain)     # the same convention, the same gate
    end

    @testset "a healthy strained run reports a range strictly inside the grid" begin
        logs, r = run_logs((; strain = sch, pressure = 0.0, strain_step = 0.05))
        p = r.points[1]
        @test lo < p.strain_min <= p.strain_max < hi
        @test p.strain_min <= r.final_strain <= p.strain_max
        @test p.strain_outside == 0.0
        @test !any(l -> pinned(string(l.message)), logs)
    end

    @testset "an oversized step is inefficient, not pinned — and is not warned about" begin
        # The discriminating arm: `strain_step` comparable to the whole domain throws a
        # large fraction of proposals out of it (ordinary Metropolis rejections — the
        # domain is part of the state space), yet the chain's marginal never approaches
        # an edge, so no boundary term is being dropped and nothing is wrong.
        logs, r = run_logs((; strain = sch, pressure = 0.0, strain_step = 0.25))
        p = r.points[1]
        @test p.strain_outside > MCs._STRAIN_OUTSIDE_WARN   # the rate alone would fire…
        margin = MCs._STRAIN_EDGE_WARN * (hi - lo)
        @test p.strain_min - lo > margin && hi - p.strain_max > margin   # …the edge saves it
        @test !any(l -> pinned(string(l.message)), logs)
    end

    @testset "a chain the pressure pins against an edge is named" begin
        for (P, edge) in [(-0.5, "upper"), (-1.0, "upper"), (1.0, "lower")]
            logs, r = run_logs((; strain = sch, pressure = P, strain_step = 0.05))
            p = r.points[1]
            hits = filter(l -> pinned(string(l.message)), logs)
            @test length(hits) == 1
            @test occursin("$(edge) edge", string(hits[1].message))
            # and the fields a post-hoc screen would read agree with the verdict
            @test p.strain_outside > MCs._STRAIN_OUTSIDE_WARN
            @test min(p.strain_min - lo, hi - p.strain_max) <
                  MCs._STRAIN_EDGE_WARN * (hi - lo)
        end
    end

    @testset "the decision rule itself: both conditions are load-bearing" begin
        # The sampled arms above kill the EDGE condition (the oversized-step run has a
        # high refusal rate and must stay silent) but not the RATE one: no sampled run
        # here has a low rate AND an extreme at the edge, so dropping the rate test
        # would survive them. Pin the predicate directly on hand-built points instead —
        # a four-cell truth table, no chain involved.
        pt(smin, smax, out) =
            MCs.TempResult(0.05, 0.05 / MCs.KB_EV, Dict{Symbol,MCs.ObservableStat}(),
                           NaN, NaN, NaN, 0.5, smin, smax, out,
                           0.3, 0.01, 0.0, NaN, NaN, 0, false)
        margin = MCs._STRAIN_EDGE_WARN * (hi - lo)
        near, far = hi - 0.2 * margin, hi - 5 * margin
        hot, cold = 4 * MCs._STRAIN_OUTSIDE_WARN, 0.2 * MCs._STRAIN_OUTSIDE_WARN
        plan = MCs.UpdatePlan([0.05]; sweeps_therm = 1, sweeps_measure = 1,
                              measure_interval = 1, or_per_metropolis = 0,
                              disp_per_metropolis = 1, step = 0.3, step_u = 0.01,
                              adapt_target = 0.5, adapt_interval = 1,
                              renorm_interval = 1, nbins = 2, carryover = true,
                              seed = 1, strain_interval = 1, strain_step = 0.05)
        fires(p) = !isempty(filter(l -> pinned(string(l.message)),
                                   first(Test.collect_test_logs() do
                                             MCs._warn_strain_boundary([p], sch, plan)
                                         end)))
        @test fires(pt(lo + 5 * margin, near, hot))     # at the edge AND refused a lot
        @test !fires(pt(lo + 5 * margin, near, cold))   # at the edge, but not refused
        @test !fires(pt(lo + 5 * margin, far, hot))     # refused a lot, but never there
        @test !fires(pt(lo + 5 * margin, far, cold))    # neither
        # the lower edge is screened too, and named as such
        @test fires(pt(lo + 0.2 * margin, hi - 5 * margin, hot))
        # a fixed-cell point carries NaN and must not be interpreted as either
        @test !fires(pt(NaN, NaN, NaN))
    end

    @testset "the range and the refusal rate survive a checkpoint" begin
        mktempdir() do dir
            path = joinpath(dir, "npt.jld2")
            r = run_mc(H; base..., strain = sch, pressure = 0.0, strain_step = 0.05,
                       checkpoint = path, checkpoint_interval = 10_000)
            p = r.points[1]
            q = MCs.jldopen(path, "r") do f
                MCs._read_point(f, "points/1")
            end
            @test q.strain_min == p.strain_min && q.strain_max == p.strain_max
            @test q.strain_outside == p.strain_outside
        end
    end
end

# The strain cadence. `_resolve_strain_moves` has refusal tests, but nothing asserted
# that the resolved number is the number of attempts the driver actually makes: making
# `_resolve_strain_moves` return 1 unconditionally left the suite green, because every
# strain gate reads `acceptance_strain` (a RATIO — unchanged by attempting ten times as
# often) or `!isnan` of it. A cadence silently stuck at 1 costs a full from-scratch
# energy per sweep and changes the chain's mixing, so it is worth a count.
#
# The count is read out of the checkpoint, which stores the raw counters: the
# measurement phase resets them at the freeze boundary, so a phase of `m` sweeps firing
# on `sweep % k == 0` must attempt exactly `fld(m, k)` times. That is arithmetic, not a
# captured number.
@testset "the strain cadence is the number of attempts, not just a ratio" begin
    zsm, zmodels = _ss_zeta_grid()
    H = TiledHamiltonian(zmodels[2]; dims = (2, 1, 1), fixed_reference = true)
    sch = MCs.StrainSchedule(zsm, H)
    obs = [Observable(:energy, 1, v -> v.energy)]
    m = 21
    mktempdir() do dir
        for k in (1, 3, 7)
            path = joinpath(dir, "cadence$(k).jld2")
            run_mc(H; kT = 0.05, sweeps_therm = 10, sweeps_measure = m, nbins = 2,
                   renorm_interval = 50, strain = sch, pressure = 0.0,
                   strain_step = 0.05, strain_interval = k, seed = 4,
                   observables = obs, evaluables = Evaluable[],
                   checkpoint = path, checkpoint_interval = 0)
            cnt, att_out = MCs.jldopen(path, "r") do f
                (f["chain/counters"], f["chain/counters"][9])
            end
            acc_s, att_s = cnt[7], cnt[8]
            @test att_s == fld(m, k)              # every k-th measurement sweep, exactly
            @test 0 <= acc_s <= att_s
            @test 0 <= att_out <= att_s           # the out-of-domain share is a subset
        end
    end
end
