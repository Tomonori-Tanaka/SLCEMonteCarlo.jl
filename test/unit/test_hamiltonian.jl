# TiledHamiltonian construction and supercell tiling (docs/specs/hamiltonian-tiling.md).
# Machine-precision gates: dims=(1,1,1) reproduces `predict_energy − intercept`; a
# periodically replicated configuration on dims=(2,2,2) carries exactly 8× the cell
# energy; the CSR instance/site bookkeeping is self-consistent; the (4π)^(body/2)
# scale is applied exactly once; self-image (chain) terms wrap correctly.

@testset "TiledHamiltonian" begin
    @testset "construction from a fitted model" begin
        H = TiledHamiltonian(_dimer_model())
        @test H.n_cell_atoms == 4
        @test H.dims == SVector(1, 1, 1)
        @test n_sites(H) == 4
        @test H.lmax == 1
        @test H.nlm == 4
        @test length(H.terms) == 1              # the canonical (undirected) 1–2 bond
        @test length(H.inst_term) == 1
        @test H.site_has_l1[1] && H.site_has_l1[2]
        @test !H.site_has_l1[3] && !H.site_has_l1[4]
        @test occursin("4 atoms × 1×1×1 = 4 sites", sprint(show, H))

        H8 = TiledHamiltonian(_dimer_model(); dims = (2, 2, 2))
        @test n_sites(H8) == 32
        @test length(H8.inst_term) == 1 * 8     # one instance per term and cell
    end

    @testset "site indexing" begin
        H = TiledHamiltonian(_dimer_model(); dims = (3, 2, 1))
        @test MC.site_index(H, 1, (0, 0, 0)) == 1
        @test MC.site_index(H, 4, (2, 1, 0)) == n_sites(H)
        @test MC.site_index(H, 2, (1, 0, 0)) == 2 + 4 * 1
        for s in [1, 5, 13, n_sites(H)]
            @test 1 <= MC.site_atom(H, s) <= 4
        end
        # round-trip: atom-fastest, cells column-major
        s = 0
        for c3 = 0:0, c2 = 0:1, c1 = 0:2, a = 1:4
            s += 1
            @test MC.site_index(H, a, (c1, c2, c3)) == s
            @test MC.site_atom(H, s) == a
        end
        @test_throws ArgumentError MC.site_index(H, 5, (0, 0, 0))
        @test_throws ArgumentError MC.site_index(H, 1, (3, 0, 0))
    end

    @testset "dims=(1,1,1) ≡ predict_energy − intercept (machine precision)" begin
        for model in (_dimer_model(), _biquadratic_model(0))
            H = TiledHamiltonian(model)
            rng = MersenneTwister(7)
            for _ = 1:3
                config = _rand_config(rng, H)
                @test total_energy(H, config) ≈
                      predict_energy(model, _config_matrix(config)) -
                      intercept(model) atol = 1e-12
            end
        end
    end

    @testset "periodic replication: supercell = N × cell energy" begin
        for model in (_dimer_model(), _biquadratic_model(0))
            H1 = TiledHamiltonian(model)
            H8 = TiledHamiltonian(model; dims = (2, 2, 2))
            H6 = TiledHamiltonian(model; dims = (3, 1, 2))
            rng = MersenneTwister(11)
            for _ = 1:3
                cell = _rand_config(rng, H1)
                E1 = total_energy(H1, cell)
                @test total_energy(H8, _tile_config(H8, cell)) ≈ 8 * E1 atol = 1e-10
                @test total_energy(H6, _tile_config(H6, cell)) ≈ 6 * E1 atol = 1e-10
            end
        end
    end

    @testset "CSR bookkeeping is self-consistent" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        n_inst = length(H.inst_term)
        @test H.inst_ptr[1] == 1 && H.inst_ptr[end] == length(H.inst_sites) + 1
        @test H.site_ptr[1] == 1 && H.site_ptr[end] == length(H.inst_sites) + 1
        # every (site → instance, slot) entry points back at that site
        for s = 1:n_sites(H), j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
            i = H.site_inst[j]
            slot = H.site_slot[j]
            @test H.inst_sites[H.inst_ptr[i] + slot - 1] == s
        end
        # and every instance member appears exactly once in the adjacency
        counted = zeros(Int, n_inst)
        for s = 1:n_sites(H), j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
            counted[H.site_inst[j]] += 1
        end
        @test all(i -> counted[i] == H.inst_ptr[i + 1] - H.inst_ptr[i], 1:n_inst)
    end

    @testset "self-image chain: wrap and site-distinctness guard" begin
        J = -0.05
        terms = _chain_terms(J)
        # dims=(1,1,1) folds the ±x images onto the anchor site itself → rejected
        @test_throws ArgumentError TiledHamiltonian(1, terms)
        # a 4-cell chain: uniform config has 4 bonds of energy J each
        H = TiledHamiltonian(1, terms; dims = (4, 1, 1))
        @test n_sites(H) == 4
        up = SVector(0.0, 0.0, 1.0)
        @test total_energy(H, MC.SpinConfig([up for _ = 1:4])) ≈ 4J atol = 1e-12
        # flipping one spin breaks its two bonds: E = 2J·(−1) + 2J·(+1) ... each site
        # has bonds to both neighbors; flipped site contributes −2J, rest +2J
        flipped = MC.SpinConfig([up, up, -up, up])
        @test total_energy(H, flipped) ≈ 2J - 2J atol = 1e-12
        # +x neighbor of the last cell wraps to cell 0 (both directed members present)
        @test length(H.inst_term) == 2 * 4
    end

    @testset "scale-once: hand contraction of a single-site l=1 field term" begin
        # V(e) = c0·Z_10(e) = c0·N1·e_z, raw coef c0/(4π)^(1/2) — the ctor applies
        # (4π)^(1/2) exactly once.
        c0 = 0.3
        n1 = SLCE.Harmonics.N1
        folded = zeros(3)
        folded[2] = 1.0                       # μ = 0 slot of l = 1
        z = SVector(0, 0, 0)
        term = MultipoleTerm(c0 / sqrt(4π), 1, [1], [z], [1], folded)
        H = TiledHamiltonian(1, [term])
        e = normalize(SVector(0.3, -0.4, 0.85))
        @test total_energy(H, MC.SpinConfig([e])) ≈ c0 * n1 * e[3] atol = 1e-14
    end

    @testset "constructor guards" begin
        z = SVector(0, 0, 0)
        good = MultipoleTerm(1.0, 2, [1, 2], [z, z], [1, 1], zeros(3, 3))
        @test_throws ArgumentError TiledHamiltonian(2, MultipoleTerm[])
        @test_throws ArgumentError TiledHamiltonian(0, [good])
        @test_throws ArgumentError TiledHamiltonian(2, [good]; dims = (0, 1, 1))
        @test_throws ArgumentError TiledHamiltonian(1, [good])            # atom 2 > 1
        bad_anchor = MultipoleTerm(1.0, 2, [1, 2], [SVector(1, 0, 0), z], [1, 1],
                                   zeros(3, 3))
        @test_throws ArgumentError TiledHamiltonian(2, [bad_anchor])
        bad_repeat = MultipoleTerm(1.0, 2, [1, 1], [z, z], [1, 1], zeros(3, 3))
        @test_throws ArgumentError TiledHamiltonian(2, [bad_repeat])
        bad_shape = MultipoleTerm(1.0, 2, [1, 2], [z, z], [1, 2], zeros(3, 3))
        @test_throws ArgumentError TiledHamiltonian(2, [bad_shape])
    end
end
