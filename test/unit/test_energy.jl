# The 4-function energy contract: exact single-spin ΔE (leave-one-out coefficients),
# their independence of the site's own spin, and the on-sphere gradient against
# central finite differences.

@testset "energy contract" begin
    H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
    rng = MersenneTwister(3)

    @testset "ΔE ≡ total-energy difference (machine precision)" begin
        config = _rand_config(rng, H)
        zrows = MC._zrows(H, config)
        c = zeros(H.nlm)
        znew = zeros(H.nlm)
        for _ = 1:6
            s = rand(rng, 1:n_sites(H))
            e2 = _rand_spin(rng)
            fill!(c, 0.0)
            MC.site_coeffs!(c, H, s, zrows)
            MC._zlm_row!(znew, e2, H.lmax)
            ΔE = MC.delta_energy(c, view(zrows, :, s), znew)

            config2 = copy(config)
            config2[s] = e2
            @test ΔE ≈ total_energy(H, config2) - total_energy(H, config) atol = 1e-12
        end
    end

    @testset "site_coeffs! is independent of the site's own spin" begin
        config = _rand_config(rng, H)
        for s in [1, n_sites(H)]
            c1 = MC.site_coeffs!(zeros(H.nlm), H, s, MC._zrows(H, config))
            config2 = copy(config)
            config2[s] = _rand_spin(rng)
            c2 = MC.site_coeffs!(zeros(H.nlm), H, s, MC._zrows(H, config2))
            @test c1 == c2
        end
    end

    @testset "site energy from c reproduces the instance sum" begin
        # c · Z(e_s) is the total energy of every instance touching s, with the other
        # sites frozen: check via the ΔE of moving e_s to a reference direction.
        config = _rand_config(rng, H)
        zrows = MC._zrows(H, config)
        s = 2
        c = MC.site_coeffs!(zeros(H.nlm), H, s, zrows)
        # move site s across three random directions; ΔE must chain consistently
        e_a, e_b = _rand_spin(rng), _rand_spin(rng)
        za, zb = zeros(H.nlm), zeros(H.nlm)
        MC._zlm_row!(za, e_a, H.lmax)
        MC._zlm_row!(zb, e_b, H.lmax)
        Δ_ab = MC.delta_energy(c, za, zb)
        Δ_a = MC.delta_energy(c, view(zrows, :, s), za)
        Δ_b = MC.delta_energy(c, view(zrows, :, s), zb)
        @test Δ_ab ≈ Δ_b - Δ_a atol = 1e-13
    end

    @testset "site_gradient vs central finite differences" begin
        config = _rand_config(rng, H)
        for s = 1:2
            g = MC.site_gradient(H, s, config)
            e = config[s]
            @test abs(dot(g, e)) < 1e-12               # tangent-projected
            # two orthonormal tangent directions
            t1 = normalize(cross(e, abs(e[3]) < 0.9 ? SVector(0.0, 0.0, 1.0) :
                                    SVector(1.0, 0.0, 0.0)))
            t2 = cross(e, t1)
            h = 1e-5
            for v in (t1, t2)
                f = t -> begin
                    c2 = copy(config)
                    c2[s] = normalize(e + t * v)
                    total_energy(H, c2)
                end
                fd = (f(h) - f(-h)) / (2h)
                @test dot(g, v) ≈ fd atol = 1e-8 rtol = 1e-5
            end
        end
    end

    @testset "config size guard" begin
        @test_throws DimensionMismatch total_energy(H, MC.SpinConfig(
            [SVector(0.0, 0.0, 1.0)]))
    end
end

@testset "program kernels ≡ reference kernels (bitwise)" begin
    # The hot kernels walk the precompiled contraction programs; the rank-generic
    # reference kernels are the spec. Same entry/factor/accumulation order ⇒ the
    # results must be `==`, not `≈`. Covers body 1 (onsite l = 2), body 2
    # (isotropic and anisotropic), body 3 with a self-image shift, and sparse
    # `folded` tensors (about half the entries zeroed → the nonzero filter).
    rng = MersenneTwister(7)
    sparse_folded(dims...) = begin
        f = randn(rng, dims...)
        f[rand(rng, length(f)) .< 0.5] .= 0.0
        f
    end
    z3 = SVector(0, 0, 0)
    x3 = SVector(1, 0, 0)
    mixed = [SpinMultipoleTerm(0.3, 1, [1], [z3], [2], sparse_folded(5)),
             SpinMultipoleTerm(-0.2, 2, [1, 2], [z3, z3], [1, 1], sparse_folded(3, 3)),
             SpinMultipoleTerm(0.1, 3, [1, 2, 1], [z3, z3, x3], [1, 1, 2],
                           sparse_folded(3, 3, 5))]
    hams = [MC.TiledHamiltonian(2, mixed; dims = (2, 2, 1)),
            TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1)),
            TiledHamiltonian(_dimer_model()),
            MC.TiledHamiltonian(1, _chain_terms(0.05); dims = (4, 1, 1))]
    for H in hams, _ = 1:3
        config = _rand_config(rng, H)
        zrows = MC._zrows(H, config)
        @test MC._total_energy(H, zrows) == MC._total_energy_ref(H, zrows)
        ok = true
        for s = 1:n_sites(H)
            ok &= MC.site_coeffs!(zeros(H.nlm), H, s, zrows) ==
                  MC._site_coeffs_ref!(zeros(H.nlm), H, s, zrows)
        end
        @test ok
    end
end
