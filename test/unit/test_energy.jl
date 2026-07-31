# The 4-function energy contract: exact single-spin ΔE (leave-one-out coefficients),
# their independence of the site's own spin, and the on-sphere gradient against
# central finite differences.

@testset "energy contract" begin
    H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
    rng = MersenneTwister(3)

    # The ONE physical oracle of the whole ΔE path — the leave-one-out coefficients
    # against a from-scratch recomputation, with no shared routine between the two
    # sides. It used to run on a single body-2 fixture, which left the triplet and
    # general branches of `site_coeffs!` (energy.jl's three-way dispatch) covered only
    # by the bitwise `_site_coeffs_ref!` comparison further down — and that reference
    # shares the entry tables, so it checks the dispatch, not the physics. Run the
    # oracle across the body orders instead: exactness at any body order is the claim
    # `ΔE = c_s·(Z(e′) − Z(e))` makes, and only body ≥ 3 exercises the leave-one-out
    # contraction over more than one surviving axis.
    @testset "ΔE ≡ total-energy difference (machine precision)" begin
        cases = [("body 2 (fitted, l ≤ 2)",
                  TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))),
                 ("body 3", MC.TiledHamiltonian(1, _threebody_terms(0.05);
                                                dims = (4, 1, 1))),
                 ("body 4", MC.TiledHamiltonian(1, _fourbody_terms(0.05);
                                                dims = (5, 1, 1))),
                 ("bodies 1+2+3 mixed", MC.TiledHamiltonian(1, _chain_terms(0.05);
                                                            dims = (4, 1, 1)))]
        for (name, Hc) in cases
            config = _rand_config(rng, Hc)
            zrows = MC._zrows(Hc, config)
            c = zeros(Hc.nlm)
            znew = zeros(Hc.nlm)
            worst = 0.0
            for _ = 1:6
                s = rand(rng, 1:n_sites(Hc))
                e2 = _rand_spin(rng)
                fill!(c, 0.0)
                MC.site_coeffs!(c, Hc, s, zrows)
                MC._zlm_row!(znew, e2, Hc.lmax)
                ΔE = MC.delta_energy(c, view(zrows, :, s), znew)

                config2 = copy(config)
                config2[s] = e2
                exact = total_energy(Hc, config2) - total_energy(Hc, config)
                @test ΔE ≈ exact atol = 1e-12
                worst = max(worst, abs(ΔE - exact))
            end
            # not merely "within 1e-12": the incremental path is exact to roundoff on
            # every one of these, and a body order that only just passed would be a
            # signal in itself
            @test worst < 1e-13
        end
    end

    # `delta_energy` keeps the ROW-DIFFERENCE form `Σ cₖ(znewₖ − zoldₖ)` rather than the
    # algebraically identical `c·znew − c·zold`, and the docstring puts the cost of the
    # latter at two to three orders of magnitude. Nothing tested it: swapping in the
    # two-dot form left test_energy / test_metropolis (drift gate included) /
    # test_overrelaxation green, because every fixture there has ΔE comparable to E.
    #
    # The separation is a summation effect, not a subtraction one. Each `znewₖ − zoldₖ`
    # is computed exactly (floating subtraction of nearby values is), so the difference
    # form sums small terms and carries an absolute error ~eps·|ΔE|; the two-dot form
    # sums two LARGE quantities and carries ~eps·Σ|cₖzₖ|, which has nothing to do with
    # how small ΔE is. The oracle is the same sum in `BigFloat` — an independent,
    # higher-precision evaluation, not another Float64 path.
    @testset "delta_energy keeps its accuracy when ΔE ≪ E" begin
        rng2 = MersenneTwister(90210)
        n = 64
        c = randn(rng2, n)
        zold = 1.0e3 .* randn(rng2, n)          # a site whose rows are large…
        znew = zold .+ 1.0e-6 .* randn(rng2, n) # …and a move that barely changes them
        exact = sum(BigFloat(c[k]) * (BigFloat(znew[k]) - BigFloat(zold[k]))
                    for k = 1:n)
        impl = MC.delta_energy(c, zold, znew)
        naive = dot(c, znew) - dot(c, zold)     # the form the docstring rejects
        e_impl = abs(BigFloat(impl) - exact)
        e_naive = abs(BigFloat(naive) - exact)
        @test abs(exact) > 1e-7                 # there IS a signal to lose
        @test e_impl < 1e-14 * abs(exact)       # correctly rounded, near enough
        @test e_naive > 1e3 * e_impl            # …and the rejected form is not
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
