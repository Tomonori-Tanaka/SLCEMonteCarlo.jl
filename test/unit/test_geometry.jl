# Geometry helpers: supercell_crystal ordering matches TiledHamiltonian site
# indexing; to_matrix/from_matrix round-trips.

@testset "geometry" begin
    @testset "supercell_crystal matches site indexing" begin
        model = _dimer_model()
        cr = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])   # rebuild the fixture crystal
        crystal = Crystal(cr, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1],
                          ["Fe"])
        dims = (2, 3, 1)
        H = TiledHamiltonian(model; dims = dims)
        sup = supercell_crystal(crystal, dims)
        @test n_atoms(sup) == n_sites(H)
        @test sup.lattice.vectors ≈ crystal.lattice.vectors *
                                    Diagonal([2.0, 3.0, 1.0]) atol = 1e-14
        cart_cell = cartesian_positions(crystal)
        cart_sup = cartesian_positions(sup)
        A = Matrix(crystal.lattice.vectors)
        for c3 = 0:0, c2 = 0:2, c1 = 0:1, a = 1:4
            s = MC.site_index(H, a, (c1, c2, c3))
            @test sup.species[s] == crystal.species[a]
            @test cart_sup[:, s] ≈ cart_cell[:, a] + A * [c1, c2, c3] atol = 1e-10
        end
        @test_throws ArgumentError supercell_crystal(crystal, (0, 1, 1))
    end

    @testset "to_matrix / from_matrix round-trip" begin
        rng = MersenneTwister(4)
        config = MC.SpinConfig([_rand_spin(rng) for _ = 1:6])
        m = MC.to_matrix(config)
        @test size(m) == (3, 6)
        @test MC.from_matrix(m) ≈ config atol = 1e-14   # door-projection roundoff
        # a moment-scaled matrix is REFUSED, never silently normalized (the
        # family's unit-direction door; this pinned the opposite until 2026-08)
        m2 = 2.5 .* m
        @test_throws ArgumentError MC.from_matrix(m2)
        # inside the 1e-6 band: accepted and projected exactly onto the sphere
        m3 = (1.0 + 1.0e-7) .* m
        @test all(abs(norm(e) - 1) < 4 * eps() for e in MC.from_matrix(m3))
        # the near-pole case the band alone cannot fix: 5e-9 off unit at the
        # pole has |z| > 1 raw; the door projects instead of letting it reach
        # the Legendre recursion
        mp = copy(m)
        mp[:, 1] = [0.0, 0.0, 1.0 + 5.0e-9]
        @test MC.from_matrix(mp)[1][3] <= 1.0
        @test_throws DimensionMismatch MC.from_matrix(zeros(2, 4))
        @test_throws ArgumentError MC.from_matrix(zeros(3, 4))
    end
end
