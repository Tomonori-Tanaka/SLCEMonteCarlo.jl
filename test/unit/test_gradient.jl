# The all-site public gradient `energy_gradient!` / `energy_gradient`: bitwise
# consistency with the per-site `site_gradient` and with `minimize.jl`'s internal
# `_gradient!` (one `_site_grad` kernel behind all three), task-count independence,
# central finite differences, the SLCE torque cross-check (τ = G × e), and the
# inactive-site convention.

@testset "energy_gradient" begin
    rng = MersenneTwister(11)

    @testset "≡ site_gradient (bitwise), tangency, finite differences" begin
        H = TiledHamiltonian(_biquadratic_model(1); dims = (2, 2, 1))
        config = _rand_config(rng, H)
        G = MC.energy_gradient(H, config)
        @test length(G) == n_sites(H)
        for s = 1:n_sites(H)
            @test G[s] == MC.site_gradient(H, s, config)
            @test abs(dot(G[s], config[s])) < 1e-12          # tangent-projected
        end
        # central finite differences in two orthonormal tangent directions
        s = 1
        e = config[s]
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
            @test dot(G[s], v) ≈ fd atol = 1e-8 rtol = 1e-5
        end
    end

    @testset "bit-identical for any ntasks" begin
        H = TiledHamiltonian(_biquadratic_model(2); dims = (3, 2, 1))
        config = _rand_config(rng, H)
        G1 = MC.energy_gradient(H, config; ntasks = 1)
        for nt in (2, 3, 7, 64)                     # 64 > n_sites: clamped, still ==
            @test MC.energy_gradient(H, config; ntasks = nt) == G1
        end
    end

    @testset "τ = G × e ≡ SLCE.predict_torque on the training cell" begin
        model = _biquadratic_model(3)
        H = TiledHamiltonian(model; dims = (1, 1, 1))
        config = _rand_config(rng, H)
        G = MC.energy_gradient(H, config)
        T = predict_torque(model, MC.to_matrix(config))
        for s = 1:n_sites(H)
            # ≈, not ==: tiled-instance vs SALC-member accumulation order differ
            τ = cross(G[s], config[s])
            @test τ ≈ SVector{3,Float64}(T[1, s], T[2, s], T[3, s]) atol = 1e-12
        end
    end

    @testset "inactive sites: exact zeros" begin
        H = TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))  # atoms 3–4 free
        @test any(!, H.site_active)
        config = _rand_config(rng, H)
        G = MC.energy_gradient(H, config)
        for s = 1:n_sites(H)
            if !H.site_active[s]
                @test G[s] == zero(SVector{3,Float64})
            end
        end
    end

    @testset "≡ minimize.jl _gradient! (bitwise)" begin
        H = TiledHamiltonian(_biquadratic_model(4); dims = (2, 1, 1))
        config = _rand_config(rng, H)
        G = Vector{SVector{3,Float64}}(undef, n_sites(H))
        MC._gradient!(G, H, config, MC._zrows(H, config), zeros(H.nlm),
                      Vector{Float64}(undef, H.lmax + 1))
        @test G == MC.energy_gradient(H, config)
    end

    @testset "argument validation" begin
        H = TiledHamiltonian(_dimer_model(); dims = (1, 1, 1))
        config = _rand_config(rng, H)
        @test_throws DimensionMismatch MC.energy_gradient!(
            Vector{SVector{3,Float64}}(undef, 2), H, config)
        @test_throws DimensionMismatch MC.energy_gradient(
            H, MC.SpinConfig([SVector(0.0, 0.0, 1.0)]))
        @test_throws ArgumentError MC.energy_gradient(H, config; ntasks = 0)
    end
end
