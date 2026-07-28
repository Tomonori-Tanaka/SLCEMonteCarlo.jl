# Inactive (non-magnetic) sites: a site no instance touches — a species with
# lmax = 0, or one whose SALC coefficients all fitted to zero — has a
# spin-independent energy. Contract: flagged in `site_active`/`n_active`, skipped by
# the update sweeps (no RNG consumed, not counted as attempts), excluded from the
# standard observables and their per-site normalizations, and kept bitwise frozen
# through sweeps, renormalization, and the ground-state descent.
#
# The dimer fixture is the natural test bed: atoms 1–2 are the coupled pair, atoms
# 3–4 are outside every cutoff — inactive.

@testset "inactive sites" begin
    Hd = TiledHamiltonian(_dimer_model())

    @testset "flags, counts, printing" begin
        @test Hd.site_active == [true, true, false, false]
        @test Hd.n_active == 2
        @test all(!, Hd.site_has_l1[3:4])
        @test occursin("(2 inactive)", sprint(show, Hd))
        H2 = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
        @test H2.site_active == [true, true, false, false, true, true, false, false]
        @test H2.n_active == 4
        # an all-coupled model reports no inactive suffix
        Hc = TiledHamiltonian(1, _chain_terms(-0.05); dims = (4, 1, 1))
        @test Hc.n_active == n_sites(Hc)
        @test !occursin("inactive", sprint(show, Hc))
    end

    @testset "coef == 0 terms are dropped up front" begin
        terms = _chain_terms(-0.05)
        zero_term = SpinMultipoleTerm(0.0, terms[1].body, terms[1].atoms, terms[1].shifts,
                                  terms[1].ls, terms[1].folded)
        Ha = TiledHamiltonian(1, terms; dims = (3, 1, 1))
        Hb = TiledHamiltonian(1, vcat(terms, [zero_term]); dims = (3, 1, 1))
        @test length(Hb.terms) == length(Ha.terms)
        rng = Xoshiro(5)
        cfg = MC.SpinConfig([MC._random_unit(rng) for _ = 1:n_sites(Ha)])
        @test total_energy(Hb, cfg) == total_energy(Ha, cfg)
        # all-zero coefficients leave no spin-dependent term: a loud error
        @test_throws ArgumentError TiledHamiltonian(1, [zero_term])
    end

    @testset "sweeps freeze inactive spins and count only active attempts" begin
        rng = Xoshiro(2)
        st = MC.ChainState(Hd, _rand_config(rng, Hd), Xoshiro(11), 0.6)
        sc = MC.SweepScratch(Hd)
        frozen = (st.config[3], st.config[4])
        for _ = 1:20
            MC.metropolis_sweep!(st, Hd, 1 / 0.02, sc)
            MC.overrelaxation_sweep!(st, Hd, 1 / 0.02, sc)
        end
        @test st.config[3] === frozen[1] && st.config[4] === frozen[2]
        @test st.att_metro == 20 * Hd.n_active
        MC._renormalize!(st, Hd)
        @test st.config[3] === frozen[1] && st.config[4] === frozen[2]
    end

    @testset "standard observables exclude inactive sites" begin
        up = SVector(0.0, 0.0, 1.0)
        x = SVector(1.0, 0.0, 0.0)
        cfg = MC.SpinConfig([up, up, x, x])     # active aligned +z, inactive +x
        @test MC._mean_spin(_view(Hd, cfg)) == up
        sub = MC._sublattice_m(_view(Hd, cfg))
        @test sub[1:6] == [0, 0, 1, 0, 0, 1]
        @test sub[7:12] == zeros(6)             # inactive sublattices exactly zero
        # evaluables receive n_active, not n_sites
        nprobe = Evaluable(:nprobe, [:energy], (m, kT, n) -> Float64(n))
        r = run_mc(Hd; kT = 0.05, sweeps_therm = 10, sweeps_measure = 40, nbins = 4,
                   evaluables = [nprobe], seed = 3)
        @test r.points[1].stats[:nprobe].mean[1] == Hd.n_active
    end

    @testset "PT: frozen through lane swaps, n_active reaches the evaluables" begin
        # a shared explicit init makes every lane's frozen inactive directions
        # identical, so they must survive any number of payload swaps bitwise
        up = SVector(0.0, 0.0, 1.0)
        x = SVector(1.0, 0.0, 0.0)
        init = MC.SpinConfig([up, up, x, x])
        nprobe = Evaluable(:nprobe, [:energy], (m, kT, n) -> Float64(n))
        r = run_pt(Hd; kT = [0.06, 0.03], init = init, sweeps_therm = 50,
                   sweeps_measure = 100, exchange_interval = 5, nbins = 4,
                   evaluables = [nprobe], seed = 9)
        @test sum(r.swap_acceptance) > 0        # swaps actually happened
        for cfg in r.final_configs
            @test cfg[3] === x && cfg[4] === x
        end
        for p in r.points
            @test p.stats[:nprobe].mean[1] == Hd.n_active
            @test p.stats[:sublattice_m].mean[7:12] == zeros(6)
        end
    end

    @testset "checkpoint resume is bit-identical with inactive sites" begin
        path = joinpath(mktempdir(), "inactive.jld2")
        kw = (; kT = 0.03, sweeps_therm = 100, sweeps_measure = 300, nbins = 4,
              seed = 21)
        a = run_mc(Hd; kw...)
        b = run_mc(Hd; kw..., checkpoint = path, checkpoint_interval = 80)
        @test a.final_config == b.final_config
        c = resume(path, Hd)
        @test a.final_config == c.final_config
        @test a.points[1].stats[:m].mean == c.points[1].stats[:m].mean
    end

    @testset "ground-state search keeps inactive spins bitwise" begin
        rng = Xoshiro(4)
        cfg = _rand_config(rng, Hd)
        # exactly-unit inactive spins: the init-boundary renormalization (e/‖e‖,
        # applied to every spin) is then bitwise a no-op, isolating the descent
        cfg[3] = SVector(1.0, 0.0, 0.0)
        cfg[4] = SVector(0.0, 1.0, 0.0)
        frozen = (cfg[3], cfg[4])
        r = minimize_energy(Hd; init = cfg)
        @test r.converged
        @test r.config[3] == frozen[1] && r.config[4] == frozen[2]
        @test MC.site_gradient(Hd, 3, r.config) == zero(SVector{3,Float64})
        @test dot(r.config[1], r.config[2]) ≈ 1 atol = 1e-8   # ferro pair aligned
    end
end
