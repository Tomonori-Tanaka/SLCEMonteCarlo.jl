# Observables and evaluables: exact values on hand-set configurations, the
# accumulation/finalize plumbing, and jackknifed evaluables against direct formulas.

@testset "observables" begin
    H = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))   # 4 atoms × 2 cells

    @testset "standard observables on a staggered configuration" begin
        up = SVector(0.0, 0.0, 1.0)
        # the ACTIVE sublattices stagger (1 up, 2 down ⇒ m cancels); the inactive
        # sublattices (3, 4 — outside every cutoff) are set up too, and must not
        # leak into the active-site mean
        config = MC.SpinConfig([MC.site_atom(H, s) == 2 ? -up : up
                                for s = 1:n_sites(H)])
        E = total_energy(H, config)
        obs = Dict(o.name => o for o in standard_observables(H))
        @test obs[:energy].f(config, E, H) == E
        @test obs[:energy2].f(config, E, H) == E^2
        @test obs[:m].f(config, E, H) ≈ SVector(0.0, 0.0, 0.0) atol = 1e-15
        @test obs[:absm].f(config, E, H) ≈ 0.0 atol = 1e-15
        sub = obs[:sublattice_m].f(config, E, H)
        @test length(sub) == 12
        @test sub[3] ≈ 1.0 atol = 1e-15      # atom 1, z
        @test sub[6] ≈ -1.0 atol = 1e-15     # atom 2, z
        @test sub[7:12] == zeros(6)          # inactive sublattices: exactly zero
        # uniform tilt: |m| = 1, m4 = m2² = 1
        tilt = normalize(SVector(1.0, 2.0, 2.0))
        uniform = MC.SpinConfig([tilt for _ = 1:n_sites(H)])
        @test obs[:absm].f(uniform, 0.0, H) ≈ 1.0 atol = 1e-14
        @test obs[:m2].f(uniform, 0.0, H) ≈ 1.0 atol = 1e-14
        @test obs[:m4].f(uniform, 0.0, H) ≈ 1.0 atol = 1e-14
    end

    @testset "accumulate → finalize: raw stats and jackknifed evaluables" begin
        rng = MersenneTwister(21)
        planned = 512
        accs = [MC.ObsAccumulator(o, planned, 32) for o in standard_observables(H)]
        cfg = _rand_config(rng, H)
        for _ = 1:planned
            cfg = _rand_config(rng, H)
            E = total_energy(H, cfg)
            for acc in accs
                MC._measure!(acc, cfg, E, H)
            end
        end
        kT = 0.05
        stats = MC._finalize_stats(accs, standard_evaluables(), kT, H.n_active,
                                   H.n_active)
        @test stats[:energy].count == planned
        @test length(stats[:m].mean) == 3
        @test length(stats[:sublattice_m].mean) == 12
        # direct check of the jackknife inputs: C/k_B from the stored bins
        e_bins = vec(MC.bin_means(accs[1].store))
        e2_bins = vec(MC.bin_means(accs[2].store))
        c_direct, _ = MC.jackknife((m1, m2) -> (m2 - m1^2) / (H.n_active * kT^2),
                                   [e_bins, e2_bins])
        @test stats[:specific_heat].mean[1] ≈ c_direct atol = 1e-12
        @test isnan(stats[:specific_heat].tau_int[1])
        @test stats[:binder].count == 32
        # susceptibility and binder are finite and sane on random spins
        @test isfinite(stats[:susceptibility].mean[1])
        @test stats[:binder].mean[1] > 0
    end

    @testset "user observables and evaluables" begin
        # a scalar user observable: the z-projection of sublattice 1
        myobs = Observable(:sub1z, 1,
                           (cfg, E, H) -> mean(cfg[s][3] for s = 1:length(cfg)
                                               if MC.site_atom(H, s) == 1))
        up = SVector(0.0, 0.0, 1.0)
        config = MC.SpinConfig([up for _ = 1:n_sites(H)])
        @test myobs.f(config, 0.0, H) ≈ 1.0 atol = 1e-15

        # an evaluable over it
        myev = Evaluable(:sub1z_sq, [:sub1z], (m, kT, n) -> m.sub1z^2)
        accs = [MC.ObsAccumulator(myobs, 64, 8)]
        for _ = 1:64
            MC._measure!(accs[1], config, 0.0, H)
        end
        stats = MC._finalize_stats(accs, [myev], 1.0, n_sites(H), n_sites(H))
        @test stats[:sub1z_sq].mean[1] ≈ 1.0 atol = 1e-12

        # guards: missing / non-scalar inputs
        bad = Evaluable(:nope, [:missing_obs], (m, kT, n) -> 0.0)
        @test_throws ArgumentError MC._finalize_stats(accs, [bad], 1.0, 8, 8)
        vec_obs = Observable(:vec3, 3, (cfg, E, H) -> SVector(1.0, 2.0, 3.0))
        vaccs = [MC.ObsAccumulator(vec_obs, 8, 4)]
        badv = Evaluable(:nope2, [:vec3], (m, kT, n) -> 0.0)
        @test_throws ArgumentError MC._finalize_stats(vaccs, [badv], 1.0, 8, 8)

        # a wrongly-declared component count is caught at measurement time
        wrong = Observable(:oops, 2, (cfg, E, H) -> 1.0)
        wacc = MC.ObsAccumulator(wrong, 8, 4)
        @test_throws DimensionMismatch MC._measure!(wacc, config, 0.0, H)
    end

    @testset "fewer planned measurements than nbins degrades, not NaNs" begin
        # planned < nbins ⇒ bin_size = 1, only `planned` bins fill; the jackknife
        # runs over those (≥ 2) instead of erroring or NaN-ing.
        up = SVector(0.0, 0.0, 1.0)
        config = MC.SpinConfig([up for _ = 1:n_sites(H)])
        obs = [Observable(:energy, 1, (c, E, h) -> E),
               Observable(:energy2, 1, (c, E, h) -> E^2)]
        accs = [MC.ObsAccumulator(o, 5, 32) for o in obs]   # planned 5 ≪ nbins 32
        for k = 1:5
            for acc in accs
                MC._measure!(acc, config, Float64(k), H)
            end
        end
        stats = MC._finalize_stats(accs, standard_evaluables()[1:1], 0.1,
                                   n_sites(H), n_sites(H))
        @test stats[:specific_heat].count == 5              # bins actually used
        @test isfinite(stats[:specific_heat].mean[1])
        # and with < 2 bins the evaluable is NaN-guarded
        acc1 = [MC.ObsAccumulator(o, 1, 32) for o in obs]
        for acc in acc1
            MC._measure!(acc, config, 1.0, H)
        end
        s1 = MC._finalize_stats(acc1, standard_evaluables()[1:1], 0.1, n_sites(H),
                                n_sites(H))
        @test isnan(s1[:specific_heat].mean[1])
    end
end
