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
        @test obs[:energy].f(_view(H, config, E)) == E
        @test obs[:energy2].f(_view(H, config, E)) == E^2
        @test obs[:m].f(_view(H, config, E)) ≈ SVector(0.0, 0.0, 0.0) atol = 1e-15
        @test obs[:absm].f(_view(H, config, E)) ≈ 0.0 atol = 1e-15
        sub = obs[:sublattice_m].f(_view(H, config, E))
        @test length(sub) == 12
        @test sub[3] ≈ 1.0 atol = 1e-15      # atom 1, z
        @test sub[6] ≈ -1.0 atol = 1e-15     # atom 2, z
        @test sub[7:12] == zeros(6)          # inactive sublattices: exactly zero
        # uniform tilt: |m| = 1, m4 = m2² = 1
        tilt = normalize(SVector(1.0, 2.0, 2.0))
        uniform = MC.SpinConfig([tilt for _ = 1:n_sites(H)])
        @test obs[:absm].f(_view(H, uniform)) ≈ 1.0 atol = 1e-14
        @test obs[:m2].f(_view(H, uniform)) ≈ 1.0 atol = 1e-14
        @test obs[:m4].f(_view(H, uniform)) ≈ 1.0 atol = 1e-14
    end

    @testset "accumulate → finalize: raw stats and jackknifed evaluables" begin
        rng = MersenneTwister(21)
        planned = 512
        accs = [MC.ObsAccumulator(o, planned, 32) for o in standard_observables(H)]
        config = _rand_config(rng, H)
        for _ = 1:planned
            config = _rand_config(rng, H)
            E = total_energy(H, config)
            for acc in accs
                MC._measure!(acc, _view(H, config, E))
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
                           v -> mean(v.config[s][3] for s = 1:length(v.config)
                                     if MC.site_atom(v.H, s) == 1))
        up = SVector(0.0, 0.0, 1.0)
        config = MC.SpinConfig([up for _ = 1:n_sites(H)])
        @test myobs.f(_view(H, config)) ≈ 1.0 atol = 1e-15

        # an evaluable over it
        myev = Evaluable(:sub1z_sq, [:sub1z], (m, kT, n) -> m.sub1z^2)
        accs = [MC.ObsAccumulator(myobs, 64, 8)]
        for _ = 1:64
            MC._measure!(accs[1], _view(H, config))
        end
        stats = MC._finalize_stats(accs, [myev], 1.0, n_sites(H), n_sites(H))
        @test stats[:sub1z_sq].mean[1] ≈ 1.0 atol = 1e-12

        # guards: missing / non-scalar inputs
        bad = Evaluable(:nope, [:missing_obs], (m, kT, n) -> 0.0)
        @test_throws ArgumentError MC._finalize_stats(accs, [bad], 1.0, 8, 8)
        vec_obs = Observable(:vec3, 3, v -> SVector(1.0, 2.0, 3.0))
        vaccs = [MC.ObsAccumulator(vec_obs, 8, 4)]
        badv = Evaluable(:nope2, [:vec3], (m, kT, n) -> 0.0)
        @test_throws ArgumentError MC._finalize_stats(vaccs, [badv], 1.0, 8, 8)

        # a wrongly-declared component count is caught at measurement time
        wrong = Observable(:oops, 2, v -> 1.0)
        wacc = MC.ObsAccumulator(wrong, 8, 4)
        @test_throws DimensionMismatch MC._measure!(wacc, _view(H, config))

        # ragged input columns (accumulators out of lockstep — unreachable through
        # the run drivers, reachable by hand) are refused, never silently truncated
        # to the shortest column: `jackknife` requires equal lengths (audit #4)
        ra = MC.ObsAccumulator(Observable(:ragA, 1, v -> 1.0), 64, 8)
        rb = MC.ObsAccumulator(Observable(:ragB, 1, v -> 2.0), 64, 8)
        for k = 1:64
            MC._measure!(ra, _view(H, config))
            k <= 32 && MC._measure!(rb, _view(H, config))   # 8 vs 4 filled bins
        end
        rag = Evaluable(:rag, [:ragA, :ragB], (m, kT, n) -> 0.0)
        err = try
            MC._finalize_stats([ra, rb], [rag], 1.0, 8, 8)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("lockstep", err.msg)
    end

    @testset "fewer planned measurements than nbins degrades, not NaNs" begin
        # planned < nbins ⇒ bin_size = 1, only `planned` bins fill; the jackknife
        # runs over those (≥ 2) instead of erroring or NaN-ing.
        up = SVector(0.0, 0.0, 1.0)
        config = MC.SpinConfig([up for _ = 1:n_sites(H)])
        obs = [Observable(:energy, 1, v -> v.energy),
               Observable(:energy2, 1, v -> v.energy^2)]
        accs = [MC.ObsAccumulator(o, 5, 32) for o in obs]   # planned 5 ≪ nbins 32
        for k = 1:5
            for acc in accs
                MC._measure!(acc, _view(H, config, Float64(k)))
            end
        end
        stats = MC._finalize_stats(accs, standard_evaluables()[1:1], 0.1,
                                   n_sites(H), n_sites(H))
        @test stats[:specific_heat].count == 5              # bins actually used
        @test isfinite(stats[:specific_heat].mean[1])
        # and with < 2 bins the evaluable is NaN-guarded
        acc1 = [MC.ObsAccumulator(o, 1, 32) for o in obs]
        for acc in acc1
            MC._measure!(acc, _view(H, config, 1.0))
        end
        s1 = MC._finalize_stats(acc1, standard_evaluables()[1:1], 0.1, n_sites(H),
                                n_sites(H))
        @test isnan(s1[:specific_heat].mean[1])
    end
end

# The two spin evaluables had no oracle at all: the suite checked `isfinite` and `> 0`,
# so `χ ∝ 1/kT²` or a stray factor in the Binder ratio would have shipped silently
# (both mutations were confirmed to leave the whole suite green). Pin them against
# closed forms instead, in the one regime where closed forms exist.
#
# FREE SPINS. At kT ≫ |J| the couplings are negligible and `m = Σe/N` is the mean of N
# iid unit vectors, so for large N it is Gaussian with per-component variance 1/(3N):
#
#   ⟨m²⟩ = 1/N;  |Σe| is Maxwell with σ² = N/3, so ⟨|m|⟩² = 4σ²(2/π)/N² = 8/(3πN)
#   ⇒  χ = N(⟨m²⟩ − ⟨|m|⟩²)/kT = (1 − 8/3π)/kT          — note the FIRST power of kT
#   ⇒  U = ⟨m⁴⟩/⟨m²⟩² = 5/3                              — the 3-d Gaussian ratio
#
# Neither number comes from this package. Tolerance: 5× the run's own binning error.
# Measured across three seeds at this kT the worst deviation was 2.6σ (χ) and 2.1σ (U),
# so the bound carries ≈ 2× headroom — while the mutations it must resolve sit ~40σ
# (χ, a whole factor of kT) and ~28σ (U, a factor of two) away.
@testset "susceptibility and Binder against their free-spin closed forms" begin
    Hf = TiledHamiltonian(_dimer_model(); dims = (4, 4, 4))    # 128 sites
    kT = 200 * abs(_dimer_J(Hf))                               # deep in the free regime
    for seed in (12, 13)
        r = run_mc(Hf; kT = kT, sweeps_therm = 2000, sweeps_measure = 20_000,
                   measure_interval = 5, nbins = 8, seed = seed)
        chi, U = r.points[1].stats[:susceptibility], r.points[1].stats[:binder]
        @test abs(chi.mean[1] - (1 - 8 / (3π)) / kT) < 5 * chi.err[1]
        @test abs(U.mean[1] - 5 / 3) < 5 * U.err[1]
        # the error bars themselves must be small enough for the above to mean anything
        @test chi.err[1] < 0.1 * chi.mean[1] && U.err[1] < 0.1 * U.mean[1]
    end
end

# The evaluable half of the entry validation. `_finalize_stats` raises the same three
# errors, but it runs AFTER the measurement phase, so before this check a mistyped
# input name cost the whole run's samples (and a resume re-threw at the same place —
# the accumulators are checkpointed, the finalized result is not). The name collision
# is the quiet one: raw stats and evaluables share one `Dict`, so an evaluable named
# after an observable silently REPLACED that observable's binning result.
@testset "evaluables are validated at entry, not after the run" begin
    Hs = TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))
    obs = standard_observables(Hs)
    run_short(; kw...) = run_mc(Hs; kT = 0.05, sweeps_therm = 2, sweeps_measure = 4,
                                measure_interval = 1, nbins = 2, seed = 1, kw...)
    # (a) an input that is not measured
    missing_input = Evaluable(:my_ratio, [:not_measured], (m, kT, n) -> 1.0)
    @test_throws ArgumentError run_short(observables = obs,
                                         evaluables = [missing_input])
    # (b) a non-scalar input (`:m` has three components)
    vector_input = Evaluable(:my_m, [:m], (m, kT, n) -> 1.0)
    @test_throws ArgumentError run_short(observables = obs, evaluables = [vector_input])
    # (c) the collision: an evaluable named after a measured observable
    shadowing = Evaluable(:energy, [:energy], (m, kT, n) -> m.energy / n)
    @test_throws ArgumentError run_short(observables = obs, evaluables = [shadowing])
    # (d) two evaluables of the same name
    dup = Evaluable(:dup, [:energy], (m, kT, n) -> m.energy)
    @test_throws ArgumentError run_short(observables = obs, evaluables = [dup, dup])
    # and the valid combination still runs
    ok = Evaluable(:e_per_site, [:energy], (m, kT, n) -> m.energy / n)
    r = run_short(observables = obs, evaluables = [ok])
    @test haskey(r.points[1].stats, :e_per_site)
    @test haskey(r.points[1].stats, :energy)          # the raw observable survives
end
