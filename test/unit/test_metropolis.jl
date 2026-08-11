# Metropolis + run_mc: exact statistical gates (dimer pair correlation, single-site
# Langevin), adaptive-step behavior, seed reproducibility, drift, annealing.
# Statistical tolerances mirror the proven SLCETools MC suite (fixed seeds).

@testset "Metropolis / run_mc" begin
    Hd = TiledHamiltonian(_dimer_model())
    J = _dimer_J(Hd)                       # < 0 (ferro)
    @test J < 0

    @testset "exact dimer gate: ⟨e₁·e₂⟩ = L(β|J|), ⟨E⟩ = J·L(β|J|)" begin
        for (i, βJ) in enumerate([1.0, 2.5])
            kt = abs(J) / βJ
            obs = vcat(standard_observables(Hd), _corr12_obs())
            r = run_mc(Hd; kT = kt, sweeps_therm = 500, sweeps_measure = 20_000,
                       measure_interval = 5, seed = 100 + i, observables = obs)
            exact = _langevin(βJ)          # ferro: positive alignment
            p = r.points[1]
            @test p.stats[:corr12].mean[1] ≈ exact atol = 0.03
            @test p.stats[:energy].mean[1] ≈ J * exact atol = 0.03 * abs(J)
            # the binning error bar is honest: within 5σ of exact
            @test abs(p.stats[:corr12].mean[1] - exact) <
                  5 * max(p.stats[:corr12].err[1], 1e-3)
            # the uncoupled sublattices (3, 4) are inactive: excluded from the
            # standard observables, so their reported moment is exactly zero
            sub = p.stats[:sublattice_m].mean
            @test sub[7:9] == zeros(3)
            @test sub[10:12] == zeros(3)
        end
    end

    @testset "single-site Langevin limit: ⟨e_z⟩ = −L(βh)" begin
        c0 = 0.3
        n1 = SLCE.Harmonics.N1
        h = c0 * n1                        # V(e) = h·e_z
        folded = zeros(3)
        folded[2] = 1.0
        term = SpinMultipoleTerm(c0 / sqrt(4π), 1, [1], [SVector(0, 0, 0)], [1], folded)
        H1 = TiledHamiltonian(1, [term])
        ez = Observable(:ez, 1, v -> v.config[1][3])
        for (i, βh) in enumerate([1.0, 2.0])
            kt = h / βh
            r = run_mc(H1; kT = kt, sweeps_therm = 500, sweeps_measure = 40_000,
                       measure_interval = 2, seed = 200 + i,
                       observables = [Observable(:energy, 1, v -> v.energy),
                                      Observable(:energy2, 1, v -> v.energy^2), ez],
                       evaluables = Evaluable[])
            @test r.points[1].stats[:ez].mean[1] ≈ -_langevin(βh) atol = 0.03
        end
    end

    @testset "adaptive step: target acceptance, frozen during measurement" begin
        # every site coupled (the dimer fixture's uncoupled sites are inactive and
        # skipped now, but an all-coupled model keeps this gate self-contained)
        H = TiledHamiltonian(1, _chain_terms(-0.05); dims = (4, 4, 1))
        r = run_mc(H; kT = 0.01, sweeps_therm = 1_000, sweeps_measure = 2_000,
                   seed = 7)
        p = r.points[1]
        @test abs(p.acceptance_metropolis - 0.5) < 0.15
        @test 1e-3 <= p.final_step <= Float64(π)
        # the frozen step does not depend on the measurement length
        r2 = run_mc(H; kT = 0.01, sweeps_therm = 1_000, sweeps_measure = 200,
                    seed = 7)
        @test r2.points[1].final_step == p.final_step
    end

    @testset "seed reproducibility" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        kw = (; kT = [0.05, 0.02], sweeps_therm = 200, sweeps_measure = 400,
              nbins = 8, seed = 3)
        a = run_mc(H; kw...)
        b = run_mc(H; kw...)
        @test a.final_config == b.final_config
        for (pa, pb) in zip(a.points, b.points)
            @test pa.stats[:energy].mean == pb.stats[:energy].mean
            @test pa.stats[:m].mean == pb.stats[:m].mean
            @test pa.acceptance_metropolis == pb.acceptance_metropolis
            @test pa.final_step == pb.final_step
        end
        c = run_mc(H; kT = [0.05, 0.02], sweeps_therm = 200, sweeps_measure = 400,
                   nbins = 8, seed = 4)
        @test c.final_config != a.final_config

        # the default seed is drawn fresh per call (independent runs, no silent
        # duplicates) and recorded, so any run can be reproduced after the fact
        kwd = (; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100, nbins = 4)
        d1 = run_mc(H; kwd...)
        d2 = run_mc(H; kwd...)
        @test d1.seed != d2.seed
        d3 = run_mc(H; kwd..., seed = d1.seed)
        @test d3.final_config == d1.final_config
        @test d3.points[1].stats[:energy].mean == d1.points[1].stats[:energy].mean
    end

    @testset "incremental-energy drift stays at machine scale" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        r = run_mc(H; kT = 0.03, sweeps_therm = 500, sweeps_measure = 5_000,
                   renorm_interval = 500, seed = 5)
        p = r.points[1]
        Escale = max(1.0, abs(p.stats[:energy].mean[1]))
        @test p.max_drift < 1e-9 * Escale
    end

    @testset "annealing ladder (warm start) and independent restarts" begin
        H = TiledHamiltonian(_dimer_model(); dims = (2, 2, 2))
        kts = abs(J) .* [4.0, 0.25]
        r = run_mc(H; kT = kts, sweeps_therm = 400, sweeps_measure = 2_000,
                   seed = 11)
        @test [p.kT for p in r.points] == kts
        @test r.points[1].temperature ≈ kts[1] / KB_EV
        E_hot = r.points[1].stats[:energy].mean[1]
        E_cold = r.points[2].stats[:energy].mean[1]
        @test E_cold < E_hot
        # cold and ordered: the coupled sublattices carry a large moment
        @test r.points[2].stats[:absm].mean[1] > 0.0
        ri = run_mc(H; kT = kts, sweeps_therm = 400, sweeps_measure = 2_000,
                    seed = 11, carryover = false)
        @test ri.points[2].stats[:energy].mean[1] < ri.points[1].stats[:energy].mean[1]
    end

    @testset "init forms and guards" begin
        H = TiledHamiltonian(_dimer_model())
        up = SVector(0.0, 0.0, 1.0)
        r = run_mc(H; kT = 1e-3, sweeps_therm = 0, sweeps_measure = 2,
                   init = [up for _ = 1:4], nbins = 2, evaluables = Evaluable[],
                   seed = 1)
        @test r isa MCResult
        # a moment-scaled init is refused, never silently normalized (the
        # family's unit-direction door; ‖e‖ = 2 pinned the opposite until 2026-08)
        @test_throws ArgumentError run_mc(H; kT = 1e-3, sweeps_therm = 0,
                                          sweeps_measure = 2, nbins = 2,
                                          evaluables = Evaluable[], seed = 1,
                                          init = [SVector(0.0, 0.0, 2.0)
                                                  for _ = 1:4])
        m = zeros(3, 4)
        m[3, :] .= 1.0
        r2 = run_mc(H; kT = 1e-3, sweeps_therm = 0, sweeps_measure = 2, init = m,
                    nbins = 2, evaluables = Evaluable[], seed = 1)
        @test r2 isa MCResult
        @test_throws DimensionMismatch run_mc(H; kT = 0.1, init = zeros(3, 5))
        @test_throws ArgumentError run_mc(H; kT = 0.1, init = zeros(3, 4))  # zero spins
        @test_throws ArgumentError run_mc(H)                                # no control
        @test_throws ArgumentError run_mc(H; kT = 0.1, temperature = 300.0)
        @test_throws ArgumentError run_mc(H; kT = -0.1)
        @test_throws ArgumentError run_mc(H; kT = 0.1, sweeps_measure = 0)
        @test_throws ArgumentError run_mc(H; kT = 0.1, step = 0.0)
        @test_throws ArgumentError run_mc(H; kT = 0.1, nbins = 1)
        @test_throws ArgumentError run_mc(H; kT = 0.1, seed = -1)
        @test_throws ArgumentError run_mc(H; kT = 0.1,
                                          observables = [Observable(:a, 1, +),
                                                         Observable(:a, 1, +)])
    end

    @testset "result printing" begin
        H = TiledHamiltonian(_dimer_model())
        r = run_mc(H; kT = 0.05, sweeps_therm = 50, sweeps_measure = 100, seed = 1)
        @test occursin("MCResult", sprint(show, r))
        long = sprint(show, MIME("text/plain"), r)
        @test occursin("E/site", long)
        @test occursin("C/kB", long)
    end
end
