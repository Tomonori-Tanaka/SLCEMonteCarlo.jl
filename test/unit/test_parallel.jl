# Parallel sweep execution (`sweep_tasks`): the sweeps scan color classes with
# per-site RNG streams and reduce the staged ΔE in fixed class order, so serial
# and any-task-count execution are the same chain, bit for bit
# (docs/specs/updates-stationarity.md U1). These gates pin that claim end to end.

@testset "parallel sweeps (sweep_tasks)" begin
    @testset "coloring is proper and partitions the active sites" begin
        for H in [TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1)),
                  MC.TiledHamiltonian(1, _chain_terms(0.05); dims = (4, 1, 1)),
                  TiledHamiltonian(_dimer_model(); dims = (2, 1, 1))]
            colors = zeros(Int, n_sites(H))
            for c = 1:H.n_colors, q = H.color_ptr[c]:(H.color_ptr[c + 1] - 1)
                colors[H.color_sites[q]] = c
            end
            ok = true
            for i in eachindex(H.inst_term)
                mem = H.inst_sites[H.inst_ptr[i]:(H.inst_ptr[i + 1] - 1)]
                ok &= allunique(colors[mem])   # no instance inside one class
            end
            @test ok
            @test sort(vec(H.color_sites)) == findall(H.site_active)
            @test H.color_ptr[end] - 1 == H.n_active
        end
    end

    @testset "serial ≡ parallel sweeps, any task count (bitwise)" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        β = 1 / 0.05
        function run_chain(ntasks)
            st = MC.ChainState(H, MC._initial_config(H, nothing, Xoshiro(3)),
                               Xoshiro(5), 0.6)
            scs = [MC.SweepScratch(H) for _ = 1:ntasks]
            for _ = 1:25
                MC.metropolis_sweep!(st, H, β, scs)
                MC.overrelaxation_sweep!(st, H, β, scs)
            end
            return st
        end
        ref = run_chain(1)
        for nt in (2, 3, 7)
            st = run_chain(nt)
            @test st.config == ref.config
            @test st.zrows == ref.zrows
            @test st.energy === ref.energy
            @test (st.acc_metro, st.att_metro, st.acc_or, st.att_or) ==
                  (ref.acc_metro, ref.att_metro, ref.acc_or, ref.att_or)
        end
        # and the incremental energy stayed exact
        @test abs(ref.energy - total_energy(H, ref.config)) < 1e-10
    end

    @testset "serial ≡ parallel with inactive sites (dimer, frozen spins)" begin
        H = TiledHamiltonian(_dimer_model())        # atoms 3–4 are inactive
        @test H.n_active < n_sites(H)
        β = 1 / 0.05
        function run_dimer(ntasks)
            st = MC.ChainState(H, MC._initial_config(H, nothing, Xoshiro(13)),
                               Xoshiro(17), 0.6)
            scs = [MC.SweepScratch(H) for _ = 1:ntasks]
            for _ = 1:20
                MC.metropolis_sweep!(st, H, β, scs)
            end
            return st
        end
        ref = run_dimer(1)
        st = run_dimer(3)
        @test st.config == ref.config
        @test st.energy === ref.energy
        @test st.att_metro == 20 * H.n_active   # inactive never counted
        # frozen spins bitwise untouched in both paths
        init = MC._initial_config(H, nothing, Xoshiro(13))
        for s in findall(.!H.site_active)
            @test st.config[s] === init[s] && ref.config[s] === init[s]
        end
    end

    @testset "run_mc: sweep_tasks 1 vs N bit-identical end-to-end" begin
        H = MC.TiledHamiltonian(1, _chain_terms(-0.02); dims = (4, 2, 1))
        args = (; kT = 0.02, sweeps_therm = 40, sweeps_measure = 80,
                or_per_metropolis = 1, nbins = 4, seed = UInt64(11))
        r1 = run_mc(H; args..., sweep_tasks = 1)
        r3 = run_mc(H; args..., sweep_tasks = 3)
        @test r1.final_config == r3.final_config
        p1, p3 = r1.points[1], r3.points[1]
        for k in keys(p1.stats)
            @test p1.stats[k].mean == p3.stats[k].mean
            @test p1.stats[k].err == p3.stats[k].err
        end
        @test p1.acceptance_metropolis == p3.acceptance_metropolis
    end

    @testset "run_pt: sweep_tasks composes with the lane pool" begin
        H = TiledHamiltonian(_biquadratic_model(1); dims = (2, 1, 1))
        args = (; kT = [0.02, 0.05, 0.1], sweeps_therm = 20, sweeps_measure = 40,
                exchange_interval = 5, nbins = 4, seed = UInt64(4))
        r1 = run_pt(H; args..., sweep_tasks = 1, ntasks = 1)
        r2 = run_pt(H; args..., sweep_tasks = 2, ntasks = 2)
        @test r1.final_configs == r2.final_configs
        @test r1.swap_acceptance == r2.swap_acceptance
    end

    @testset "find_ground_state: sweep_tasks bit-identical" begin
        H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))
        kw = (; nstarts = 2, anneal_sweeps = 10, kT = [0.5, 0.1], seed = 7)
        g1 = find_ground_state(H; kw..., sweep_tasks = 1)
        g2 = find_ground_state(H; kw..., sweep_tasks = 2)
        @test g1.energies == g2.energies
        @test g1.configs == g2.configs
        @test g1.best == g2.best
    end

    @testset "checkpoint carries sweep_tasks and site streams (bit-identity)" begin
        H = MC.TiledHamiltonian(1, _chain_terms(-0.02); dims = (4, 1, 1))
        dir = mktempdir()
        path = joinpath(dir, "ck.jld2")
        # seed ≥ 2^63 also gates the UInt64 seed round-trip (Int() would throw)
        args = (; kT = 0.02, sweeps_therm = 10, sweeps_measure = 30, nbins = 4,
                seed = 0x8000000000000003, sweep_tasks = 2)
        full = run_mc(H; args...)
        run_mc(H; args..., checkpoint = path)
        c = resume(path, H)          # completed-run checkpoint → the same result
        @test c.final_config == full.final_config
        @test c.points[1].stats[:energy].mean == full.points[1].stats[:energy].mean
    end

    @testset "joint drivers: sweep_tasks / ntasks bit-identical end-to-end" begin
        # P6 for the compound sweep that now carries a displacement pass. The sweep
        # kernel's own serial ≡ parallel gate is in test_joint.jl; this is the driver
        # level, where the pass scheduling, the second proposal width's adaptation and
        # (for PT) the payload swap all participate.
        terms, L = _einstein_terms(2.5)
        H = MC.TiledHamiltonian(1, terms, L; dims = (2, 2, 2),
                                fixed_reference = true)
        obs = [Observable(:energy, 1, (c, E, h) -> E)]
        kw = (; sweeps_therm = 120, sweeps_measure = 200, nbins = 4,
              renorm_interval = 40, step_u = 0.3, seed = 99, observables = obs,
              evaluables = Evaluable[])
        a = run_mc(H; kT = 0.4, kw..., sweep_tasks = 1)
        b = run_mc(H; kT = 0.4, kw..., sweep_tasks = 4)
        @test a.points[1].stats[:energy].mean == b.points[1].stats[:energy].mean
        @test a.points[1].disp_rms == b.points[1].disp_rms
        @test a.points[1].acceptance_disp == b.points[1].acceptance_disp
        @test a.points[1].final_step_u == b.points[1].final_step_u
        p = run_pt(H; kT = [0.4, 0.3, 0.2], kw..., exchange_interval = 10,
                   ntasks = 1)
        q = run_pt(H; kT = [0.4, 0.3, 0.2], kw..., exchange_interval = 10,
                   ntasks = 3, sweep_tasks = 2)
        @test p.swap_acceptance == q.swap_acceptance
        for i = 1:3
            @test p.points[i].stats[:energy].mean == q.points[i].stats[:energy].mean
            @test p.points[i].disp_rms == q.points[i].disp_rms
            @test p.points[i].acceptance_disp == q.points[i].acceptance_disp
        end
        # the run really did move the displacements — otherwise this gate is vacuous
        @test p.points[1].disp_rms > 0.0
    end

    @testset "validation" begin
        H = TiledHamiltonian(_dimer_model())
        @test_throws ArgumentError run_mc(H; kT = 0.1, sweeps_therm = 1,
                                          sweeps_measure = 2, sweep_tasks = 0)
        st = MC.ChainState(H, MC._initial_config(H, nothing, Xoshiro(1)),
                           Xoshiro(2), 0.5)
        @test_throws ArgumentError MC.metropolis_sweep!(st, H, 1.0,
                                                        MC.SweepScratch[])
    end
end