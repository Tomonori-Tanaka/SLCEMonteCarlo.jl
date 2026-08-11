# Checkpoint / resume: bit-identity gates (==, never ≈) for MC and PT, Xoshiro
# round-trips, and the schema guards.

# Everything result-shaped must be bit-equal between two runs.
function _assert_same_result(a, b)
    @test length(a.points) == length(b.points)
    for (pa, pb) in zip(a.points, b.points)
        @test pa.kT == pb.kT
        @test sort(collect(keys(pa.stats))) == sort(collect(keys(pb.stats)))
        for k in keys(pa.stats)
            @test pa.stats[k].mean == pb.stats[k].mean
            @test pa.stats[k].err == pb.stats[k].err
            @test isequal(pa.stats[k].tau_int, pb.stats[k].tau_int)
            @test pa.stats[k].count == pb.stats[k].count
        end
        # `isequal`, not `==`: an acceptance is NaN where the move type had no attempt
        # (a displacement-only model has no spin-active site, and vice versa)
        @test isequal(pa.acceptance_metropolis, pb.acceptance_metropolis)
        @test isequal(pa.acceptance_or, pb.acceptance_or)
        @test isequal(pa.acceptance_disp, pb.acceptance_disp)
        @test pa.final_step == pb.final_step
        @test isequal(pa.final_step_u, pb.final_step_u)
        @test pa.max_drift == pb.max_drift
        @test isequal(pa.disp_rms, pb.disp_rms)
        @test isequal(pa.disp_max, pb.disp_max)
        @test pa.escaped == pb.escaped
    end
    # the end-of-run cell scale rides every bit-identity gate too (`nothing` on
    # every fixed-cell run this file drives — the field must still agree)
    a isa MCResult && @test isequal(a.final_strain, b.final_strain)
    a isa PTResult && @test isequal(a.final_strains, b.final_strains)
    return nothing
end

# The S12 interrupted-writer pattern (strain-move.md S12): a completed mc file
# ALWAYS ends at the completed marker — `_mc_loop!` writes an unconditional
# end-of-temperature boundary checkpoint — so a resume built on a finished run
# returns the stored result without re-running a sweep, and a
# resume-equals-uninterrupted assertion on it compares the file with itself.
# Mid-run continuation teeth therefore REQUIRE a writer that actually stops: the
# poison observable throws at the n-th measurement, leaving the file at the last
# periodic (or boundary) write, and `resume` completes the run from there. The
# benign twin re-supplies the name at resume (the checkpoint validates observable
# names/ncomps, never functions) and returns the same 0.0 the poison did before
# it fired, so the continued statistics stay bit-comparable. The PT gates below
# need none of this: `_pt_run!` has no end-of-run write, so their files land
# mid-measure by interval arithmetic (asserted where it matters).
function _poison_pair(n::Int)
    cnt = Ref(0)
    return Observable(:poison, 1,
                      v -> (cnt[] += 1) >= n ? error("poison interrupt") : 0.0),
           Observable(:poison, 1, v -> 0.0)
end

function _interrupted(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("poison", err.msg)
    return nothing
end

_mc_progress(path) = MC.jldopen(path, "r") do f
    (f["progress/temp_index"], f["progress/phase"], f["progress/sweep"])
end

@testset "checkpoint / resume" begin
    dir = mktempdir()
    H = TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))

    @testset "model_fingerprint facade" begin
        # the public facade IS the internal fingerprint (dependent-package tier)
        @test MC.model_fingerprint(H) === MC._fingerprint(H)
        @test MC.model_fingerprint(H) ===
              MC.model_fingerprint(TiledHamiltonian(_biquadratic_model(0);
                                                    dims = (2, 1, 1)))
        @test MC.model_fingerprint(H) !==
              MC.model_fingerprint(TiledHamiltonian(_biquadratic_model(0);
                                                    dims = (2, 2, 1)))
    end

    @testset "Xoshiro word round-trip" begin
        rng = Xoshiro(1234)
        rand(rng, 17)
        words = MC._rng_words(rng)
        @test length(words) == fieldcount(Xoshiro)
        rng2 = MC._rng_from_words(words)
        @test all(rand(rng, UInt64) == rand(rng2, UInt64) for _ = 1:100)
        @test_throws ErrorException MC._rng_from_words(UInt64[1, 2])
    end

    @testset "MC: checkpointing does not perturb, resume is bit-identical" begin
        poison, benign = _poison_pair(300)   # temp-2 measurement 100 of 200
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3], sweeps_therm = 200, sweeps_measure = 400,
              measure_interval = 2, nbins = 8, renorm_interval = 100, seed = 42,
              observables = obs)
        path = joinpath(dir, "mc.jld2")
        a = run_mc(H; kw...)                                # no checkpointing
        b = run_mc(H; kw..., checkpoint = path, checkpoint_interval = 150)
        _assert_same_result(a, b)                           # writing consumes no RNG
        @test a.final_config == b.final_config
        # mid-run continuation, via the interrupted writer (see `_poison_pair`):
        # the poison run dies at temp-2 measure sweep 200, its file holds the
        # periodic write at measure sweep 100, and resume must complete THAT run
        # into `a`, bit for bit
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path, checkpoint_interval = 150))
        ti, phase, sweep = _mc_progress(path)
        @test ti == 2 && phase == "measure" && 0 < sweep < 400   # genuinely mid-run
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
        @test c isa MCResult
    end

    @testset "MC: resume from a thermalization-phase checkpoint" begin
        # the poison fires at temp-2's FIRST measurement, so the file's last
        # periodic write (global sweep 520 = temp-2 therm sweep 120) sits inside
        # thermalization — the phase this gate exists to resume from
        poison, benign = _poison_pair(101)
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3], sweeps_therm = 300, sweeps_measure = 100,
              measure_interval = 1, nbins = 8, seed = 7, observables = obs)
        path = joinpath(dir, "mc_therm.jld2")
        a = run_mc(H; kw...)
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path, checkpoint_interval = 260))
        ti, phase, sweep = _mc_progress(path)
        @test ti == 2 && phase == "therm" && 0 < sweep < 300
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "MC: boundary-only checkpoints (interval 0) and carryover=false" begin
        # poison at temp-3 measurement 50: with interval 0 the file's last write
        # is the end-of-temp-2 boundary, so resume replays temp 3 IN FULL —
        # thermalization, measure phase, and the carryover=false restart (whose
        # random redraw must come out of the restored RNG bit-identically)
        poison, benign = _poison_pair(250)
        obs = [standard_observables(H); benign]
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 100, sweeps_measure = 100,
              nbins = 4, carryover = false, seed = 3, observables = obs)
        path = joinpath(dir, "mc_boundary.jld2")
        a = run_mc(H; kw...)
        _interrupted(() -> run_mc(H; kw...,
                                  observables = [standard_observables(H); poison],
                                  checkpoint = path))       # interval 0
        ti, phase, sweep = _mc_progress(path)
        @test ti == 3 && phase == "therm" && sweep == 0   # end-of-temp-2 boundary
        c = resume(path, H; observables = obs)
        _assert_same_result(a, c)
        @test a.final_config == c.final_config
    end

    @testset "PT: resume is bit-identical" begin
        kw = (; kT = [0.5, 0.3, 0.2], sweeps_therm = 150, sweeps_measure = 300,
              exchange_interval = 7, nbins = 8, seed = 11)
        path = joinpath(dir, "pt.jld2")
        a = run_pt(H; kw...)
        b = run_pt(H; kw..., checkpoint = path, checkpoint_interval = 120)
        _assert_same_result(a, b)
        @test a.final_configs == b.final_configs
        @test a.swap_acceptance == b.swap_acceptance
        c = resume(path, H)
        @test c isa PTResult
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
        @test a.swap_acceptance == c.swap_acceptance
    end

    @testset "PT: resume from the phase-boundary checkpoint (interval 0)" begin
        kw = (; kT = [0.5, 0.2], sweeps_therm = 100, sweeps_measure = 200,
              exchange_interval = 9, nbins = 4, seed = 13)
        path = joinpath(dir, "pt_boundary.jld2")
        a = run_pt(H; kw...)
        run_pt(H; kw..., checkpoint = path)
        c = resume(path, H)
        _assert_same_result(a, c)
        @test a.final_configs == c.final_configs
    end

    @testset "resume of a completed run is idempotent" begin
        # a job-array retry loop may call resume on a checkpoint whose run already
        # finished — it must return the finished result unchanged (MC and PT)
        path = joinpath(dir, "done_mc.jld2")
        a = run_mc(H; kT = [0.5, 0.3], sweeps_therm = 100, sweeps_measure = 200,
                   nbins = 8, seed = 9, checkpoint = path,
                   checkpoint_interval = 50)
        b = resume(path, H)
        @test b.final_config == a.final_config
        @test all(b.points[i].stats[:energy].mean == a.points[i].stats[:energy].mean
                  for i in eachindex(a.points))
        pp = joinpath(dir, "done_pt.jld2")
        c = run_pt(H; kT = [0.5, 0.3, 0.2], sweeps_therm = 100,
                   sweeps_measure = 200, nbins = 8, seed = 9, checkpoint = pp,
                   checkpoint_interval = 50)
        d = resume(pp, H)
        @test d.final_configs == c.final_configs
        @test d.swap_acceptance == c.swap_acceptance
    end

    @testset "joint: displacement state survives a resume bit-exactly" begin
        # Schema v3's reason for existing. The Einstein oscillator is used rather than
        # the joint fixture because it is BOUNDED: an escaping chain would reach the
        # same (wrong) numbers on both paths and prove nothing about the format.
        terms, L = _einstein_terms(2.5)
        Hj = TiledHamiltonian(1, terms, L; dims = (2, 2, 1), fixed_reference = true)
        @test MC.has_disp(Hj)
        obs = [Observable(:energy, 1, v -> v.energy),
               Observable(:energy2, 1, v -> v.energy^2)]
        kw = (; kT = [0.4, 0.25], sweeps_therm = 300, sweeps_measure = 400,
              nbins = 8, renorm_interval = 50, step_u = 0.3, seed = 5,
              observables = obs, evaluables = Evaluable[])
        path = joinpath(dir, "joint.jld2")
        a = run_mc(Hj; kw...)
        b = run_mc(Hj; kw..., checkpoint = path, checkpoint_interval = 170)
        _assert_same_result(a, b)
        # the resume half runs on an INTERRUPTED writer (see `_poison_pair`): the
        # poison dies at temp-2 measure sweep 200, the file holds the periodic
        # write at measure sweep 190, and the displacement state must survive
        # the genuinely mid-run rebuild
        poison, benign = _poison_pair(600)
        obsb = [obs; benign]
        aP = run_mc(Hj; kw..., observables = obsb)
        _interrupted(() -> run_mc(Hj; kw..., observables = [obs; poison],
                                  checkpoint = path, checkpoint_interval = 170))
        ti, phase, sweep = _mc_progress(path)
        @test ti == 2 && phase == "measure" && 0 < sweep < 400
        c = resume(path, Hj; observables = obsb, evaluables = Evaluable[])
        _assert_same_result(aP, c)
        # the displacement STATE itself, not only the statistics computed from it:
        # `_assert_same_result` compares floats derived at renormalization points, and
        # a resume that rebuilt `disps` a hair differently would still pass those
        @test a.final_config == b.final_config == c.final_config
        @test a.final_disps == b.final_disps == c.final_disps
        @test length(a.final_disps) == Hj.n_sites && any(!iszero, a.final_disps)
        # the run really did sample displacements — otherwise the gate above is vacuous
        @test all(p -> 0.0 < p.acceptance_disp <= 1.0, a.points)
        @test all(p -> p.disp_rms > 0.0 && p.final_step_u > 0.0, a.points)
        @test !any(p -> p.escaped, a.points)
        # PT too: the payload swap moves displacements between lanes, so a resumed
        # ladder must land every lane on the frame its replica was pinned to
        pp = joinpath(dir, "joint_pt.jld2")
        d = run_pt(Hj; kw..., sweeps_therm = 200, sweeps_measure = 200,
                   exchange_interval = 20)
        run_pt(Hj; kw..., sweeps_therm = 200, sweeps_measure = 200,
               exchange_interval = 20, checkpoint = pp, checkpoint_interval = 90)
        e = resume(pp, Hj; observables = obs, evaluables = Evaluable[])
        _assert_same_result(d, e)
        @test d.final_configs == e.final_configs
        @test d.final_disps == e.final_disps
        @test d.final_disps[1] != d.final_disps[2]     # not a shared reference

        # the already-completed fast path (`resume` on a finished run) constructs the
        # result on its own code path, so it needs its own joint gate — a job-array
        # retry loop hits it every time the run finished before the retry did
        done = joinpath(dir, "joint_done.jld2")
        f = run_mc(Hj; kw..., checkpoint = done, checkpoint_interval = 50)
        g = resume(done, Hj; observables = obs, evaluables = Evaluable[])
        @test g.final_config == f.final_config
        @test g.final_disps == f.final_disps
        @test any(!iszero, g.final_disps)
    end

    @testset "schema and mismatch guards" begin
        path = joinpath(dir, "guard.jld2")
        run_mc(H; kT = 0.5, sweeps_therm = 50, sweeps_measure = 60, nbins = 4,
               seed = 1, checkpoint = path, checkpoint_interval = 40)
        # fingerprint mismatch: different dims
        H2 = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        @test_throws ErrorException resume(path, H2)
        # observable mismatch
        @test_throws ErrorException resume(path, H; observables = [
            Observable(:energy, 1, v -> v.energy)])
        # missing file
        @test_throws ArgumentError resume(joinpath(dir, "nope.jld2"), H)
        # negative interval guard
        @test_throws ArgumentError run_mc(H; kT = 0.5, checkpoint = path,
                                          checkpoint_interval = -1)
        # a corrupted stored configuration is refused by the non-projecting
        # door (validated without projecting — restore must stay bit-exact)
        cpath = joinpath(dir, "corrupt.jld2")
        cp(path, cpath)
        MC.jldopen(cpath, "r+") do f
            m = f["chain/config"]
            delete!(f, "chain/config")
            m[:, 1] .*= 2.5
            f["chain/config"] = m
        end
        @test_throws ArgumentError resume(cpath, H)
    end
end

# Every field of `ChainState` survives the file, checked exhaustively rather than by
# sample. The escape detector's block ladder was the gap: `disp_blk_sum`, `disp_blk_n`,
# `disp_blk_cap`, `escape_strikes`, `disp_ref_ms` and `escape_warned` could all be
# restored as zero and the whole 1091-assertion checkpoint suite stayed green, because
# nothing reads them back — yet the source comment names the consequence (a chain
# checkpointed often enough never accumulates the consecutive strikes that report an
# escape, and `escape_warned` reset would turn an already-reported escape into a clean
# verdict). The other five escape floats were covered; these six were not.
#
# Exhaustive over `fieldnames`, so a field added later has to round-trip or fail here.
@testset "the whole chain state round-trips, field by field" begin
    H = TiledHamiltonian(first(_joint_model()); dims = (2, 2, 1))
    st = MC.ChainState(H, _rand_config(MersenneTwister(77), H), Xoshiro(77), 0.35;
                       disps = [SVector{3,Float64}(0.01i, -0.02i, 0.03i)
                                for i = 1:H.n_sites], step_u = 0.04)
    # distinctive, non-default values everywhere — a zero-restore must not coincide
    st.energy = -12.5
    st.strain, st.strain_min, st.strain_max = 1.03, 0.97, 1.05
    st.frozen = true
    st.acc_metro, st.att_metro = 11, 23
    st.acc_or, st.att_or = 5, 9
    st.acc_disp, st.att_disp = 7, 13
    st.acc_strain, st.att_strain, st.att_strain_out = 3, 17, 4
    st.max_drift = 1.5e-12
    st.com_removed[1] = SVector(0.11, -0.22, 0.33)
    st.disp_rms, st.disp_max, st.disp_rms0 = 0.021, 0.052, 0.019
    st.disp_checks = 29
    st.disp_ms_sum, st.disp_blk_sum, st.disp_ref_ms = 1.3e-3, 4.1e-4, 3.7e-4
    st.disp_blk_n, st.disp_blk_cap = 6, 8
    st.escape_strikes, st.escape_warned = 2, true

    dir2 = mktempdir()
    path = joinpath(dir2, "chain.jld2")
    MC.jldopen(path, "w") do f
        MC._write_chain(f, "chain", st)
    end
    st2 = MC.jldopen(path, "r") do f
        MC._read_chain(f, "chain", H)
    end

    rngwords(r) = MC._rng_words(r)
    bad = Symbol[]
    for f in fieldnames(MC.ChainState)
        a, b = getfield(st, f), getfield(st2, f)
        same = f === :rng ? rngwords(a) == rngwords(b) :
               f === :site_rngs ? all(rngwords.(a) .== rngwords.(b)) :
               a == b
        same || push!(bad, f)
    end
    @test bad == Symbol[]
    # the loop above is only meaningful if the values were non-default to begin with
    @test st.escape_warned && st.escape_strikes != 0 && st.disp_blk_n != 0
    @test st.disp_blk_cap != 1 && st.disp_ref_ms != 0.0 && st.disp_blk_sum != 0.0
    @test st.att_strain_out != 0 && st.strain_min != 1.0
end
