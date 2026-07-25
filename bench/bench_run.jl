# End-to-end runs — what a user actually waits on: `run_mc` (with the observable /
# binning machinery) and `run_pt` (thread scaling).
#
#   julia --project=bench bench/bench_run.jl [sweeps] [n_bcc] [n_2141]
#
# run_mc: reports sweeps/s and isolates the measurement overhead by comparing
# measure_interval = 1 against 10 (the gap is observables + binning, not the sweep
# kernels). run_pt: an 8-rung ladder on the bcc lattice — compare wall time against
# ntasks = 1 (the ladder is bit-identical regardless, so the speedup is pure thread
# scaling; run with `julia -t N` to give it threads).

using SLCEMonteCarlo
using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))

sweeps = argn(1, 500)
n_bcc  = argn(2, 8)
n_2141 = argn(3, 2)

bench_header("end-to-end runs — bcc Fe $(n_bcc)³ / Nd2Fe14B $(n_2141)³, " *
             "$sweeps measure sweeps")

# Warm up compile on a tiny run, then time a single full run (a full bench_one-style
# multi-trial pass would multiply minutes-scale runs).
function timed_run(label, f, tiny, total_sweeps, nsites)
    tiny()
    t = @elapsed f()
    @printf("%-36s  %8.2f s   %8.1f sweeps/s   %8.1f ns/attempt\n",
            label, t, total_sweeps / t, 1e9 * t / (total_sweeps * nsites))
    return t
end

therm = sweeps ÷ 5

function mc_report(name, H)
    println()
    println("--- $name: ", describe(H))
    for interval in (1, 10)
        timed_run("run_mc (measure_interval=$interval)",
                  () -> run_mc(H; kT = BENCH_KT, sweeps_therm = therm,
                               sweeps_measure = sweeps, or_per_metropolis = 1,
                               measure_interval = interval, nbins = 8, seed = 7),
                  () -> run_mc(H; kT = BENCH_KT, sweeps_therm = 2,
                               sweeps_measure = 4 * interval,
                               measure_interval = interval, nbins = 2, seed = 7),
                  # or_per_metropolis = 1 ⇒ 2 lattice sweeps per compound sweep
                  2 * (therm + sweeps), H.n_active)
    end
    return nothing
end

mc_report("bcc Fe (light kernel)",
          TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc)))
mc_report("Nd2Fe14B (heavy kernel)",
          TiledHamiltonian(nd2fe14b_model(); dims = (n_2141, n_2141, n_2141)))

# --- parallel tempering: thread scaling on the bcc lattice ---------------------
println()
H = TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc))
rungs = 8
kts = [BENCH_KT * (2.0^(r / (rungs - 1))) for r = (rungs - 1):-1:0]  # 2×kT → kT
println("--- run_pt: $rungs rungs on bcc Fe $(n_bcc)³ (bit-identical for any ntasks)")
for nt in unique((1, Threads.nthreads()))
    timed_run("run_pt (ntasks=$nt)",
              () -> run_pt(H; kT = kts, ntasks = nt, sweeps_therm = therm,
                           sweeps_measure = sweeps, exchange_interval = 10,
                           nbins = 8, seed = 7),
              () -> run_pt(H; kT = kts, ntasks = nt, sweeps_therm = 2,
                           sweeps_measure = 4, exchange_interval = 2, nbins = 2,
                           seed = 7),
              rungs * (therm + sweeps), H.n_active)
end
