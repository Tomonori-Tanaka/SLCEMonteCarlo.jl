# Sweep throughput — the MC hot path (`metropolis_sweep!` / `overrelaxation_sweep!`).
#
#   julia --project=bench bench/bench_sweeps.jl [nsweeps] [n_bcc] [n_2141]
#
# Reports ms/sweep, **ns per attempt** (the size-independent figure of merit — compare
# against the kernel lower bound from bench_kernels.jl; the gap is proposal/RNG/copy
# bookkeeping), and **allocs per sweep** (nonzero = an optimization red flag: the
# sweep kernels are designed to be allocation-free). Frozen step, fixed seeds,
# kT = BENCH_KT (0.025 eV) so accepted-move bookkeeping is realistically mixed in.

using SLCEMonteCarlo
using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))

nsweeps = argn(1, 50)
n_bcc   = argn(2, 8)
n_2141  = argn(3, 2)

bench_header("sweeps — bcc Fe $(n_bcc)³ / Nd2Fe14B $(n_2141)³, $nsweeps sweeps, " *
             "kT=$(BENCH_KT) eV")

function sweep_report(name, H, nsweeps)
    println()
    println("--- $name: ", describe(H))
    β = 1 / BENCH_KT
    for (label, sweep!) in (("metropolis_sweep!", metropolis_sweep!),
                            ("overrelaxation_sweep!", overrelaxation_sweep!))
        st, sc = chain_state(H)
        sweep!(st, H, β, sc)                          # warm-up / compile
        allocs = @allocations sweep!(st, H, β, sc)
        t = @elapsed for _ = 1:nsweeps
            sweep!(st, H, β, sc)
        end
        acc = label == "metropolis_sweep!" ? st.acc_metro / max(st.att_metro, 1) :
              st.acc_or / max(st.att_or, 1)
        @printf("%-24s  %9.3f ms/sweep   %8.1f ns/attempt   allocs/sweep=%-6d acc=%.2f\n",
                label, 1e3 * t / nsweeps, 1e9 * t / (nsweeps * H.n_active), allocs, acc)
    end
    # color-parallel execution (bit-identical to serial; needs julia -t ≥ ntasks)
    for ntasks in (2, 4, Threads.nthreads())
        ntasks <= Threads.nthreads() || continue
        st, _ = chain_state(H)
        scs = [SLCEMonteCarlo.SweepScratch(H) for _ = 1:ntasks]
        metropolis_sweep!(st, H, β, scs)              # warm-up / compile
        t = @elapsed for _ = 1:nsweeps
            metropolis_sweep!(st, H, β, scs)
        end
        @printf("%-24s  %9.3f ms/sweep   %8.1f ns/attempt   (%d colors)\n",
                "metropolis, $(ntasks) tasks", 1e3 * t / nsweeps,
                1e9 * t / (nsweeps * H.n_active), H.n_colors)
    end
    return nothing
end

sweep_report("bcc Fe (light kernel)",
             TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc)), nsweeps)
sweep_report("Nd2Fe14B (heavy kernel)",
             TiledHamiltonian(nd2fe14b_model(); dims = (n_2141, n_2141, n_2141)),
             nsweeps)
sweep_report("Nd2Fe14B nbody=3 (triplet kernel)",
             TiledHamiltonian(nd2fe14b3_model(); dims = (n_2141, n_2141, n_2141)),
             nsweeps)
