# Ground-state search — the all-site gradient pass (`_gradient!`, one pass ≈ one
# Metropolis sweep by design), the BB descent (`minimize_energy`), and the
# multi-start anneal + polish (`find_ground_state`, threaded).
#
#   julia --project=bench bench/bench_minimize.jl [n_bcc] [n_2141] [nstarts] [anneal]
#
# `find_ground_state` parallelizes over starts — run with `julia -t N` for the
# threaded number. The bench trims the search (nstarts = 4, anneal_sweeps = 50);
# the interesting per-kernel figures are the `_gradient!` pass and ms/iter.

using SLCEMonteCarlo
using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))

n_bcc   = argn(1, 8)
n_2141  = argn(2, 2)
nstarts = argn(3, 4)
anneal  = argn(4, 50)

bench_header("ground-state search — bcc Fe $(n_bcc)³ / Nd2Fe14B $(n_2141)³, " *
             "nstarts=$nstarts, anneal_sweeps=$anneal")

function minimize_report(name, H)
    println()
    println("--- $name: ", describe(H))
    config   = rand_config(H)
    ms    = MC._MinimizeScratch(H)
    t_g = @belapsed MC._gradient!($(ms.G), $H, $config, $(ms.zrows), $(ms.c), $(ms.plm))
    @printf("%-28s  %10.3f ms/pass   %8.1f ns/site\n",
            "_gradient! (all sites)", 1e3 * t_g, 1e9 * t_g / n_sites(H))

    minimize_energy(H; init = config, maxiter = 5)       # warm-up / compile
    t = @elapsed r = minimize_energy(H; init = config)
    @printf("%-28s  %10.3f s   iters=%-5d  %8.3f ms/iter   converged=%s\n",
            "minimize_energy", t, r.iterations, 1e3 * t / max(r.iterations, 1),
            r.converged)

    t = @elapsed g = find_ground_state(H; nstarts = nstarts, anneal_sweeps = anneal,
                                       seed = 7)
    @printf("%-28s  %10.3f s   E=%.6g   converged %d/%d\n",
            "find_ground_state", t, g.energy, count(g.converged_starts), nstarts)
    return nothing
end

minimize_report("bcc Fe (light kernel)",
                TiledHamiltonian(bcc_fe_model(); dims = (n_bcc, n_bcc, n_bcc)))
minimize_report("Nd2Fe14B (heavy kernel)",
                TiledHamiltonian(nd2fe14b_model(); dims = (n_2141, n_2141, n_2141)))
