# Statistical profile of a hot path — line-level bottleneck identification.
#
#   julia --project=bench bench/bench_profile.jl [target] [fixture] [seconds]
#
# `target`  ∈ sweep (default) | or | total_energy | gradient | minimize
# `fixture` ∈ 2141 (default — the heavy-kernel regime, where optimization pays) | bcc
# `seconds` ≈ sampling budget (default 5; the workload loops until it is exceeded).
#
# Prints the profile twice: as a call tree (where the time sits structurally) and
# flat by line count (the top individual lines). Julia-only frames (C = false).

using SLCEMonteCarlo
using SLCE
using Profile
include(joinpath(@__DIR__, "fixtures.jl"))

target  = length(ARGS) >= 1 ? ARGS[1] : "sweep"
fixture = length(ARGS) >= 2 ? ARGS[2] : "2141"
budget  = argf(3, 5.0)

H = fixture == "bcc" ? TiledHamiltonian(bcc_fe_model(); dims = (8, 8, 8)) :
    TiledHamiltonian(nd2fe14b_model(); dims = (2, 2, 2))

bench_header("profile — target=$target, fixture=$fixture (~$(budget) s)")
println(describe(H))

st, sc = chain_state(H)
β = 1 / BENCH_KT
cfg = rand_config(H)
ms = MC._MinimizeScratch(H)

# One unit of the chosen workload (compiled before profiling starts).
work = if target == "sweep"
    () -> metropolis_sweep!(st, H, β, sc)
elseif target == "or"
    () -> overrelaxation_sweep!(st, H, β, sc)
elseif target == "total_energy"
    () -> MC._total_energy(H, st.zrows)
elseif target == "gradient"
    () -> MC._gradient!(ms.G, H, cfg, ms.zrows, ms.c, ms.plm)
elseif target == "minimize"
    () -> minimize_energy(H; init = cfg, maxiter = 50)
else
    error("unknown target $(repr(target)) (sweep | or | total_energy | gradient " *
          "| minimize)")
end

work()                                              # warm-up / compile
t1 = @elapsed work()
reps = max(1, round(Int, budget / max(t1, 1e-9)))
println("one unit ≈ $(round(1e3 * t1, digits = 3)) ms → $reps reps")

Profile.clear()
@profile for _ = 1:reps
    work()
end

println("\n================ call tree (mincount ≥ 1% of samples) ================")
n_samples = length(Profile.fetch())
Profile.print(; format = :tree, C = false, maxdepth = 18,
              mincount = max(5, n_samples ÷ 100), noisefloor = 2.0)

println("\n================ flat, top lines by sample count ======================")
Profile.print(; format = :flat, C = false, sortedby = :count,
              mincount = max(5, n_samples ÷ 50))
