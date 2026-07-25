# `TiledHamiltonian` construction — supercell unfolding + CSR adjacency build —
# and the index memory it retains.
#
#   julia --project=bench bench/bench_tiling.jl [n_bcc] [n_2141]
#
# Defaults: bcc Fe 16³ (8192 sites) and Nd2Fe14B 4³ (4352 sites — the manual-smoke
# size; the real l02 model built in 0.01 s / 7.8 MB). Construction should stay
# integer-bookkeeping cheap; memory is the CSR arrays + one folded tensor per term.

using SLCEMonteCarlo
using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))

n_bcc  = argn(1, 16)
n_2141 = argn(2, 4)

bench_header("TiledHamiltonian construction — bcc Fe $(n_bcc)³ / Nd2Fe14B $(n_2141)³")

function tiling_report(name, model, dims)
    println()
    println("--- $name, dims = $dims")
    bench_one("TiledHamiltonian", () -> TiledHamiltonian(model; dims = dims))
    H = TiledHamiltonian(model; dims = dims)
    println("    ", describe(H))
    @printf("    retained size: %.2f MiB\n", Base.summarysize(H) / 2^20)
    return nothing
end

tiling_report("bcc Fe", bcc_fe_model(), (n_bcc, n_bcc, n_bcc))
tiling_report("Nd2Fe14B", nd2fe14b_model(), (n_2141, n_2141, n_2141))
