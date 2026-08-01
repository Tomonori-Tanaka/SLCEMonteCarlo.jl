# Energy-contract kernel microbenchmarks — the sweep inner loop, decomposed.
#
#   julia --project=bench bench/bench_kernels.jl [n_bcc] [n_2141]
#
# Per-call cost of each stage a single-spin attempt is built from: the tesseral
# tabulation (`_zlm_row!`), the leave-one-out accumulation (`site_coeffs!` — the
# expected dominant cost, ∝ site adjacency × nnz(folded)), the ΔE dot
# (`delta_energy`), plus the re-anchoring `_total_energy` and the diagnostics-path
# `site_gradient` (which rebuilds zrows — not on the sweep path). Nonzero allocs in
# the in-place kernels are an optimization red flag.
#
# `n_bcc` / `n_2141` are the cubic tiling sizes (defaults 8 → 1024 sites, 2 → 544).

using SLCEMonteCarlo
using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))

n_bcc  = argn(1, 8)
n_2141 = argn(2, 2)

bench_header("energy kernels — bcc Fe $(n_bcc)³ / Nd2Fe14B $(n_2141)³")

# One full leave-one-out pass over every active site (what a sweep does, minus
# proposals — inactive sites are skipped there too).
function all_site_coeffs!(c, H, zrows)
    for s = 1:n_sites(H)
        H.site_active[s] || continue
        fill!(c, 0.0)
        site_coeffs!(c, H, s, zrows)
    end
    return c
end

function kernel_report(name, H)
    println()
    println("--- $name: ", describe(H))
    config   = rand_config(H)
    zrows = MC._zrows(H, config)
    c     = zeros(H.nlm)
    znew  = zeros(H.nlm)
    e     = config[1]
    # The busiest site (a non-magnetic sublattice — B in Nd2Fe14B — has adjacency 0,
    # so a "middle site" sample can be meaningless; the mean below is the sweep-
    # relevant number).
    smax = argmax(s -> H.site_ptr[s + 1] - H.site_ptr[s], 1:n_sites(H))
    adj  = H.site_ptr[smax + 1] - H.site_ptr[smax]

    plm = Vector{Float64}(undef, H.lmax + 1)
    t_z = @belapsed MC._zlm_row!($znew, $e, $(H.lmax), $plm)
    a_z = @allocations MC._zlm_row!(znew, e, H.lmax, plm)
    @printf("%-28s  %10.1f ns   allocs=%d\n", "_zlm_row!", 1e9 * t_z, a_z)

    t_cm = @belapsed (fill!($c, 0.0); site_coeffs!($c, $H, $smax, $zrows))
    a_c = @allocations site_coeffs!(c, H, smax, zrows)
    @printf("%-28s  %10.1f ns   allocs=%d\n",
            "site_coeffs! (adjacency $adj)", 1e9 * t_cm, a_c)

    t_ca = @belapsed all_site_coeffs!($c, $H, $zrows)
    t_c = t_ca / H.n_active
    @printf("%-28s  %10.1f ns\n", "site_coeffs! (mean/active site)", 1e9 * t_c)

    zold = view(zrows, :, smax)
    t_d = @belapsed delta_energy($c, $zold, $znew)
    a_d = @allocations delta_energy(c, zold, znew)
    @printf("%-28s  %10.1f ns   allocs=%d\n", "delta_energy", 1e9 * t_d, a_d)

    attempt = t_z + t_c + t_d
    @printf("%-28s  %10.1f ns   (sweep lower bound %.2f ms)\n",
            "Σ per-attempt kernels (mean)", 1e9 * attempt, 1e3 * attempt * H.n_active)

    t_E = @belapsed MC._total_energy($H, $zrows)
    a_E = @allocations MC._total_energy(H, zrows)
    @printf("%-28s  %10.3f µs   allocs=%d\n", "_total_energy", 1e6 * t_E, a_E)

    t_zr = @belapsed MC._zrows($H, $config)
    @printf("%-28s  %10.3f µs\n", "_zrows (full rebuild)", 1e6 * t_zr)

    t_g = @belapsed site_gradient($H, $smax, $config)
    @printf("%-28s  %10.3f µs   (diagnostic path — rebuilds zrows)\n",
            "site_gradient", 1e6 * t_g)
    return nothing
end

kernel_report("bcc Fe (light kernel)", TiledHamiltonian(bcc_fe_model();
                                                        dims = (n_bcc, n_bcc, n_bcc)))
kernel_report("Nd2Fe14B (heavy kernel)", TiledHamiltonian(nd2fe14b_model();
                                                          dims = (n_2141, n_2141, n_2141)))
kernel_report("Nd2Fe14B nbody=3 (triplet kernel)",
              TiledHamiltonian(nd2fe14b3_model(); dims = (n_2141, n_2141, n_2141)))
