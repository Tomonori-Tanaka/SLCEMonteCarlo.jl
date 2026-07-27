# SLCEMonteCarlo.jl benchmarks

Performance benchmarks for the MC hot paths, built to **localize bottlenecks**, not
just to time end-to-end runs: the sweep cost is decomposed into its kernels, every
in-place path reports its allocation count (nonzero = red flag), and a profiler
script gives line-level attribution. Each script is standalone and runs in the
`bench/` environment (its own `Project.toml`/`Manifest.toml`, which develops this
package and `../SLCE.jl`).

## Setup (once)

```bash
julia --project=bench -e 'using Pkg; \
    Pkg.develop([PackageSpec(path="."), PackageSpec(path="../SLCE.jl")]); \
    Pkg.add(["BenchmarkTools", "Spglib", "StaticArrays", "Statistics", "Printf", \
             "Random", "LinearAlgebra", "Profile"]); Pkg.instantiate()'
```

## Run

```bash
julia --project=bench bench/bench_kernels.jl  [n_bcc] [n_2141]           # attempt kernels, per call
julia --project=bench bench/bench_sweeps.jl   [nsweeps] [n_bcc] [n_2141] # ns/attempt + allocs/sweep
julia --project=bench bench/bench_tiling.jl   [n_bcc] [n_2141]           # TiledHamiltonian ctor + memory
julia --project=bench bench/bench_run.jl      [sweeps] [n_bcc] [n_2141]  # run_mc + run_pt end-to-end
julia -t 8 --project=bench bench/bench_run.jl                            # ... with PT thread scaling
julia --project=bench bench/bench_minimize.jl [n_bcc] [n_2141] [nstarts] # gradient + ground state
julia --project=bench bench/bench_profile.jl  [target] [fixture] [secs]  # line-level hotspots
```

### GPU (`bench/gpu/`, its own environment — it carries CUDA)

```bash
julia --project=bench/gpu bench/bench_gpu.jl [n_2141_max] [nsweeps]  # the G6 go/no-go sweep
julia --project=bench/gpu bench/gpu/smoke_joint.jl                   # the JOINT device smoke
```

`bench_gpu.jl` runs pure-spin fixtures and falls back to the KA-CPU backend when no
GPU is present (a correctness smoke — its throughput is not a GPU number).
`smoke_joint.jl` covers what that cannot: it launches the displacement kernel, and it
**refuses** to run without a functional CUDA device, because the KA-CPU test suite is
structurally blind to the one bug class this exists to catch (`@index` returns `Int32`
on CUDA and `Int` on the CPU backend — see the G7 field note). **Run it on a device
after any kernel-adjacent change.** PBS jobs for kugui are alongside:
`job_i1accs.pbs` (30 min debug queue), `job_joint_smoke.pbs` (the joint smoke),
`job_f1accs.pbs` (the production size sweep).

## How to find a bottleneck

1. `bench_sweeps.jl` — is the ns/attempt near the kernel lower bound printed by
   `bench_kernels.jl`? A large gap = proposal/RNG/copy bookkeeping, not the energy
   kernels. Nonzero allocs/sweep = something in the hot loop allocates.
2. `bench_kernels.jl` — which stage dominates an attempt: the tesseral row
   (`_zlm_row!`), the leave-one-out accumulation (`site_coeffs!` — expected, it
   scales with site adjacency × nnz of the folded tensors), or the ΔE dot?
3. `bench_profile.jl sweep 2141` — line-level attribution inside the winner
   (call tree + flat). Other targets: `or`, `total_energy`, `gradient`, `minimize`.
4. `bench_run.jl` — the measure_interval 1 vs 10 gap is the observables/binning
   overhead; the run_pt ntasks 1 vs N ratio is pure thread scaling (results are
   bit-identical by design, so wall time is the only difference).

## Fixtures (`fixtures.jl`)

Two synthetic-coefficient `SLCEModel` models span the kernel-cost regimes:

- **`bcc_fe_model()`** — 2-atom bcc Fe training cell, isotropic `l = 1` pair basis
  (nlm = 4, 16 shift-carrying directed pair terms, site adjacency 16): the
  light-kernel / large-lattice regime, where throughput is bookkeeping-bound.
  Tiled 8³ → 1024 sites by default.
- **`nd2fe14b_model()`** — the 68-atom Nd₂Fe₁₄B cell (`assets/nd2fe14b.toml`,
  same structure asset as SLCE's bench), 9 sublattice species (B
  non-magnetic, `lmax = 0`), `l ≤ 2`, every resolvable pair → ~9400 terms and site
  adjacency ~276 (cf. the real l02 production model, 4692 terms): the heavy-kernel
  regime where `site_coeffs!` dominates. Tiled 2³ → 544 sites by default
  (4³ = 4352 sites is the manual-smoke size).

Coefficients are seeded random (timing is value-independent); `BENCH_KT = 0.025 eV`
gives realistically mixed acceptance so the accepted-move bookkeeping is included.
Helpers: `rand_config`, `chain_state` (frozen step), `describe`, `bench_one` /
`bench_header` / `argn` / `argf` (as in SLCE's bench).

## Recording results

Append a before/after entry to [`../.claude/bench_log.md`](../.claude/bench_log.md)
when you touch a hot path (`energy.jl`, `updates.jl`, `hamiltonian.jl` ctor,
`minimize.jl`, the observables/binning measurement path). Note machine, Julia
version, thread count, and the fixture/dims.
