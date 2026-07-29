# Parallelism: what runs where

```@meta
CurrentModule = SLCEMonteCarlo
```

This page states exactly how the package is parallelized, what it deliberately
does **not** do, and the practical recipes for scaling beyond one machine.

## How it is implemented

The primary parallel construct in the package is the replica-exchange lane pool
of [`run_pt`](@ref): one *lane* per ladder rung, swept concurrently with
`Threads.@spawn` (shared memory — start Julia with `julia -t N`; the other
construct is the in-sweep `sweep_tasks` below). Each lane exclusively owns its
chain state, scratch buffers, RNG, adaptive step, and measurement accumulators,
so lanes never contend; an exchange synchronizes **only the two lanes of each
attempted pair** (the ladder globally re-syncs just at checkpoint writes and
phase ends) and swaps only the chain *payload* — the physical state: spins,
displacements and their re-centring record, the running energy, and on an NPT run
the cell scale together with the lane's coefficient clone (all reference swaps, no
copy). `ntasks = 1` selects the serial reference schedule;
any `ntasks ≥ 2` (the default with threads available) runs every lane as its own
task.

Two properties worth relying on:

- **Bit-determinism is thread-count-independent.** For a fixed seed, `run_pt`
  returns bit-identical results whether it runs on 1 thread or 32 (gated in the
  test suite; RNG discipline in `docs/specs/pt-threads-determinism.md`). The
  promise is scoped to one package + Julia version — it is a testing discipline,
  not a cross-version guarantee (spec P6).
- **Scaling is near-linear in rungs** up to the physical core count, because the
  only synchronization is the pairwise exchange handshake every
  `exchange_interval` sweeps — a slow lane stalls its neighbors, not the ladder.

## In-sweep parallelism: `sweep_tasks`

A single chain's sweeps can also run in parallel: `run_mc`, `run_pt`, and
`find_ground_state` take `sweep_tasks` — the number of concurrent tasks executing
each lattice sweep. The Hamiltonian's sites are greedily colored so that no two
sites of one color share a cluster instance; a color class is then updated
concurrently (the single-site kernels of a class are exactly independent), with a
barrier between classes. Because every site owns its RNG stream and the accepted
ΔE are reduced in a fixed order, the result is **bit-identical for any
`sweep_tasks`** — parallelism is an execution detail, not a different chain
(spec: `updates-stationarity.md` U1; gate: `test/unit/test_parallel.jl`).

When to use which:

- **PT with at least as many rungs as cores**: leave `sweep_tasks = 1` — the lane
  pool already saturates the machine.
- **A single temperature, a short ladder, or ground-state annealing on a big
  supercell**: `sweep_tasks = <P-core count>` parallelizes the sweep itself
  (measured ~3× on 4 performance cores for ≳4000-site models; small models
  amortize the per-class barrier less well).
- **Mixing both** (`ntasks · sweep_tasks`): keep the product within the thread
  count. Prefer performance cores — the class barrier synchronizes to the slowest
  core, so efficiency-core stragglers hurt more than they help.

A `run_mc` temperature **collection** still runs sequentially because that is its
semantics: each temperature warm-starts from the previous one (annealing).

There is **no MPI support**, and `run_mc`/`run_pt` are CPU-only drivers. A
chain-level device Metropolis sweep exists and is exported — see the
[GPU guide](gpu.md).

## What this cannot do

**A single PT ladder cannot span more than one node.** Lanes communicate through
shared memory, so one `run_pt` call is bounded by the cores (and memory) of one
machine. Since the rung count must grow like `√(n_sites·C)` for a fixed
temperature span, a very large supercell can genuinely need more rungs than one
node has cores — that regime is out of scope for v0. The planned route is a thin
distributed layer *on top of* the deterministic single-node core (an MPI driver
holding a block of adjacent rungs per rank, or a Carlo.jl adapter as a package
extension) — not an MPI rewrite of the core.

Until then: rungs beyond the core count still *work* (tasks queue over the
available threads); they just stop adding wall-clock speedup.

## Recipe: temperatures in parallel within one process

Independent temperatures (no annealing warm start wanted) parallelize with plain
`Threads.@threads` — `TiledHamiltonian` is immutable and safely shared; all
mutable state is created inside each `run_mc` call:

```julia
kts = collect(range(0.5, 0.05; length = 16))
results = Vector{MCResult}(undef, length(kts))
Threads.@threads for i in eachindex(kts)
    results[i] = run_mc(H; kT = kts[i], seed = UInt64(1000 + i))
end
```

Three rules:

1. **Distinct seeds per task** (or rely on the random default) — never share one.
2. This trades away the annealing warm start: every temperature equilibrates
   from a random configuration. Fine at moderate `T`; near freezing, prefer a
   serial annealing ladder or [`run_pt`](@ref).
3. If checkpointing, give every task its **own path** (one writer per file).

## Recipe: job arrays across nodes

Sweeps over sizes, models, seeds, or temperature blocks are embarrassingly
parallel — use the scheduler's job array, one independent Julia process per
element. No MPI needed. A SLURM sketch:

```bash
#!/bin/bash
#SBATCH --array=1-16
#SBATCH --cpus-per-task=8        # threads for run_pt lanes
#SBATCH --time=24:00:00
julia -t $SLURM_CPUS_PER_TASK --project driver.jl $SLURM_ARRAY_TASK_ID
```

with a `driver.jl` along the lines of

```julia
using SLCEMonteCarlo, SLCE, JLD2

task = parse(Int, ARGS[1])
model = SLCE.load(SLCEModel, "model.toml")

sizes = [(4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10)]
seeds = 1:4
L, s = sizes[cld(task, 4)], seeds[mod1(task, 4)]     # task → (size, seed)

H = TiledHamiltonian(model; dims = L)
r = run_pt(H; temperature = range(200, 1400; length = 24),
           sweeps_measure = 10^6, seed = UInt64(s),
           checkpoint = "pt_L$(L[1])_s$(s).jld2", checkpoint_interval = 50_000)

# persist the numbers you need as plain data (post-process on the login node)
jldsave("result_L$(L[1])_s$(s).jld2";
        kts = [p.kT for p in r.points],
        energy = [p.stats[:energy].mean[1] for p in r.points],
        energy_err = [p.stats[:energy].err[1] for p in r.points],
        binder = [p.stats[:binder].mean[1] for p in r.points],
        seed = r.seed)
```

Because every run checkpoints to its own file, a walltime kill costs nothing:
resubmit the same array element and have the driver call
[`resume`](@ref)`("pt_….jld2", H)` when the checkpoint exists — the restart is
bit-identical to an uninterrupted run.

```julia
path = "pt_L$(L[1])_s$(s).jld2"
r = isfile(path) ? resume(path, H) :
    run_pt(H; temperature = range(200, 1400; length = 24),
           sweeps_measure = 10^6, seed = UInt64(s),
           checkpoint = path, checkpoint_interval = 50_000)
```

## Choosing the layer

| Need | Tool |
|---|---|
| One system, one ladder through `T_c` | `run_pt` on one node (`julia -t N`) |
| Independent temperatures, no annealing | `Threads.@threads` over `run_mc` calls |
| Sizes × seeds × models | job array of independent processes |
| One ladder larger than a node | out of scope in v0 (planned: MPI/Carlo layer) |
