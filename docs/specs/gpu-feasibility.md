# Decision record — GPU port feasibility (assessment)

Status: assessment (2026-07-15). The Phase-1 prototype it staged has since
landed — see `gpu-prototype.md` for the implementation record. This record
captures the survey, the measured big-cell baseline, and the agreed strategy so a
future implementation starts from decisions, not from scratch. Companion records:
`hamiltonian-tiling.md` (the tables a kernel would consume), `updates-stationarity.md`
U1 (the coloring argument), `pt-threads-determinism.md` P3/P6 (the determinism
discipline a port must re-establish).

## F1 — verdict and scope

A CUDA port is viable and structurally well matched: the color-parallel sweep is
already the checkerboard decomposition GPUs need (generalized to arbitrary cluster
topology), per-site RNG streams and the fixed-order `_reduce_dE` reduction are the
right determinism substrate, and chain state is compact (`config` + `zrows` +
scalars). The payoff exists only for **large supercells** (the Nd₂Fe₁₄B-class
`nbody = 3` production regime) or **many-replica ensembles**; small pair-dominated
cells (sub-ms sweeps on CPU) would not beat the tuned CPU fast paths. There is no
published GPU Metropolis for cluster-expansion Hamiltonians (CASM, icet, smol, CLAMM
are all CPU-only) — no prior art to copy, none proving it a bad idea. The nearest
structural analogs are GPU Heisenberg spin-glass codes (Metropolis/heat-bath +
over-relaxation + PT all on-device, arXiv:1204.6192) and MD force kernels
(gather-sparse-terms-then-reduce).

## F2 — measured baseline (Apple M4, julia 1.12.6, kT = 0.025 eV)

`nd2fe14b3_model()` (nbody = 3, cutoff 3.5 Å, 16792 terms, mean adjacency 762,
98.1 % body-3 entries — mirrors production l044/l064/l066):

| dims | sites (active) | colors | tables | serial | 4 tasks |
|---|---|---|---|---|---|
| 4³ | 4352 (4096) | 9 | 99 MiB | 79.5 ms/sweep | 22.3 ms/sweep |
| 8³ | 34816 (32768) | 9 | 594 MiB | 631.6 ms/sweep | 167.3 ms/sweep |

~19.3 µs per attempt serial, task scaling 3.6–3.8× at 4 tasks. Three consequences:

- **9 colors** (the cutoff keeps conflict degree low; the 34–37-color figure in
  bench_log #4 is the *all-pairs* nbody = 2 fixture) → ~3600 active sites per color
  at 8³, and only 9 kernel launches per sweep — launch overhead is negligible.
- `zrows` at 8³ is 9 × 34816 × 8 B ≈ 2.5 MB — **fits entirely in an A100's 40 MB
  L2**, so the gather-heavy neighbor reads become cache hits; the kernel is
  compute/stream-bound, not scattered-DRAM-bound. Tables scale linearly with cells
  (16³ ≈ 4.7 GiB — fits A100 HBM).
- A standard campaign (2000 + 10000 sweeps) at 8³ is ~33 min per temperature point
  at 4 tasks; a 20-rung PT ladder is days on a laptop. A 5–30× GPU win converts
  days to hours — that is the actual motivation, not headline speedups.

## F3 — parallelization mapping

- **Within a replica**: color-serial kernel launches; within a color, **one block
  per site with threads parallel over the site's adjacency entries** (the MD-force
  pattern), warp/block reduction into the `nlm` accumulator `c`. Term-parallelism
  absorbs the `z == 0.0` skips as idle lanes instead of divergent sites and makes
  occupancy independent of sites-per-color (works from 4³ up). The site-parallel
  alternative (one thread per site) needs ≥10⁴ sites per color and suffers
  adjacency-length divergence; keep it as a fallback for light models.
- **Across replicas**: PT rungs alone (tens) cannot fill a GPU (literature
  consensus). Combine rung × color parallelism, or pivot the massive axis to
  population annealing (10⁴–10⁶ replicas, the GPU-native ensemble method,
  arXiv:1703.03676) — the natural upgrade of `find_ground_state`'s thermal cycling.
  The PT payload swap stays an O(1) device-pointer/buffer-index swap.
- **Not applicable**: double-checkerboard shared-memory tiling and multi-hit
  updates (assume compact regular stencils; multi-hit also changes dynamics),
  multispin coding (discrete spins only).

## F4 — determinism (interaction with P6)

A GPU port changes the RNG stream → **breaking, one CHANGELOG line** (P6 allows
this explicitly). The replacement discipline is *stronger* than the CPU one:
**stateless counter-based Philox4x32-10 keyed by (seed, replica, site, sweep)** —
every draw a pure function of logical coordinates, independent of thread
scheduling, block shape, and even CPU-vs-GPU backend (KernelAbstractions' CPU
backend can replay draws bit-for-bit). Precedents: Random123 (SC'11), HOOMD-blue,
PAising. Two hard rules carried over from the CPU design: **no floating-point
atomics** in accumulation paths (fixed-order/tree reductions only — the GPU analog
of `_reduce_dE`'s fixed color-order sum), and the acceptance uniform drawn
per-proposal from the keyed stream so consumption stays scheduling-independent.
CUDA.jl's built-in device RNG (Philox2x32, warp-shared state) is **not** suitable —
its streams are tied to warps, hence to scheduling; implement the keyed generator
directly (~30 lines, portable under KernelAbstractions).

## F5 — precision: Float64 end-to-end

The literature-standard FP32-spins + FP64-accumulators split exists to dodge
consumer GPUs' 1:64 FP64 penalty. The actual target (kugui A100, FP64:FP32 = 1:2,
bandwidth-dominated workload) makes it unnecessary, and every exact-equality gate,
the `1e-8·max(1,|E|)` drift gate, and the `z/p/ck == 0.0` skip contracts are
FP64-calibrated — FP32 would invalidate the whole verification culture for ~2×.
Decision: Float64 throughout; revisit only if a consumer-GPU target ever matters.

## F6 — stack and dev loop

**KernelAbstractions.jl + CUDA.jl backend** (KA is load-bearing JuliaGPU
infrastructure since GPUArrays v11; CUDA.jl v6.x supports julia 1.12). Apple
silicon has **no FP64 in Metal at all** (kernel Float64 is an `InvalidIRError`,
no emulation in Metal.jl), so the M4 dev loop is: same kernels on **KA's CPU
backend in Float64** for correctness/bit gates locally; compile-and-bench on kugui
(`F1accs`/`L1accs` queues, A100 ×4 per acc node, one per PBS vnode — the `@slce`
env from the deployment is the runtime). Device RNG and warp primitives are the
two things KA does not abstract — the keyed Philox (F4) sidesteps both.

## F7 — go/no-go and staging

- Phase 1 (prototype): one KA kernel pair — term-parallel `site_coeffs!` +
  accept/write — driving a color-serial Metropolis sweep on device, keyed Philox,
  `zrows`/tables resident; bit-gate vs CPU reference through the KA-CPU backend;
  bench on A100 at 4³/8³.
- **Exit bar, fixed in advance: < 5× over the tuned CPU path on the 8³ nbody = 3
  fixture → stop, keep the CPU code.** (Realistic literature-deflated expectation
  is 5–30×; the 100–1000× headlines are vs 2008-era single scalar cores.)
- Phase 2 (only on go): observables/measurement on device, PT rung × color, then
  population annealing as the ensemble layer.

## References

- M. Weigel, *Performance potential for simulating spin models on GPU*,
  J. Comput. Phys. 231, 3064 (2012), arXiv:1101.1427 — checkerboard, RNG hazards,
  Heisenberg DP 94× / SP 366× vs one 2008 core.
- T. Yavors'kii, M. Weigel, Eur. Phys. J. ST 210, 159 (2012), arXiv:1204.6192 —
  heat-bath + over-relaxation + PT for continuous spins, fully on-device.
- L. Barash, M. Weigel, M. Borovský, W. Janke, L. Shchur, Comput. Phys. Commun.
  220, 341 (2017), arXiv:1703.03676 — PAising: population annealing on GPU,
  per-thread keyed Philox, deterministic fixed-order reductions (no FP atomics).
- J. Salmon, M. Moraes, R. Dror, D. Shaw, SC'11, DOI 10.1145/2063384.2063405 —
  Random123 counter-based RNGs.
- CLAMM, arXiv:2506.17800 — closest CPU code (CE + magnetic CE MC); no GPU.
