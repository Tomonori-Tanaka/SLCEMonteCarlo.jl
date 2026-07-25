# Decision record — GPU Metropolis prototype (Phase 1 of the F7 staging)

Status: landed on `feat/gpu-prototype` (Phase 1 of `gpu-feasibility.md` F7).
Owner: `src/gpu/*.jl`; gates in `test/unit/test_gpu.jl`; bench in
`bench/bench_gpu.jl`. The CPU paths are untouched and remain the production
default; the GPU path is `public` (unexported) until the go/no-go decision (G6).

## G1 — scope

Device Metropolis sweep only: `gpu_metropolis_sweep!` / `gpu_run_sweeps!` over a
`GPUTiledHamiltonian` + `GPUChainState` on any KernelAbstractions backend (the
package never references a GPU runtime — the caller passes `CUDABackend()`;
`CPU()` is the reference backend the test gates run on). No over-relaxation, no
PT, no on-device observables, no checkpointing, no step adaptation: `to_host!`
drops the state back into a `ChainState`, where every host facility applies
unchanged. Renormalization round-trips through the host (`gpu_run_sweeps!`),
which is seam-free because `normalize` is IEEE-exact arithmetic and the host and
device tesseral rows are bitwise-identical (G4).

## G2 — keyed RNG (Philox4x32-10, `src/gpu/philox.jl`)

Every draw is a pure function of logical coordinates — no RNG state exists:

```
key = (lo32(seed), hi32(seed))          seed::UInt64, recorded in GPUChainState
ctr = (site, sweep, slot, 0)            site/sweep::Int32 (1-based; sweep counts
                                        completed sweeps + 1), slot ∈ 0:2
```

The fourth counter word is reserved zero — replica ids / update-kind tags get
their own subspace later without moving any existing stream. Slot map (3 blocks
per proposal): slot 0 words 1–2 → flip uniform (vs `_FLIP_FRACTION`), words 3–4
→ accept uniform; slot 1 → Box–Muller axis normals n₁, n₂; slot 2 → n₃ and the
Gaussian angle n₄. Because a slot's value is fixed by its coordinates,
branch-dependent *consumption* is meaningless — the CPU path's "accept uniform
drawn only when ΔE > 0" contract disappears wholesale. Uniform bit convention:
top 52 bits, centered — `(w >>> 12 + 0.5)·2⁻⁵²`, strictly open (0, 1) with exact
endpoints `2⁻⁵³` and `1 − 2⁻⁵³` (a 53-bit variant rounds its top value to exactly
1.0; `rand()`'s `[0, 1)` convention is NOT used). Gate: the three Random123
`kat_vectors` known answers, edge-word openness, stream separation.

The GPU path is therefore a **different Markov chain** than any CPU run (new RNG
scheme — the P6 "breaking, one CHANGELOG line" case, scoped to the GPU path
only; CPU streams are byte-identical to before).

## G3 — determinism contract

- **(a) Within one backend**: for fixed (seed, backend, workgroup size, package
  + Julia version), runs are bitwise identical and scheduling-independent by
  construction — keyed draws, per-site-disjoint writes, and a fixed lane-ordered
  reduction. The workgroup size is part of the contract; the pinned default is
  **WS = 128** (power of two enforced). Gates: full-sweep bitwise vs the keyed
  reference, repeated-run identity.
- **(b) Across backends** (KA-CPU ↔ CUDA): bitwise identity holds for the
  algebraic kernels only — the tesseral row and the entry-walk products/sums are
  `+, *, /, sqrt` and an integer-power complex multiply chain, all IEEE-exact —
  but full trajectories differ in ULPs because Box–Muller and the accept test
  use backend libm (`log`/`cos`/`sin`/`exp`). The backend-independent gates are
  the incremental-energy drift gate (physics-exact) and statistics.
- **The bitwise anchor** is `_metropolis_sweep_keyed_ref!` (bottom of
  `gpu_sweep.jl`): a plain serial host sweep of the identical keyed scheme,
  sharing `_keyed_proposal` / `_entry_walk_partial` / the lane-ordered fold with
  the kernel — on the CPU backend (same libm) the kernel must match it bitwise,
  and does (config, zrows, energy, counters; gated at ws ∈ {4, 32} on pair-,
  triplet-, and inactive-site-bearing fixtures).

## G4 — kernel shape (`src/gpu/gpu_sweep.jl`, `src/gpu/zlm_device.jl`)

Color-serial launches (launch-queue ordering makes per-color host syncs
unnecessary; one `synchronize` per sweep), one workgroup per site of the color,
threads term-parallel over the site's adjacency entries:

- **Direct-ΔE accumulation** — no materialized coefficient vector: `znew`
  depends only on the proposal, so lane 1 computes it first and the walk folds
  `w·p·(znew[tgt] − zrows[tgt, s])` per entry. Halves local memory and removes a
  reduction pass; the shared-`c` variant (needed for a future device
  over-relaxation) is the noted follow-up. Consequence: even on the CPU backend
  the summation order differs from `site_coeffs!` + `delta_energy`, so vs the
  existing CPU kernels the gate is a scaled 1e-12 tolerance; bitwise is owned by
  the keyed reference (G3).
- The `z == 0.0` / `p == 0.0` skips are kept verbatim — they are part of the
  bitwise contract (adding an exact 0.0 can flip a −0.0 partial).
- **Reduction**: per-lane strided partials (`lane, lane+ws, …`), then a
  lane-ordered serial fold by lane 1 — deterministic for pinned ws; a pairwise
  tree is a possible later optimization (measure first), which would be a
  contract change (same-backend trajectories move).
- **Device tesseral row** (`_zlm_row_device!`): a bitwise-faithful replication
  of `_zlm_row!` → `Harmonics.Zlm_unsafe` → `LegendrePolynomials.dnPl`
  (`_unsafednPl!`'s exact recursion order, `_plm_norm`'s division loop, and the
  `Base.power_by_squaring` value path as `_zlm_cpow` — the upstream code cannot
  compile in a kernel because of its throw/`no_offset_view` wrappers, which are
  value-neutral and dropped). `Val{LMAX}` specialization gives static stack
  buffers; lmax ≤ 6 supported. Gates: dense bitwise equality against the host
  row (lmax 0:6, poles/axes/equator + 2000 seeded directions, both as a direct
  call and through a KA-CPU kernel), `_zlm_cpow ≡ ^` exhaustively for n = 1:6.
- KA CPU-backend discipline: plain locals do not survive `@synchronize` (the
  body splits into blocks), so `lane`/`s` are recomputed per segment and all
  cross-segment state lives in `@localmem`; `@index` sits at segment top level.
- Host driver: per-sweep `dE`/`acc` copy-back (≈ 0.4 MB at 8³ — negligible) and
  the fixed-color-order `_reduce_dE` fold reused verbatim; acceptance counters
  are integer sums of per-site flags (no atomics of any kind on the device).

## G5 — gates (`test/unit/test_gpu.jl`, all on the CPU backend)

Philox known answers; device-row bitwise (direct + through a kernel); direct-ΔE
walk vs `site_coeffs!`+`delta_energy` (scaled 1e-12) on pair AND triplet
(`_threebody_terms` — asserted to hit `site_col < 0`) programs; full-sweep
bitwise vs the keyed reference (4 fixtures × ws ∈ {4, 32}, config/zrows/energy/
counters over 5 sweeps); repeated-run identity + seed sensitivity; inactive
sites bitwise frozen through a renormalizing `gpu_run_sweeps!`; drift gate
(`≤ 1e-8·max(1, |E|)` after 200 unrenormalized sweeps); the exact dimer
statistics gate (`⟨e₁·e₂⟩ = L(β|J|)`, atol 0.03 — same convention as
`test_metropolis.jl`). On a CUDA device the meaningful subset is repeated-run
identity, the drift gate, and statistics — wired into `bench/bench_gpu.jl`'s
smoke rather than the CI suite.

## G6 — A100 measurements and go/no-go: **GO**

Measured 2026-07-16 on kugui `F1accs` (A100-SXM4-40GB, driver 560.35.03 /
CUDA 12.6, EPYC host; ws = 128, kT = 0.025 eV, 100 sweeps per point; CPU
baseline = the tuned `metropolis_sweep!` with 4 sweep tasks on the SAME node).
Device correctness gates (repeat-run bitwise identity, drift ≤ 1e-8·scale)
passed on device; acceptance rates match the CPU sampler at every size
(0.21/0.21):

| model | device ms/sweep | cpu-4T ms/sweep | ratio |
|---|---|---|---|
| bcc 16³ (light-kernel control) | 0.21 | 0.86 | 4.1× |
| Nd₂Fe₁₄B nbody=3 4³ | 2.84 | 46.9 | 16.5× |
| Nd₂Fe₁₄B nbody=3 8³ (**the bar**) | **10.88** | **327.6** | **30.1×** |
| Nd₂Fe₁₄B nbody=3 16³ | 78.7 | 2813 | 35.7× |

**Verdict: GO — 30.1× at the fixed 8³ bar (≥ 5×).** The 16³ tables (4.45 GiB,
0.32 s upload) fit the 40 GB part comfortably; throughput still improves with
size (300 ns/attempt at 16³). Campaign scale: a 12k-sweep 8³ point drops from
65 min (same-node CPU-4T) to 2.2 min; 16³ from 9.4 h to 16 min. Operational
notes: compute nodes have no internet — pin `CUDA.set_runtime_version!` to the
driver's version (12.6) and instantiate on the login node first; the debug-queue
smoke caught a real cross-backend bug (`@index` is Int32 on CUDA, Int on the CPU
backend) — keep the smoke-before-bench procedure. Phase-2 candidates (in the
order they will matter): on-device observables (measurement currently costs a
copy-back), PT rung × color, population annealing, promotion of the API from
`public` to exported (**done 2026-07-19**: the sweep API — `GPUTiledHamiltonian`,
`GPUChainState`, `gpu_metropolis_sweep!`, `gpu_run_sweeps!`, `to_host!` — is
exported; the G7 gradient tier stays public-unexported as the dependent-package
seam).

### Production-model validation (2026-07-16, kugui F1accs)

Physics gate on a real fitted model, not a fixture: the Nd₂Fe₁₄B l02 model
(isotropic bilinear, 179 SALCs) refit from the original EMBSET with SLCE
(rmse vs the Magesty fit 3.4 meV), tiled to 8³ (34,816 sites, 38 colors, mean
adjacency 147). The tuned CPU sampler (4 sweep tasks) and the A100 kernel ran
as independent chains with identical measurement code (1500 therm + 3000 sweeps,
sampling every 5; errors from `LogBinner`):

| kT (eV) | quantity | cpu-4T | gpu | agreement |
|---|---|---|---|---|
| 0.05 | E/site (eV) | −0.086822 ± 6.1e-5 | −0.086799 ± 5.8e-5 | 0.28σ |
| 0.05 | \|m\| | 0.4165 ± 0.0081 | 0.4059 ± 0.011 | 0.78σ |
| 0.12 | E/site (eV) | −0.022378 ± 2.0e-5 | −0.022420 ± 2.1e-5 | 1.5σ |
| 0.12 | \|m\| | 0.0094 ± 2.4e-4 | 0.0089 ± 1.6e-4 | 1.6σ |

All observables agree within 1.6σ; the end-of-run drift gate
(|E_incremental − E_recomputed| ≤ 1e-8·scale) passed on device at both
temperatures. Real-model speedup: 1.34 vs 19.4 ms/sweep = **14.4× at 8³**
(bilinear kernels are lighter than the nbody=3 fixture's, hence below the
30× of the table above); 16³ (278,528 sites) runs at 7.45 ms/sweep on device.

### Production-model validation — l044, nbody = 3 (2026-07-17, kugui F1accs)

The heavy production target: the Nd₂Fe₁₄B l044 model (nbody = 3, body-2/3
`lsum = 4`, 4672 SALCs — refit from the original EMBSET with SLCE's
per-body-lsum BasisSpec, rmse vs the Magesty fit 14.4 meV), 405,312 multipole
terms and mean adjacency 18,852 (128× l02's). The statistics gate ran at 3³
(1836 sites — the CPU chains dominate the walltime at ~0.9 s/sweep), CPU-8T
and A100 as independent chains with identical measurement code (2000 therm +
1000 sweeps, sampling every 5; errors from `LogBinner`):

| kT (eV) | quantity | cpu-8T | gpu | agreement |
|---|---|---|---|---|
| 0.05 | E/site (eV) | −0.111804 ± 1.7e-4 | −0.111126 ± 2.0e-4 | 2.6σ |
| 0.05 | \|m\| | 0.8148 ± 0.0011 | 0.8119 ± 0.00099 | 2.0σ |
| 0.12 | E/site (eV) | −0.010138 ± 6.9e-5 | −0.009945 ± 7.3e-5 | 1.9σ |
| 0.12 | \|m\| | 0.0333 ± 0.00099 | 0.0320 ± 0.0011 | 0.87σ |

Worst case 2.6σ (E at kT = 0.05), attributed to residual thermalization in
the stiff ordered phase rather than a sampler difference: the half-chain
means show the two chains approaching equilibrium from opposite sides
(cpu |m| 0.8123 → 0.8173 still rising, gpu 0.8131 → 0.8107 easing down; an
earlier 600-therm run split 4σ, 2000 therm brought it to 2σ), the E offset
direction is consistent with the |m| offset (the cpu chain is the more
ordered one), the fast-relaxing kT = 0.12 point agrees at 0.87–1.9σ, and the
drift gate passes exactly at every point.

Real-model speedups: 3³ 905 → 85 ms/sweep = 10.6×; **8³ 15.11 s → 396 ms =
38.1×** (heavier kernels widen the GPU lead past l02's 14.4×, consistent
with the 30× nbody=3 fixture above); 10³ (68,000 sites) 807 ms/sweep on
device. Measured table footprint 0.36 MiB/site + ~0.4 GiB fixed: 8³ ≈ 12.7
GiB and 10³ ≈ 24 GiB fit the 40 GB part, 16³ ≈ 99 GiB does not. Campaign
scale: a 12k-sweep 8³ point drops from ~50 h (CPU-8T) to 79 min.

Operational: the first submission burned its walltime on a **silent KA-CPU
fallback** — an `rsync --delete` deploy had removed the kugui-only
`bench/gpu/LocalPreferences.toml` CUDA pin. The pin now lives machine-global
in `~/.julia/environments/v1.12/` (LocalPreferences.toml **plus** a
`[extras]` entry for CUDA_Runtime_jll in that env's Project.toml — without
the extras entry the preference resolves to `nothing`), outside the rsync'd
tree; GPU job scripts additionally export `SCE_REQUIRE_CUDA=1`, which the
bench scripts turn into a fail-fast error when CUDA is not functional.

## G7 — phase 2: device all-site gradient (`src/gpu/grad_device.jl`, `src/gpu/gpu_gradient.jl`)

The entry point SLCEDynamics' GPU LLG consumes: `gpu_energy_gradient!` —
all-site, tangent-projected `G[s] = ∂E/∂e_s`, the device twin of the host
`energy_gradient!` (public tier, unexported, with `GPUGradientScratch` and
`gpu_zlm_rows!`).

- **Gradient row**: `_grad_zlm_device` is the operation-order-faithful replica
  of `Harmonics.grad_Zlm_unsafe` → `_barP`/`_dbarP` → `dnPl` →
  `_grad_zlm_assemble`. The two genuinely new pieces: the `dnPl` trivial-zero
  branch (`l < n` returns a +0.0 literal BEFORE touching the cache —
  `_zlm_dnpl_or0`; the host's `parity·norm·(+0.0)` then yields **−0.0** for odd
  parity, which the `===` gate checks), and the `_zlm_cpow` `p == 0` branch
  (`zxy^(n−1)` at n = 1; the previous code walked `trailing_zeros(0)` off the
  exponent — a real latent bug, unreachable from the value row).
- **Kernel shape**: one workgroup per site, NO coloring (read-only pass,
  per-site disjoint `G[s]` writes → a single launch over all sites). Lane 1
  fills the 3×nlm gradient-row table into `@localmem` (the `znew` analog —
  `∇Z(e_s)` is fixed during the pass); all lanes run `_entry_walk_grad` — the
  structural clone of `_entry_walk_partial` (same three-way `site_col`
  dispatch, same zero-skips) folding a 3-vector partial per lane; lane 1 does
  the lane-ordered component fold. Shared memory: `3·(ws + nlm)·8` ≈ 4.2 KB at
  ws = 128 / lmax 6 (a materialized-coefficient variant would need 50 KB — the
  direct fold is why). Inactive sites: empty adjacency range → fold of +0.0s →
  exactly `(0, 0, 0)`, no `site_active` on device.
- **Rows rebuild**: LLG moves every spin per stage, so `zrows` is rebuilt from
  the configuration per gradient call (`_zlm_rows_kernel!`, one thread per
  site, bitwise ≡ host `_zrows` by the G4 row identity). The scratch is owned
  upstream (`GPUGradientScratch`); `refresh_zrows = false` is the MC-side
  convenience (`GPUChainState` rows are current by the sweep invariant).
- **Determinism**: bitwise for fixed (backend, workgroupsize); the whole
  pipeline (row + walk + fold) is `+ − * /` + correctly-rounded `sqrt` — **no
  libm, no RNG** — so unlike the sweep the device output is expected to match
  the serial `_gradient_lane_ref!` bitwise on EVERY backend. CI gates it on
  the CPU backend; the A100 smoke claims it on CUDA with a documented fallback
  (scaled tolerance ≤ 1e-12·max(1, maxₛ‖G_host‖) + a note here) should FMA
  contraction ever appear. `muladd`/`@fastmath` are forbidden in the pipeline.
- **Gates** (test_gpu.jl): dense bitwise grad row vs `grad_Zlm_unsafe`
  (l ≤ 6, poles/axes/equator + 2000 dirs, `===` per component); `_zlm_cpow`
  n = 0:6; rows rebuild bitwise; kernel ≡ lane reference bitwise at
  ws ∈ {4, 32} incl. triplet/general programs and inactive-site zeros;
  scaled tolerance vs host `energy_gradient!` + tangency ≤ 1e-13.
- **Perf (measured 2026-07-19, kugui A100-SXM4-40GB, job 858227)**: fixture
  Nd₂Fe₁₄B nbody=3 8³ (34,816 sites, ws = 128): **T_grad = 3.74 ms/eval**,
  grad/sweep ratio **1.11** (device sweep 3.38 ms — the cost model held: one
  gradient eval ≈ one sweep). **GR9 confirmed: bitwise vs `_gradient_lane_ref!`
  on CUDA** — the fallback tolerance path stayed unused. Sweep go/no-go
  re-confirmed 16.9× the same run. Field note: the first A100 attempt (job
  858226) caught a CUDA-only compile bug — the CUDA backend's `@index` returns
  Int32 and the raw group index made `_entry_walk_grad`'s Int-typed call a
  compile-time MethodError (`a9ff0e4`); the KA-CPU gates cannot see this class
  (their `@index` returns Int) — device-only smoke stays mandatory after any
  kernel-adjacent change.
