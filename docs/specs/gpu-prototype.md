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

## G8 — phase 3: the displacement channel (`src/gpu/disp_device.jl`, M4 slice 3f)

**Device rows (3f/1).** `_solid_row_device!` replicates
`SLCE.SolidHarmonics.solid_harmonics!`; `_disp_rows_device!` then applies the
layout's `(k, l)` blocks with the `|u|^{2k}` prefactor, the device twin of
`_disp_rows!`. Unlike the tesseral case this is a near-verbatim copy rather than an
operation-order transcription: the upstream core (`_solid_harmonics_impl!`) is
already pure scalar arithmetic with no throws and no allocation, so only the
gradient half and the `lmax` validation are dropped and `lmax` is lifted to a `Val`.
`lmax ≤ 6`, as on the spin side.

**The bitwise scope shrinks here, deliberately.** `_solid_row_device!` is
`+ - * / sqrt` and is gated bitwise. `_disp_rows_device!`'s `k ≥ 1` blocks are NOT
bitwise-identical to the host: `r2 = dot(u, u)` is a mul-add chain whose last bit
depends on whether LLVM contracts it into an FMA, and that decision differs between
the host (harmonic batch = a separate call) and the device (batch inlined, as a
kernel requires, which puts a second `x²+y²+z²` in the same scope to be
canonicalized/CSEd against). Measured: ~22 % of random `u`, exactly 1 ulp of `r2`,
amplified by the exponent. Neither reordering the two computations nor transcribing
`dot` as an explicit `muladd` chain removes it. `r2^k` adds a second, smaller effect
at `k ≥ 2` (`Float64 ^ Int` routes through libm `pow`, not `power_by_squaring`).

**Decision: accept it; do not chase it.** A single shared `r2` would fix it, but
`r2` is computed independently in `SLCE.site_rows!` (the upstream authority) and in
`_disp_rows!`, so the fix is an upstream convention change that moves SLCE's numerics
by an ulp. Not worth it, because bitwise identity here buys **test sharpness, not
accuracy**: 1 ulp is relative 1e-16 against MC statistical errors of ~1e-3, and the
gates that actually catch bugs — kernel ≡ keyed reference, serial ≡ parallel — are
**device-vs-device** and stay bitwise (the reference calls the same
`_disp_rows_device!`). What is genuinely lost is the G4 claim that the
renormalization host round-trip is seam-free: on a `k ≥ 1` block it injects a 1-ulp
perturbation, which the existing incremental-energy drift gate already covers.

Gates (`test/unit/test_gpu.jl`): `_solid_row_device!` bitwise vs upstream over
lmax 0:6 × (u = 0, axes, 8 decades of magnitude, 2000 seeded), directly and through
a KA-CPU kernel; `_disp_rows_device!` bitwise on every `k = 0` row and within
`kmax + 1` ulp elsewhere, over three layouts spanning the `(k, l)` block structure;
`_disp_layout_tables` against the layout it flattens; the pure-spin layout as a no-op.

**Joint device tables and the joint-safe spin sweep (3f/2).** `GPUTiledHamiltonian`
no longer requires a pure-spin `H`. Three changes make the existing Metropolis
kernel correct on a joint model, and each is a device form of something the host
already does:

1. **Full row table.** `GPUChainState.zrows` is `nrows × n_sites` (was `nlm ×
   n_sites`) and the state carries `disps` and `step_u`; `to_host!`/`_from_host!`
   round-trip the displacements with the rest.
2. **Row range.** `_entry_walk_partial` takes `lo:hi` and selects the same entries as
   `delta_energy(c, zold, znew, lo, hi)` (energy.jl). A single-channel move rewrites
   one block, the rows it did not touch contribute `c_k · 0`. The spin kernel passes
   `1:(LMAX+1)²`, which **is** `H.nlm` (`RowLayout.disp_offset` — now asserted in the
   constructor, since a caller-supplied padded `RowLayout` would otherwise truncate
   the device ΔE), so the write-back and the ΔE stay inside the spin block and the
   site's own displacement rows are read as the constant factors they are. On a
   pure-spin `H` every entry is in range and the walk is the pre-M4 one bit for bit.

   **The buffer convention differs from `delta_energy`, deliberately.**
   `delta_energy` indexes a full-length `znew` absolutely (that is how
   `_attempt_disp!` calls it, with `sc.znew` of length `H.nrows`); the walk takes the
   moved block alone, indexed relatively (`znew_block[tgt - lo + 1]`), because a
   kernel's trial row is `@localmem` sized to exactly one block. They coincide at
   `lo = 1`. `length(znew_block) == hi - lo + 1` is the contract and cannot be
   checked on a device — **3f/3 hazard**: a displacement kernel written by analogy
   that hands the walk a full-length row with `lo = nlm+1` would read the spin block
   as displacement rows, silently and with no shape error.
3. **Per-channel schedules.** `_channel_colors` slices the one coloring by
   `site_has_spin` / `site_has_disp` into `spin_sites`/`disp_sites` (device) with
   `spin_ptr`/`disp_ptr` launch ranges (host) — the host sweeps' skip predicates,
   moved to launch time. A joint model can have a site active in one channel only (a
   force-constant-only ligand carries no moment), and a sweep that visited it anyway
   would be always-accepted noise that biases the acceptance statistics. On a
   pure-spin `H` the two predicates coincide with `site_active`, so `spin_sites ==
   H.color_sites` and `spin_ptr == H.color_ptr` **verbatim** — the pre-M4 launch
   schedule, unchanged.

The **gradient** path stays spin-only: `_entry_walk_grad` has no row range and the
derivative is with respect to the spin direction alone. The guard therefore moved
from the shared constructor onto **every** gradient entry point —
`GPUGradientScratch`, `gpu_zlm_rows!`, both `gpu_energy_gradient!` overloads, and
`_gradient_lane_ref!` — and each needs its own: `gsc` is no evidence about the
Hamiltonian it is handed with (a scratch built from the spin restriction of the same
model has the matching `(nlm, n_sites)` shape and passes the dimension check), and
the documented `refresh_zrows = false` fast path skips `gpu_zlm_rows!` entirely. An
unguarded call walks the joint program table — whose `sent_tgt` reaches past `nlm` —
against a spin-sized gradient row, an `@inbounds` out-of-bounds read that returns
garbage rather than throwing. Each of the five is gated separately.

**The driver refuses the wrong ensemble.** (Superseded by 3f/3 below, which makes the
displacement sweep available and so lets the driver default to the right thing:
`fixed_lattice` is gone, replaced by `disp_per_metropolis`. The reasoning is kept
because the constraint it encodes did not change.) On a joint model the device moves
the spins only, so the chain samples `π(ê | u)` at the uploaded lattice, not the joint
distribution — a different ensemble, and displacement observables off it are
conditional on a lattice nobody equilibrated. `gpu_run_sweeps!` throws unless the
caller passes `fixed_lattice = true`. This is the device analogue of
`_resolve_disp_passes` (run.jl), which likewise refuses to let `disp_per_metropolis`
default to 0 on a joint model; the difference is that the host CAN default to the
right thing and here we cannot yet, so the caller acknowledges instead. Prose in a
docstring is not the same instrument. `gpu_metropolis_sweep!` stays unjudged — it is
the primitive, and interleaving host `displacement_sweep!`s around it is how a joint
chain is driven until 3f/3.

Two further guards fell out of the same review:

- `_require_spin_sites` (hamiltonian.jl, the spin mirror of `_require_disp`) rejects
  `n_spin_active == 0`. A lattice-only model was accepted by the constructor, produced
  an empty `spin_sites`, and completed a "successful" run that attempted nothing —
  a `0/0` acceptance over a bit-for-bit unchanged state. That the `lmax == -1`
  degenerate case never reached `Val(H.lmax)` was luck of the empty-launch guard, not
  design.
- `H.nlm == (H.lmax+1)^2` is asserted in the constructor (see item 2 above).

"Fixed lattice" is exact only up to the centre-of-mass gauge: with
`renorm_interval > 0` the host `_renormalize!` re-centres each displacement-coupling
component, so `st.disps` moves by a rigid per-component shift recorded in
`com_removed`. The energy is unaffected (measured drift 0.0) — it is a gauge choice,
not a different lattice — but the docstring says so, and the drift testset now pins
both halves: `disps + com_removed` reproduces the upload exactly, and with
renormalization off `disps` is bit-identical.

Gates (`test/unit/test_gpu.jl`, joint fixtures = the hand-built
`_channel_split_terms` and a fitted `_joint_model`): the pure-spin schedule identity
above, asserted field by field; the channel lists against the host predicates, with
`n_spin_active < n_active` on the split fixture so the split is not vacuous; the
restricted walk vs `site_coeffs!` + `delta_energy(…, 1, nlm)`, with two non-vacuity
counters (spin sites whose programs target displacement rows at all, and sites whose
skipped rows carry nonzero coefficients); the five-sweep kernel ≡ keyed reference
bitwise, plus — independently of both — displacements, displacement rows, and
spin-inactive directions bitwise unchanged; the drift gate with renormalization off
and on, including the gauge claim above; the `fixed_lattice` and empty-sweep refusals
(and that a pure-spin `H` needs no opt-in); the padded-`RowLayout` rejection; and
`ArgumentError` from every gradient entry point on a joint `H`, under both
`refresh_zrows` values and with a scratch whose shape matches.

Two whole-slice checks outside the suite: a stash/restore comparison of pure-spin
device trajectories (4 fixtures × ws ∈ {4, 32}, hashed over config/rows/energy) came
back identical to `5471379`, and SLCEDynamics.jl — which consumes the gradient tier —
passes its 3493 assertions unchanged.

**The displacement kernel and the compound driver (3f/3).** `_disp_kernel!` is
`_metro_kernel!` with the other channel's proposal, row filler, row range and
write-back — same workgroup shape, same lane-1 ownership, same lane-ordered fold,
same accept rule — walking `tb.disp_sites` with `lo:hi = nlm+1:nrows`. The trial row
is filled by `_disp_rows_device!` (3f/1) from a `@localmem` solid-harmonic batch
workspace, exactly as the host fills it from `SweepScratch.rbuf`. Its serial twin is
`_displacement_sweep_keyed_ref!`, so the same kernel ≡ reference bitwise gate covers
both channels.

**One trial-row convention.** 3f/2 shipped the walk with a block-relative `znew`, and
the review flagged the resulting asymmetry with `delta_energy` as a hazard for exactly
this slice. Rather than document it, the convention is now single: `znew` is
`H.nrows` long everywhere — `SweepScratch.znew` on the host, `@localmem Float64 NROWS`
in both kernels — and indexed absolutely, with only `lo:hi` written and read. The
spin kernel therefore carries a few unused floats on a joint model; that is cheaper
than two conventions. Pure-spin device trajectories are unchanged bit for bit
(re-verified against `5471379` after the switch).

**Per-move-kind RNG counters.** `GPUChainState` counts `sweep_index` and `disp_index`
separately, and the slot map (philox.jl) gives the displacement proposal slots 3–5
against the spin proposal's 0–2. Both are needed and for different reasons: disjoint
SLOTS keep the two moves independent within one step (a shared accept uniform would
make them accept and reject together), and a separate COUNTER keeps several
displacement passes in one compound step from drawing the identical shift. Slot 5's
second Box–Muller normal is discarded — a spare half-block beats a mixed accessor.

**The driver compounds by default.** `gpu_run_sweeps!` takes `disp_per_metropolis`
and resolves it through `_resolve_disp_passes` — the host drivers' own function, so
the rule is shared rather than mirrored: `nothing` means one pass on a joint model and
none on a pure-spin one. The `fixed_lattice` opt-in of 3f/2 is gone; freezing the
lattice is now spelled `disp_per_metropolis = 0`, which is the same acknowledgement in
the vocabulary the host already uses. A lattice-only model (no spin-active site) has
its Metropolis pass skipped by the driver rather than erroring, while the spin
primitive still refuses it; a model with nothing to sweep in either channel throws.

Gates: the slot map's disjointness and the exact `u + step_u·(g₁,g₂,g₃)` shift;
`gpu_displacement_sweep!` ≡ `_displacement_sweep_keyed_ref!` bitwise over five sweeps
on both joint fixtures × two workgroup sizes, with — independently — spins, spin rows
and displacement-inactive sites bitwise unchanged; compound scheduling (both counters,
both attempt tallies, several passes per step, the pure-spin refusal of a displacement
pass, the lattice-only skip-and-throw pair); the joint drift gate now with the lattice
moving, plus the frozen-lattice conditional and its centre-of-mass gauge; and the
**Einstein oscillator against its closed form** — `⟨|u|²⟩ = 3kT/(2a)` to 5 %, the one
displacement gate in the suite that checks the device chain against an external truth
rather than against another implementation of the same arithmetic.
Mutation-tested: widening the displacement walk's row range to the whole table, and
sharing the spin RNG counter, fail 22 and 21 assertions respectively.

**Distributional validation of the compound joint chain** (scratch tier — too slow for
CI, same tier as the stash/restore trajectory comparison). The device compound chain
and the host `metropolis_sweep!` + `displacement_sweep!` chain are different Markov
chains (keyed Philox vs per-site Xoshiro), so they can only be compared in
distribution. On a bounded joint fixture — spin pair + onsite spin–displacement
coupling + a well on **every** atom, `harmonic_stability` min eigenvalue 4.0 with no
negative mode — at kT = 0.05, 4000 thermalization and 6000 measurement compound
sweeps, four seeds each: `⟨E⟩/site` agrees to **0.12σ** and `⟨|u|²⟩/site` to **0.0σ**,
with acceptance rates matching to three digits (0.81/0.73 both sides).

Two traps this measurement walked into first, both worth remembering:

- **The committed joint test fixtures are not dynamically stable in the displacement
  channel.** `_joint_model` has no well at all, and `_channel_split_terms` has one on
  atom 2 only while atom 1 carries a coupling *linear* in `u`. Both are fine for the
  implementation-vs-implementation gates (bitwise vs the keyed reference, short drift
  runs) but a distributional comparison on them compares two escaped, non-stationary
  chains: the first attempt reported `⟨|u|²⟩ ≈ 10⁴` (|u| ~ 100 Å) on both sides and a
  meaningless 4.5σ. A distributional check needs a bounded model.
- **Naive `std/√N` error bars.** With τ_int ≈ 8 they understate the error by ~4×,
  which turned agreement into a spurious 8–9σ disagreement. Use the package's own
  `LogBinner`/`std_error`.

**Review follow-ups (3f/3).** Four things the numerical review surfaced, recorded
because each is a rule rather than a one-off fix:

- **`has_disp` vs `n_disp_active`, again.** `_resolve_disp_passes` gates on the row
  LAYOUT; `gpu_displacement_sweep!` gates on the SITES. A joint basis whose
  displacement couplings all fitted to zero separates them, and the first version of
  the driver threw out of its own default on such a model — which the host runs fine
  as a no-op sweep. The driver now resolves first (so an explicit displacement pass on
  a Hamiltonian with no displacement ROWS is still refused) and clamps to zero passes
  when there is no displacement-active SITE. Zero passes is not a wrong-ensemble risk
  there: with nothing depending on `u`, the frozen-`u` conditional IS the joint
  distribution. Gated.
- **The device driver owes the U8 diagnostics too.** Now that the device chain moves
  the lattice, `_warn_gpu_escape_cadence` mirrors `run.jl`'s `_warn_escape_cadence`
  (same `_escape_min_checks` arithmetic) and additionally warns when
  `renorm_interval ≤ 0` — which on the device path is legal and turns the escape
  detector off entirely along with re-centring. Step adaptation stays out of scope
  (G1), but the guide now says so for `step_u` as well as `step`, and says where the
  acceptance data actually lives (`gst`, not `st`).
- **Widening the trial row moved a shape error into a silent one.** With a
  block-sized buffer, a wrong `(lo, hi)` was a bounds error; with the full-length one
  it is an uninitialized-`@localmem` read (genuinely uninitialized on a GPU). The
  range check is still the only thing preventing it, so the spin kernel now zero-fills
  its unused half — cheap insurance, empty loop on a pure-spin model. The
  displacement-block tiling assertion (`disp_starts[end] + 2l + 1 == nrows`) joins the
  spin-block padding assertion for the same reason: the write-back copies the whole
  block.
- **Two gates were softer than they looked.** The Einstein statistics gate started
  from a random lattice already within 20 % of the answer (now the clamped-ion state,
  so `⟨|u|²⟩` has to be built up; rtol tightened 0.05 → 0.04), and the renormalizing
  arm of the joint drift gate used an interval dividing the sweep count, so the last
  action before the comparison re-anchored the energy it was about to check (now 30
  into 200). Recorded limits of the Einstein gate: it is step-independent by
  construction, and its one `(k, l) = (1, 0)` row exercises neither an `l ≥ 1` solid
  harmonic nor the `k ≥ 2` libm-`pow` path.

**A100 smoke of the joint path — PASSED** (2026-07-27, kugui `i1accs` job 867813,
A100-SXM4-40GB, driver 560.35.03 / CUDA 12.6, CUDA.jl 6.2.1, Julia 1.12.6; script
`bench/gpu/smoke_joint.jl`, job `bench/gpu/job_joint_smoke.pbs`). A separate script
from `bench_gpu.jl` because that one runs pure-spin fixtures and so never launches
`_disp_kernel!` at all — and device-only because the KA-CPU suite cannot see the G7
class of bug by construction. It refuses to fall back to the CPU backend: a smoke
that silently ran on the host would report success for exactly the thing it exists
to check.

All 16 checks green on a 128-site joint fixture (spin 64, disp 128, nrows 8):

| check | result |
|---|---|
| `_disp_kernel!` compiles and runs on CUDA | ok (the G7 failure mode did not recur) |
| `_metro_kernel!` under the new `Val(NROWS)` signature | ok |
| channel isolation, both directions, bitwise | ok |
| determinism: same seed ⇒ identical, different seed ⇒ different | ok |
| joint drift gate, `renorm_interval` 0 / 30 | 7.1e-15 / **0.0** |
| Einstein `⟨|u|²⟩ = 3kT/(2a)` on device | **−0.02 %** |
| CUDA ≡ KA-CPU in distribution, `⟨E⟩` / `⟨|u|²⟩` | 0.97σ / 1.19σ (τ ≈ 9) |
| acceptance rates across backends | 0.779/0.725 both, to three digits |

The cross-backend agreement is the gate G3(b) actually promises — Box–Muller and the
accept test go through backend libm, so bitwise identity is not available there. That
the acceptance rates match to three digits says the two trajectories stay very close
in practice regardless.
