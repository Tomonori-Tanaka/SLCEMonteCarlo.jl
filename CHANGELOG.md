# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — the displacement sampler's preconditions (M4 slice 3c, part 1)

- **Displacement-coupling components.** Two displacement-active sites are joined when
  some instance carries a displacement axis on both; a uniform shift of **each
  component separately** is an exact symmetry under the acoustic sum rule, so the flat
  space is `3 × n_disp_comps`, not 3. (The 2×2×2 tiling of the joint test fixture has
  eight components — a single global centre of mass would have left seven relative
  drifts unbounded.) Exposed as `H.disp_comp_ptr`/`disp_comp_sites`/`n_disp_comps`.
- **`site_has_disp` / `n_disp_active`**, completing the per-channel activity split
  started in 3b.
- **A translation-invariance gate.** `TiledHamiltonian` measures how flat the
  uniform-shift direction actually is — `H.translation_residual`, the energy change
  under a rigid shift relative to that under a generic distortion of the same size — and
  **refuses** a displacement model whose direction is not flat. The measurement is on
  the Hamiltonian itself, not inferred from the fit, so it works on every construction
  path (a hand-built term list has no `asr_residual`). The error names both ways
  forward: refit with `fit(...; asr = true)`, or pass **`fixed_reference = true`** if
  the absolute position is genuinely physical (a substrate-clamped slab, a pinning
  defect), which disables re-centring and puts the displacement guard in the absolute
  frame.

### Added — joint spin–lattice ingest and energy (M4 slice 3b)

- `TiledHamiltonian` now accepts **joint (spin + displacement) models**. A term's
  tensor axes are `TermSlot`s rather than sites — one site may carry both a spin and
  a displacement axis — and the row numbering is the upstream sampler-row contract
  `SLCE.row_layout(model)`, whose `SPIN` block is `Harmonics.lm_index` at offset 0.
  New constructor `TiledHamiltonian(n_cell_atoms, ::Vector{DecoratedTerm}, ::RowLayout;
  dims)`, and `TiledHamiltonian(model; dims)` routes a joint model to it.
- `total_energy(H, config, disps)` — the joint energy, with `disps` a
  `Vector{SVector{3,Float64}}` of Cartesian displacements per site. It is **required**
  on a Hamiltonian with displacement rows (omitting it throws rather than silently
  meaning `u = 0`) and ignored otherwise. Gates: `dims = (1,1,1)` against
  `predict_energy(model, e, u) − intercept(model)`, 2×2×2 periodic replication = 8×,
  and a bitwise check of this package's row filler against `SLCE.site_rows!`.
- The constructor enforces **at most one tensor axis per `(site, channel)`** (upstream's
  `SiteDecor` rule). Two axes of the same channel on one site would make even a
  single-channel move drop a cross term, and such a term has a pure-spin layout — so no
  displacement guard would catch it.
- `site_has_spin` / `n_spin_active` alongside `site_active` / `n_active`: on a joint
  model a displacement-only ligand is active for scheduling but carries no moment, so
  the spin sweeps, `_renormalize!` and the spin observables read the spin predicate.
  Identical to the old fields on any pure-spin model.
- `has_disp(H)` (public, unexported) and the new fields `H.nrows`, `H.disp_lmax`,
  `H.layout`. `H.lmax`/`H.nlm` keep describing the **SPIN block alone**, so every
  spin-only consumer's scratch sizing is unchanged.
- `site_coeffs!` is exact for a move that changes **one channel** of a site (all this
  package's updates); a simultaneous spin+displacement move on one site misses the
  cross term `Δz·Δr`, which is documented and gated as such.

### Changed — BREAKING: `ScaledTerm.ls` → `ScaledTerm.slots`

- `ScaledTerm` (public, unexported) carries `slots::Vector{TermSlot}`, one per tensor
  axis, instead of the per-site angular-momentum list `ls`. `_ContractionPrograms`
  gained `efac_site` and its `sfac_slot` now means "member **site position**" (the same
  integer as before on any pure-spin term). Nothing a pure-spin model produces moved:
  the two ingest surfaces flatten to **byte-identical** program arrays (gate:
  `test_joint.jl` "MultipoleTerm ≡ DecoratedTerm (bitwise)"), and the checkpoint
  fingerprint is unchanged for every pure-spin model, pinned against an in-test copy of
  the pre-M4 formula.
- The sweeps, `minimize_energy`/`find_ground_state`, `energy_gradient!`/`site_gradient`
  and the GPU device path **refuse** a joint Hamiltonian (`_require_spin_only`) rather
  than sample it at an implied `u = 0`. Displacement moves are slice 3c.

### Fixed

- `model_fingerprint` now mixes `layout.disp_factors` on a joint Hamiltonian. Without
  it two models differing only by a displacement radial power `k` — hence by a factor
  `|u|²` — collided, because `TermSlot.row0` is a layout-relative block start and
  `(k, l) → row0` is not injective across layouts. Unchanged for every pure-spin model.
- Passing nonzero `disps` to a Hamiltonian without displacement rows now throws instead
  of silently evaluating the clamped-ion energy (all-zero is still accepted).
- The GPU reference kernels `_gradient_lane_ref!` / `_metropolis_sweep_keyed_ref!` now
  refuse a joint Hamiltonian too. The former is called by qualified name from
  SLCEDynamics, so it is a real cross-package entry point.
- The fixtures, benches, assets and guide pages still used the upstream `isotropy`
  keyword, which `SLCE.jl` replaced by `soc` with an **inverted** meaning
  (`isotropy = true` ⇔ `soc = false`). The whole unit suite errored at fixture
  construction; all call sites are migrated (including `bench/fixtures.jl`'s own
  keyword and `bench/assets/nd2fe14b.toml`'s `[interaction]` key).


### Changed — BREAKING: package renamed SCEMonteCarlo.jl → SLCEMonteCarlo.jl (SLCE family, M0)

- The whole family is renamed to the **spin–lattice cluster expansion (SLCE)**
  stem per `docs/specs/spin-lattice-ce-design.md` §2 (SLCE.jl /
  SLCEMonteCarlo.jl / SLCEDynamics.jl / SLCETools.jl). Package + module name
  changed; **UUID kept** (path-dev Manifests stay resolvable). Old model /
  checkpoint artifacts are unaffected (persistence schemas carry versions,
  not package names).

### Changed

- The GPU sweep API — `GPUTiledHamiltonian`, `GPUChainState`,
  `gpu_metropolis_sweep!`, `gpu_run_sweeps!`, `to_host!` — is now **exported**
  (was public-unexported pending the A100 go/no-go): the GO (30.1× at the 8³
  bar) and the l02/l044 production validations landed. The gradient tier
  (`gpu_energy_gradient!` / `GPUGradientScratch` / `gpu_zlm_rows!`) stays
  public-unexported as the dependent-package seam. New docs guide
  (`docs/src/guide/gpu.md`) + API section.

### Added

- `philox_block` / `philox_normal2` (public, unexported): thin facade over the
  GPU path's keyed philox4x32-10 (Random123 known-answer-gated), so dependent
  packages (SCESpinDynamics' thermal noise) share one stream definition.
  Counter-layout contract: MC streams keep `ctr[4] == 0`; consumers must claim
  a nonzero `ctr[4]` domain tag.
- GPU phase 2 — device all-site gradient (`gpu_energy_gradient!` /
  `GPUGradientScratch` / `gpu_zlm_rows!`, public unexported): the device twin
  of `energy_gradient!` for dependent packages' GPU dynamics (SCESpinDynamics
  LLG). One un-colored launch (one workgroup per site, direct-∇Z fold with the
  sweep's entry-walk dispatch and zero-skips); the gradient row
  `_grad_zlm_device` is a bitwise-faithful replica of
  `Harmonics.grad_Zlm_unsafe` incl. the `dnPl` trivial-zero branch (signed
  zeros gated with `===`). The whole pipeline is libm-free, so the kernel is
  bitwise against its serial lane reference (`_gradient_lane_ref!`, the
  one-arithmetic-contract pattern) on the CPU backend in CI — and **confirmed
  bitwise on CUDA** (A100, 2026-07-19: T_grad 3.74 ms/eval at the nbody=3 8³
  fixture, grad/sweep ratio 1.11 — decision record G7). Fixes a latent
  `_zlm_cpow(z, 0)` bug (undefined behavior via `trailing_zeros(0)` —
  unreachable from the value row, hit by the gradient row's `zxy^(n−1)` at
  n = 1).

### Fixed

- `_grad_kernel!` failed to compile on CUDA (`InvalidIRError`): the CUDA
  backend's `@index` returns Int32 and the raw group index had no
  `_entry_walk_grad` method. Value-identical `Int(g)` conversion, mirroring
  `_metro_kernel!`'s documented pattern; caught by the A100 smoke (the KA-CPU
  gates cannot see this class).
- `model_fingerprint` (public, unexported): facade over the checkpoint format's
  stable FNV-1a model fingerprint, so dependent packages' checkpoint files
  (SCESpinDynamics) carry the same model-identity check.
- `energy_gradient!` / `energy_gradient` (public, unexported): all-site,
  tangent-projected, exact-at-any-body-order gradient `G[s] = ∂E/∂e_s`, built on
  the same per-site kernel as `minimize.jl`'s descent (`_site_grad`, bitwise-gated
  against `site_gradient`). Read-only site loop threaded over `ntasks` with
  task-local scratch — bit-identical for any task count, no coloring. One call
  costs about one Metropolis sweep. This is the effective-field/torque entry
  point for dependent packages (`τ_s = G[s] × e_s` matches
  `SCEFitting.predict_torque`; `B_s = −G[s]/(magmom_s·μ_B)` is the caller's —
  moment magnitudes never enter this package).

### Fixed

- `find_ground_state` never hands back a non-finite "winner" silently: a run
  where some starts diverge warns with the count, and an all-non-finite run
  errors instead of returning start 1 as the minimum.
- `_random_unit` redraws on an (astronomically improbable) near-zero Gaussian
  draw instead of emitting a NaN spin; RNG consumption on the no-retry path is
  unchanged, so bit-determinism is unaffected.
- Docs: stale "schema v1" labels in `SPEC.md`/`CLAUDE.md`/the checkpoint design
  record corrected to the current schema v2 (the code's version check was
  already v2).

### Changed

- **`reduce_cell` matches translation copies in canonical site order** (sorted
  `(reduced atom, shift)`, re-anchored, `ls`/`folded` carried through the
  permutation): SCEFitting's canonical SALC members (one term per physical
  instance, up to `N!`× fewer terms — see SCEFitting's persist-v4 change) anchor
  translation copies at different member sites, which the previous
  anchored-form-only grouping could not identify. The periodicity census now
  accepts `q·|det M|` copies for a term list carrying `q` identical summands
  per instance (e.g. hand-built directed pairs), emitting `q` representative
  copies. Fitted-model reductions are unchanged in meaning; hand-built directed
  term lists now reduce to canonical-form representatives.
  `docs/specs/cell-reduction.md` updated. Downstream effect of the SCEFitting
  change: `TiledHamiltonian` on a canonical (or reloaded pre-v4) model has
  ~`N!`× fewer instances/adjacency entries — measured 7.7–8.7× faster sweeps
  and ~6× smaller GPU tables on the l044 production model. Energies agree with
  the previous representation to summation-order (last-ulp) level, so
  checkpoint resume across this boundary is not bit-identical.

### Added

- **GPU Metropolis prototype** (`docs/specs/gpu-prototype.md`; Phase 1 of the
  `gpu-feasibility.md` staging): a KernelAbstractions device sweep —
  `GPUTiledHamiltonian` / `GPUChainState` / `gpu_metropolis_sweep!` /
  `gpu_run_sweeps!` / `to_host!` (`public`, unexported until the A100 go/no-go)
  — with color-serial launches, one workgroup per site, threads term-parallel
  over the adjacency entries, and a direct-ΔE fold. The package references no
  GPU runtime (the caller passes the backend; `CPU()` is the gated reference).
  RNG: **the GPU path draws from a stateless Philox4x32-10 stream keyed by
  (seed, site, sweep)** — a new RNG scheme, so a GPU chain is a different
  trajectory than any CPU chain (P6 scope note; **CPU streams are unchanged**).
  Determinism: bitwise reproducible per (seed, backend, workgroup size,
  version) and scheduling-independent by construction; the device tesseral row
  is a bitwise-faithful replication of the host `_zlm_row!` (gated dense,
  lmax 0:6). Gates: full-sweep bitwise vs a keyed serial reference, repeat-run
  identity, frozen inactive sites, the drift gate, and the exact dimer
  statistics gate. New deps: KernelAbstractions, Adapt (hard — the CPU-backend
  gates run in the default suite); CUDA appears only in the `bench/gpu` env.

### Changed

- **`run_pt` lanes now synchronize pairwise at exchange boundaries** instead of
  through a whole-ladder barrier every segment (`pt-threads-determinism.md`
  P3/P4): with `ntasks ≥ 2` every lane runs as its own task for a whole async
  block (between checkpoint writes / phase ends) and an exchange boundary
  handshakes only the two lanes of each attempted pair — a straggling lane stalls
  its neighbors, not the ladder. The exchange uniforms are pre-drawn per block in
  the serial consumption order, so results are **bit-identical** to before and to
  `ntasks = 1` (trajectories, checkpoints, and resume are unaffected). `ntasks`
  values ≥ 2 no longer chunk lanes (they all mean "one task per lane"; the Julia
  scheduler multiplexes when rungs exceed threads). A dying lane task poisons the
  block so the original exception surfaces (wrapped in `@sync`'s
  `CompositeException`) instead of livelocking. Measured:
  ~5–13 % on mixed P/E cores, largest at `exchange_interval = 1`
  (`.claude/bench_log.md` #7).

- **Pair/triplet fast paths in `site_coeffs!`** (`docs/specs/hamiltonian-tiling.md`
  T5): a body-2 (body-3) site program has one (two) factors per entry, always on
  the other member slots, so the constructor now precomputes the hoisted neighbor
  columns per adjacency entry (`site_col`/`site_col2`, with the sign of `site_col`
  tagging the path so the pair path stays exactly as fast as before) and the
  factor rows per entry (`pent_row`/`pent_row2`); the kernel walks purely
  sequential streams plus the `zrows` gathers. **Bitwise identical**
  (`(1.0·z₁)·z₂… ≡ z₁·z₂…`, same skip and accumulation order — run-level
  fingerprints match) — trajectories, fixed-seed tests, and checkpoints are
  unaffected. Roughly halves `site_coeffs!` on both the pair-heavy and the
  triplet-heavy (production l044-like `nbody = 3`) regimes and cuts Nd₂Fe₁₄B-scale
  sweeps ~2× on top of the color-parallel numbers; an adjacency locality sort was
  measured first and rejected (≤2 % — the cost is the indirection chain, not cache
  capacity). New triplet-heavy bench fixture `nd2fe14b3_model`. Numbers in
  `.claude/bench_log.md` (#5, #6).

- **Color-parallel sweeps** (`sweep_tasks` on `run_mc` / `run_pt` /
  `find_ground_state`) — **breaking for reproducibility and for the checkpoint
  schema**. The update sweeps now scan the sites in the Hamiltonian's color-class
  order (a greedy proper coloring of the "shares a cluster instance" conflict
  graph, precomputed in the `TiledHamiltonian` constructor): sites within one
  class have exactly independent single-spin kernels, so a class is updated by
  `sweep_tasks` concurrent tasks with a barrier between classes. Every site now
  owns its proposal/accept RNG stream (`ChainState.site_rngs`, derived from the
  chain RNG) and the accepted ΔE are staged per site and reduced in fixed class
  order, so the trajectory is **bit-identical for any `sweep_tasks`** (and any
  `ntasks` / thread count — gates in `test/unit/test_parallel.jl`, spec
  `updates-stationarity.md` U1). Measured: ~3× per sweep at 4 performance cores
  for ≳4000-site models (`.claude/bench_log.md` #4); keep
  `ntasks · sweep_tasks` within the thread count under PT. Breaking: the scan
  order and RNG streams change every fixed-seed trajectory, and checkpoints move
  to **schema v2** (adds `chain/site_rngs`, `plan/sweep_tasks`; v1 files are
  rejected).

- **Sweeps and the minimizer are now allocation-free**: the tesseral-row
  tabulation (`_zlm_row!`) and the minimizer gradient (`_gradient!`) thread a
  reusable associated-Legendre recursion workspace (new `plm` buffer on
  `SweepScratch` / `_MinimizeScratch`) through to SCEFitting's new cache-threaded
  `Zlm_unsafe` / `grad_Zlm_unsafe`, eliminating the 2 heap allocations per
  harmonic evaluation that LegendrePolynomials' `dnPl` default made on every call
  (the whole of the remaining sweep allocations after the contraction-program
  change). Values are **bit-identical** — trajectories, fixed-seed tests, and
  checkpoints are unaffected. Numbers in `.claude/bench_log.md` (#3).

- **Energy kernels rebuilt on precompiled sparse contraction programs**
  (`docs/specs/hamiltonian-tiling.md` T5): the `TiledHamiltonian` constructor now
  flattens each template's nonzero `folded` entries into flat index/weight arrays,
  and `site_coeffs!` / `_total_energy` walk those instead of the rank-generic
  contraction — eliminating the per-instance dynamic dispatch (~2–3 heap
  allocations per instance per visit) that dominated every sweep. The programs are
  built in the reference kernels' exact loop and operation order, so results are
  **bitwise identical**: trajectories, fixed-seed tests, and checkpoints are
  unaffected (not a reproducibility-breaking change). The rank-generic kernels
  remain in `energy.jl` as the readable reference, pinned by a new bitwise
  equivalence gate in `test/unit/test_energy.jl`. Numbers in
  `.claude/bench_log.md` (#2).

- **The bit-reproducibility promise is now explicitly scoped** (new authoritative
  section: `docs/specs/pt-threads-determinism.md` P6): guaranteed for a fixed seed
  within one package + Julia version and independent of the thread count — a
  testing discipline (resume ≡ uninterrupted, `ntasks` race gate, non-flaky CI),
  **not** a cross-version guarantee. Julia does not stabilize `rand`/`randn`
  streams across releases; RNG-stream-changing package improvements remain allowed
  (recorded as breaking); ULP-level summation-order details of derived observables
  are outside the promise. README, module docstring, parallelism guide, and
  CLAUDE.md now point at the scoped statement. No code change.

- **Inactive (non-magnetic) sites are now skipped and excluded** (e.g. boron in
  Nd₂Fe₁₄B — any site no cluster instance touches, including sites whose SALC
  coefficients all fitted to zero; `coef == 0` terms are dropped in the
  `TiledHamiltonian` constructor). Such sites are flagged
  (`TiledHamiltonian.site_active`, `n_active`) and: the update sweeps skip them
  (previously they free-random-walked — every move accepted — consuming RNG,
  inflating the measured acceptance, and biasing the adaptive step toward the
  ceiling), the standard observables exclude them (previously `:m`/`:absm`/`:m2`/
  `:m4`, χ and the Binder cumulant were diluted by their random directions, and
  `:sublattice_m` reported their noise; inactive sublattices now report exactly
  zero), per-site normalizations (C, χ, evaluable `n`) use `n_active`, and
  renormalization plus the ground-state descent keep them **bitwise frozen** at
  their initial direction. They remain part of the state (`n_sites`, `config`,
  checkpoints, the `3 × n_atoms` I/O layout). **Breaking for reproducibility**:
  models containing inactive sites consume a different RNG stream, so fixed-seed
  trajectories and acceptance statistics differ from previous versions
  (all-magnetic models are unaffected). Conventions recorded in
  `docs/specs/updates-stationarity.md` (U1) and
  `docs/specs/binning-observables.md` (B3); gates in `test/unit/test_inactive.jl`.

- `run_mc` / `run_pt` default `seed` is now drawn fresh per call
  (`rand(UInt64)`) instead of the fixed `0`, so repeated default runs are
  independent samples rather than silent duplicates. Reproducibility is opt-in
  (pass an explicit `seed`) and never lost: the seed actually used is recorded
  in the result and in checkpoints.

### Added

- Benchmark suite (`bench/`, own environment): bottleneck-oriented scripts —
  `bench_kernels` (the single-spin attempt decomposed: `_zlm_row!` /
  `site_coeffs!` / `delta_energy`, plus `_total_energy` and the diagnostics
  paths), `bench_sweeps` (ns/attempt and allocs/sweep for Metropolis and
  overrelaxation), `bench_tiling` (`TiledHamiltonian` construction + retained
  memory), `bench_run` (`run_mc` with measurement-overhead isolation; `run_pt`
  thread scaling), `bench_minimize` (gradient pass, BB descent, multi-start
  search), and `bench_profile` (line-level `Profile` tree/flat reports per
  target). Two fixtures span the kernel regimes: a 2-atom bcc Fe `l = 1` model
  (light kernel, large lattice) and a synthetic-coefficient Nd₂Fe₁₄B model
  (`bench/assets/nd2fe14b.toml`, ~9400 terms, site adjacency ~276 — the real
  l02 production regime). Baselines and first findings (per-instance
  dynamic-dispatch allocations in the energy kernels) in `.claude/bench_log.md`.
- Ground-state search: `minimize_energy` (deterministic Riemannian
  Barzilai–Borwein projected-gradient descent on the sphere product, nonmonotone
  Armijo safeguard, no optimizer dependency, no RNG in the descent) and
  `find_ground_state` (multi-start simulated annealing with optional thermal
  cycling — `cycles`/`reheat` — polished by the same descent; threads-parallel and
  bit-identical for a fixed seed regardless of `ntasks`), both returning
  `GroundStateResult` with the per-start energy table as a degeneracy diagnostic.
  Includes the PT-polish recipe (`inits = pt.final_configs, anneal_sweeps = 0`),
  an executed docs guide with figures, and the decision record
  `docs/specs/ground-state-search.md`.
- Docs: a parallelism guide — how the Threads lane pool works, the explicit
  limits (no MPI/GPU; one PT ladder is bounded by one node), and multi-node
  recipes (`Threads.@threads` over temperatures, SLURM job arrays with blind
  `resume` retries — backed by a new idempotent-resume gate).
- Docs: executed figures in the parallel-tempering guide — four annealed chains vs
  four PT runs on a random-anisotropy model (one chain freezes into a basin 100×
  its own error bar away; PT rescues every seed), and the ladder diagnostics
  (swap acceptance collapsing with system size, recovering with rung count). The
  docs build now runs with `-t 4` so PT examples sweep lanes in parallel.
- Docs: an executed tutorial (`tutorials/cubic_heisenberg.md`) — the ferromagnetic
  transition of a simple-cubic classical Heisenberg model, with figures (energy,
  specific heat, magnetization, susceptibility, Binder-cumulant crossing on the
  literature ``k_BT_c/|J| = 1.443``) computed at docs-build time, plus a
  user-defined staggered-magnetization observable on the antiferromagnetic
  counterpart. CairoMakie + Spglib added to the docs environment.
- Cell reduction: `reduce_cell` / `ReducedCell` — re-express a model fitted on a
  supercell in a user-chosen smaller (or re-based, non-diagonal `M` included) cell,
  after verifying the lattice relation, the atomic mapping, and that every fitted
  term has its full set of translation copies; `TiledHamiltonian(red; dims)` then
  counts `dims` in reduced-cell units, decoupling finite-size checks from the
  training-cell granularity (`docs/specs/cell-reduction.md`).
- Geometry/I-O helpers: `supercell_crystal` (site ordering matched to
  `TiledHamiltonian`), `to_matrix` / `from_matrix`.
- Checkpoint/restart: `checkpoint`/`checkpoint_interval` on `run_mc`/`run_pt` +
  `resume` — versioned plain-data JLD2 schema (model fingerprint, Xoshiro words,
  accumulator cascades), atomic writes, and bit-identical resumed runs
  (`docs/specs/checkpoint-schema.md`).
- `run_pt`: replica exchange (parallel tempering) over threads — one lane per
  ladder rung, payload-swap exchanges every `exchange_interval` sweeps
  (thermalization and measurement alike), per-lane adaptive steps and
  accumulators, and bit-identical results for a fixed seed regardless of
  `ntasks`/thread count (`docs/specs/pt-threads-determinism.md`).
- Overrelaxation sweeps (`or_per_metropolis`): deterministic involutive reflection
  about the local `l=1` field axis + Metropolis correction — exactly microcanonical
  on pure-`l=1` channels, exact for any body order via the accept step
  (stationarity: `docs/specs/updates-stationarity.md`).
- `run_mc`: single-temperature and warm-started ladder (annealing) runs of
  single-spin Metropolis with the exact `ΔE = c_s·ΔZ` kernel, symmetric
  flip+Rodrigues proposal, thermalization-only adaptive step (frozen during
  measurement), periodic renormalize + energy re-anchoring with drift tracking,
  and bit-reproducible seeding. Results as `MCResult` / `TempResult` with a
  summary-table printer.
- Composable measurement layer: `Observable` / `Evaluable` with the standard set
  (`E`, `E²`, `m`, `|m|`, `m²`, `m⁴`, per-sublattice magnetization) and derived
  `C/k_B`, |m|-connected `χ`, Binder `U = ⟨m⁴⟩/⟨m²⟩²` (conventions:
  `docs/specs/binning-observables.md`).
- Error analysis: streaming `LogBinner` (log-binning plateau errors + `τ_int`,
  O(levels) memory), `BinStore` + leave-one-bin-out `jackknife` for derived
  quantities.
- `TiledHamiltonian`: the fitted SCE unfolded onto an `N₁×N₂×N₃` supercell from the
  public `multipole_terms` introspection (per-site integer `shifts`, toroidal wrap),
  with template-once + CSR-instance memory layout and the `(4π)^(body/2)` scale
  applied exactly once. Supports self-image (`AllImages`) clusters when `dims` keeps
  the images distinct sites.
- The 4-function energy contract: `total_energy`, `site_coeffs!` (leave-one-out
  coefficients — exact single-spin `delta_energy` for any body order), and
  `site_gradient` (on-sphere, via `Harmonics.grad_Zlm_unsafe`).
- Package scaffold: module skeleton, temperature control (`KB_EV`, `resolve_kt` —
  kelvin XOR model-energy-unit keywords), test harness (`TEST_MODE`
  default/all/unit/aqua/jet with Aqua + JET), Documenter docs skeleton.
