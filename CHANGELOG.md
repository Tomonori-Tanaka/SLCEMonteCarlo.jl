# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — BREAKING: the fourth naming batch (upstream)

- **`SLCE.SlotRef` → `SLCE.Slot`.** It is not a reference to a slot — it *is* the
  slot (site index + site factor), and `Ref` means a mutable box in Julia. This
  package reads it in the adjacency builder (`hamiltonian.jl`), the cell reduction
  (`reduce.jl`), the fixtures and the GPU smoke bench; all follow. Nothing else in
  SLCE.jl's fourth batch reaches this package.

### Added — the documentation is published

- **<https://tomonori-tanaka.github.io/SLCEMonteCarlo.jl/dev/>** — the Documenter site is
  now deployed to GitHub Pages by the `documentation build` CI job (`deploydocs`
  in `docs/make.jl`, `permissions: contents: write` on the job). It was being
  built on every push and then thrown away.
- **Per-line source links work**: `remotes = nothing` / `edit_link = nothing` are
  gone in favour of the real repository, so every docstring on the site links to
  its own lines on GitHub and each page has an "Edit on GitHub" link.
- README carries a docs badge, a CI badge and the site URL.

### Changed — BREAKING: the family-wide naming batch

Follows SLCE.jl's third naming batch (see its `CHANGELOG`); landed in all four
repositories together.

- **`MultipoleTerm` / `multipole_terms` → `SpinMultipoleTerm` /
  `spin_multipole_terms`** (upstream rename; `TiledHamiltonian`'s frozen pure-spin
  path and every fixture follow).
- **`has_disp` is now a method of `SLCE.has_disp`, not a second generic.** This
  package defined its own `has_disp(::TiledHamiltonian)` while the core defined
  `has_disp(::SiteDecor)`; they asked the same question at two granularities but were
  different functions, so a user loading both packages had two `has_disp` and no way
  to call one of them unqualified. Imported and extended, exactly as `n_atoms` already
  was for `ReducedCell`. Behaviour is unchanged; `SLCEMonteCarlo.has_disp(H)` still
  resolves.
- **`KB_EV` / `resolve_kt` moved to `SLCE`** and are re-exported / re-published from
  here unchanged — `export KB_EV` and `public resolve_kt` still hold, and the values
  are bit-identical (the two definitions were character-for-character the same). This
  package no longer owns a second copy of the kelvin ↔ model-energy conversion.
- **Prose: `SCE` → `SLCE`, "spin–lattice cluster expansion".** The module docstring
  and README expanded the acronym as "symmetry-adapted cluster expansion", which was
  wrong — *symmetry-adapted* describes the basis, not the expansion. No identifier
  changed. `docs/specs/` decision records keep their original wording.

### Added — the device displacement sweep (M4 slice 3f/3)

- **`gpu_displacement_sweep!`** (exported) — the displacement half of the device
  Metropolis path. `_disp_kernel!` is `_metro_kernel!` with the other channel's
  proposal, row filler, row range and write-back: same workgroup shape, same lane-1
  ownership, same lane-ordered fold, same accept rule, walking `disp_sites` with
  `lo:hi = nlm+1:nrows`. The trial row is filled by `_disp_rows_device!` (3f/1) from a
  `@localmem` solid-harmonic batch workspace, as the host fills it from
  `SweepScratch.rbuf`; the serial twin `_displacement_sweep_keyed_ref!` extends the
  kernel ≡ reference bitwise gate to both channels.
- **`gpu_run_sweeps!` compounds the two sweeps** — one Metropolis pass then
  `disp_per_metropolis` displacement passes — and resolves the count through
  `_resolve_disp_passes`, the host drivers' own function rather than a copy of its
  rule: `nothing` means one pass on a joint model and none on a pure-spin one. This
  **replaces the `fixed_lattice` opt-in of 3f/2**; freezing the lattice is now
  `disp_per_metropolis = 0`, the same acknowledgement in the vocabulary `run_mc`
  already uses. A lattice-only model has its Metropolis pass skipped by the driver
  (the spin primitive still refuses it); a model with nothing to sweep in either
  channel throws.
- **Per-move-kind RNG counters and slots.** The displacement proposal takes Philox
  slots 3–5 against the spin proposal's 0–2, and `GPUChainState` counts `sweep_index`
  and `disp_index` separately. Both are needed, for different reasons: disjoint SLOTS
  keep the two moves independent within one compound step (a shared accept uniform
  would make them accept and reject together), and a separate COUNTER keeps several
  displacement passes in one step from drawing the identical shift.
- **One trial-row convention, replacing 3f/2's two.** `znew` is `H.nrows` long
  everywhere — `SweepScratch.znew` on the host, `@localmem Float64 NROWS` in both
  kernels — and indexed absolutely, with only `lo:hi` written and read. The
  block-relative kernel buffer of 3f/2 saved a few floats per workgroup and bought
  the silent hazard the review had flagged for exactly this slice. Pure-spin device
  trajectories are unchanged bit for bit (re-verified against `5471379`).
- Gated by: the slot map's disjointness and the exact `u + step_u·(g₁,g₂,g₃)` shift;
  the displacement sweep ≡ its keyed reference bitwise over five sweeps on both joint
  fixtures × two workgroup sizes, with — independently — spins, spin rows and
  displacement-inactive sites bitwise unchanged; compound scheduling (both counters,
  both attempt tallies, several passes per step, the pure-spin refusal, the
  lattice-only skip-and-throw pair); the joint drift gate with the lattice now moving,
  plus the frozen-lattice conditional and its centre-of-mass gauge; and the
  **Einstein oscillator against its closed form**, `⟨|u|²⟩ = 3kT/(2a)` to 5 % — the
  one displacement gate that checks the device chain against an external truth rather
  than another implementation of the same arithmetic.
- Validated outside the suite (scratch tier, recorded in G8): on a **bounded** joint
  fixture the device compound chain and the host `metropolis_sweep!` +
  `displacement_sweep!` chain agree to 0.12σ in `⟨E⟩` and 0.0σ in `⟨|u|²⟩` over four
  seeds each, with acceptance rates matching to three digits.
- **A100 smoke PASSED** (kugui job 867813, A100-SXM4-40GB / CUDA 12.6 / CUDA.jl
  6.2.1): all 16 checks green on real hardware — `_disp_kernel!` compiles and runs
  (the G7 `@index` failure mode did not recur), channel isolation bitwise, device
  determinism, drift 7.1e-15 / 0.0, the Einstein closed form to −0.02 %, and
  CUDA ≡ KA-CPU in distribution at 0.97σ / 1.19σ with acceptance rates identical to
  three digits. New: `bench/gpu/smoke_joint.jl` and `bench/gpu/job_joint_smoke.pbs`
  — a separate script from `bench_gpu.jl`, which runs pure-spin fixtures and so never
  launches the displacement kernel at all. It refuses to fall back to the CPU
  backend, since a smoke that silently ran on the host would report success for
  precisely the thing it exists to check.
- **`has_disp` (row layout) vs `n_disp_active` (sites), again.**
  `_resolve_disp_passes` gates on the first, `gpu_displacement_sweep!` on the second,
  and a joint basis whose displacement couplings all fitted to zero separates them:
  the driver's own default threw on such a model, which the host runs fine as a no-op
  sweep. It now resolves first — so an explicit displacement pass on a Hamiltonian
  with no displacement *rows* is still refused — and clamps to zero passes when there
  is no displacement-active *site*.
- **The device driver carries the U8 diagnostics too**, now that the device chain
  moves the lattice: `_warn_gpu_escape_cadence` mirrors `run.jl`'s
  `_warn_escape_cadence` (same `_escape_min_checks` arithmetic) and additionally warns
  when `renorm_interval ≤ 0`, which on this path is legal and switches off the escape
  detector and re-centring together. Step adaptation stays out of scope, but the guide
  now says so for `step_u` as well as `step`, says where the acceptance data actually
  lives (`gst`, not `st` — `to_host!` does not copy counters), and warns that both RNG
  counters restart at zero, so a re-uploaded "resume" under the same seed replays
  rather than continues.
- **Two device-side shape guards, not one.** Widening the trial row to `NROWS` turned
  a wrong `(lo, hi)` from a bounds error into an uninitialized-`@localmem` read, so
  the spin kernel zero-fills its unused half; and the constructor now also rejects a
  `RowLayout` whose displacement blocks do not tile `nlm+1:nrows` (the write-back
  copies the whole block), the mirror of 3f/2's padded-spin-block assertion.
- Two gates were softer than they looked and were sharpened: the Einstein statistics
  gate now starts from the clamped-ion state rather than a random lattice already
  within 20 % of the answer (rtol 0.05 → 0.04), and the renormalizing arm of the joint
  drift gate uses an interval that does not divide the sweep count, so the run no
  longer ends on the `_renormalize!` that re-anchors the energy it is about to check.

### Added — joint device tables and a joint-safe spin sweep (M4 slice 3f/2)

- **`GPUTiledHamiltonian` accepts joint spin–lattice Hamiltonians.** The
  `_require_spin_only` guard is gone from the constructor; `disp_lmax ≤ 6` joins the
  existing `lmax ≤ 6` gate. `_GPUTables` gains the `RowLayout` displacement blocks
  (`fac_k`/`fac_l`/`fac_start`) and, in place of the single `color_sites`, one color
  list per channel.
- **`GPUChainState` carries the lattice.** `zrows` is the full `nrows × n_sites` row
  table, and the state holds `disps` and `step_u` alongside the spins plus
  per-move-type acceptance counters; `to_host!`/`_from_host!` round-trip the
  displacements with the rest.
- **Row-range-restricted ΔE.** `_entry_walk_partial` takes `lo:hi` — the device form
  of `delta_energy(c, zold, znew, lo, hi)`. A single-channel move rewrites one block,
  the rows it did not touch contribute `c_k · 0`, and `znew` holds that block alone, so
  entries targeting the other block are skipped rather than read out of a
  half-written buffer. The spin kernel passes `1:(LMAX+1)²`, which **is** `H.nlm`.
- **Per-channel sweep schedules.** `_channel_colors` slices the one coloring by
  `site_has_spin` / `site_has_disp`, moving the host sweeps' skip predicates to launch
  time: a joint model can have a site active in one channel only (a
  force-constant-only ligand carries no moment), and visiting it in a spin sweep would
  be always-accepted noise that biases the acceptance statistics.
- **Nothing changes on a pure-spin model, by construction.** There `site_has_spin ≡
  site_active`, so `spin_sites == H.color_sites` and `spin_ptr == H.color_ptr`
  verbatim, `disp_sites` is empty, and every entry is in range — the pre-M4 launch
  schedule and the pre-M4 walk, bit for bit (asserted field by field).
- **The gradient path stays spin-only.** `_entry_walk_grad` has no row range and
  differentiates with respect to the spin direction alone, so the guard moved from the
  shared constructor onto **every** gradient entry point — `GPUGradientScratch`,
  `gpu_zlm_rows!`, both `gpu_energy_gradient!` overloads, `_gpu_gradient_rows!`,
  `_gradient_lane_ref!` — and each needs its own, for two reasons: `gsc` is no
  evidence about the Hamiltonian it is handed with (a scratch from the spin
  restriction of the same model has the matching `(nlm, n_sites)` shape), and the
  documented `refresh_zrows = false` fast path skips `gpu_zlm_rows!`. Unguarded, such
  a call walks the joint program table against a spin-sized gradient row — an
  `@inbounds` out-of-bounds read returning garbage instead of throwing. Each entry
  point is gated separately.
- **`gpu_run_sweeps!` refuses the wrong ensemble by default.** (`fixed_lattice` was
  superseded within this same unreleased cycle by 3f/3's `disp_per_metropolis`, which
  encodes the same constraint in the host's vocabulary; the reasoning below is what
  carried over.) On a joint model the
  device moves the spins only, so the chain samples `π(ê | u)` at the uploaded
  lattice — not the joint distribution — and every displacement observable off such a
  run is conditional on a lattice nobody equilibrated. The driver now **throws** on a
  joint `H` unless the caller passes `fixed_lattice = true`, the device analogue of
  `_resolve_disp_passes`' refusal (run.jl) to let `disp_per_metropolis` default to 0.
  `gpu_metropolis_sweep!` stays unjudged — it is the primitive, and interleaving host
  `displacement_sweep!`s around it is how a joint chain is driven today.
  "Fixed" is up to the centre-of-mass gauge: with `renorm_interval > 0`,
  `_renormalize!` re-centres each displacement component, so `st.disps` moves by a
  rigid per-component shift recorded in `com_removed` (gated: `disps + com_removed`
  reproduces the upload exactly, and with renormalization off `disps` is bit-identical).
- **A sweep with nothing to sweep now throws.** `_require_spin_sites` (hamiltonian.jl,
  the spin-channel mirror of `_require_disp`) rejects a Hamiltonian with
  `n_spin_active == 0` — a lattice-only model, or a joint basis whose spin couplings
  all fitted to zero. Such a run previously "succeeded" having attempted no move,
  reporting a `0/0` acceptance over a bit-for-bit unchanged state.
- **The device SPIN block width is asserted, not assumed.** The kernel spells it
  `(LMAX+1)²` (trial row, ΔE range, write-back) where the host reads `H.nlm`;
  `row_layout` makes those equal, but `TiledHamiltonian` accepts a caller-supplied
  `RowLayout` and checks `disp_offset` only against the slot rows. A padded layout
  would give a truncated device ΔE and a partial write into the displacement block, so
  `GPUTiledHamiltonian` now rejects one.
- **The walk's trial-row convention is block-relative**, unlike `delta_energy`'s
  absolute indexing of a full-length buffer — a kernel's trial row is `@localmem`
  sized to one block. The argument is named `znew_block` and the difference is written
  down in three places, because the check cannot compile on a device and a
  displacement kernel written by analogy would read the spin block as displacement
  rows with no shape error to catch it.
- Gated by (joint fixtures = the hand-built `_channel_split_terms`, whose atom 1 sits
  in both channels and atom 2 in the displacement channel alone, and a fitted
  `_joint_model`): the pure-spin schedule identity; the channel lists against the host
  predicates with `n_spin_active < n_active` so the split is not vacuous; the
  restricted walk vs `site_coeffs!` + `delta_energy(…, 1, nlm)` with two non-vacuity
  counters; five sweeps kernel ≡ keyed reference bitwise, plus — independently —
  displacements, displacement rows and spin-inactive directions bitwise unchanged; the
  drift gate with renormalization off and on; and the gradient rejections.

### Added — device displacement rows (M4 slice 3f/1)

- **`_solid_row_device!` / `_disp_rows_device!`** (`src/gpu/disp_device.jl`) — the
  displacement-channel counterpart of the device tesseral row, the first piece of the
  GPU port of the displacement channel. Unlike the tesseral case this is a
  near-verbatim copy rather than an operation-order transcription: the upstream
  `SLCE.SolidHarmonics._solid_harmonics_impl!` is already pure scalar arithmetic with
  no throws and no allocation, so only the gradient half and the `lmax` validation are
  dropped and `lmax` is lifted to a `Val` for static stack buffers.
- **The bitwise scope shrinks, deliberately** (decision record G8). The solid-harmonic
  row is gated bitwise against upstream. The `|u|^{2k}` prefactor is not: `r2 =
  dot(u, u)` is a mul-add chain whose last bit depends on LLVM's FP-contraction
  decision, which differs between a host function that *calls* the harmonic batch and
  a device function that *inlines* it (as a kernel requires) — ~22 % of random `u`, one
  ulp, amplified by the exponent; reordering and an explicit `muladd` transcription
  both fail to remove it. A single shared `r2` would fix it but is an upstream
  `SLCE.site_rows!` convention change, and bitwise identity here buys **test
  sharpness, not accuracy**: one ulp is relative 1e-16 against MC statistical errors of
  ~1e-3, and the gates that catch bugs (kernel ≡ keyed reference, serial ≡ parallel)
  are device-vs-device and stay bitwise. What is lost is G4's "the renormalization host
  round-trip is seam-free" on `k ≥ 1` blocks, already covered by the drift gate.
- Gated by: bitwise equality with `SLCE.SolidHarmonics.solid_harmonics!` over
  `lmax = 0:6` × (`u = 0`, axes, eight decades of magnitude, 2000 seeded), both
  directly and through a KA-CPU kernel; `_disp_rows_device!` bitwise on every `k = 0`
  row and within `kmax + 1` ulp elsewhere, over three layouts spanning the `(k, l)`
  block structure; the `u = 0` exactness of the polynomial form; `_disp_layout_tables`
  against the layout it flattens; and a pure-spin layout as a no-op.

### Added — cell reduction for joint models (M4 slice 3e)

- **`reduce_cell` accepts joint spin–lattice models.** The verified re-expression of a
  supercell-fitted Hamiltonian in a user-chosen smaller cell — the mechanism that
  decouples the finite-size-scaling grid from the fitting cell — was pure-spin only:
  it read the model through `multipole_terms`, which now refuses a basis with a
  displacement sector. It reduces through `SLCE.DecoratedTerm` instead, so a joint
  model can be tiled at any reduced-cell multiple, and
  `reduce_cell(crystal, terms, layout, sub_lattice)` takes a hand-built decorated list.
- **`ReducedCell{T}`** is now parameterized on the term type and carries a `layout`
  field: `ReducedCell{MultipoleTerm}` with `layout === nothing` (the row numbering is
  derived from the terms, exactly as before M4), `ReducedCell{DecoratedTerm}` with the
  model's `SLCE.RowLayout` — a slot's `(channel, k, l)` can only be placed by the
  numbering the model itself defines. `TiledHamiltonian(red; dims)` dispatches on that
  and now forwards `fixed_reference`.
- **What generalizes, and what does not.** A pure-spin term's tensor axis *is* its
  site; a decorated term's is a slot, several may share a site, and `folded`'s axes are
  slots. The canonical anchored form therefore applies the site sort to `atoms`/`shifts`
  and — through `invperm` — to every slot's site, then re-sorts the slots into canonical
  `(channel, site, k, l)` order with `folded` carried by `permutedims`. The physics is
  unchanged: a lattice translation moves no displacement vector and rotates no spin, so
  orbit members still share `coef`, `scale` and the aligned `folded` exactly.
- **`scale` is carried verbatim, never re-derived.** Deriving it from `length(atoms)`
  is precisely the silent mis-scaling `DecoratedTerm.scale` exists to prevent — on the
  fixture, two of three hand-built terms have a `scale` that `(4π)^(body/2)` gets
  wrong. Every copy on one anchored key must declare the identical value, reported as
  its own error (two spellings of one number — `sqrt(4π)` vs `(4π)^0.5` — are 1 ulp
  apart and would otherwise surface as a periodicity failure). Deliberately not
  checked: whether the declared value follows the rule at all, since
  `TiledHamiltonian` treats the field as declared data too.

### Fixed — the periodicity census was weaker than its own contract

- **`reduce_cell` now verifies the translation copies per coset, not in total.** The
  census demanded `count % |det M| == 0`, which a class living in a *single* coset
  satisfies: four copies of one term all anchored in the same coset of a `|det M| = 4`
  reduction passed, and were emitted as one representative — that is, as a term sitting
  in *every* reduced cell. A different Hamiltonian, silently. It now requires the same
  count `q` in **each** coset, with the coset read off the absolute anchor by exact
  integer arithmetic (`mod.(adj(M)·σ₁, |det M|)`, a complete invariant of `σ + Mℤ³`).
  Not reachable from a fitted model's own term list; reachable by stitching two lists
  together — and "verified, never assumed" is the entire claim of this function.
- Degenerate hand-built terms now raise `ArgumentError` instead of escaping as a
  `BoundsError` (a body-0 term) or a `DimensionMismatch` (a `folded` extent that
  disagrees with its axis's `l`).
- Gated by: exact field-for-field recovery of a hand-built mixed-channel chain through
  unfold → reduce, with the `(4π)^(n_spin_slots/2)` pin re-derived from each term's own
  slot list; a **3-body** mixed chain whose translation copies are genuine 3-cycles —
  the only shape that can distinguish `invperm(perm)` from `perm`, every ≤ 2-body site
  permutation being an involution, which is why the rest of the suite (all 2-body)
  cannot gate the new relabel; a non-diagonal `M` carrying a displacement axis; a fitted
  joint model reduced 2× against `predict_energy` and a non-commensurate tiling, plus an
  **odd** reduced-cell count; a pinned non-vacuity count (exactly 29 of that model's 62
  terms need both permutations); and the refusals — broken coefficient, lopsided cosets,
  scale disagreement, body vs `atoms`, `folded` rank and extent, out-of-cluster slot
  site, and the `ReducedCell` constructor's own two invariants.

### Added — displacement observables and the harmonic screen (M4 slice 3d)

- **`MCView`** — the single argument every [`Observable`](@ref) now receives
  (`v.config`, `v.disps`, `v.energy`, `v.H`). **BREAKING**: `f(config, energy, H)`
  becomes `f(v)`; every observable definition, here and downstream in
  `SLCEDynamics`, is rewritten. The sampled state grows with the model —
  displacements arrived in M4 — and a positional contract would break every
  observable ever written each time it did.
  `disps` is **emptied** on a Hamiltonian with no displacement channel, whatever the
  producer passed, so a displacement observable throws on a pure-spin model instead
  of reporting a confident zero.
- **Displacement observables** on a model with a displacement-active site: `:u2`
  (mean square displacement — a *binned* observable with an autocorrelation-aware
  error bar, unlike `TempResult.disp_rms`), `:u4`, and `:sublattice_u2` /
  `:sublattice_u4` per training-cell atom (`:sublattice_u2` is the isotropic
  Debye–Waller input `B_a = 8π²⟨u²⟩_a/3`). Every one is a moment of `u_s − ū_c`:
  the centre of mass is removed **inside the observable**, along the flat directions
  only, because the sampler re-centres at renormalization points while measurements
  fire far more often — relying on the sampler's frame biases `⟨u²⟩` by an amount
  linear in `renorm_interval` (over 100 % at the defaults on a small cell), always
  positive and shrinking with system size, so it would read as a finite-size effect.
- **`:u_moment_ratio`** = `⟨u⁴⟩/⟨u²⟩²`, the anharmonicity screen. Read against
  **temperature**: for a harmonic model every `σ_s² ∝ T`, so the ratio is
  temperature-independent whatever the crystal. Its *level* is
  `(5/3)·mean(σ⁴)/(mean σ²)² ≥ 5/3` by Jensen — 5/3 only when every
  displacement-active site samples the same isotropic Gaussian, and a two-sublattice
  Einstein crystal with a 4× stiffness contrast measures 2.26 while being exactly
  harmonic. The clean 5/3 test is the per-sublattice ratio, which is why both
  sublattice moments are measured. Gated on a heterogeneous fixture against the
  Jensen prediction and on the per-sublattice ratio against 5/3.
- **`force_constant_matrix` / `harmonic_stability`** — the displacement Hessian of
  the **tiled** Hamiltonian at a given spin configuration, by central differences of
  `total_energy`, and its spectrum. This is the screen that runs *before* a joint
  run, complementing the escape detector, which only reports the case that has
  already diverged: on the package's own unbounded fixture it finds negative
  eigenvalues without sampling anything. An eigenvalue below `−tol` is a proof of
  failure; a clean spectrum proves nothing (deciding global non-negativity of a
  quartic form is NP-hard), and the docstring says so. Deliberately finite
  differences rather than a second analytic derivation of the force constants:
  `SLCE.force_constants` already owns that convention, and this one answers the
  different question of what the sampler actually evaluates.
  The returned `tol` is **measured, not modelled**: a translation-invariant model has
  `3·n_disp_comps` exact zero eigenvalues and finite differences scatter each across
  zero, so an untolerated `count(< 0, λ)` reports the documented proof of failure on
  a healthy model about half the time. The floor is read off the residual of the
  rigid-shift directions the construction gate certified flat — where the exact
  answer is zero, so whatever comes out is pure roundoff — and it tracks the acoustic
  scatter to ~12 % across three decades of `step`. `acoustic_residual` is likewise
  normalized **per displacement-coupling component**, not against the global scale,
  so a dominant or pinned component cannot mask a broken sum rule in another.
- **`MCResult.final_disps` / `PTResult.final_disps`** — the chain's last displacement
  configuration, so a continuation run can warm-start from it instead of silently
  restarting at the clamped-ion state. Empty on a pure-spin model.

### Added — the drivers take the displacement channel (M4 slice 3c, part 3)

- **`run_mc` / `run_pt` now sample joint spin–lattice models.** The
  `_require_spin_only` guard they carried through slice 3c/2 is gone; a compound
  sweep is one Metropolis spin sweep, then `or_per_metropolis` overrelaxation
  sweeps, then `disp_per_metropolis` displacement sweeps. Any fixed composition of
  π-stationary kernels is π-stationary, so the counts are efficiency knobs rather
  than physics. Decision record: `docs/specs/updates-stationarity.md` U9.
- **`disp_per_metropolis = nothing`** resolves from the model — `1` on a joint
  model, `0` on a pure-spin one. A constant default of `0` would sample a joint
  model at its clamped-ion displacements, a different ensemble, with nothing saying
  so. Explicitly requesting `0` on a joint model selects that clamped-ion ensemble
  deliberately; requesting a nonzero count on a pure-spin model is an error, not a
  no-op. Pure-spin runs are therefore bit-identical to the pre-M4 sampler (the
  displacement sweep is never entered, so no randomness is consumed).
- **`step_u` gained its own adaptation** (`run_mc`/`run_pt` keyword, default
  `0.01`). `_adapt_step!` now drives both proposal widths, each on its own
  acceptance counters and its own clamp, and returns both. `step`'s clamp is the
  radian range `(1e-3, π)`; `step_u`'s `(1e-6, 1.0)` is a runaway guard on a
  length — on an unbounded-below model every displacement proposal is downhill, so
  an unclamped adaptation would grow the width geometrically and make an escape
  look like a well-tuned chain.
- **`run_mc`/`run_pt` gained `disps`** — the chain's initial displacements (a
  `3 × n_sites` matrix or a vector of 3-vectors, copied verbatim), defaulting to the
  clamped-ion state.

### Fixed — what lifting the spin-only guard exposed

- **`:specific_heat` was normalized by the wrong site count on a joint model.**
  `_finalize_stats` handed every evaluable `H.n_spin_active`, which was invisible
  while the two counts coincided (they do on every pure-spin model). On a joint
  model the total energy carries the lattice heat capacity — ≈ 1.5 `k_B` per
  displacement-active site — and on a displacement-only model the count is zero.
  `Evaluable` now declares a `scope` (`:energy` → `n_active`, `:spin` →
  `n_spin_active`, default `:spin`) and `standard_evaluables()` marks
  `:specific_heat` as `:energy`. Gate: an Einstein oscillator's `C = 1.5 k_B` per
  atom, exactly, at any temperature. Convention doc:
  `docs/specs/binning-observables.md` B3.
- **The default observable set produced a table of `NaN`s on a displacement-only
  model** (every magnetization observable is `0/0` there). `standard_observables(H)`
  now omits them, and the new `standard_evaluables(H)` — what the drivers default to
  — drops the evaluables that read them.
- **The escape detector was re-anchored only at the thermalization→measurement
  boundary**, so a temperature ladder carried the previous rung's anchor. Its
  anchors are r.m.s. values and `rms ∝ √T`, so a *bounded* model on
  `kT = [0.005, 1.0]` false-alarmed with a warning asserting the model is
  unnormalizable. `_reset_escape!` is now called at every phase boundary — the
  freeze, each temperature's thermalization entry, and `_reset_config!`.
- **The block test could not fire at the package defaults.** Three consecutive
  strikes need 15 renormalization checks in a phase; `sweeps_measure = 10_000` with
  `renorm_interval = 1_000` gives 10, leaving only the absolute 10× guard — which
  catches a fast escape and misses a diffusive one. Joint runs below the bar are now
  warned **up front**, and `TempResult.disp_checks` records what the phase got, so
  `escaped == false` can be told apart from *not screened*.
- **A saturated `step_u` clamp was silent** — which is the failure mode the clamp
  exists to prevent, since on an unbounded model the width pins at the ceiling during
  thermalization. Both bounds now warn at the freeze boundary, with the acceptance as
  the discriminator (ceiling + high acceptance ⇒ runaway; floor + low ⇒ frozen
  channel).

### Changed

- **BREAKING — `TempResult` gained six fields**: `acceptance_disp`, `final_step_u`,
  `disp_rms`, `disp_max`, `disp_checks`, and `escaped`. `disp_rms` is the phase
  **average** `√⟨|u|²⟩` (not a snapshot) and `disp_max` an extreme value over the same
  phase; `disp_checks` is how many observations both rest on, and whether `escaped`
  means anything. The `MCResult`/`PTResult` summary tables print the displacement
  columns on a joint run and flag escaped points inline. All are `NaN`/`0`/`false` on
  a pure-spin model.
- **BREAKING — `Evaluable` gained a `scope` keyword** (see Fixed above) and
  `standard_evaluables(H)` was added.
- **BREAKING — checkpoint schema v2 → v3**: the chain's `disps`, `com_removed`,
  `step_u`, six acceptance counters and the escape-detector accumulators, plus
  `plan/disp_per_metropolis` and `plan/step_u0`. `_read_chain` / `_write_chain` no
  longer refuse a joint Hamiltonian; a joint resume is gated bit-identical for MC
  and PT alike. v2 files are rejected by the version check (pre-release breaking
  change). Schema doc: `docs/specs/checkpoint-schema.md`.

### Added — the displacement sampler (M4 slice 3c, part 2)

- **`displacement_sweep!(st, H, β, sc)`** — one single-site displacement Metropolis
  sweep, in the same color-class order as the spin sweeps and with the same serial
  `SweepScratch` / parallel `Vector{SweepScratch}` forms and the same
  bit-determinism for any task count (P6). The proposal is an isotropic Gaussian
  shift of width `ChainState.step_u` — a **length**, in the model's units, not an
  angle — drawn from the site's own RNG stream. Sites with no displacement axis are
  skipped, so on a pure-spin model the sweep is a no-op that consumes no randomness.
  Decision record: `docs/specs/updates-stationarity.md` U7.
- **`ChainState` carries the displacements**: `disps` (Cartesian, one per site,
  all-zero and never read on a pure-spin model), the proposal width `step_u`, the
  `acc_disp`/`att_disp` counters, and `com_removed` — the accumulated centre-of-mass
  shift taken out of each displacement-coupling component, so a site's uncentred
  position is `disps[s] + com_removed[c]`. The constructor takes them as keywords
  (`disps`, `step_u`); `disps = nothing` starts from the clamped-ion state `u = 0`.
  Displacements and their re-centring record are part of the replica-exchange
  payload — a swap moves a whole physical state, frame included.
- **Re-centring.** `_renormalize!` now removes each displacement-coupling
  component's mean displacement before re-anchoring the energy, projected onto the
  directions the construction gate measured as flat. Along those the projection costs no
  energy; what it buys is that the displacement observables measure against a fixed
  frame instead of a diffusing one, and that the solid-harmonic rows stay
  well-conditioned. It
  lives in `_renormalize!` and **not** in the sweep on purpose: a mean over sites is
  an order-dependent reduction, and inside the barrier-separated color loop it would
  make the trajectory depend on `sweep_tasks`.
- **`delta_energy(c, zold, znew, lo, hi)`** — the row-range-limited form. A
  single-channel move rewrites one block of the row table, and the rows it did not
  touch contribute exactly zero, so restricting the sum is not an approximation; it
  also means the proposal buffer never has to hold the untouched half. `SweepScratch`
  now sizes `c`/`znew` to `H.nrows` and carries the solid-harmonic batch workspace
  `rbuf` (empty on a pure-spin model). All of this is a bitwise no-op there.

- **An escape detector, because an unbounded target is otherwise invisible.** The
  cluster expansion is a finite polynomial in `u`, so `exp(−βE)` is a probability
  measure only when the leading even form is positive definite — and nothing upstream
  guarantees that. When it fails the chain has no stationary distribution and runs
  downhill, and *no pre-existing diagnostic notices*: the ΔE bookkeeping stays exact
  (drift `1e-14`, so the drift warning is silent by construction) and the acceptance
  sits at 0.97–0.99. `_check_escape!` runs inside `_renormalize!` and compares the
  block-averaged centre-of-mass-free mean square displacement against the previous
  block's, plus an absolute guard against order-of-magnitude growth — the one property
  that separates the two cases is recurrence.
  `ChainState` reports `disp_rms`/`disp_max`. Detection only, deliberately: a
  displacement bound would convert a loud failure into a quiet one, with the reported
  numbers set by the bound rather than by the Hamiltonian (measured). Decision record:
  `docs/specs/updates-stationarity.md` **U8**, which also records why umbrella sampling
  — not a wall — is the right instrument where the boundary genuinely carries weight.
- **An analytic gate for the displacement sampler.** `_einstein_terms` builds
  `E = a|u|²`, whose Boltzmann distribution is an isotropic Gaussian with
  `⟨|u|²⟩ = 3kT/(2a)` exactly. Every other displacement test compares the sampler with
  itself (incremental vs from-scratch, serial vs parallel); this one compares it with an
  external truth, and agrees to 0.54 %. It doubles as the equilibrated control that
  pins the escape detector's false-positive rate (zero strikes and zero warnings over
  8 seeds × 120 checks).

### Fixed

- **The flatness verdict is now per (Cartesian direction, displacement-coupling
  component)** — `H.comp_free`, a `3 × n_disp_comps` matrix, with `_recenter!`
  projecting the mean onto exactly the flat subspace. `_translation_residuals` probed a
  single rigid direction, so a component whose flat directions are a proper subspace —
  a substrate-clamped slab, pinned along its normal and free in the plane, which is the
  example the spec itself gives — was reported as wholly pinned, re-centring skipped it
  entirely, and its in-plane centre of mass then random-walked without bound. Nothing
  would have caught that: free diffusion sits below the escape detector's threshold by
  construction. Gated by a fixture in which a one-body `l = 1` displacement axis pins
  exactly one direction and leaves two free, for each tesseral slot.
- **The escape detector's growth test is block-averaged with a capped window**
  (`_ESCAPE_WINDOW = 16`, three strikes) rather than an unbounded doubling ladder, plus
  an absolute guard at `10×` the phase's first measurement. The ladder had a structural
  blind spot — comparisons only at checks 1, 2, 4, 8, …, so an escape starting after a
  quarter of the run never gets two consecutive comparisons and is never reported, and
  a spin-driven escape (the model becomes unbounded only once the spins order) is
  exactly that case. It also let sub-threshold growth accumulate for ever, and its
  false-positive rate is governed by the effective number of degrees of freedom
  carrying `⟨|u|²⟩` — a single dominant soft mode has `ν ≈ 1` at any system size — not
  by the site count. Measured on the Einstein control: zero strikes and zero warnings
  over 8 seeds × 120 checks.
- **The escape warning is once per chain phase, not once per session.** `maxlog = 1` is
  per callsite for the whole logger, so a single spurious strike anywhere would have
  silenced every genuine report from every other temperature and every replica-exchange
  lane for the rest of the run.
- **The flatness verdict was previously per displacement-coupling component**
  rather than global (`translation_residual`/`translation_invariant` remain as the
  worst-entry summary). `_translation_residual` already measured it per component, but a
  single global `Bool` meant that on a mixed model — one flat
  component, one genuinely pinned — re-centring was disabled for the flat component too,
  so its frame drifted without bound. That costs the reporting convention *and*
  numerical conditioning (`_disp_rows!` evaluates the solid harmonics at the absolute
  `u`, so drift makes the energy a difference of large near-cancelling terms).
- **A flat one-site displacement component is now refused.** Its energy does not depend
  on that site's displacement at all — it is a pure gauge coordinate, and sampling it
  spends randomness on always-accepted moves, dilutes the acceptance statistics, and
  writes a meaningless displacement into every reported configuration. A fitted model
  cannot produce one (the ASR zeroes a lone displacement axis), so its presence is
  information.

### Changed

- `_recenter!` is now justified by **π-invariance** rather than by energy neutrality
  (which is strictly weaker): the chain is a skew product whose quotient marginal is
  Markov, so re-centring transforms only an unmeasured passenger. That argument holds
  **only while every state-space restriction is gauge-invariant**, which is now recorded
  as a standing constraint on anything added downstream.
- `run_mc` / `run_pt` refuse a displacement Hamiltonian (the guard moved off the
  `ChainState` constructor, which now accepts one): the drivers schedule no
  displacement pass yet, so they would silently sample the clamped-ion ensemble.
  `displacement_sweep!` is callable directly in the meantime. Checkpoint resume
  refuses one too — schema v2 stores no displacement state.
- `_renormalize!` takes a `SweepScratch` rather than a bare `plm` workspace (it now
  needs the solid-harmonic buffer as well).

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
