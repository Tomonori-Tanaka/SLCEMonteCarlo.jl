# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Full classical spin Monte Carlo for fitted SCE models from
[`SLCE.jl`](../SLCE.jl) — the from-scratch successor of the frozen
`SpinClusterMC.jl` (Magesty-XML + Carlo.jl), with no API-compatibility constraint.
Core capabilities: tile the fitted training-cell Hamiltonian onto an `N₁×N₂×N₃`
supercell (`TiledHamiltonian`, via `MultipoleTerm.shifts`) — optionally after a
*verified* re-expression in a user-chosen smaller cell (`reduce_cell`, so `dims` is
not locked to training-cell multiples) — single-spin Metropolis
with an adaptive step, overrelaxation, annealing sweeps (`run_mc`), replica exchange
over threads (`run_pt`), numerical ground-state search (`minimize_energy` /
`find_ground_state`), composable observables with log-binning errors + jackknife
evaluables, and bit-reproducible JLD2 checkpoint/restart (reproducibility scope —
same package + Julia version, thread-count-independent, trajectory not
observable-ULPs: `docs/specs/pt-threads-determinism.md` P6). Self-contained core —
**no Carlo.jl dependency** (a Carlo adapter could later be a weakdep extension).

Relation to siblings: `SLCETools.jl` keeps the single-training-cell *configuration
samplers* (MFA + light Metropolis); this package is for thermodynamics-grade runs
(observables, error bars, T sweeps). `SpinClusterMC.jl` is a read-only design
reference — its pain points (God-struct, module-level global caches, split
temperature-unit conventions, hard-coded observables, per-instance payload
duplication, positional hand-rolled serialization) are what this design avoids.

This package reads a fitted model **only** through `SLCE`'s public surface:
`decorated_terms` / `multipole_terms`, `row_layout` / `row_index` / `site_rows!`,
`restrict`, `n_atoms(model)`, `intercept`, `SLCE.load(SLCEModel, …)`,
`Lattice`/`Crystal`/`cartesian_positions`, `SLCE.Harmonics` (`Zlm_unsafe`,
`lm_index`, `num_lm`, `grad_Zlm_unsafe`) and `SLCE.SolidHarmonics`
(`solid_harmonics!`, `solid_harmonic_index`) — never SALC-basis internals and never
`model.basis.crystal` (not public tier; geometry helpers take an explicit `Crystal`).
During development the dependency is a path-dev: `Pkg.develop(path="../SLCE.jl")`.

## Numerical / physics conventions

- **Spin directions are unit vectors.** Internal state is `SpinConfig =
  Vector{SVector{3,Float64}}` (one entry per supercell site); the 3×n matrix layout
  of the siblings appears only at the I/O boundary (`to_matrix`/`from_matrix`).
- **Real (tesseral) spherical harmonics `Zₗₘ`** from `SLCE.Harmonics`
  (`lm_index(l, m) = l² + l + m + 1` ordering) on spin axes; 4π-free real solid
  harmonics `|u|^{2k} Rₗₘ(u)` from `SLCE.SolidHarmonics` on displacement axes.
  `multipole_terms`/`decorated_terms` return the **raw** fitted `jϕ`; the scale —
  `(4π)^(n_spin_slots/2)`, one `√(4π)` per **spin slot**, which reduces to
  `(4π)^(body/2)` only when every site holds exactly one spin factor — is applied
  **exactly once**, in the `TiledHamiltonian` constructor (`ScaledTerm.coef`), read off
  `DecoratedTerm.scale` and never re-derived from the cluster shape. Never re-apply
  downstream.
- **Displacements** are Cartesian, one `SVector{3,Float64}` per site, in the model's
  length units, and are a **required** argument on a Hamiltonian with displacement rows
  (`has_disp(H)`) — an omitted `disps` would silently mean `u = 0`, which is a different
  physical state, not a default.
- **Energies** are in the model's energy units (eV for DFT-fitted models), `j0`
  (intercept) excluded everywhere — MC only needs differences; the reconstruction
  gate is `total_energy(H₁ₓ₁ₓ₁, cfg[, u]) == predict_energy(model, cfg[, u]) −
  intercept(model)`.
- **Supercell tiling**: `MultipoleTerm.shifts` are per-site integer training-cell
  lattice translations (`shifts[1] = 0` anchored). One instance per template term and
  supercell cell `t`, member `i` at `site_index(atom_i, mod.(t + shifts[i], dims))`.
  Each directed member is one plain summand — no ½ or 1/N factors.
- **Temperature**: absolute only, exactly one of `temperature` [K] XOR `kT`
  [model energy units]; `KB_EV` is the exact CODATA ratio. β enters only in accept
  steps; coefficients and energies stay in model units.
- **ΔE locality**: every instance's member *sites* are distinct after the toroidal
  wrap (asserted per term in the ctor — minimum-image models have distinct atoms
  outright; `AllImages` models may reuse an atom across images and need `dims` large
  enough), so the leave-one-out coefficient vector `c_s` is independent of `e_s` and
  `ΔE = c_s·(Z(e′) − Z(e))` is exact for any body order.

## Coupled ("linked") code sites — change one, check all

- **`hamiltonian.jl` ↔ the core's introspection contract** (`SLCE`'s
  `slce/introspect.jl`): field meanings of `MultipoleTerm` (coef/body/atoms/shifts/
  ls/folded) and `DecoratedTerm` (coef/**scale**/body/atoms/shifts/slots/folded), the
  raw-coef scale rule, and the shifts anchoring. Gates:
  `test_hamiltonian.jl` (dims=(1,1,1) ≡ `predict_energy − intercept`; 2×2×2
  periodic-replication = 8× cell energy; scale-once) and `test_joint.jl` (the same two
  with displacements).
- **Channels: `ScaledTerm.slots` ↔ `SLCE.row_layout` ↔ `_disp_rows!` ↔
  `SLCE.site_rows!` ↔ the merged site programs ↔ `_fingerprint`**
  (`hamiltonian.jl`, `energy.jl`, `checkpoint.jl`, `docs/specs/hamiltonian-tiling.md`
  T6). A tensor axis is a **slot**, not a site: one site may carry a spin *and* a
  displacement axis, so `sfac_slot`/`site_slot` mean member **site position** and
  `efac_site` maps an energy-program factor to it. Four things move together:
  (1) the row numbering is `SLCE.row_layout(model)` — never invented here, and the
  frozen `MultipoleTerm` path deliberately keeps deriving its own
  (`_spin_row_layout`) so a pure-spin model's rows are the pre-M4 integers, with
  `H.lmax`/`H.nlm` = the SPIN block alone and `H.nrows` = all rows;
  (2) the scale is `(4π)^(n_spin_slots/2)` read off `DecoratedTerm.scale`, NOT
  `(4π)^(body/2)` — they differ exactly on a term with a spin-free site, and a
  fixture whose sites all carry spin makes the check vacuous;
  (3) `_disp_rows!` batches the solid harmonics once per site while
  `SLCE.site_rows!` re-batches per `(k, l, m)` — the gate is **bitwise** equality, and
  a change to `SolidHarmonics`' recurrences or normalization breaks it together with
  the upstream force constants and ASR;
  (4) `_fingerprint` mixes the slot layout only for non-identity layouts, so pre-M4
  checkpoints still identify their Hamiltonian — pinned against an in-test copy of the
  old formula — but it must ALSO mix `layout.disp_factors` on a joint model, because
  `TermSlot.row0` is a layout-relative block start and `(k, l) → row0` is not injective
  across layouts (a `degree = 3:5` sector's `(1,1),(2,1)` blocks start where a `1:3`
  sector's `(0,1),(1,1)` do, so two models differing only by a factor `|u|²` would
  collide);
  (5) `disp_scale` (a `BasisSpec` field upstream currently refuses unless `== 1.0`) is
  carried by neither `RowLayout` nor `DecoratedTerm`, so this package cannot see it —
  the day upstream implements it, `_disp_rows!` must move with `predict_energy`.
  The ctor enforces **at most one axis per `(site, channel)`** (upstream's `SiteDecor`
  rule): two axes of the SAME channel on one site would make even a single-channel move
  drop a cross term, and such a term has a pure-spin layout so `_require_spin_only`
  would not catch it. `site_coeffs!` is exact for **single-channel** moves only (a
  simultaneous spin+displacement move on one site misses `Δz·Δr`); the pair/triplet
  fast paths hoist columns only when every axis of the site yields the same column
  tuple (`_hoisted_columns` compares them — canonical axis order does NOT guarantee
  same-site axes are adjacent). Gates: `test_joint.jl` throughout; every spin-only
  entry point must keep calling `_require_spin_only` until slice 3c lands.
- **Upstream BREAKING spec keywords ↔ this package's fixtures/benches/docs/assets**:
  `SLCE`'s `BasisSpec` keywords are consumed in `test/unit/fixtures.jl`,
  `bench/fixtures.jl`, `bench/assets/*.toml` (`[interaction]`) and every
  `docs/src/**.md` `@example` block. The `isotropy` → `soc` rename (with its
  **inverted** sense) broke all four at once and was invisible until the suite was
  run — a downstream rename is not done when SLCE's own tests are green.
- **`energy.jl` 4-function contract ↔ `updates.jl` ↔ SLCETools' `mc/metropolis.jl`**:
  `site_coeffs!`/`delta_energy` are the site-generalized siblings of SLCETools'
  `_accumulate_site_term!` kernel (same `μ = idx − l − 1` mapping, rank-specialized
  barrier). Gates: `test_energy.jl` ΔE ≡ total-energy difference (1e-12).
- **`lm_index` ordering ↔ `zlm_row!` ↔ the overrelaxation l=1 axis extraction**
  (`updates.jl`): the tesseral l=1 components map to Cartesian axes; a reorder
  upstream breaks the OR axis. Gate: pure-l=1 OR proposals have `ΔE ≡ 0` and
  acceptance 1 (`test_overrelaxation.jl`).
- **`reduce.jl` ↔ `hamiltonian.jl` tiling ↔ `geometry.jl` ordering ↔ SLCE's
  canonical members**: `reduce_cell` emits raw-coefficient `MultipoleTerm`s (the
  `(4π)^(body/2)` scale still happens once, in the `TiledHamiltonian` ctor),
  anchored `shifts[1] = 0`, and a reduced `Crystal` whose atom order matches
  `site_index` so `supercell_crystal(red.crystal, dims)` pairs with
  `TiledHamiltonian(red; dims)`. Translation copies are grouped in **canonical
  site order** (sorted `(reduced atom, shift)`, re-anchored, `ls`/`folded`
  permuted along) because canonical model terms carry one summand per instance,
  anchored wherever sorting put it; the census accepts `q·|det M|` copies for
  `q` identical summands per instance. The invariance and verification contract
  lives in `docs/specs/cell-reduction.md`. Gates: `test_reduce.jl` (exact
  canonical-form recovery, energy identity via site permutation).
- **`energy.jl` `_site_grad` ↔ `site_gradient` ↔ `energy_gradient!` ↔
  `minimize.jl` `_gradient!`**: one per-site gradient kernel (`_site_grad`) backs
  the public all-site `energy_gradient!` (the field/torque entry point for
  dependent packages — task-count bit-identity rests on task-local `c`/`plm`
  scratch in `_gradient_chunk!`) and the descent's `_gradient!`; both must stay
  arithmetically identical to the public per-site `site_gradient` (same `(l, m)`
  loop over `lm_index` order, same `ck == 0` skip). Gates: the bitwise `==`
  consistency tests in `test_gradient.jl` / `test_minimize.jl` and the
  `predict_torque` cross-check (`τ = G × e`); an `lm_index` reorder upstream
  breaks them together with the OR axis (previous bullet).
- **Checkpoint writer ↔ reader ↔ schema doc** (`checkpoint.jl`,
  `docs/specs/checkpoint-schema.md`): plain-data JLD2 schema v2, Xoshiro capture via
  `fieldnames`, accumulator state. Gate: bit-identical resume (`test_checkpoint.jl`).
  The public `model_fingerprint` facade over `_fingerprint` is pinned by dependent
  packages' checkpoint formats (SLCEDynamics) — changing the mixing changes
  every stored fingerprint (schema-version territory there too).
- **Observable conventions** (C/χ/U definitions) live in ONE place:
  `docs/specs/binning-observables.md`; `observables.jl` and the guide pages follow it.
- **Coloring ↔ sweeps ↔ stationarity spec** (`hamiltonian.jl` `_color_sites` /
  `color_ptr`/`color_sites`, `updates.jl`, `docs/specs/updates-stationarity.md`
  U1): the sweeps assume every color class is instance-disjoint (exactly
  independent single-site kernels) and bit-determinism for any `sweep_tasks` rests
  on per-site RNG streams (`ChainState.site_rngs`, checkpoint schema v2) + the
  fixed-order ΔE reduction (`_reduce_dE`). Touch the coloring, the sweep loops, or
  the reduction and re-run `test/unit/test_parallel.jl` (serial ≡ parallel `==`).
- **Device tesseral row ↔ host `_zlm_row!` ↔ upstream recursions**
  (`src/gpu/zlm_device.jl`): `_zlm_row_device!` is a deliberate, operation-order-
  faithful reimplementation of `_zlm_row!` → `Harmonics.Zlm_unsafe` →
  `LegendrePolynomials.dnPl` (+ `Base.power_by_squaring` as `_zlm_cpow`), because
  the upstream path cannot compile in a GPU kernel. Any upstream change to those
  functions (a normalization, a recursion reorder, an SLCE `Harmonics`
  edit) breaks the dense bitwise gate in `test/unit/test_gpu.jl` — update the
  device file together with it.
- **GPU kernel ↔ keyed reference ↔ slot map ↔ workgroup-size pin**
  (`src/gpu/gpu_sweep.jl`, `src/gpu/philox.jl`, `docs/specs/gpu-prototype.md`
  G2–G4): `_metro_kernel!` and `_metropolis_sweep_keyed_ref!` implement ONE
  arithmetic contract (proposal slots, `_entry_walk_partial` dispatch + zero
  skips, lane-ordered fold, accept rule). Touch any of them — or the Philox slot
  layout, or the pinned default ws — and the other side plus the G-record move
  together; gate: the full-sweep bitwise section of `test/unit/test_gpu.jl`.
- **Device gradient ↔ lane reference ↔ upstream grad recursions**
  (`src/gpu/grad_device.jl`, `src/gpu/gpu_gradient.jl`, G7): `_grad_kernel!`
  and `_gradient_lane_ref!` implement ONE arithmetic contract (the gradient-row
  table, `_entry_walk_grad`'s dispatch/skips — structurally
  `_entry_walk_partial` — and the lane-ordered component fold); the row
  `_grad_zlm_device` is the operation-order-faithful replica of
  `Harmonics.grad_Zlm_unsafe`/`_grad_zlm_assemble`/`_barP`/`_dbarP` and of
  LegendrePolynomials' `dnPl` `l < n` trivial-zero branch (`_zlm_dnpl_or0`;
  signed zeros are part of the `===` gate). The pipeline is deliberately
  libm-free — keep `muladd`/`@fastmath` out. `_gradient_lane_ref!` is called by
  qualified name from SLCEDynamics' GPU-LLG composite gate — renaming it is
  a cross-package break. Gates: the G7 sections of `test/unit/test_gpu.jl`.
- **Inactive-site convention** — TWO predicates since M4 slice 3b, and they differ on a
  joint model: `site_active`/`n_active` ("touched by any instance, either channel")
  drives the coloring and the sweep schedule, while `site_has_spin`/`n_spin_active`
  ("touched by a SPIN axis") is what the spin sweeps, `_renormalize!`, the spin
  observables and their per-site normalizations read. A displacement-only ligand (boron
  with `lmax = 0` plus a displacement sector) is the case that separates them —
  conflating them divides `m = Σ_s e_s / n` by a count including a frozen random
  direction. They coincide exactly on a pure-spin model, so every switch was a bitwise
  no-op there. Update sweeps **skip**, standard observables **exclude**, and
  sweeps/renormalization/descent keep the spins **bitwise frozen** — these move
  together, since skipping without excluding turns a frozen random direction into a
  constant observable bias. Touch `updates.jl`,
  `observables.jl`, `state.jl` `_renormalize!`, `minimize.jl` `_gradient!`/
  `_minimize!`, or `energy.jl` `energy_gradient!`/`_gradient_chunk!` (inactive
  sites → exactly zero) and re-check `test/unit/test_inactive.jl` +
  `test_gradient.jl`.

## Tests

| Command | Purpose |
|---|---|
| `julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `cd docs && make build` | docs (checkdocs = :exports) |

Statistical gates use fixed seeds with tolerances proven in SLCETools' MC suite.
Manual smoke (not CI): Nd₂Fe₁₄B l02 model (`~/jijs/magesty/2-14-1/nd2fe14b/1x1x1/
magesty/l02/test`, rebuild via its `fit_mfa.jl` recipe), dims=(4,4,4), short PT
across the ordering temperature. Last run (2026-07-11, v0 completion): 1×1×1 and
64× counting gates at ~1e-13; construction 0.01 s / 7.8 MB index; 8 rungs ×
900 sweeps × 4352 sites in 38 s on 8 threads; ferrimagnetic projections
Nd ≈ −0.50 / Fe ≈ +0.69 at 250 K. Note: 8 rungs over 250–1300 K give *zero*
swaps at this size (rung count must scale like √(n_sites·C) — documented in the
PT guide), so use denser ladders for production.

## References

- `SPEC.md` — architecture, primary types, public API.
- `docs/specs/*.md` — decision records (tiling, update stationarity,
  binning/observable conventions, PT determinism, checkpoint schema).
- `references/` — supporting literature (notes tracked, PDFs local-only).
