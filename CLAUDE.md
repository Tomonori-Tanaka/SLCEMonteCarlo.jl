# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Full classical spin Monte Carlo for fitted SLCE models from
[`SLCE.jl`](../SLCE.jl) — the from-scratch successor of the frozen
`SpinClusterMC.jl` (Magesty-XML + Carlo.jl), with no API-compatibility constraint.
Core capabilities: tile the fitted training-cell Hamiltonian onto an `N₁×N₂×N₃`
supercell (`TiledHamiltonian`, via `SpinMultipoleTerm.shifts`) — optionally after a
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
`decorated_terms` / `spin_multipole_terms`, `row_layout` / `row_index` / `site_rows!`,
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
  `spin_multipole_terms`/`decorated_terms` return the **raw** fitted `jϕ`; the scale —
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
- **Supercell tiling**: `SpinMultipoleTerm.shifts` are per-site integer training-cell
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
  `slce/introspect.jl`): field meanings of `SpinMultipoleTerm` (coef/body/atoms/shifts/
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
  frozen `SpinMultipoleTerm` path deliberately keeps deriving its own
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
  same-site axes are adjacent). Gates: `test_joint.jl` throughout. The MC drivers took
  the displacement channel in slice 3c/3; what still calls `_require_spin_only` is
  everything with no displacement move of its own — `minimize_energy` /
  `find_ground_state`, the gradient entry points, and the GPU kernels — and each must
  keep doing so, or it reports a clamped-ion answer as the answer.
- **The displacement channel of the sampler: `ChainState.disps` ↔ `SweepScratch` ↔
  `_attempt_disp!` ↔ `_recenter!` ↔ `_renormalize!` ↔ `_swap_payload!` ↔ the checkpoint
  schema** (`state.jl`, `updates.jl`, `pt.jl`, `checkpoint.jl`,
  `docs/specs/updates-stationarity.md` U7). Five things that must move together:
  (1) the two proposal widths have **different dimensions** — `step` is an angle
  (radians, clamped to `(1e-3, π)` by `_adapt_step!`) and `step_u` is a **length** in the
  model's units; never feed one through the other's adaptation or clamp;
  (2) a single-channel move writes only its own block of `SweepScratch.znew`, so its ΔE
  **must** use the range-limited `delta_energy(c, zold, znew, lo, hi)` — the full-range
  form would read the stale half of the buffer, and the row-difference form itself is
  load-bearing (`c·znew − c·zold` loses two to three digits);
  (3) `_recenter!` lives in `_renormalize!` and **nowhere else** — a mean over sites is
  an order-dependent reduction, and inside the barrier-separated color loop it would
  make the trajectory depend on `sweep_tasks` (P6). It is per displacement-coupling
  component, over `site_has_disp` sites only, and skipped when
  `translation_invariant == false`;
  (4) `disps` and `com_removed` are part of the replica-exchange **payload** (a swap
  moves a whole physical state, frame included), unlike the RNG streams and the
  proposal widths, which stay with the lane;
  (5) the drivers must SCHEDULE the displacement pass: `disp_per_metropolis` resolves
  from the model (`nothing` ⇒ 1 joint / 0 pure-spin) via `_resolve_disp_passes`, and
  a constant default of 0 would make a joint model silently sample the clamped-ion
  ensemble — plausible spin numbers for the wrong distribution. The checkpoint schema
  (v3+) carries `disps`/`com_removed`/`step_u`/the counters/the escape accumulators, so
  a resume must stay bit-identical on a joint model too;
  (6) **every restriction of the displacement state space must be gauge-invariant**
  (a function of `u_s − ū_c`, not of `u_s`). `_recenter!` is stationarity-preserving
  because the chain is a skew product whose quotient marginal is Markov — an argument
  that holds only under that hypothesis. A gauge-DEPENDENT restriction (an absolute
  radius bound, say) makes the region move relative to the state and silently breaks it,
  and no existing gate would notice, since "re-centring is energy-neutral" stays true.
  The flatness verdict itself is per component (`H.comp_invariant`) for the same reason:
  the components are independent symmetries.
  Related: `_check_escape!` is the only diagnostic that measures **recurrence**, which
  is what an unbounded target destroys — the ΔE drift warning and the acceptance rate
  are both silent by construction there (U8).
  Gates: `test_joint.jl` "joint displacement sampling" (exact ΔE against a
  from-scratch recomputation, serial ≡ parallel bitwise, re-centring energy-neutral
  and per component, pure-spin no-op consuming no randomness), "joint drivers: pass
  scheduling and the second proposal width", and `test_checkpoint.jl` "joint:
  displacement state survives a resume bit-exactly".
- **Coefficient hot-swap: `_push_term_programs!`'s skip ↔ `sent_base`/`sent_term` ↔
  `set_coefficients!` ↔ `keep_zero_terms` ↔ the GPU tables and the checkpoint
  fingerprint** (`hamiltonian.jl`, `coefficients.jl`, `gpu/gpu_hamiltonian.jl`,
  `checkpoint.jl`). The site-program skip tests **`folded[idx]`, never
  `coef · folded[idx]`** — that is what makes the entry SET coefficient-independent, and
  it is the whole reason a rewrite can be an in-place fused multiply instead of a
  rebuild (0.033 ms vs 13–96 ms, and `dims`-independent). It is byte-identical to the
  old predicate for any `coef ≠ 0`; re-introducing the `coef` factor would leave a
  zero-coefficient term with no program. The invariant
  `sent_w[i] == term_coef[sent_term[i]] · sent_base[i]` must be re-established after
  **every** write to any of the three, and the energy program deliberately does NOT
  carry the coefficient (it multiplies per instance), so it needs no rewrite — add a
  weight array and it must join the invariant or be documented as raw.
  `term_source`/`term_scale`/`n_input_terms` exist so a rewrite is indexed by the
  caller's term list and re-applies the SAME scale the ctor did, never re-derived from
  the cluster shape (design record §7). Everything structural — `site_active`,
  `site_has_spin`/`site_has_disp` and their counts, the coloring, the displacement
  components, `comp_free` — is a property of the term list the Hamiltonian was BUILT
  with; that is what `keep_zero_terms` freezes, and why a nonzero coefficient for a
  pruned term is a loud error naming the flag rather than a silent no-op. Two
  consumers go stale by design and are documented, not fixed: a `GPUTiledHamiltonian`
  keeps its old device tables (re-upload), and `_fingerprint` mixes `t.coef`, so a
  pre-swap checkpoint correctly refuses to resume. Gates:
  `test_coefficients.jl` (byte-identical no-op round trip, swap ≡ fresh build on the
  triplet fast path, the named refusal, `keep_zero_terms` byte-neutral without zeros,
  and the flatness re-check with its opt-out).
- **`StrainSchedule` ↔ `keep_zero_terms` ↔ upstream `decorated_terms(; keep_zero)` ↔
  `set_coefficients!`** (`strain.jl`, `hamiltonian.jl`, SLCE `slce/introspect.jl`): the
  hot-swap is indexed by the caller's term list, and BOTH halves of the support freeze are
  required. Upstream prunes exactly-zero coefficients by default, so its index → SALC map
  is a function of the fit; `keep_zero_terms` here freezes the support of whatever list it
  is handed, so it now passes `keep_zero` through to the emission — a flag applied only at
  ingestion cannot resurrect a term never emitted. Two grid points whose sparse fits zero
  DIFFERENT keys otherwise emit lists of equal length with shifted maps, and every length
  check passes while each coefficient lands on a neighbouring cluster. `StrainSchedule`'s
  constructor re-checks this term by term (atoms and images, across the grid and against
  `H`) rather than trusting it. **The schedule holds no scale and no `H`**: the current
  scale is chain state and the coefficients are `H`'s, and they must be written together —
  keeping the schedule immutable is what stops a caller installing one volume's
  coefficients against another volume's displacements. `n_cells` counts ATOMS
  (`H.n_sites ÷ n_atoms`), never `prod(dims)`, which `reduce_cell` breaks; `D` is
  `3·n_disp_active − count(comp_free)`, not `3(N−1)`, because re-centring is per
  (direction, component) and there are `n_disp_comps` independent centres of mass.
  Gates: `test_strainschedule.jl`.
- **The NPT strain move: `strain_move!` ↔ `strain_delta_energy`/`_strain_log_weight` ↔
  `ChainState.strain` ↔ the run drivers ↔ the checkpoint schema ↔ `sync_coefficients!`**
  (`strain.jl`, `run.jl`, `pt.jl`, `checkpoint.jl`, `gpu/gpu_hamiltonian.jl`,
  `docs/specs/strain-move.md`). Eight rules that move together. (1) The elastic energy
  has ONE source — the grid's `j0(s)`, times `n_cells` — and there is deliberately no
  elastic-term keyword anywhere; adding one double-counts with every gate green
  (the reconstruction identity in `test_strainschedule.jl` is the fence). (2) The
  proposal draw (`_strain_y`/`_strain_s_of_y`) and the acceptance exponent
  (`_strain_s_exponent`: `3·N_mob + 3` for `:logvolume`, `3·N_mob + 2` for `:scale`,
  with `N_mob = n_disp_active` the displacement-active SITE count) branch on the
  SAME symbol and sit adjacent — drawing in one arm and weighting with the other is
  off by `(V′/V)^{2/3}`, below the statistical gate's resolution; the closed-form
  and white-box-replay gates exclude it exactly, so touch one branch and re-check
  both. The power is NOT the COM-reduced `D/3` (the 2026-07-29 re-correction —
  quotiented flat directions keep their `∝ s` measure factor, the ideal-gas COM
  factor; the toy gate's flat/pinned pair at equal `N_mob` and unequal
  `count(comp_free)` is the mutation fence). (3) The `(H, chain)` contract — `H`
  carries the schedule's coefficients at `st.strain` before and after every move,
  accepted or not (the reject path restores by re-running the same deterministic
  Horner pass; every argument is validated BEFORE the first write) — is what the
  sweep layer, `run_mc`'s reference install, `carryover = false`'s cell reset, and
  `resume`'s scale reinstall all assume; break it anywhere and the sweeps sample one
  volume's displacements against another's coefficients. Both drivers hand `H` back
  at the REFERENCE on return, and the pairing check compares a structural term
  fingerprint (`_schedule_term_fp`), not counts alone — two same-shape models pass
  every count. (4) On a strained run the checkpoint `model_fingerprint` is defined
  AT THE REFERENCE scale: the writer captures it while `H` holds `s = 1` (order of
  `_make_checkpointer` vs the install in `run_mc` is load-bearing) and `resume`
  reinstalls the reference from the supplied schedule before comparing — reorder
  either and every strained resume refuses or, worse, accepts a wrong model. (5) An
  accepted rescale is exactly affine, so `disps`, `com_removed` AND the escape
  detector's length statistics rescale together (`_rescale_escape!` — a reset would
  disarm the block ladder permanently at the default cadence), and ΔE's two sides
  come from one estimator (`e_old` recomputed from scratch, drift carried across).
  (6) a host `set_coefficients!` under a live
  `GPUTiledHamiltonian` needs `sync_coefficients!` — `sent_w` is the ONE
  coefficient-dependent device array, and forgetting the call is undetectable from
  the device side (the bitwise gate in `test_gpu.jl` holds the contract). (7) On a
  strained `run_pt` (strain-move.md S10), each lane sweeps its own
  `_coefficient_clone` of `H` — the clone copies EXACTLY the fields
  `set_coefficients!` writes (`terms`, `progs.term_coef`, `progs.sent_w`) and
  shares every other array by reference, so a new coefficient-carrying array in
  the programs must join the clone or every PT lane silently shares it; the
  lane's `H` reference is exchange PAYLOAD (swapped with the state in
  `_attempt_swap!` — separating them re-opens the coefficients-vs-scale desync
  the clone exists to prevent); the swap weight `_swap_dweight` must carry
  `ΔE + n_cells·Δj0 + P·ΔV` as a SUM OF DIFFERENCES (per-lane totals differenced
  compute at `ulp(|W|)` instead of `ulp(|ΔW|)`, a loss growing with `n_cells` —
  and the bracket gate's hand logw shares the association, so change one side
  and re-pair the other) and NOTHING volume-power-shaped (the `V^{N_mob}`
  factor is β-independent and cancels in the exchange ratio — adding it "for
  consistency" biases every swap), and its fixed-cell arm must stay the bare
  chain-energy difference bitwise (the NVT trajectory pin); the caller's `H`
  never enters a lane and stays at the reference for the whole run. Exact rule mutations are
  caught only by the bracket gate in `test_strainschedule.jl` ("PT + strain"
  testsets) — the statistical rung-marginal gate measured ≤ 0.6σ under the NVT
  mutation, so do not lean on it for the rule. The two
  fixed-cell byte-neutrality pins (pre-wiring `run_mc` and `run_pt` trajectories
  asserted bit-identical in
  `test_strainschedule.jl`) catch any of these leaking into unstrained runs —
  they are capture-platform-scoped (the `_ss_grid` fixture inherits LAPACK's
  ASR null-space basis, so the pins fire on a model-fingerprint match and
  macOS aarch64 asserts they fired; strain-move.md S6).
  (8) The NPT target's `W = E_config + n_cells·j0(s) + P·V(s)` now lives in
  THREE places — the strain move's acceptance pieces (`strain_delta_energy` +
  `_strain_log_weight`), the exchange weight (`_swap_dweight`), and
  `npt_observables`' `:enthalpy` — and a change to the target must move all
  three. The observable's formula is pinned at machine precision against the
  zeta fixture's ANALYTIC `j0 = 40η²` / `V = n_cells·27·s³` (the anchor gate in
  `test_strainschedule.jl` "npt_observables"), which is what actually kills a
  dropped or misplaced term — the FDT cross-gate there
  (`var(W)/(k_BT)² ≡ d⟨W⟩/d(k_BT)`, 4σ) cannot resolve one (the config-only
  mutation is ≈ 3–6 % of C on that fixture), the same
  statistics-cannot-carry-the-rule split as rule (7)'s bracket.
  `npt_observables`' closures must STAY pure (immutable schedule + resolved
  constants; the per-view guard is IDENTITY on the shared `inst_term` array —
  clones share it, any foreign `H`, same-shape included, does not — rather than
  `v.H === H`): that purity is the entire reason they are legal under
  `run_pt`'s concurrently-measuring lane clones while `pressure_diagnostics`
  is refused by name. On a FIXED-CELL run both drivers refuse
  `:enthalpy`/`:enthalpy2` at entry (`_refuse_npt_observables`) — keep that in
  sync if the observable names change.
- **The §8(ζ) pressure diagnostic: `_energy_with_coefs` ↔ `_total_energy`, and the
  exact-derivative trio ↔ their Horner sources** (`strain.jl`,
  `docs/specs/strain-move.md` S9): `energy_volume_derivative` is exact via (a) the
  differentiated Horner passes (`_strain_dcoefficients!`/`_strain_dj0` must mirror
  `strain_coefficients!`/`strain_j0` — change one pass's abscissa handling and the
  derivative silently drifts by `_sch_dz_ds`'s chain-rule factor) and (b) Euler's
  theorem on the per-template displacement degree `Σ (2k + l)`
  (`_term_disp_degrees` reads `TermSlot.row0` against `layout.disp_starts` — a row
  numbering change moves both together). `_energy_with_coefs` is a deliberate
  MIRROR of `_total_energy`'s loop (external coefficient vector) so the pinned hot
  path stays untouched — edit one and re-sync the other. The kernel must keep
  reading `J(s)` from the schedule, never `H.progs.term_coef` (purity is gated).
  `pressure_diagnostics`' `Evaluable` gets its point's own `kT` — do not bake a kT
  into the closure. Gates: the two §8(ζ) testsets in `test_strainschedule.jl`
  (FD exactness, upstream `grid_strain_derivative` cross-check at `u = 0`, and the
  statistical identity on an Einstein-well fixture — kept bounded-below on
  purpose; random ASR'd coefficients are indefinite and the chain escapes).
- **Upstream BREAKING spec keywords ↔ this package's fixtures/benches/docs/assets**:
  `SLCE`'s `BasisSpec` keywords are consumed in `test/unit/fixtures.jl`,
  `bench/fixtures.jl`, `bench/assets/*.toml` (`[interaction]`) and every
  `docs/src/**.md` `@example` block. The `isotropy` → `soc` rename (with its
  **inverted** sense) broke all four at once and was invisible until the suite was
  run — a downstream rename is not done when SLCE's own tests are green.
- **Names this package borrows rather than owns** (`src/SLCEMonteCarlo.jl` header,
  `src/units.jl`): `has_disp` and `n_atoms` are **methods of `SLCE`'s generics**
  (`import SLCE: n_atoms, has_disp`), and `KB_EV` / `resolve_kt` are `using`-ed from
  `SLCE` and merely re-exported. Do not re-define any of them here. A second generic
  of the same name compiles and passes both suites while leaving a user who loads
  both packages holding two different `has_disp`; a second `const KB_EV` is a unit
  conversion that can drift with nothing to catch it (both were the case until the
  2026-07-28 family naming batch). Adding a `has_disp`-like predicate for a new type
  → extend the upstream generic, or ask upstream to add one.
- **`energy.jl` 4-function contract ↔ `updates.jl` ↔ SLCETools' `mc/metropolis.jl`**:
  `site_coeffs!`/`delta_energy` are the site-generalized siblings of SLCETools'
  `_accumulate_site_term!` kernel (same `μ = idx − l − 1` mapping, rank-specialized
  barrier). Gates: `test_energy.jl` ΔE ≡ total-energy difference (1e-12).
- **`lm_index` ordering ↔ `zlm_row!` ↔ the overrelaxation l=1 axis extraction**
  (`updates.jl`): the tesseral l=1 components map to Cartesian axes; a reorder
  upstream breaks the OR axis. Gate: pure-l=1 OR proposals have `ΔE ≡ 0` and
  acceptance 1 (`test_overrelaxation.jl`).
- **`reduce.jl` ↔ `hamiltonian.jl` tiling ↔ `geometry.jl` ordering ↔ SLCE's
  canonical members**: `reduce_cell` emits raw-coefficient terms (the `(4π)`
  scale still happens once, in the `TiledHamiltonian` ctor),
  anchored `shifts[1] = 0`, and a reduced `Crystal` whose atom order matches
  `site_index` so `supercell_crystal(red.crystal, dims)` pairs with
  `TiledHamiltonian(red; dims)`. Translation copies are grouped in **canonical
  site order** (sorted `(reduced atom, shift)`, re-anchored, axis labels and
  `folded` permuted along) because canonical model terms carry one summand per
  instance, anchored wherever sorting put it; the census requires the same `q`
  copies **in every coset** (labelled by `mod.(adj(M)·σ₁, nc)`, a complete
  integer invariant) — a total divisible by `|det M|` is not enough, since a
  class living in one coset satisfies it and would be emitted into every reduced
  cell. `ReducedCell{T}` mirrors the
  term surface it was fed: a joint model reduces as `DecoratedTerm`, where the
  site sort **also** relabels each slot's site and re-sorts the slots into
  canonical `(channel, site, k, l)` order (`folded` axes carried by
  `permutedims`), the reduction carries the model's `RowLayout`, and `scale` is
  taken from the representative **verbatim** — never re-derived from the cluster
  shape, which is exactly SLCE design-record §13 risk 3. `reduce_cell(model, …)`
  picks the arm by the same `isempty(row_layout(model).disp_factors)` test
  `TiledHamiltonian(model)` uses, so the pure-spin path stays byte-identical.
  The invariance and verification contract lives in
  `docs/specs/cell-reduction.md`. Gates: `test_reduce.jl` (exact canonical-form
  recovery, energy identity via site permutation, the `(4π)^(n_spin_slots/2)`
  pin on a hand-built mixed-channel list, a **3-body** 3-cycle case — the only
  shape that can tell `invperm(perm)` from `perm`, since every ≤ 2-body site
  permutation is an involution — and a pinned non-vacuity count that the
  slot-permutation path actually fires).
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
  `docs/specs/checkpoint-schema.md`): plain-data JLD2 schema v3, Xoshiro capture via
  `fieldnames`, accumulator state. Gate: bit-identical resume (`test_checkpoint.jl`).
  The public `model_fingerprint` facade over `_fingerprint` is pinned by dependent
  packages' checkpoint formats (SLCEDynamics) — changing the mixing changes
  every stored fingerprint (schema-version territory there too).
- **`MCView` ↔ every observable ↔ SLCEDynamics' `_measure!`** (`observables.jl`,
  `SLCEDynamics/src/run.jl`): an observable receives ONE argument, a view of the
  sampled state, precisely so that adding a channel does not break every definition
  ever written. Two rules make it safe: the view's inner constructor **empties**
  `disps` on a Hamiltonian with no displacement channel (a pure-spin chain carries an
  all-zero vector, and passing it through would make `sum(abs2, v.disps[s])` return a
  confident 0.0 on a model that describes no displacements), and SLCEDynamics builds
  its view with an empty `disps` for the same reason. Add a field to `MCView` and
  every producer must decide explicitly what goes in it — a zero placeholder is how a
  wrong number becomes a confident one.
- **Displacement observables ↔ `_recenter!`'s cadence** (`observables.jl`
  `_mean_u_moment` / `_sublattice_u_moment` / `_component_mean`, `state.jl`
  `_recenter!`): the observables subtract the component centre of mass **themselves**,
  with the same flat-direction projection `_recenter!` uses. Relying on the sampler's
  frame instead biases `⟨u²⟩` by `⟨|ū|²⟩`, linear in `renorm_interval` (measurements
  fire every `measure_interval`, re-centring every `renorm_interval`), always positive
  and shrinking with system size — indistinguishable from a finite-size effect. If the
  projection rule changes on one side it must change on both, or `⟨u2⟩` and
  `TempResult.disp_rms²` stop being the same quantity. Gate: `test_joint.jl`
  "displacement observables see the centre-of-mass-free frame" (three `renorm_interval`
  spanning 10², plus agreement with `disp_rms²`).
- **Displacement channel gates: `n_disp_active > 0`, never `has_disp`**
  (`observables.jl`, `hamiltonian.jl` `_require_disp`): `has_disp` is a property of the
  row LAYOUT, and a joint basis whose displacement couplings all fitted to zero has
  displacement rows and not one site whose energy depends on its displacement. Gating
  on the layout there measures `:u2 = 0.0` and returns a 0×0 Hessian — the confident
  zero and the "clean" empty spectrum the design exists to prevent. The spin side has
  always gated on `n_spin_active`; mirror it.
- **`harmonic_stability`'s verdict tolerance ↔ the acoustic modes**
  (`stability.jl`): a translation-invariant model has `3·n_disp_comps` EXACT zero
  eigenvalues, and finite differences scatter each across zero at the `ε|E|/h²` floor,
  so an untolerated `count(< 0, λ)` reports the documented proof of failure on a
  healthy model about half the time, with a step-dependent count. `tol` is measured
  from the residual of the certified-flat rigid-shift directions (exact answer zero ⇒
  pure roundoff) and returned so the verdict is auditable. Change the stencil or the
  step and the floor moves with it — which is the point of measuring rather than
  hard-coding. Gate: the same count at three `step` spanning 10³.
- **Observable conventions** (C/χ/U definitions) live in ONE place:
  `docs/specs/binning-observables.md`; `observables.jl` and the guide pages follow it.
  Since M4 there are **two site counts** and an `Evaluable` must declare which it
  needs (`scope = :energy` ⇒ `n_active`, `:spin` ⇒ `n_spin_active`). They coincide on
  every pure-spin model, so a wrong `scope` is invisible until a joint model runs —
  and then it is a plausible finite number, not a failure. Gate: an Einstein
  oscillator's `C = 1.5 k_B` per atom.
- **Coloring ↔ sweeps ↔ stationarity spec** (`hamiltonian.jl` `_color_sites` /
  `color_ptr`/`color_sites`, `updates.jl`, `docs/specs/updates-stationarity.md`
  U1): the sweeps assume every color class is instance-disjoint (exactly
  independent single-site kernels) and bit-determinism for any `sweep_tasks` rests
  on per-site RNG streams (`ChainState.site_rngs`, in the checkpoint schema) + the
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
- **Device displacement rows ↔ host `_disp_rows!` ↔ `SLCE.SolidHarmonics`**
  (`src/gpu/disp_device.jl`, G8): `_solid_row_device!` copies
  `_solid_harmonics_impl!` (already device-safe scalar arithmetic — only the
  gradient half is dropped) and IS gated bitwise; `_disp_rows_device!` adds the
  `|u|^{2k}` prefactor and is bitwise only on `k = 0` rows. The `k ≥ 1` gap is
  one ulp of `r2 = dot(u, u)`, whose FMA contraction differs between a host
  function that calls the harmonic batch and a device function that inlines it —
  an accepted scope reduction (G8), NOT a bug to fix by touching upstream
  `SLCE.site_rows!`. Bug-catching bitwise-ness is preserved because kernel and
  keyed reference both call the SAME device row. Change the upstream recurrences
  or `_disp_rows!`'s block loop → update this file and the ulp bound together.
- **GPU kernel ↔ keyed reference ↔ slot map ↔ workgroup-size pin**
  (`src/gpu/gpu_sweep.jl`, `src/gpu/philox.jl`, `docs/specs/gpu-prototype.md`
  G2–G4/G8): TWO pairs, one contract each — `_metro_kernel!` ↔
  `_metropolis_sweep_keyed_ref!` and `_disp_kernel!` ↔
  `_displacement_sweep_keyed_ref!` (proposal slots, `_entry_walk_partial` dispatch
  + zero skips, lane-ordered fold, accept rule). Touch any of them — or the Philox
  slot layout, or the pinned default ws — and the other side plus the G-record move
  together; gate: the full-sweep bitwise sections of `test/unit/test_gpu.jl`.
  The two move kinds need BOTH disjoint Philox slots (independence within one
  compound step) AND separate `GPUChainState` counters (`sweep_index` /
  `disp_index` — several displacement passes per step must not repeat a draw);
  neither substitutes for the other.
- **Device channel plumbing ↔ the host sweeps' skips and row ranges**
  (`src/gpu/gpu_hamiltonian.jl`, `src/gpu/gpu_sweep.jl`, `src/updates.jl`, G8):
  `_entry_walk_partial`'s `lo:hi` IS `delta_energy(c, zold, znew, lo, hi)`
  (energy.jl), full-length `znew` and absolute indexing included — ONE convention,
  host and device, `SweepScratch.znew` and `@localmem Float64 NROWS` alike. Do not
  reintroduce a block-relative buffer to save localmem: a `lo = nlm+1` walk handed one
  reads the spin block as displacement rows, silently. `_channel_colors`'
  `spin_sites`/`disp_sites` are the `site_has_spin`/`site_has_disp` skips of
  `metropolis_sweep!`/`displacement_sweep!` moved to launch time. Change a host
  skip predicate or a move's row block and the device schedule/range must move
  with it. `_table_arrays` is the single field list both `GPUTiledHamiltonian`
  (device) and `_host_tables` (reference/tests) build from — add a `_GPUTables`
  field there, not in either constructor. Gates: the joint sections of
  `test/unit/test_gpu.jl`, including the pure-spin `spin_sites == H.color_sites`
  identity that keeps every pre-M4 bitwise gate meaningful.
  Since the shared `GPUTiledHamiltonian` constructor no longer rejects joint models,
  **each** gradient entry point carries `_require_spin_only` itself —
  `GPUGradientScratch`, `gpu_zlm_rows!`, BOTH `gpu_energy_gradient!` overloads,
  `_gpu_gradient_rows!`, `_gradient_lane_ref!`. Neither `gsc` nor a neighbouring
  guard substitutes: a scratch from the spin restriction of the same model has the
  matching shape, and `refresh_zrows = false` skips `gpu_zlm_rows!`. Add a gradient
  entry point → add the guard AND its `@test_throws`.
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
