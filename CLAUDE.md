# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

**Before writing, reviewing, or renaming code here, read
[`STYLE_GUIDE.md`](STYLE_GUIDE.md).** Its §1 is the SLCE-family naming contract —
mirrored verbatim in all five packages, canonical copy in `SLCE.jl` — and the
sections after it are this package's own deltas. **Read
[`ARCHITECTURE.md`](ARCHITECTURE.md)** when you need the dependency graph, the
include layering, or a reading order through the source.

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

The register below is an **index**. Each line names what must move together and
the gate that proves it did; the reasoning — why the invariant holds, which
alternatives were rejected and how they fail, and the measured numbers behind
every tolerance — lives in **`docs/specs/coupled-sites.md`**, keyed by the same
bolded lead phrase. Read the entry there before changing anything on its line;
adding a coupling means writing the entry there and the index line here.

- **`hamiltonian.jl` ↔ the core's introspection contract** — `SpinMultipoleTerm` /
  `DecoratedTerm` field meanings, the raw-coefficient scale rule, shifts anchoring.
  Gates: `test_hamiltonian.jl`, `test_joint.jl`.
- **Channels: `ScaledTerm.slots` ↔ `SLCE.row_layout` ↔ `_disp_rows!` ↔
  `SLCE.site_rows!` ↔ the merged site programs ↔ `_fingerprint`** — a tensor axis is a
  SLOT, not a site. Five things move together: the row numbering, the
  `(4π)^(n_spin_slots/2)` scale, the bitwise row-batch equality, the fingerprint's
  layout mixing, and `disp_scale`. Gates: `test_joint.jl` throughout.
- **The displacement channel of the sampler: `ChainState.disps` ↔ `SweepScratch` ↔
  `_attempt_disp!` ↔ `_recenter!` ↔ `_renormalize!` ↔ `_swap_payload!` ↔ the checkpoint
  schema** — six rules, including that the two proposal widths have different
  DIMENSIONS, that a single-channel move must use the range-limited `delta_energy`,
  that `_recenter!` lives in `_renormalize!` and nowhere else, and that every
  restriction of the displacement state space must be gauge-invariant. Gates:
  `test_joint.jl` "joint displacement sampling" / "joint drivers…",
  `test_checkpoint.jl`.
- **Coefficient hot-swap: `_push_term_programs!`'s skip ↔ `sent_base`/`sent_term` ↔
  `set_coefficients!` ↔ `keep_zero_terms` ↔ the GPU tables and the checkpoint
  fingerprint** — the site-program skip tests `folded[idx]`, never `coef · folded[idx]`.
  Gates: `test_coefficients.jl`.
- **`StrainSchedule` ↔ `keep_zero_terms` ↔ upstream `decorated_terms(; keep_zero)` ↔
  `set_coefficients!`** — both halves of the support freeze are required. Gates:
  `test_strainschedule.jl`.
- **The NPT strain move: `strain_move!` ↔ `strain_delta_energy`/`_strain_log_weight` ↔
  `ChainState.strain` ↔ the run drivers ↔ the checkpoint schema ↔
  `sync_coefficients!`** — the longest entry: one elastic-energy source, the
  proposal/exponent branch pair, the `(H, chain)` contract, the reference-scale
  fingerprint, `_rescale_escape!`'s detector-vs-reporting split, the volume-grid
  boundary screen, the GPU sync, PT lane clones and the exchange weight, the
  three-place NPT target, and the warm start. Gates: `test_strainschedule.jl`.
- **The §8(ζ) pressure diagnostic: `_energy_with_coefs` ↔ `_total_energy`, and the
  exact-derivative trio ↔ their Horner sources**. Gates: the two §8(ζ) testsets in
  `test_strainschedule.jl`.
- **Upstream BREAKING spec keywords ↔ this package's fixtures/benches/docs/assets** —
  a downstream rename is not done when SLCE's own tests are green.
- **Names this package borrows rather than owns** — `has_disp`, `n_atoms`, `KB_EV`,
  `resolve_kt` are SLCE's; never re-define them here.
- **`energy.jl` 4-function contract ↔ `updates.jl` ↔ SLCETools' `mc/metropolis.jl`**.
  Gates: `test_energy.jl` ΔE ≡ total-energy difference.
- **`lm_index` ordering ↔ `zlm_row!` ↔ the overrelaxation l=1 axis extraction**. Gate:
  `test_overrelaxation.jl` (pure-l=1 OR has `ΔE ≡ 0`, acceptance 1).
- **`reduce.jl` ↔ `hamiltonian.jl` tiling ↔ `geometry.jl` ordering ↔ SLCE's canonical
  members** — canonical site order, the per-coset census, `ReducedCell{T}` mirroring
  its term surface. Gates: `test_reduce.jl`.
- **`energy.jl` `_site_grad` ↔ `site_gradient` ↔ `energy_gradient!` ↔ `minimize.jl`
  `_gradient!`** — one kernel behind three surfaces. Gates: the bitwise consistency
  tests in `test_gradient.jl` / `test_minimize.jl`.
- **Checkpoint writer ↔ reader ↔ schema doc** — plain-data JLD2, schema v6. Gate:
  bit-identical resume (`test_checkpoint.jl`).
- **`MCView` ↔ every observable ↔ SLCEDynamics' `_measure!`** — an observable takes ONE
  argument so a new channel does not break every definition ever written; the view
  EMPTIES `disps` on a model with no displacement channel.
- **Displacement observables ↔ `_recenter!`'s cadence** — the observables subtract the
  component centre of mass themselves; the guide is the third site. Gate: `test_joint.jl`
  "displacement observables see the centre-of-mass-free frame".
- **Displacement channel gates: `n_disp_active > 0`, never `has_disp`** — `has_disp` is a
  property of the row LAYOUT; mirror the spin side's `n_spin_active`.
- **A run must be able to move something: `_require_moves` ↔ `_resolve_or_passes` ↔
  `_resolve_disp_passes` ↔ `_compound_sweep!`'s spin skip ↔ the sweep entry guards** — a
  lattice-only Hamiltonian is first-class, and it is the case every spin-shaped
  assumption fails on. Gates: `test_joint.jl` "joint drivers: pass scheduling…".
- **`n_disp_comps > 1` is announced at construction, and this is the only screen for
  it** — a warning, not a refusal, and deliberately no `maxlog`. Gate: `test_joint.jl`
  "disjoint displacement components are announced".
- **`harmonic_stability`'s verdict tolerance ↔ the acoustic modes** — `tol` is MEASURED
  from the certified-flat directions, not hard-coded. Gate: the same count at three
  `step` spanning 10³.
- **Observable conventions** (C/χ/U definitions) live in ONE place,
  `docs/specs/binning-observables.md`; since M4 an `Evaluable` must declare its `scope`
  (`:energy` ⇒ `n_active`, `:spin` ⇒ `n_spin_active`). Gate: an Einstein oscillator's
  `C = 1.5 k_B` per atom.
- **Coloring ↔ sweeps ↔ stationarity spec** — class instance-disjointness plus per-site
  RNG streams and the fixed-order ΔE reduction. Gate: `test_parallel.jl` (serial ≡
  parallel `==`).
- **Device tesseral row ↔ host `_zlm_row!` ↔ upstream recursions** — an
  operation-order-faithful replica; any upstream change breaks the dense bitwise gate in
  `test_gpu.jl`.
- **Device displacement rows ↔ host `_disp_rows!` ↔ `SLCE.SolidHarmonics`** — bitwise on
  `k = 0`; the `k ≥ 1` one-ulp gap is an accepted scope reduction (G8), not a bug.
- **GPU kernel ↔ keyed reference ↔ slot map ↔ workgroup-size pin** — two pairs, one
  contract each; the move kinds need BOTH disjoint Philox slots and separate counters.
- **Device channel plumbing ↔ the host sweeps' skips and row ranges** — `lo:hi` IS
  `delta_energy`'s range, one convention host and device; each gradient entry point
  carries `_require_spin_only` itself.
- **Device gradient ↔ lane reference ↔ upstream grad recursions** — one arithmetic
  contract, deliberately libm-free; `_gradient_lane_ref!` is called by name from
  SLCEDynamics.
- **Inactive-site convention** — TWO predicates since M4 slice 3b: `site_active`/
  `n_active` (either channel) drives the coloring and schedule, `site_has_spin`/
  `n_spin_active` drives the spin sweeps, `_renormalize!`, the spin observables and
  their normalizations. Update sweeps SKIP, standard observables EXCLUDE, and the spins
  stay bitwise frozen — these move together. Re-check `test_inactive.jl` +
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
- `docs/specs/coupled-sites.md` — the reasoning behind every line of the index
  above: why each invariant holds, which alternatives were rejected and how they
  fail, and the measured numbers behind each tolerance. **This file carries the
  index only.** Detail belongs there, so that what is loaded into every session
  stays instructions; when an entry grows past a line or two here, move the
  growth there.
- `docs/specs/*.md` — decision records by topic (tiling, update stationarity,
  binning/observable conventions, PT determinism, checkpoint schema, the strain
  move, cell reduction, the GPU prototype).
- `references/` — supporting literature (notes tracked, PDFs local-only).
