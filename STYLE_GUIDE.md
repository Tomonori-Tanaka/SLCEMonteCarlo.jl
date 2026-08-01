# Style guide

**Read this file before writing, reviewing, or renaming code in this package.**

> The shared Julia style lives in `~/Packages/CLAUDE.md` ("Julia style"): official
> Julia + DFTK guides, `for i = 1:n` vs `for x in xs`, explicit named tuples,
> ≤ 92 columns, hot-path `SVector`/`MVector`/`@views`/`@inbounds`. Section 1
> below is the SLCE-family **naming contract** and is identical in all five
> packages; the sections after it are this package's own deltas.

## 1. Suite-wide naming

> **This section is mirrored verbatim in `SLCE.jl`, `SLCEMonteCarlo.jl`,
> `SLCEDynamics.jl`, `SLCETools.jl` and `SphericalTensorFC.jl`.** The canonical
> copy is `SLCE.jl/STYLE_GUIDE.md`: change it there and propagate to the other
> four in the same session. A copy that has drifted is a defect, not a local
> preference.

### 1.1 The rule

DFTK's, adopted verbatim: **"Avoid shortening variable names only for its own
sake"**, under **"readability over consistency"**. This code is read far more
often than it is written, and the reader is usually its author some months later.
A name has to earn its length, and most of the time spelling the word out is what
earns it:

| Write | Not |
|---|---|
| `estimator` | `est` |
| `config`, `configs` | `cfg`, `cfgs` |
| `index` (or `i` in `for i = 1:n`) | `idx` |
| `result` | `res` |
| `selection` | `sel` |
| `positions` | `pos` |
| `checkpoint` | `ck` |
| `overrelaxation` | `or` |
| `workgroupsize` | `ws` |

The rule binds hardest on the **public surface** — exported names, `public`
names, struct fields, keyword arguments — because those are read by people who
cannot see the definition. A loop counter that lives for three lines is free.

### 1.2 When an abbreviation is right

Keep the short form when **at least one** of these holds. Otherwise spell it out.

1. **It is the literature symbol.** `l`, `m`, `k`, `L`, `q`, `u`, `f`, `d`, `Φ`,
   `kT`, `β`, `τ`, `ε`, `ω`. A physicist reads these faster than any expansion,
   and expanding them severs the correspondence with `references/` and with the
   manuscripts the package implements. This is *readability over consistency*
   doing its job, not an exception to it.
2. **It is a field acronym**: SALC, ASR, LLG, MC, PT, NPT, SOC, PBC, MFA, vMF,
   DFT, GPU, CV, GCV, OLS, OMP, FDT, QTB, S(q,ω). Written `UpperCamelCase` in a
   type name (`ASRReparam`, `MCView`), lowercase in a function name
   (`build_asr`, `run_mc`, `run_pt`).
3. **It is owned by an external contract.** A StatsAPI generic (`coef`, `dof`,
   `nobs`, `r2`, `fit`, `predict`, `residuals`, `coeftable`, `islinear`); a file
   format token (`NAT`, `NKD`, `NORDER`, `PREFIX`, `MAGMOM`, `SAXIS`,
   `FORCE_CONSTANTS`, `M_CONSTR`); a foreign package's own field spelling
   (Sunny's `latvecs`). Renaming these de-registers us from the contract — see
   §1.8.
4. **It is formula-local.** A single letter whose scope is a few lines and whose
   meaning is written in the equation immediately above it: `X`, `y`, `β` in a
   regression kernel, `A` for a constraint matrix, `Z` for its null space.

**The standing keep-list**, which no rename pass may expand without a reason
written down here:

`l m k L Lf L_S ls lmax pmax lsum nbody Lseq · q Zlm Rlm Plm dnPl · kT beta tau
kappa rho alpha sigma · u f d Φ · X y beta_p X_E y_E X_T y_T X_F y_F ·
coef dof nobs r2 rmse_* rss_* gcv · nat nkd (inside the I/O adapters only) ·
SALC ASR LLG MC PT NPT SOC PBC MFA vMF DFT GPU CV GCV OLS OMP`

`nbody` is on the list because "N-body" is the standard phrase for the axis it
names, not a contraction of "number of bodies".

### 1.3 Counting names

A **public** name that counts something is `n_<thing>`, with the underscore:
`n_atoms`, `n_sites`, `n_active`, `n_spin_active`, `n_disp_active`, `n_cells`,
`n_salcs`, `n_ops`, `n_alive`, `n_holdout`. A local may write `n` plus the noun
if it is obvious in place.

The suite currently also carries `nfolds`, `nbins`, `ntasks`, `nsteps`, `nlm`,
`nrows`, `nlambda`, `nterms`, `ngroups`, `nfft`, `nsegments`, `nrealizations`,
`nsup`, `ncell`, `nacc`, `natt`, `nspin`, `nsite`. These are **debt, not
precedent** (§1.9): match the surrounding file when editing one, use `n_<thing>`
for anything new.

### 1.4 One concept, one spelling

The suite is five packages around one method, so a concept that crosses a package
boundary must not change its name on the way. The canonical spellings:

| Concept | Canonical | Also seen (debt, §1.9) |
|---|---|---|
| Fitted coefficient vector of a model | `coefficients` | `jphi` (SLCE), `coeffs` (SphericalTensorFC) |
| Constant/intercept term of the expansion | `j0` field, `intercept(model)` accessor | — |
| Per-term scalar prefactor | `coefficient`; `scaled_coefficient` when a scale is folded in | `coef`, `scaled_coef` |
| Atoms in the reference cell | `n_atoms` | `nat`, `natoms` |
| Body order — the count | `nbody` (keep-list) | `n_body` |
| Body order — the label on one object | `body_order` | `body` |
| Per-site decoration labels | `SiteDecor` with `spin_l`, `disp_k`, `disp_l` | bare `k`, `l` (SphericalTensorFC) |
| A term's per-site factor list | `slots` | `factors` |
| Which training datum a design row came from | `<channel>_datum` | `torque_config`, `force_config`, `energy_structure` |
| Resolved truncation rows on a spec | `sector_rules` | `sectors` |
| Species labels on a spec | `species_labels` | `labels` |
| Thermal energy, model units | `kT` (and `kTs`, `resolve_kT`) | `kt`, `kts`, `resolve_kt` |
| Absolute temperature, kelvin | `temperature` | — |
| Supercell repeat counts | `dims` | `dim` |
| Per-atom magnetic moments | `magmoms` | `magmom` |
| Spin configuration, `3 × n_atoms` unit vectors | `directions` on a training datum; `config`/`configs` on a sampler state | `spins` as a *field* name |
| Displacement field | `displacements` in public fields; `u` formula-locally | `disps` in public fields |
| Move bookkeeping | `n_accepted_<move>` / `n_attempted_<move>` on state; `acceptance_<move>` on results | `acc_*`, `att_*`, `nacc`, `natt` |
| Overrelaxation, anywhere in a public name | `overrelaxation` | `or` |
| Effective degrees of freedom | `effective_dof` | `edof` |

Two rules follow from the table rather than sitting in it:

- **A field name and its persisted key are separate decisions.** `sector_rules`
  is the field; `"sectors"` is the document key, kept because renaming a key
  strands saved files while renaming a field costs one edit. Never move a
  persisted key to follow a field.
- **A constructor that is a function is `snake_case`**, even when it returns one
  type: `spin_datum`, `lattice_datum`, `joint_datum`. An `UpperCamelCase`
  spelling promises a type, and it kept promising one after `SpinDatum` stopped
  being a type.

### 1.5 Reserved short locals

Four short names are sanctioned because they appear thousands of times in one
package's hot paths, and each one belongs to exactly one meaning **across the
whole suite**:

| Name | Means | Owner |
|---|---|---|
| `H` | `TiledHamiltonian` | SLCEMonteCarlo (and its consumers) |
| `st` | `ChainState` | SLCEMonteCarlo (and its consumers) |
| `sc` | `SweepScratch` | SLCEMonteCarlo |
| `spec` | `BasisSpec` | SLCE, SphericalTensorFC |

Do not reuse them for anything else: a reader who works across two packages in
one sitting should not have to re-learn a two-letter name. A supercell is
`supercell`, a run specification is `run_spec`, a scratch buffer of any other
type is `scratch` or a typed name.

### 1.6 Case, suffixes, and tiers

Beyond the baseline (`UpperCamelCase` types, `snake_case` functions and
variables, `UPPER_SNAKE_CASE` constants, trailing `!` for mutation):

- **A leading `_` marks the private tier, and it covers types and constants too**
  — `_RunSpec`, `_PTLane`, `_ASR_RTOL_ZERO` — not only functions. A `_` name is
  never in `export` and never in `public`.
- **The public surface is two-tiered everywhere**: a lean `export` list plus a
  `public`-keyword tier reached by qualification, machine-checked with
  `Base.ispublic` / Aqua. A new public-but-unexported name goes in the `public`
  block or it is not public.
- **Predicates are `has_*` or `is_*`** — `has_disp`, `has_torque`, `is_soc_free`
  — never a bare adjective and never a `?` suffix.
- **Physical constants carry their unit in the name**: `KB_EV`, `HBAR_EV_FS`,
  `MU_B_EV_T`, `GPA_PER_EV_A3`, `ANGSTROM_PER_BOHR`, `EV_PER_RY`.
- **A `const` that aliases a type keeps `UpperCamelCase`** (`SpinConfig`,
  `_WigCache`) — it is a type name, not a value.
- **Type parameters**: a single capital for a type (`{T}`, `{R}`, `{N}`, `{D}`),
  `UPPER_SNAKE` for a compile-time *value* (`{LMAX}`, `{NROWS}`), and a comment
  for an abbreviated container tag (`{VI,VB,VF}`) saying what each stands for.
  Always brace the `where`.

### 1.7 Unicode

**Unicode in prose, ASCII in identifiers.** Docstrings, comments and the guides
write `Zₗₘ`, `Φ`, `Ā`, `∇Zₗₘ` — that is what makes them match the papers. The
identifiers those docstrings describe are `Zlm`, `grad_Zlm`, `Phi`, `Abar`.
Exported names and keyword arguments are ASCII without exception, so that a user
can type them.

One carve-out, in SLCETools only and already documented there: Unicode locals
(`τ`, `κ`, `β`, `ρ`, `θ`, `φ`) are preferred inside the mean-field kernels, where
they match `docs/specs/mfa-sampling.md` symbol for symbol.

### 1.8 Names that must not be renamed

Renaming any of these is a compatibility break, not a style improvement:

- **Persisted document keys** — SLCE's TOML model schema (`jphi`, `j0`, `ls`,
  `nbody`, `sectors`, `folded`, `shifts`, `pair_cutoff`, `isotropy`, …), the JLD2
  checkpoint keys of SLCEMonteCarlo (schema v6) and SLCEDynamics (schema v3),
  including the integrator names persisted as strings.
- **External format tokens** — ALAMODE (`NAT`, `NKD`, `NORDER`, `DFSET`,
  `FORCE_CONSTANTS`), phonopy, VASP (`MAGMOM`, `SAXIS`, `M_CONSTR`, `POSCAR`),
  Sunny's field spellings.
- **StatsAPI generics** we implement (`coef`, `fit`, `nobs`, `dof`, `coeftable`,
  `islinear`, `residuals`, `predict`, `r2`) and the `coeftable` column labels.
- **Names one package borrows from another** — `n_atoms`, `has_disp`, `KB_EV`,
  `resolve_kt` are SLCE's generics, extended downstream and never re-defined; a
  rename is a five-repository change. `SLCEMonteCarlo._gradient_lane_ref!` is
  `_`-prefixed but called by qualified name from SLCEDynamics, so it is public in
  practice.
- **The `:spins` observable name and its component layout** in SLCEDynamics —
  validated by name when a checkpoint resumes.

When a struct field on this list has a bad name, rename the **field** and leave
the key string alone; that is what `sector_rules` ↔ `"sectors"` already does.

### 1.9 Known deviations

The code does not yet satisfy §1.2–§1.4 everywhere. These are the deviations on
record, so that this guide describes the rule *and* the debt instead of quietly
disagreeing with the source:

| Deviation | Where | Why it is still there |
|---|---|---|
| `jphi` / `coeffs` for the coefficient vector | SLCE, SphericalTensorFC | field of exported types, named in docs, examples, and SLCEMonteCarlo's fixtures; needs a deprecation path |
| `kt` beside `kT` for one quantity | SLCEDynamics (`_RunSpec.kt`, `kts`), SLCE (`resolve_kt`) | `resolve_kt` is borrowed by three packages |
| `disps` in public fields | SLCE, SLCEMonteCarlo, SphericalTensorFC | also a JLD2 checkpoint key |
| `acc_*` / `att_*` / `nacc` / `natt` | SLCEMonteCarlo, SLCETools | checkpoint keys |
| `or` for overrelaxation in `or_per_metropolis`, `acceptance_or` | SLCEMonteCarlo | user-facing keyword, documented in the guides |
| `nfolds`, `nbins`, `ntasks`, `nsteps`, … | all | public keywords |
| `body` vs `nbody` vs `sites` for one axis | SLCE, SphericalTensorFC | `sites` is a schema-v6 document key |
| `gzee` / `gth` / `pref` | SLCEDynamics | `gzee` is a field of the exported `LLGProblem`, and the three are one naming family — moving `gth` alone would split it |
| `nat`, `nkd` outside the I/O adapters | SphericalTensorFC | `SuperCell.nat` is a field of an exported type |
| `midx`, `cidx`, `colidx`, `idxp`, `idxm` | SLCE | each needs its own word, not a mechanical expansion |
| `iw`, `crow`, `qsv`, `ns_t`, `dtf`, `dtm` | SLCEDynamics | formula-local; each needs reading before it can be named |

Fixing one of these is a deliberate, separately reviewed change — never a
drive-by edit inside an unrelated commit, and never a partial sweep that leaves
two spellings where there was one.

**Already done (2026-08-01), so do not "restore" the short form**: the
internal-only sweep renamed `est` → `estimator`, `idx` → `index`, `cfg`/`cfgs` →
`config`/`configs`, `res` → `result`, `sel` → `selection` / `selector`,
`ck` → `checkpointer`, `tr` → `trace`, `sc` → `scratch` (SLCEDynamics) /
`supercell` (SphericalTensorFC), `spec` → `run_spec` (SLCEDynamics), `filt` →
`noise_filter`, `bq` → `biquad`, `ebar` → `spin_means`, plus the internal helper
functions `_ck_*` → `_checkpoint_*`, `_edof*` → `_effective_dof_*`, `_sch_*` →
`_schedule_*`, `_sm_*` → `_grid_*`, `_ex_*` → `_exchange_*`, `_gar_weights!` →
`_group_adaptive_weights!`, `_fp_mix` → `_fingerprint_mix`, and the one-word
helpers (`_wig`, `_frob`, `_tclose`, `_jnum`, `_mat3`, `_ge0`, `_det3`, `_adj3`,
`_mfslice`). No exported name, `public` name, struct field or persisted key was
touched.

## 2. Argument order

Energy/update kernels take `(output-or-state, hamiltonian, site, …)` — the mutated
argument first (Julia convention), the `TiledHamiltonian` second.

## 3. Package-specific naming

- This package **owns all three** reserved locals of §1.5: `H` is a
  `TiledHamiltonian`, `st` a `ChainState`, `sc` a `SweepScratch`.
- "site" always means a supercell site index in `1:n_sites(H)`; "atom" always means
  a training-cell atom index in `1:H.n_cell_atoms`. Never mix the two words.
- Internal helpers carry a leading underscore; the public-but-unexported tier
  (`SLCEMonteCarlo.site_coeffs!`, …) does not.
- **Two activity predicates, never interchangeable**: `site_active` / `n_active`
  (a site moves in *either* channel) and `site_has_spin` / `n_spin_active` (the
  spin channel only). A name that says "active" without naming its channel is
  ambiguous — say which one it means.
- A tensor axis is a **slot**, not a site: `ScaledTerm.slots`, `TermSlot`, and the
  `(4π)^(n_spin_slots/2)` scale all count slots. A site may carry two slots.
