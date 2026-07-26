# Decision record — binning error analysis and observable conventions

Status: landed (M2). Owner: `src/binning.jl`, `src/observables.jl`;
gates in `test/unit/test_binning.jl`, `test/unit/test_observables.jl`.
**This file is the authoritative statement of the C/χ/U conventions** — the code
and the guide pages follow it.

## B1 — streaming log-binning, no stored time series

`LogBinner` keeps, per cascade level `k`, the `(count, Σ, Σ²)` of bin means of size
`2^(k-1)` plus one pending half-pair — O(levels) memory. The reported error is the
naive standard error at the **deepest level with ≥ 32 bins** (the plateau proxy);
`τ_int = ((err_pl/err_naive)² − 1)/2`.

Known accuracy: with 32 bins the plateau error itself fluctuates by
`≈ 1/√(2·31) ≈ 13%` (τ, quadratic in it, by ≈ 26%) — fine for error bars, and the
reason the AR(1) test gate uses matching tolerances. Rejected alternative: storing
the full time series for a windowed autocorrelation estimator — accuracy is not
needed at that price (a 32-replica PT run × 10⁶ measurements would cost GB).

## B2 — jackknife over a fixed bin layout for derived quantities

Nonlinear evaluables (C, χ, U) are estimated by leave-one-bin-out jackknife over
`BinStore` bin means: `nbins` (default 32) equal-size bins with
`bin_size = max(1, planned ÷ nbins)` fixed **up front**, so every bin has equal
weight; a trailing remainder (< 1 bin) is dropped. Bias-corrected estimate
`n_b·f(m̄) − (n_b−1)·θ̄`, error `√((n_b−1)/n_b · Σ(θᵢ−θ̄)²)`. For a linear `f` this
reproduces the plain mean and error exactly (machine-precision gate).

## B3 — observable conventions (authoritative)

**Active sites.** A site no cluster instance touches (a species with `lmax = 0`,
e.g. boron, or one whose SALC coefficients all fitted to zero) has a
spin-independent energy: it is flagged inactive (`TiledHamiltonian.site_active`,
count `n_active`), skipped by the update sweeps (frozen at its initial direction;
attempting it would always accept, biasing the acceptance statistics and the step
adaptation), **excluded from every standard observable**, and per-site
normalizations use `n_active`. *Why exclusion and skipping come together*: a frozen
spin included in `:m` would add a constant bias, a free-walking one dilution noise —
either contaminates; excluded, its updates are pure waste. `:sublattice_m` reports
exactly zero for inactive sublattices (a frozen random unit vector would look like
perfect order). The `f(config, energy, H)` signature hands custom observables
`H.site_active` for the same masking.

**Two counts, once displacements exist.** `n_active` counts sites active in
**either** channel; `n_spin_active` counts the magnetic ones. They coincide on every
pure-spin model — every member site of an instance carries an axis, and on a pure-spin
model that axis is a spin — so the distinction was invisible before M4. On a joint
spin–lattice model it is not: the total energy carries a lattice contribution
(≈ 1.5 `k_B` per displacement-active site) that no magnetic site accounts for, and a
displacement-only model has `n_spin_active == 0`. **Energy-derived** quantities
therefore normalize by `n_active`, **magnetization-derived** ones by `n_spin_active`;
an `Evaluable` declares which it needs through its `scope` field (`:energy` /
`:spin`, default `:spin`).

Raw set (`standard_observables(H)`): `:energy`, `:energy2` (total, model units, `j0`
excluded), `:m` (3-vector `Σ_spin_active e / n_spin_active`), `:absm`, `:m2`, `:m4`,
`:sublattice_m` (per training-cell atom, cell-averaged 3-vector, flattened).
Directions only — moment magnitudes (μ_B) are not part of the fitted model. On a model
with `n_spin_active == 0` the magnetization observables are **omitted** rather than
measured: every one is `0/0`, and a column of `NaN`s passes every finiteness check a
caller is likely to write. `standard_evaluables(H)` drops the ones that read them.

Derived (`standard_evaluables`, jackknifed):

- **Specific heat, per active site, in units of k_B** (`scope = :energy`):
  `C/k_B = (⟨E²⟩ − ⟨E⟩²) / (n_active (k_BT)²)`.
  *Why*: intensive (comparable across `dims`); k_B units avoid eV/K clutter and are
  the lattice-MC standard. On a joint model this is the **spin + lattice** heat
  capacity — the classical harmonic limit alone contributes 3/2 per
  displacement-active site, which is also the gate (`_einstein_terms`).
- **Susceptibility, |m|-connected, per spin-active site** (`scope = :spin`):
  `χ = n_spin_active (⟨m²⟩ − ⟨|m|⟩²) / k_BT`.
  *Why*: on a finite system with continuous symmetry `⟨m⟩ = 0` exactly, so the
  textbook connected form degenerates to `n⟨m²⟩/kT`, which grows ∝ N below T_c
  instead of peaking; the |m|-connected form peaks at the transition in both phases
  (the finite-size-scaling standard). The high-T form is user-composable from `:m2`.
- **Binder cumulant, plain ratio**: `U = ⟨m⁴⟩/⟨m²⟩²` — → 1 (ordered), → 5/3
  (disordered, 3-component Gaussian). Chosen over `1 − U/3`-style variants because
  there is no convention factor to get wrong; `U(T)` crossings locate `T_c`
  identically.

## B4 — composability

`Observable(name, ncomp, f(config, energy, H))` and
`Evaluable(name, inputs, f(means::NamedTuple, kT, n); scope)` are plain structs the
run drivers accept as vectors — nothing is hard-coded into the sweep (the
SpinClusterMC pain point). `n` is the count `scope` selects (above). Evaluable inputs
must be scalar observables (validated).
Ferrimagnetic order parameters compose from `:sublattice_m` components.
