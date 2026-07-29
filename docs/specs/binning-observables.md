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
perfect order). The `MCView` a custom observable receives carries `v.H`, so
`v.H.site_active` gives it the same masking.

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

**Displacement set** (added M4 slice 3d, on a model with at least one
**displacement-active** site — `n_disp_active > 0`, NOT merely `has_disp`, since a
joint basis whose displacement couplings all fitted to zero has displacement rows and
nothing to measure): `:u2`, `:u4`, `:sublattice_u2`, `:sublattice_u4`, every one of
them a moment of `u_s − ū_c` — the site's displacement minus the centre of mass of
its displacement-coupling component along that component's FLAT directions.
Sublattices with no displacement axis report exactly zero, the `:sublattice_m`
convention.

**The subtraction happens inside the observable**, not by relying on the sampler's
`_recenter!`. That runs at renormalization points while measurements fire every
`measure_interval`, so at almost every measurement the frame has drifted since it was
last removed and `mean|u|² = mean|u−ū|² + |ū|²` picks up the free random walk. The
excess is linear in `renorm_interval` (measured 0.0041 / 0.081 / 0.400 at 10 / 200 /
1000), always positive, and shrinking with system size — it would read as a
finite-size effect. Only the flat directions are removed: a pinned direction's
absolute frame is physical. A displacement *vector* mean is deliberately absent:
`⟨u − ū⟩ ≡ 0` identically.

`:u2` and `TempResult.disp_rms²` are then the same quantity measured twice: the
observable is binned and carries an autocorrelation-aware error bar, the diagnostic
is a phase average over renormalization points with none. Where they can be
compared, they must agree — the Einstein-oscillator gate checks both against
`3kT/(2a)` at once.

Derived (`standard_evaluables`, jackknifed):

- **Specific heat, per active site, in units of k_B** (`scope = :energy`):
  `C/k_B = (⟨E²⟩ − ⟨E⟩²) / (n_active (k_BT)²)`.
  *Why*: intensive (comparable across `dims`); k_B units avoid eV/K clutter and are
  the lattice-MC standard. On a joint model this is the **spin + lattice** heat
  capacity — the classical harmonic limit alone contributes 3/2 per
  displacement-active site, which is also the gate (`_einstein_terms`).
  **On a strained (NPT) run this is configurational-only — neither `C_V` nor the
  NPT `C_P`.** The β-conjugate state energy of the NPT target is
  `W = E_config + n_cells·j0(s) + P·V(s)` (strain-move.md S10), whose `j0` and
  `P·V` halves fluctuate with the sampled volume; `var(E_config)` alone measured
  3.4 % low on the Einstein-well fixture, and `C_P − C_V = TVα²B` reaches tens of
  percent on real solids near melting. `:energy` likewise omits the varying
  `j0(s)`. On a strained run use `npt_observables` (next bullet) instead.
- **Configurational enthalpy and isobaric specific heat** (`npt_observables(sch,
  H; pressure...)`, strained runs only — an optional add-on, not part of the
  standard set): raw `:enthalpy = W` and `:enthalpy2 = W²` with the same
  `W = E_config + n_cells·j0(s) + P·V(s)` the strain move and
  the strained-PT exchange rule weigh, and the evaluable (`scope = :energy`)
  `:npt_specific_heat = (⟨W²⟩ − ⟨W⟩²)/(n_active (k_BT)²)`.
  *Why this is the isobaric C*: the sampled measure `p ∝ V^{N_mob} e^{−βW}` has a
  β-independent Jacobian and a β-independent (grid-truncated) volume domain, so
  `C/k_B = var(W)/(k_BT)² = d⟨W⟩/d(k_BT)` exactly — gated by a finite-difference
  cross-check in `test_strainschedule.jl`; the formula itself is pinned at
  machine precision against the fixture's analytic `j0`/`V`. Read it as the
  physical `C_P` only while the sampled `p(V)` sits well inside the grid (the
  truncated measure is a volume-constrained system), and note v0's cell shape is
  frozen (hydrostatic-only). Configurational like everything here: no momenta,
  so for an absolute value add the classical kinetic term analytically —
  `(3/2)·n_disp_active/n_active` in the reported per-active-site units (`3/2`
  per *mobile* atom; the counts differ on a model with spin-only sites). The
  factory must be given the run's own schedule and pressure — the view carries
  neither (the Hamiltonian itself is identity-checked per view; lane clones
  pass). Details: `strain-move.md` S11.
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
- **Displacement moment ratio**: `⟨u⁴⟩/⟨u²⟩²`. *Why a ratio*: for a harmonic model
  every `σ_s² ∝ T`, so it is **temperature-independent** whatever the crystal — a
  signature that needs no reference value and cannot be faked by a mis-set `kT`,
  which makes it the continuous companion to the escape detector
  (`updates-stationarity.md` U8) and to the harmonic screen (`harmonic_stability`).

  *Why not 5/3*: `:u2` and `:u4` each average over sites BEFORE the ratio is taken,
  so the harmonic level is `(5/3)·mean_s(σ_s⁴)/(mean_s σ_s²)² ≥ 5/3` by Jensen, with
  equality only when every displacement-active site samples the same **isotropic**
  Gaussian. Measured on a two-sublattice Einstein crystal (exactly harmonic, exactly
  isotropic per site): 1.665 at `(a₁,a₂) = (2.5, 2.5)`, **2.255** at `(2.5, 10)`,
  **3.028** at `(1, 20)` — against the Jensen predictions 1.667 / 2.267 / 3.031. Site
  anisotropy shifts it too (`1 + 2 tr(Σ²)/(tr Σ)²`, which is 5/3 only for cubic site
  symmetry). The clean 5/3 test is the **per-sublattice** ratio
  `⟨u⁴⟩_a/⟨u²⟩_a²`, since translation-equivalent sites share a covariance and Jensen
  collapses to equality — measured 1.66 in all three cases above. That is why both
  sublattice moments are in the standard set.

## B4 — composability

`Observable(name, ncomp, f(v::MCView))` and
`Evaluable(name, inputs, f(means::NamedTuple, kT, n); scope)` are plain structs the
run drivers accept as vectors — nothing is hard-coded into the sweep (the
SpinClusterMC pain point). `n` is the count `scope` selects (above). Evaluable inputs
must be scalar observables (validated).
Ferrimagnetic order parameters compose from `:sublattice_m` components.
