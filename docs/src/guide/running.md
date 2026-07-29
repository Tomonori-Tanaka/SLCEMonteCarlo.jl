# Running chains

```@meta
CurrentModule = SLCEMonteCarlo
```

[`run_mc`](@ref) drives one Markov chain: per temperature it **thermalizes**
(`sweeps_therm` sweeps, with the proposal step adapting), freezes the kernel, then
**measures** (`sweeps_measure` sweeps, recording every `measure_interval`-th).
One sweep is one single-spin Metropolis attempt per **active** site, scanned in
the Hamiltonian's color-class order (inactive, non-magnetic sites stay frozen —
see [`TiledHamiltonian`](@ref)) with the exact `ΔE` of the fitted Hamiltonian —
any body order, no linearization — optionally followed by `or_per_metropolis`
overrelaxation sweeps and, on a joint spin–lattice model, `disp_per_metropolis`
displacement sweeps. `sweep_tasks` executes each sweep on several concurrent
tasks with a bit-identical result (see the parallelism guide).

## Annealing vs independent chains

A temperature **collection** runs in the given order with the chain carried over
(fresh thermalization at each value):

```julia
r = run_mc(H; temperature = [1200, 900, 600, 450, 300], seed = 1)   # annealing
ri = run_mc(H; temperature = [1200, 300], carryover = false)  # independent chains
```

High → low ordering is an annealing run — the standard way to reach a
low-temperature ordered state from a random start. `carryover = false` restarts
each temperature from a fresh random configuration instead.

## Seeding

By default every run draws a fresh `seed = rand(UInt64)`, so repeated runs are
independent samples — you cannot silently average the same chain twice. Pass an
explicit `seed` when you need bit-reproducibility (tests, docs, debugging).
Either way the seed actually used is recorded in `MCResult.seed` /
`PTResult.seed` and in every checkpoint, so any run can be reproduced after the
fact.

## The adaptive step

The Metropolis rotation scale `step` adapts every `adapt_interval` thermalization
sweeps toward `adapt_target` (default 0.5) acceptance, then **freezes** for the
whole measurement phase (a kernel that keeps responding to chain history would
bias expectations and break bit-reproducible restarts). Per temperature,
`TempResult.final_step` reports the frozen value and
`TempResult.acceptance_metropolis` the measured acceptance.

On a joint model the displacement proposal width `step_u` adapts the same way, on
its own acceptance counters (`TempResult.final_step_u` / `acceptance_disp`). The
two are **not** interchangeable: `step` is an angle in radians, `step_u` a length
in the model's units (Å for a DFT-fitted model), and nothing in the Hamiltonian
converts one to the other.

Two caveats:

- On a lattice with **decoupled sites** (e.g. atoms outside every cluster) the
  acceptance has a floor — free sites always accept — so the target may be
  unreachable and the step pins at its bound. Harmless, but read the acceptance
  accordingly.
- At very low temperature the antipodal-flip component (20% of proposals) is
  almost always rejected; the rotation component alone carries the adaptation.

## Diagnostics to check before trusting numbers

```julia
p = r.points[1]
p.acceptance_metropolis      # ~adapt_target if adaptation had room
p.max_drift                  # incremental-energy drift; ~1e-12·|E| is healthy
p.stats[:energy].tau_int     # integrated autocorrelation time (in measurements)
```

If `τ_int` is large, raise `measure_interval` (cheaper statistics per stored
measurement) or mix in overrelaxation; if the energy trace should be trend-free
but `⟨E⟩` differs between seeds far beyond its error bar, the chain is trapped —
see the metastability discussion in `docs/specs/updates-stationarity.md` and
reach for [`run_pt`](@ref).

## Overrelaxation

`or_per_metropolis > 0` mixes deterministic reflection sweeps between Metropolis
sweeps: each spin reflects about its local `l = 1` field axis and the move is
accepted with the exact-ΔE Metropolis rule. For exchange-dominated (`l = 1`)
models the reflections are energy-conserving and always accepted — fast, free
decorrelation. For strongly anisotropic models the `l ≥ 2` remainder makes
reflections cost energy and the OR acceptance can collapse at low temperature —
check `TempResult.acceptance_or`.

## Displacement sweeps (joint spin–lattice models)

A model fitted with displacement sectors samples its atomic displacements too:
each displacement sweep attempts one isotropic Gaussian shift of width `step_u`
per displacement-active site, with the same exact-ΔE Metropolis rule.
`disp_per_metropolis` sets how many of them follow each Metropolis spin sweep and
**defaults to the model**: `1` on a joint model, `0` on a pure-spin one. Any fixed
value samples the same ensemble (only the correlation times change); passing `0`
on a joint model deliberately selects the clamped-ion ensemble instead, and
passing a nonzero value to a pure-spin model is an error rather than a silent
no-op.

```julia
r = run_mc(Hjoint; temperature = 300, step_u = 0.05, disp_per_metropolis = 2,
           renorm_interval = 200)
p = r.points[1]
p.acceptance_disp    # displacement acceptance (NaN if no attempt was made)
p.disp_rms           # √⟨|u|²⟩ over the phase, centre-of-mass-free
p.disp_max           # largest single displacement anywhere in the phase
p.disp_checks        # how many observations the two above rest on
p.escaped            # ← check this before trusting any displacement number
```

`disp_rms` is a phase **average** and `disp_max` an extreme value over the same phase
(so it grows with `disp_checks` and is not comparable between runs of different
length); both are `NaN` when `disp_checks == 0`, which happens when the measurement
phase is shorter than `renorm_interval`. They are diagnostics, not observables — they
carry no error bars.

On a joint model the standard set is joint-aware: `:specific_heat` is normalized by
every active site (it is the **spin + lattice** heat capacity — the classical harmonic
limit alone contributes 3/2 `k_B` per displacement-active site), while `χ` and the
Binder cumulant stay per **spin**-active site. On a displacement-only model the
magnetization observables are omitted rather than reported as `NaN`.

!!! warning "The truncated expansion need not be bounded below"
    `E(u)` is a finite polynomial, so `exp(−βE)` is a probability measure only when
    the leading even form is positive definite — nothing in the fit guarantees it.
    When it fails the chain has no stationary distribution and simply runs
    downhill, with an *exact* incremental energy and an acceptance near 1: no
    pre-existing diagnostic sees it. The sampler therefore measures recurrence
    directly, warns, and sets `TempResult.escaped`. A set point is not "noisy
    data" — it is a run whose displacement observables mean nothing. See
    `docs/specs/updates-stationarity.md` U8, and screen the model's dynamical
    stability (`SLCE.dynamical_matrix`) at the sampled spin configurations.

    The detector's block test needs **15 renormalization checks per measurement
    phase** before it can raise its three consecutive strikes, and the default
    `renorm_interval = 1_000` against `sweeps_measure = 10_000` gives only 10 —
    a joint run below the bar is warned about up front, and should lower
    `renorm_interval`. Under it, `escaped == false` means *not screened*, not
    *clean*; `disp_checks` is what tells the two apart.

The centre of mass of each displacement-coupling component is a flat direction
wherever the acoustic sum rule holds, so the sampler re-centres at every
renormalization; displacements are reported in that centre-of-mass-free frame. A
model built with `fixed_reference = true` (a substrate-clamped slab, a pinning
defect) keeps its absolute frame along the directions the construction gate
measured as pinned.

## NPT: sampling the cell volume (strain moves)

Passing a [`StrainSchedule`](@ref) turns a joint run isothermal–isobaric: an
outer Metropolis move rescales the whole cell over a `SLCE.StrainedModels` volume
grid, installing the grid's interpolated coefficients in place
([`set_coefficients!`](@ref)) and rescaling the displacements affinely. Without
it, a run samples the **constant-strain (fixed cell) ensemble** — a different
ensemble (`F(T, ε)`: no volume Jacobian, no `P·V` term), which is what a
fixed-geometry magnetostriction calculation wants; the two specific heats differ.

```julia
sm  = SLCE.StrainedModels(models, scales)   # fitted at, e.g., s ∈ [0.98, 1.02]
H   = TiledHamiltonian(sm.models[2]; dims = (8, 8, 8), keep_zero_terms = true)
sch = SLCEMonteCarlo.StrainSchedule(sm, H)

r = run_mc(H; temperature = 300, strain = sch, pressure_GPa = 0.0,
           observables = [standard_observables(H);
                          Observable(:scale, 1, v -> SLCEMonteCarlo.strain(v))])
r.points[1].acceptance_strain     # the outer move's acceptance
```

The pressure follows the `temperature` XOR `kT` discipline: **exactly one** of
`pressure_GPa` (GPa) or `pressure` (model units, eV/Å³), explicitly — `0.0` is a
physical choice, not a default — converted once at resolution and never again
downstream. `strain_interval` (default: one attempt per compound sweep),
`strain_proposal` (`:logvolume` or `:scale`) and `strain_step` (default: a tenth
of the grid's domain; fixed for the run) tune the move; a poor width shows up in
`acceptance_strain`, never in the sampled ensemble. Hydrostatic pressure only.

[`run_pt`](@ref) takes the same NPT keywords: every lane becomes an
isothermal–isobaric chain at the **same** pressure, sweeping its own coefficient
clone of `H` (the lanes sit at different volumes concurrently), and the exchange
rule generalizes to `(βᵢ−βⱼ)(Wᵢ−Wⱼ)` with `W = E + n_cells·j0(s) + P·V(s)` — the
strain travels with the swapped payload. One caveat: `pressure_diagnostics` stays
a `run_mc`-only check (its scratch assumes one serial chain; under `run_pt` it
refuses loudly).

One reading caveat for every strained run: `:energy` and `:specific_heat` are
**configurational-only** — they omit the fluctuating `n_cells·j0(s) + P·V(s)`
half of the NPT state energy, so the reported `C` is neither `C_V` nor the NPT
`C_P` (see the observables guide).

Inside an NPT run every measurement's [`MCView`](@ref) carries the cell scale
(`SLCEMonteCarlo.strain(v)`, with `SLCEMonteCarlo.has_strain(v)` false on a
fixed-cell run — a confident `1.0` would let a magnetostriction observable
average a constant), so volume statistics are ordinary observables, as in the
`:scale` example above. Checkpoints work as usual; resuming needs the same grid:
`resume(path, H; strain = sch)`. The details — the energy contract, the measured
Jacobian exponent, and the reference-scale fingerprint — live in
`docs/specs/strain-move.md`.

### Checking the run: the sampled pressure

[`pressure_diagnostics`](@ref) packages the mechanical-equilibrium identity as
observables: on an equilibrated NPT chain the jackknifed `:pressure` evaluable
(`N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩`, in eV/Å³) must equal the applied pressure
within its statistical uncertainty (a few error bars — a 1–2σ deviation is
ordinary fluctuation) — a persistent disagreement means the elastic `j0` bookkeeping, the
virial, the coefficient interpolation or the `P·V` term is wrong. The `dE/dV` estimator
([`energy_volume_derivative`](@ref)) is exact, not a finite difference: the
coefficient drift is the schedule's differentiated interpolant, and the
displacement response is Euler's theorem on each factor's homogeneity degree.

```julia
pd = SLCEMonteCarlo.pressure_diagnostics(sch, H)
r  = run_mc(H; temperature = 300, strain = sch, pressure_GPa = 1.0,
            observables = [standard_observables(H); pd.observables],
            evaluables  = [standard_evaluables(H); pd.evaluables])
r.points[1].stats[:pressure]      # ×160.2176634 (GPA_PER_EV_A3) for GPa
```

Trust it only when the sampled volume distribution sits well inside
[`strain_domain`](@ref) — the identity holds up to boundary terms of the bounded
grid, and a chain pressed against a grid edge is answering a different question.
