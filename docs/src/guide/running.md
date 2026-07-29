# Running chains

```@meta
CurrentModule = SLCEMonteCarlo
```

[`run_mc`](@ref) drives one Markov chain: per temperature it **thermalizes**
(`sweeps_therm` sweeps, with the proposal step adapting), freezes the kernel, then
**measures** (`sweeps_measure` sweeps, recording every `measure_interval`-th).
One sweep is one single-spin Metropolis attempt per **spin-active** site, scanned
in the Hamiltonian's color-class order (inactive and non-magnetic sites stay
frozen — see [`TiledHamiltonian`](@ref)) with the exact `ΔE` of the fitted
Hamiltonian —
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

## Beyond the spin sweep

Two fields join the chain once the model has them, each with its own page:

- **Displacements** — a joint spin–lattice model samples its atomic displacements
  in the same compound sweep (`disp_per_metropolis`, `step_u`, `renorm_interval`,
  and the boundedness screens): [joint spin–lattice models](joint.md).
- **The cell volume** — a [`StrainSchedule`](@ref) adds an outer isothermal–isobaric
  strain move and changes which ensemble is being sampled, which changes how the
  standard observables must be read: [NPT and strain moves](npt.md).
