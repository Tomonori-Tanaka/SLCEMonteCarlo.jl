# Decision record — the public surface: adiabatic only, coupled sampling deferred

Status: **decided 2026-07-31, not implemented.** Nothing in the code changes yet; this
records the decision and the work it implies. Owner: `SPEC.md`'s public API section,
`README.md`, `docs/src/**`, and the `export` / `public` tiers in
`src/SLCEMonteCarlo.jl`.

## The decision

The **published** capability is the one that keeps the spin and lattice degrees of
freedom **separated** — the adiabatic route, of which magnon–phonon coupling is the
motivating case. Concretely:

- spins sampled at **fixed geometry** (the clamped-ion sub-model);
- the lattice read out **at a fixed spin configuration** (force constants, `D(q)`, the
  harmonic screen);
- the coupling obtained as the dependence of one on the other — e.g. run the spin chain
  at temperature `T`, then evaluate `dynamical_matrix` on the sampled magnetic state to
  get the phonon shift with magnetization.

**NPT stays public, in its quasi-harmonic form**: a volume grid carrying `J(V)`, with
the spins sampled at each volume and the **displacements frozen**. The cell is then a
slow external parameter that the fast degrees of freedom equilibrate against, which is
exactly the quasi-harmonic approximation and gives magnetic thermal expansion. What is
deferred is sampling `u` and `s` *together*.

**Deferred, not removed:** the joint spin+displacement sampler stays in the tree, keeps
its tests, and keeps working. It is simply not part of the supported surface.

## Why — and why it is NOT a correctness argument

The 2026-07-31 package audit left the joint sampler's *correctness* in unusually good
shape: exact incremental ΔE at every body order, detailed balance checked against
spherical quadrature, bitwise serial ≡ parallel, and the NPT acceptance rule verified
against an independent quadrature of the analytic marginal. The reasons to hold it back
are all of the other kind — **failures that are silent**:

1. **The target need not be a probability measure.** A truncated cluster expansion is a
   finite polynomial in `u`, so `exp(−βE)` is normalizable only if the leading even form
   is positive definite, and nothing upstream guarantees that. `updates-stationarity.md`
   U8's escape detector *detects* this; it cannot fix it. Publishing the joint sampler
   means publishing a path on which a user can obtain confident numbers from a
   non-normalizable distribution.
2. **Single-site displacement moves critically slow down at the soft modes.** Long-
   wavelength phonons are effectively unsampled at production cell sizes, and the failure
   is quiet — the acceptance rate looks healthy throughout. This is the problem that HMC
   or collective/normal-mode moves exist to solve.
3. **The model may not describe a connected crystal at all.** A cutoff that does not
   reach across the training cell, or the upstream self-image gap, splits the
   displacement couplings into disjoint components; the construction-time warning added
   on 2026-07-31 is the *only* screen for it, and `translation_invariant`, the acoustic
   residual and `harmonic_stability` all call such a model healthy.

Freezing `u` removes 1 and 2 outright, and makes 3 a non-issue for what is published.

## What the adiabatic route needs — almost nothing new

The pieces are the mature ones. Upstream `SLCE.restrict(model, :spin)` returns the exact
clamped-ion sub-model; the spin sampler on it is the oldest and best-gated path here;
`force_constants` / `dynamical_matrix` (SLCE M4) give the lattice side at a fixed spin
state; `SLCEDynamics.jl` gives `S(q, ω)`. The work is scope marking and documentation,
not implementation.

## To do (none of it started)

1. **Mark the surface.** `SPEC.md`'s public API section, `README.md`, and the
   `export` / `public` tiers: state which entry points are supported. Demote the
   joint-specific ones out of `export` where that does not break a dependent package
   (check SLCEDynamics and SLCETools first — `_gradient_lane_ref!` is already a
   cross-package by-name call, so the tiers are load-bearing).
2. **Status banner on the guide pages.** `docs/src/guide/joint.md` gets an explicit
   "present, exercised, not part of the supported surface, and here is why" note
   pointing at this record. `docs/src/guide/npt.md` states the quasi-harmonic scope and
   that displacement sampling is out.
3. **A documented quasi-harmonic recipe.** The NPT-with-frozen-`u` path is now the
   headline lattice capability, so it needs a worked example end to end: volume grid →
   spin chain per volume → thermal expansion. Today the guide shows the joint form.
4. **An adiabatic magnon–phonon walkthrough**, using only the separated pieces:
   spin MC at `T` → `dynamical_matrix` on the sampled state → the phonon shift. This is
   the page that says what the package is *for* under the new scope.
5. **Decide the refusal.** With `u` sampling out of scope, should `disp_per_metropolis >
   0` warn (or refuse) unless an opt-in is passed? A quiet feature nobody is told about
   is safer than a loud one, but a silent one users stumble into is the worst of the
   three. This is a design question, not a mechanical change.

## The prerequisite for ever un-deferring

Both the HMC route and any future lattice **dynamics** need the same missing piece:
**displacement forces**. Every gradient entry point today is spin-only
(`_require_spin_only` at `energy.jl:278` and `energy.jl:345`, and in `minimize.jl`); the
second derivative exists (`force_constant_matrix`) but the first does not. So "reuse a
dynamics method for the displacement channel" starts with implementing the displacement
gradient — which is also step one for putting lattice dynamics into `SLCEDynamics.jl`.
That shared prerequisite is the reason to write it once, well, rather than twice.
