# Joint spin–lattice models

```@meta
CurrentModule = SLCEMonteCarlo
```

A model fitted with displacement sectors carries a second sampled field: the
atomic displacements ``\boldsymbol u_s``. [`TiledHamiltonian`](@ref) tiles those
rows exactly like the spin rows, [`has_disp`](@ref) reports whether they are
there, and every energy call needs both fields
(`total_energy(H, config, disps)` — omitting `disps` on a joint Hamiltonian
throws rather than silently meaning ``u = 0``).

This page covers what changes once displacements exist. The cell **volume** is a
third field, sampled only if you ask for it — see [NPT and strain moves](npt.md).

## Displacement sweeps

Each displacement sweep attempts one isotropic Gaussian shift of width `step_u`
per displacement-active site, with the same exact-ΔE Metropolis rule as the spin
move. `disp_per_metropolis` sets how many of them follow each Metropolis spin
sweep and **defaults to the model**: `1` on a joint model, `0` on a pure-spin
one. Any value ``≥ 1`` samples the same ensemble (only the correlation times
change); passing `0` on a joint model deliberately selects the clamped-ion
ensemble instead, and passing a nonzero value to a pure-spin model is an error
rather than a silent no-op.

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

`disp_rms` is a phase **average** and `disp_max` an extreme value over the same
phase (so it grows with `disp_checks` and is not comparable between runs of
different length); both are `NaN` when `disp_checks == 0`, which happens when the
measurement phase is shorter than `renorm_interval`. They are diagnostics, not
observables — they carry no error bars.

The displacement proposal width `step_u` adapts during thermalization on its own
acceptance counters, exactly as the rotation width `step` does — and the two are
**not** interchangeable (an angle in radians vs a length in the model's units).
See [the adaptive step](running.md#The-adaptive-step).

## The frame the displacements live in

Where a displacement-coupling component's rigid shift is a symmetry, the chain
random-walks along it freely. Whether it *is* one is **measured** — per component
and per Cartesian direction (`H.comp_free`), at construction and again at every
grid node of a volume schedule — not inferred from the acoustic sum rule, which
alone does not imply it; on a joint model that measurement is taken at one probe
spin configuration, so treat it as a screen rather than a proof. The sampler
removes the drift along the directions it found flat every `renorm_interval`
sweeps, and `disp_rms` / `disp_max` are recorded immediately afterwards — but the
`v.disps` a measurement sees carries whatever frame the **last renormalization**
left, and measurements fire far more often than renormalizations do (by default
every sweep against every 1000).

A custom displacement observable therefore has to subtract the component centre
of mass **itself**, exactly as `:u2` and `:u4` do: gauge invariance is a property
of the observable, not of when it happens to be called. Reading ``|u_s|^2`` raw
instead adds the free walk's ``⟨|\bar u|^2⟩``, which on a translation-invariant
fixture is linear in `renorm_interval`, always positive, and shrinking with system
size — it reads as a finite-size effect rather than as a bug. The recipe is in
[observables and evaluables](observables.md#Displacement-observables);
`docs/specs/updates-stationarity.md` U7 is the argument.

A model built with `fixed_reference = true` (a substrate-clamped slab, a pinning
defect) keeps its absolute frame along the directions the construction gate
measured as pinned — those are not subtracted, by either the standard set or
`_recenter!`.

## Screening the model before the run

!!! warning "The truncated expansion need not be bounded below"
    `E(u)` is a finite polynomial, so `exp(−βE)` is a probability measure only when
    the leading even form is positive definite — nothing in the fit guarantees it.
    When it fails the chain has no stationary distribution and simply runs
    downhill, with an *exact* incremental energy and an acceptance near 1: no
    pre-existing diagnostic sees it.

Two independent screens, and both are needed:

**Before the run — the harmonic spectrum.** [`harmonic_stability`](@ref) reports
the displacement Hessian's spectrum ([`force_constant_matrix`](@ref)) at a given
spin configuration:

```julia
hs = harmonic_stability(H, config)          # default expansion point: u = 0
(hs.min_eigenvalue, hs.n_negative, hs.acoustic_residual)
```

An eigenvalue below `−tol` is a **proof of failure**. A clean spectrum proves
**nothing** — deciding global non-negativity of a quartic form is NP-hard, so no
harmonic screen settles the question for a model with quartic displacement terms.
The tolerance is not cosmetic either: a translation-invariant model has
`3·n_disp_comps` exact zero eigenvalues and the finite differences scatter each of
them across zero, so `tol` is derived from that measured floor (and returned, so
the verdict stays auditable). `acoustic_residual` is a different quantity — the
acoustic sum rule as a measured residual, ~0 on a translation-invariant model and
O(1) on one with a pinned frame.

One expansion point gives no verdict at all: `harmonic_stability` **throws** when
the Hessian vanishes identically there, which is what a displacement sector whose
leading term is of degree ≥ 3 produces at ``u = 0`` — reporting a clean spectrum
would be maximally reassuring about the model carrying the least harmonic
information. Screen at a displaced expansion point (`disps = …`) instead.

**During the run — the escape detector.** The sampler measures displacement
*recurrence* directly at every renormalization, warns, and sets
`TempResult.escaped`. A set point is not "noisy data" — it is a run whose
displacement observables mean nothing. The detector's block test needs **15
renormalization checks per measurement phase** before it can raise its three
consecutive strikes, and the default `renorm_interval = 1_000` against
`sweeps_measure = 10_000` gives only 10 — a joint run below the bar is warned
about up front and should lower `renorm_interval`. Under the bar, `escaped ==
false` means *not screened*, not *clean*; `disp_checks` is what tells the two
apart. Design record: `docs/specs/updates-stationarity.md` U8.

A third, cheap screen is the [`:u_moment_ratio`](observables.md) evaluable, read
against temperature: a harmonic model's ratio is temperature-independent.

## What the standard observables do on a joint model

`standard_observables(H)` / `standard_evaluables(H)` are joint-aware and add
`:u2`, `:u4`, `:sublattice_u2`, `:sublattice_u4` and `:u_moment_ratio`. Two
normalizations part company once displacements exist: `:specific_heat` is per
**active** site and covers **spin + lattice**, while ``χ`` and the Binder cumulant
stay per **spin**-active site. That heat capacity is **configurational**: no
momenta are sampled, so a classical harmonic lattice contributes ``3/2\,k_B`` per
displacement-active site rather than the Dulong–Petit ``3\,k_B``. For an absolute
value add ``(3/2)\,n_{\mathrm{disp\,active}}/n_{\mathrm{active}}``. On a
displacement-only model the magnetization observables are omitted rather than
reported as `NaN`. Details, and how a custom observable declares which site count
it wants: [observables and evaluables](observables.md).

## Surfaces that are still spin-only

Three parts of the package refuse a joint Hamiltonian loudly rather than sampling
it at ``u = 0``:

| surface | why |
|---|---|
| [`minimize_energy`](@ref), [`find_ground_state`](@ref) | the descent is on the product of unit spheres; there is no displacement coordinate in it |
| [`site_gradient`](@ref), `energy_gradient!` | spin-direction gradients only (the seam `SLCEDynamics.jl` consumes) |
| the GPU **gradient** tier (`gpu_energy_gradient!`) | same contract as the host gradient — unlike the GPU *sweep* tier, which accepts a joint Hamiltonian |

The route for all three is the model's clamped-ion sub-model,
`SLCE.restrict(model, :spin)` — an exact restriction, not an approximation: it is
the ``u = 0`` section of the same fit.
