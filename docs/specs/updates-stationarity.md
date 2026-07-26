# Decision record — update schemes and their stationarity

Status: landed (M3–M4; U1 rewritten for colored sweeps 2026-07-15). Owner:
`src/updates.jl`, `src/run.jl`; gates in `test/unit/test_metropolis.jl`,
`test/unit/test_overrelaxation.jl`, `test/unit/test_parallel.jl`.

## U1 — color-ordered site scan, active sites only, task-count-independent

Active sites are updated in the deterministic **color-class order** of the
Hamiltonian (`TiledHamiltonian.color_ptr`/`color_sites`: a greedy proper coloring
of the conflict graph *site s ~ site t ⇔ some instance touches both*, classes
scanned in order, sites ascending within a class). Inactive sites (no adjacent
instance — `site_active`; a species with `lmax = 0`, or every coefficient fitted
to zero) are uncolored and never visited. Each single-site kernel is π-reversible
(below); a composition of π-stationary kernels in any fixed order is π-stationary
(the composition itself is not reversible, which is irrelevant for sampling).

**Why a class may be updated concurrently.** Two sites in one class share no
instance, so each one's leave-one-out coefficients — hence its proposal axis and
its exact ΔE — are independent of the other's spin: the single-site kernels of a
class commute, and executing them simultaneously equals executing them in *some*
serial order. Concurrency is therefore an execution detail, not a different chain.

**Why it is bit-deterministic for any task count** (`sweep_tasks` — gate:
`test_parallel.jl`, serial ≡ 2/3/7 tasks `==`): (1) every site owns its
proposal/accept RNG stream (`ChainState.site_rngs`, derived from the chain RNG at
construction), so no draw depends on which task visits the site; (2) accepted ΔE
are staged per site and reduced in the fixed class order — one shared loop, so the
floating-point summation order never changes; (3) acceptance counters are integer
sums. No RNG is consumed for site selection.

Trade-off accepted: the scan order and the RNG stream differ from the historical
sequential `1:n_sites` scan (a breaking, CHANGELOG-noted trajectory change), and
`ChainState` carries `n_sites` Xoshiro streams (checkpoint schema v2).

**Why skipping inactive sites is sound.** Their conditional distribution is uniform
and independent of everything else, and no standard observable reads them (see
binning-observables B3), so the sampled marginal on active sites is untouched.
Updating them would consume RNG on always-accepted moves and put a floor under the
measured acceptance, biasing the U3 step adaptation toward the ceiling. They are
kept **bitwise frozen** (sweeps, renormalization, and the ground-state descent all
skip them), so the reported configurations carry the input directions verbatim —
per config payload: under PT, replica-exchange swaps move whole configurations
(frozen spins included) between lanes. Consequence: adding/removing an inactive
species changes the RNG stream only through the site count, not through wasted
draws.

## U2 — Metropolis proposal and the RNG-consumption contract

The proposal is the symmetric two-component mixture proven in SLCETools: antipodal
flip with probability 0.2 (inter-lobe ergodicity on bimodal single-site potentials)
+ Rodrigues rotation by `step·randn` about a uniform random axis (sign-symmetric
angle × uniform axis ⇒ symmetric). Acceptance `ΔE ≤ 0 || rand < exp(−βΔE)`, with
the uniform drawn **only when `ΔE > 0`** — the RNG-consumption contract every
kernel follows, so trajectories are a pure function of `(seed, schedule)`.
ΔE is exact for any body order (`ΔE = c_s·ΔZ`, `c_s` independent of `e_s`).

## U3 — adaptive step, thermalization only

`step ← clamp(step·exp((a − target)/2), 1e-3, π)` every `adapt_interval` sweeps on
a windowed acceptance `a`, **only during thermalization**. At the measurement
boundary the step freezes (`ChainState.frozen`). Why: a step that keeps responding
to chain history makes the transition kernel history-dependent — the chain is no
longer a fixed π-reversible kernel and measured expectations carry a finite-run
adaptation bias; freezing also keeps checkpoint resume bit-identical. The frozen
value is reported per temperature as `final_step`.

Fixture caveat baked into the gate: on a lattice with decoupled (free) sites the
acceptance has a floor (free sites always accept), so the adaptation target must be
tested on an all-coupled model.

## U4 — overrelaxation = deterministic involutive reflection + Metropolis accept

Proposal: `e′ = S(e) = 2(e·ĥ)ĥ − e`, with `ĥ` the direction of the local `l = 1`
field read off the leave-one-out coefficients (tesseral slots
`Z_{1,-1} ∝ y, Z_{1,0} ∝ z, Z_{1,1} ∝ x` ⇒ `h = (c₄, c₂, c₃)`). Accept with the
standard Metropolis rule on the exact ΔE.

**Stationarity.** `S` is a deterministic involution (`S∘S = id`) and an isometry of
the sphere (unit Jacobian), and its axis depends only on the *other* spins (`c_s`
is `e_s`-independent). For such proposals Metropolis acceptance satisfies detailed
balance pointwise: `π(e)·A(e→Se) = min(π(e), π(Se)) = π(Se)·A(Se→e)`. Hence each
site kernel is π-reversible and the sweep is π-stationary.

Two limits: (a) **pure `l = 1` site channel** — the reflection conserves `e·h`,
so `ΔE ≡ 0` and every move is accepted: exactly classical microcanonical
overrelaxation, with zero special-casing (machine gate; this also pins the tesseral
axis extraction — a wrong component order breaks `ΔE ≡ 0`); (b) **general SCE** —
the `l ≥ 2` / multi-body remainder is corrected exactly by the accept step.

A wrong axis is a *correctness no-op* (any `e`-independent axis + MH accept is
stationary) — it only costs acceptance/decorrelation efficiency.

**Not ergodic alone** (pure case conserves `e·h` per move): only ever mixed into
compound sweeps — 1 Metropolis sweep + `or_per_metropolis` OR sweeps. Sites with no
`l = 1` channel (precomputed `site_has_l1` mask) or vanishing field are skipped and
not counted as attempts.

**Efficiency reality check** (from the gate work): on a strongly anisotropic model
the l=1 reflection is nearly random w.r.t. the dominant `l ≥ 2` energy, and the OR
acceptance collapses at low temperature — OR pays off in exchange-dominated
(l=1-heavy) systems, which is its classical use case.

## U5 — energy-drift policy

Every `renorm_interval` compound sweeps and at each thermalization→measurement
boundary: renormalize all spins, rebuild the tesseral rows, recompute the total
energy, record `|E_incr − E_recomp|` into `max_drift` (reported per temperature),
warn once per run above `1e-8·max(1, |E|)`, and re-anchor. The schedule is
deterministic, so it does not interfere with bit-reproducible resume.

## U6 — metastability is the fixture's problem, not the sampler's

The random anisotropic two-site fixture freezes into seed-dependent basins below
`kT ≈ 0.15` (two *pure Metropolis* chains disagree far beyond error bars while each
reports small `τ_int` — the classic broken-ergodicity failure of within-basin error
bars). Statistical cross-checks between update schemes must run at temperatures
where the fixture demonstrably equilibrates (`kT = 0.5`). This is also the
motivation for parallel tempering (M5).

## U7 — the displacement move, and where the frame is re-centred

Status: landed (M4 slice 3c/2). Gate: `test/unit/test_joint.jl` "joint
displacement sampling".

**The move.** `displacement_sweep!` scans the same color classes as the spin
sweeps, attempting one isotropic Gaussian shift `u′ = u + step_u·(g₁, g₂, g₃)` per
**displacement-active** site (`site_has_disp`). The proposal is symmetric, so the
acceptance is the bare Boltzmann rule of U2 with the identical RNG-consumption
contract (the uniform is drawn only when `ΔE > 0`). U1's concurrency argument
transfers verbatim: two sites in one class share no instance, so one's
leave-one-out coefficients are independent of the other's displacement.

**Why the ΔE is exact rather than a linearization.** It is the same leave-one-out
contraction the spin move uses, restricted to the site's displacement rows. The
constructor enforces **at most one axis per (site, channel)**, so a move that
changes one channel leaves the other channel's row a constant factor already folded
into `c_s`, and rows the move did not touch contribute `c_s,k · 0`. The
range-limited `delta_energy(c, zold, znew, nlm+1, nrows)` therefore drops only
terms that are exactly zero — it is not an approximation, and it also means the
proposal buffer never has to hold the untouched half of the row table. Keep the
**row-difference** form: `c·znew − c·zold` cancels two large sums and loses two to
three digits of the number that drives the accept step.

**The flat direction.** Where a component's rigid shift is a symmetry, the sampler
random-walks along it: `|ū| ≈ σ√(3pM/N)` after `M` sweeps. The acoustic sum rule alone
does **not** give that symmetry — for `E = Σ_c E_c(u_c)` over disjoint components it
only forces `Σ_c v_c = 0` on the per-component shift responses — so flatness is
**measured** per component and per Cartesian direction (`_translation_residuals`,
`H.comp_free`), never assumed. It is measured at one probe *spin* configuration, which
on a joint model is a screen rather than a proof: the effective force constants are
spin-dependent, so a model flat at the probe need not be flat everywhere the chain
goes. `_recenter!` removes each component's
mean at every renormalization; `ChainState.com_removed` records the accumulated shift,
so the uncentred position of a site in component `c` is `disps[s] + com_removed[c]`.

**Why re-centring is exact — and the standing constraint it imposes.** Energy
neutrality is the weaker claim; what is needed is that the projection preserves the
sampled measure. Split the state as (gauge coordinate `ū`) × (quotient state `w`,
`Σw = 0`). `E` does not depend on `ū`; the single-site proposal projected onto the
quotient is symmetric with a law that does not depend on `ū`; therefore the chain is a
**skew product** — the `w`-marginal is itself Markov and reversible for the intended
measure, and `ū` is a decoupled passenger that nothing measures. `_recenter!`
transforms only the passenger, hence preserves stationarity exactly.

> **Standing constraint.** That argument holds only while every restriction of the
> state space is a **gauge-invariant** function of the displacements. A
> gauge-DEPENDENT restriction — an absolute radius bound, say — makes the region
> move relative to the state and breaks it. Anything added downstream must be
> expressed in centre-of-mass-relative coordinates.

**What re-centring buys, given that it is free.** (1) The reporting convention:
`⟨|u|²⟩` against a drifting frame never plateaus, and any odd-in-`u` spin–lattice
correlator picks up a first-order contamination; measured relative to a single atom
instead it would be `2σ²(1−c(r))`, an EXAFS-style relative MSD comparable neither with
a Debye–Waller factor nor with the upstream force constants. (2) **Numerical
conditioning**: `_disp_rows!` evaluates the solid harmonics at the *absolute* `u`, so
under drift a slot's row grows like `|ū|^{2k+l}` (its total degree) while the
instance sum stays invariant — the
energy becomes a difference of large near-cancelling terms (measured ~30× loss of
significance at `|ū| ≈ 0.27` against a relative amplitude of 0.05, and the drift grows
like `√M`). The centre-of-mass-free representative is the minimum-norm point of the
gauge orbit. (3) `com_removed`'s growth is a
free null-model check that the direction really is flat.

Alternatives rejected: COM-preserving pair moves conserve `Σ_{a ∈ color} u_a` per
colour class, freezing `3(n_colors − 1)` quantities (not ergodic), and nearby pairs
share instances so a single-channel ΔE would drop a `Δr·Δr′` cross term; a global
(rather than per-component) mean mixes independent symmetries; gauge-fixing one site
per component is exactly stationary but its region breaks the supercell's space group,
so `⟨u_q u_{q'}^*⟩` loses its `q`-diagonality and phonon extraction from displacement
correlations is contaminated in a way analysis cannot remove; doing nothing is what the
paragraphs above rule out.

**Per component, and never inside the sweep.** Two components share no instance, so
their rigid shifts are *independent* symmetries — a global mean would leave each
component's own mean nonzero. And the reduction is a mean over sites: an
order-dependent floating-point sum, which inside the barrier-separated color loop
would make the result depend on `sweep_tasks` and break P6. `_renormalize!` — a
single-threaded, fixed-order, deterministically scheduled point — is the only place
it may live.

**When the frame is physical.** `translation_invariant == false` (a substrate-
clamped slab, a pinning defect — expressible as surviving one-body displacement
terms) means the construction gate measured the uniform-shift direction as *not*
flat. Re-centring would then change the energy, so it is skipped; such a
Hamiltonian must be built with `fixed_reference = true`, which is also the
acknowledgement that the absolute frame is meant. The verdict is kept **per
component and per Cartesian direction** (`H.comp_free`), because both axes matter: the
components are independent symmetries, so one pinned component must not disable
re-centring for a flat one; and a single component's flat directions can be a proper
subspace — the slab above is pinned along its normal and free in the plane, and a
single-direction verdict would leave the in-plane centre of mass drifting without
bound, below the escape detector's threshold by construction.

## U8 — an unbounded target is detected, not constrained

Status: landed (M4 slice 3c/2, detection only). Gates:
`test/unit/test_joint.jl` "displacement sampler: external gates".

**The problem.** The cluster expansion is a **finite polynomial in `u`**, so
`exp(−βE)` is a probability measure only when the leading even form is positive
definite. Nothing upstream guarantees that — not the fit, not `TiledHamiltonian`
(whose translation gate certifies flatness, not boundedness), not the sampler. When it
fails, the chain has no stationary distribution and simply runs downhill.

**Why nothing already in the sampler notices.** Measured on the joint test fixture at
`dims = (2,2,2)`: `E` falls `+3.9 → −2.2 → −416 → −11742` while `max|u|` reaches
25.4 Å on a 3.0 Å lattice, and throughout, the incremental-energy drift stays at
`1e-14` (the ΔE bookkeeping is exact, so U5's warning is silent *by construction*) and
the acceptance sits at 0.97–0.99 (every proposal is downhill). This is the classic
improper-target pathology — every single-site conditional is proper while the joint
distribution is not — and the only property that separates it from equilibrium is
**recurrence**, which no pre-existing diagnostic measures.

Note what does *not* catch it. The maximum displacement degree of that fixture is 2 —
**even** — so an odd-degree rule passes it; its dynamical matrix carries one imaginary
branch (`−1.7147` at every `q`, alongside the three acoustic zeros), which for a purely
harmonic model is a *proof* that `Z = ∞`. A minimum-pair-distance floor does not catch
it either: at `E = −416` the minimum pair separation is 0.9386 Å against a 1.0 Å
reference, only 6 % below — the escape is a large-amplitude collective distortion that
preserves close contacts.

**The detector.** `_check_escape!` runs inside `_renormalize!` — single-threaded,
fixed-order, deterministically scheduled, hence P6-safe — and compares the
centre-of-mass-free r.m.s. displacement against its value one *doubling* of the
observation window ago. A stationary chain holds that ratio at 1.0; free diffusion of a
flat direction gives `√2 ≈ 1.41`; the measured escape grows `rms ∝ M^1.6`, i.e. `≈ 3.0`
per doubling. The threshold is 1.7 with two consecutive strikes required, so a chain
equilibrating from the clamped-ion start does not trip it. Both error rates are gated:
on an Einstein oscillator (analytic control, below) zero strikes; on the unbounded
fixture the warning fires well inside the run. Note the false-positive rate is set by
the effective number of degrees of freedom carrying `⟨|u|²⟩`, **not** by the site count
— a single dominant soft mode has `ν ≈ 1` at any system size — which is why the blocks
are averaged and capped rather than doubled indefinitely, and why three strikes are
required.

**Known limitation.** The one-site gauge refusal catches a lone pure-gauge site. It does
not catch a site inside a larger component whose coefficient slice is identically zero:
the component's rigid shift is then not flat (its partner carries the dependence), so
neither the refusal nor re-centring applies, and the site's own diffusion sits below the
detector's threshold. The general predicate is a per-component null-space rank check,
not a size test.

**Why detection and not a bound.** A displacement bound `|u| ≤ u_max` does make the
chain uniformly ergodic, but on an unbounded model it converts a loud failure into a
quiet one: the result is then set by `u_max` rather than by the Hamiltonian. Measured
on the same fixture, `rms|u|/u_max = 0.724` at `u_max = 0.25` and `0.7275` at
`u_max = 0.40` (a uniform ball gives `√(3/5) = 0.7746`), and `max|u|/rms|u| = 1.30` and
`1.19` (a Maxwell-distributed well gives `1.51` at 8 samples and `≈ 2.7` at 10⁴;
compare at matched sample counts) — two independent statistics
agreeing that the density is flat out to the wall. Both runs also sit at
`rms|u|/d_nn = 0.181` and `0.291` against a Lindemann melting criterion of 0.10. A
bound would have reported those numbers without comment.

**Where the boundary genuinely matters (not implemented — roadmap).** When the region
near the bound carries real thermodynamic weight, the correct instrument is not a wall
but **umbrella sampling / Hamiltonian replica exchange** over a collective displacement
coordinate (the amplitude along a soft eigenvector of `D(q)`, or the gauge-invariant
`Σ_s |u_s − ū|²` which is `O(1)`-updatable from the two running scalars `Σ|u|²` and
`Σu`): it *measures* the free-energy profile instead of truncating it, and it is the
same move the SCPH/SCHA family already makes (sample a proper auxiliary distribution,
reweight). `run_pt`'s lane/payload-exchange machinery generalizes to it. Deferred
deliberately; recorded here so the omission is a decision and not an oversight.

**Analytic control.** `_einstein_terms` builds `E = a|u|²` as a single `(k, l) = (1, 0)`
displacement axis. `exp(−βa|u|²)` is an isotropic Gaussian, so `⟨|u|²⟩ = 3kT/(2a)`
exactly — the displacement sampler's only gate against an external truth rather than
against its own bookkeeping (measured agreement: 0.54 %).
