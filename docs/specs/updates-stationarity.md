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
