# Decision record — the NPT strain move (M5-4 slice 2)

Status: landed 2026-07-29. Owner: `src/strain.jl` (schedule, energy contract,
move), `src/run.jl` (driver wiring), `src/checkpoint.jl` (schema v4),
`src/gpu/gpu_hamiltonian.jl` (`sync_coefficients!`). Gates:
`test/unit/test_strainschedule.jl`, the sync section of `test/unit/test_gpu.jl`.
Upstream context: SLCE `docs/specs/spin-lattice-ce-design.md` §8 (the target
ensemble and its corrected Jacobian), §9a (volume grids), §12 gate (l).

## S0 — what the move is

An outer-loop isothermal–isobaric Metropolis move over a `SLCE.StrainedModels`
volume grid: propose a new linear cell scale `s′` by a symmetric step in a chosen
variable, rescale the displacements **affinely** (`u → (s′/s)·u`, fixed scaled
coordinates), install the grid's interpolated coefficients for `s′` via the
in-place `set_coefficients!` hot-swap, and accept with

    ln A = (3·N_mob + c)·ln(s′/s) − β·[ΔE_config + n_cells·Δj0 + P·ΔV]

where `N_mob = n_disp_active` is the displacement-active site count, and `c = 3`
(`:logvolume` proposals) or `c = 2` (`:scale`) is the proposal-variable factor
`ln|dV/dy|`. It never runs inside the color-parallel sweep layer — one serial
move per `strain_interval` compound sweeps.

**Corrected 2026-07-29 (review), superseding design-record §8's own correction.**
The first version carried the volume power `D/3` with
`D = 3·N_mob − count(comp_free)` — the dimension of the COM-projected space the
sampler works in. That undercounts: the flat COM directions are gauge directions
whose RANGE is the cell (∝ `s` per direction), so quotienting them out (as
`_recenter!` does) hides their measure factor without removing it. In scaled
coordinates `d^{3N}r = V^N d^{3N}x` — the ideal-gas COM factor, and
Frenkel–Smit's `N·ln(V′/V)` with `N` all mobile atoms. The physical exponent is
`N_mob`, independent of `comp_free`; `D` survives only as the sampled dimension
(`StrainSchedule.d_dim`, diagnostics). The omission was O(`n_comps·kT/⟨V⟩`) in
the sampled pressure — negligible on a production cell, percent-level on the
test fixtures, and invisible to every gate that measured the implemented rather
than the physical marginal, which is why the toy gate now varies `N_mob` and
holds `count(comp_free)` fixed *and* varied.

## S1 — the energy contract has one elastic source

`ΔE = ΔE_config + n_cells·Δj0 + P·ΔV`, with `j0(s)` interpolated per TRAINING
cell and `P·V` on the SUPERCELL volume. There is **no elastic-term keyword
anywhere in the package**, deliberately: each grid point's fit already contains
its own cell's lattice energy in `j0(ε)`, so an explicit `(V/2)·εᵀCε` would
double-count silently. The structural absence is the guard. Gate: the strained
reconstruction identity — configurational + `n_cells·j0(s)` equals the upstream
`predict_energy` at both scales — plus ΔE against from-scratch totals.

## S2 — the proposal arm is one value, not two knobs

The proposal variable and the acceptance power are branches of the SAME symbol
(`_strain_y` / `_strain_s_of_y` / `_strain_s_exponent`, adjacent in `strain.jl`),
because §8(β)'s live trap is drawing in one arm's variable while weighting with
the other's — off by `(V′/V)^{2/3}`, invisible on a production cell and
percent-level on the fixtures. Excluded twice: hand-derived closed forms per arm
(`(3N+3)`/`(3N+2)` powers of `ln(s′/s)`), and a white-box replay in which every
accept decision over both arms must match a hand-paired reconstruction from a
cloned RNG. The statistical marginal cannot resolve the 1/3-power difference at
test cost, which is why the replay exists.

## S3 — the exponent is measured, not asserted

On the §8(γ) toy — constant coefficients across the grid, `j0 ≡ 0`, `P = 0`,
`u ≡ 0` — the stationary volume marginal is the bare Jacobian `p(V) ∝ V^{N_mob}`.
Three chains: the translation-flat and on-site-pinned (2,1,1) fixtures share
`N_mob = 4` while their `count(comp_free)` differ by 6, so both landing on `V⁴`
is exactly the mutation gate against the rejected COM-reduced convention (which
separates them by two powers); the flat fixture at (1,1,1) has `N_mob = 2`,
which shows the exponent moves with the system rather than being asserted. Each
chain's empirical mean volume must match its own power and reject `±1` — and,
where it differs, the rejected convention's `d_dim/3`.

## S4 — the (H, chain) contract

`H`'s coefficients are the schedule's at `st.strain` — before and after every
move, accepted or rejected. The reject path restores by re-running the same
Horner pass, which is deterministic, hence bit-identical; no rollback buffer.
The sweep layer runs in between with no knowledge the cell moves. Consequences:

- an out-of-domain proposal is **rejected, never clamped** (a truncating clamp
  is an asymmetric proposal biasing the chain toward the boundary), consuming
  exactly one normal draw; in-domain attempts consume one normal + one uniform,
  all from the chain-level `st.rng`;
- an accepted move rescales `disps` AND `com_removed` (the re-centring record is
  absolute lengths in the same frame) and RESCALES the escape detector's length
  statistics by the same λ (`_rescale_escape!`) — they are exactly covariant
  under the affine map. §8(θ)'s original reading ("an accepted strain move is a
  phase boundary and must reset") is superseded: a reset on every accepted move
  disarms the detector's block ladder permanently at the default
  one-attempt-per-sweep cadence, blinding U8's only unboundedness diagnostic on
  precisely the runs where a volume runaway is easiest;
- ΔE's two sides come from ONE estimator: `e_old` is recomputed from scratch
  before the swap (the incrementally tracked `st.energy` would put the
  accumulated drift into the acceptance ratio asymmetrically), and the drift is
  carried across an accepted move unchanged so `max_drift` accounting still sees
  it;
- the chain's scale is replica-exchange **payload** (swapped with the state),
  though in v0 PT never strains (S6).

## S5 — ASR hard-error at conversion, not per proposal

`StrainSchedule(sm, H)` builds every grid node **at `H`'s own dims** (with
`fixed_reference = true`, so the build itself never refuses on flatness) and
compares flat patterns ONE-SIDEDLY: every (direction, component) that `H`
re-centres must be flat at every node — a node may be *flatter* than `H`
(sampling a flat direction is merely diffusive), but a node that *pins* a
re-centred direction would get a biasing projection at that volume. Three
corrections over the first version (2026-07-29 review): the check runs at `H`'s
dims because the component partition depends on the supercell (per-component
flatness is strictly stronger than the global sum rule); the gate is
`any(comp_free)`, not `translation_invariant`, because a partially pinned
`fixed_reference` slab is still re-centred along its free directions; and build
errors are wrapped neutrally rather than relabelled as flatness failures.
Flatness is linear in the coefficients, so flat at every node ⇒ flat for every
interpolated model — which is what licenses `strain_move!`'s
`recheck_translation = false` per proposal.

## S6 — driver wiring and the two refusals

`run_mc` resolves `strain_interval` exactly like `disp_per_metropolis`
(`nothing` = what the schedule's presence implies; contradictions are named) and
pressure by the `temperature` XOR `kT` discipline: exactly one of `pressure_GPa`
/ `pressure`, `0.0` included — a physical choice, never a default — converted
once through the exact `GPA_PER_EV_A3 = 160.2176634` and never again downstream.
The width `strain_step` is FIXED for the run (the outer move fires too rarely to
adapt on; a bad width shows in `acceptance_strain`, not in the ensemble).
`carryover = false` resets the cell with the chain (coefficients first, then the
state, so the energy recompute sees them).

`run_pt` takes the same NPT keywords since the PT + strain slice (S10): the two
obstacles this section originally recorded — one shared `TiledHamiltonian`
mutated in place, and the NVT swap rule — are resolved by per-lane coefficient
clones and the generalized exchange weight respectively.

The run-time pairing check compares counts AND a structural term fingerprint
(FNV over each template term's input index, atoms and images, captured at
conversion): the constructor's deep per-term check ran against SOME Hamiltonian,
and counts alone cannot tell two same-shape models apart — the exact silent
mis-assignment the schedule exists to prevent. `run_mc` and `resume` hand `H`
back at the REFERENCE scale on return; leaving it at the chain's final scale
would silently rescale the caller's next `model_fingerprint` / `total_energy` /
fixed-cell run.

GPU: `sync_coefficients!(gH)` re-uploads the ONE device array a host coefficient
swap moves (`sent_w`; every structural table is coefficient-independent because
the site-program skip tests `folded`, never `coef·folded`). Gated bitwise
against the keyed reference on the new weights. Audit 2026-08-01 #3 closed the
composition hole: `strain_move!(...; gpu = gH)` syncs inside the move on every
path that rewrote `H` (identity-checked — a wrapper of a coefficient clone would
upload the wrong lane's weights), and a value-blind coefficient epoch
(`progs.coef_epoch`, bumped by every `set_coefficients!`, recorded by the wrapper
at upload/sync) makes a forgotten sync a refusal at the next device sweep /
gradient entry instead of a silent two-Hamiltonians chain. The per-move device
cost has NOT been measured on real hardware yet — measure on the next kugui A100
session before production NPT+GPU runs.

Fixed-cell byte-neutrality is pinned, not assumed: a short fixed-cell `run_mc`
trajectory captured at commit `d038b86` (pre-wiring) is asserted bit-identical
in the driver testset. If an intentional sampler change moves it, recapture; any
other movement is the regression the pin exists to catch. Both trajectory pins
(this one and S10's `run_pt` twin) are **capture-platform-scoped**: the `_ss_grid`
fixture inherits `build_asr`'s SVD null-space basis, and LAPACK's choice inside
that subspace is platform-dependent — ubuntu x64 CI gets a rotated basis, i.e. a
genuinely different model (it also lands one coefficient on exact 0.0, which the
default-term-list testset measures rather than assumes). The pins therefore fire
where the model fingerprint matches its capture, and macOS aarch64 (the capture
platform, dev machine and mac CI runner alike) asserts that they fired, so the
detector cannot die silently.

## S7 — checkpoint schema v4

See `checkpoint-schema.md` C2/C4: `chain/strain` + 8-entry counters +
`plan/strain_*` + `grid_fingerprint`, v3 refused by name. The subtle pin: on a
strained run `model_fingerprint` is taken at the REFERENCE scale (the
fingerprint mixes coefficient values, and a strained chain's move with its
volume), so the writer captures it while `H` carries `s = 1` and `resume`
reinstalls the reference from the caller-supplied schedule BEFORE comparing,
then installs the checkpointed scale afterwards. `_fingerprint`'s mixing itself
is untouched — SLCEDynamics' checkpoint format depends on it.

## S8 — scope refusals and deferred work

- **Hydrostatic only**: `P·V(ε)` is a state function with no strain-measure
  ambiguity; a general applied stress is work-conjugate to a specific Seth–Hill
  measure (2nd Piola–Kirchhoff ↔ Green–Lagrange, Biot ↔ Biot) and would reopen
  design-record §9e part (1).
- **Isotropic grid only in v0**: the schedule interpolates `SLCE.StrainedModels`
  volume grids (`s·A₀`); the anisotropic-channel K(ε) split (§9b/§9c) inherits
  the same machinery when it lands upstream.
- **The chain starts at `s = 1`** unless `strain_init` says otherwise (either
  way inside the domain); the strained warm start — DONE, see S12.
- **Mechanical-equilibrium identity observable** — DONE, see S9.
- **PT + strain** — DONE, see S10.
- **`:energy` / `:specific_heat` are configurational-only on a strained run**:
  the β-conjugate state energy of the NPT target is
  `W = E_config + n_cells·j0(s) + P·V(s)` (exactly `_swap_dweight`'s content),
  and both `j0(s)` and `P·V(s)` fluctuate with the sampled volume, so the
  reported `C = var(E_config)/(n·kT²)` is neither `C_V` nor the NPT `C_P`
  (measured 3.4 % low on the Einstein-well fixture; `C_P − C_V = TVα²B` is
  percent-to-tens-of-percent on a real solid near melting). The strain-aware
  `W` observable — DONE, see S11 (`npt_observables`).

## S9 — the §8(ζ) mechanical-equilibrium observable (landed 2026-07-29)

`energy_volume_derivative` + `pressure_diagnostics` (`src/strain.jl`); gates in
`test/unit/test_strainschedule.jl` (the two §8(ζ) testsets).

**The identity.** Stationarity of the NPT marginal
`p(V) ∝ V^{N_mob}·e^{−β(E_total + P·V)}`, integrated by parts in `V` at fixed
(spins, scaled displacements):

    N_mob·kT·⟨1/V⟩ − ⟨dE_total/dV⟩ = P_applied

up to boundary terms of the schedule's bounded domain — negligible exactly when
the volume distribution is confined well inside the grid, the only regime a
production NPT run is meaningful in. It is the one production-scale check of the
`j0`/virial/`P·V` bookkeeping: a lost `n_cells` factor or a dropped virial bias
it far beyond its error bar. Its sensitivity to the volume-power convention
itself is only `kT·⟨1/V⟩` per unit of `N_mob` — on the shipped fixture ~1.3σ,
i.e. often within an error bar — so `N_mob` vs `D/3` stays pinned by the §8(γ)
marginal gate, not by this identity (the docstring says so too).

**The estimator is exact, not a finite difference**, via two facts. (1) The
coefficients are stored polynomials in the centred abscissa — `dJ/ds` is the
same Horner pass differentiated (`_strain_dcoefficients!` / `_strain_dj0`, with
the chain-rule factor `_sch_dz_ds` through the abscissa). (2) At fixed scaled
coordinates the displacement content responds as `Φ(s·w)`, and every
displacement factor `|u|^{2k} R_{l,m}(u)` is homogeneous of degree `2k + l`, so
by Euler's theorem the virial `Σ_sites u·∂Φ/∂u` of a template is `deg·Φ` with
`deg = Σ_disp-slots (2k + l)` — a per-template integer recovered from the slot's
`row0` via the layout's `disp_starts` (`_term_disp_degrees`). No displacement
gradient is ever formed; the walk is `_total_energy`'s loop with the effective
coefficient `g_k = scale_k·(dJ_k/ds + deg_k·J_k/s)` (`_energy_with_coefs`, a
deliberate mirror kept in lockstep so the pinned hot path stays untouched).
NOT `grid_strain_derivative` — that upstream surface is `u = 0`,
coefficient-drift only, and allocates a model per call; it survives here as the
`u = 0` cross-gate. That gate pins the η-convention factor (`dE/dη = s·dE/ds`)
and the drift path, not the interpolant itself (both sides build the same
centred Vandermonde) and not the Euler half (all displacement rows are zero at
`u = 0` — the finite-difference gate owns that, over ALL THREE abscissas, since
`_sch_dz_ds`'s `:volume`/`:logvolume` chain-rule branches exist nowhere else).

**Purity**: the kernel reads `J(s)` from the schedule, never from `H`'s
currently installed coefficients — gated by installing two different scales
around one call. **Gauge**: the view's displacements are COM-reduced; the virial
is gauge-invariant exactly because certified-flat directions have zero force
sum, so on a healthy schedule the reduction is invisible.

**Packaging**: `pressure_diagnostics(sch, H)` returns two raw observables
(`:strain_dEdV`, `:strain_invV`) plus the jackknifed `:pressure` evaluable —
the `Evaluable` receives its point's own `kT`, which is what lets one
diagnostic serve a multi-temperature run; jackknife over paired bins carries
the covariance of the two means. Scratch is captured per instance and sized for
the paired `H` (a view from any other Hamiltonian is refused), serial by
construction — i.e. a strained `run_mc`. Under a strained `run_pt` (S10) the
diagnostic stays unusable: the lanes measure concurrently (a shared-scratch
race) and their views carry per-lane clones, which the identity check refuses
loudly instead of racing silently.

**The statistical gate** (an Einstein-well fixture — the displacement energy
must be bounded below or there is no stationary distribution to test; random
ASR'd coefficients are a generically indefinite quadratic form, and the first
fixture draft escaped exactly as U8 warns): kT = 0.05, P = 0.01 eV/Å³, domain
s ∈ [0.9, 1.1] with the volume distribution ≥ 5σ from both edges. Measured:
P_sampled = 0.01110 ± 0.00074 (1.5σ; a 10-seed review replication gave
0.00980 ± 0.00028, −0.7σ, seed scatter 1.19× the jackknife errors — honest
bars; a 120k-sweep chain gave 0.010133 ± 0.000125, +1.1σ).
Decomposition: ideal +0.0038, drift −0.0007, virial +0.0039, j0 −(−0.0105) —
the j0 and virial halves are the ones this gate constrains (≫ 5σ / ~7σ if
broken); the drift (~1σ) is the FD gate's job. It does NOT discriminate the
rejected `D/3` convention (the fixture is fully pinned, so
`count(comp_free) = 0` makes the two coincide); that mutation is §8(γ)'s toy.

## S10 — PT + strain (landed 2026-07-29)

`run_pt` takes the full NPT keyword set; owner `src/pt.jl` (`_PTLane`,
`_swap_dweight`, `_attempt_swap!`), `src/coefficients.jl` (`_coefficient_clone`),
`src/checkpoint.jl` (schema v5); gates in the three "PT + strain" testsets of
`test/unit/test_strainschedule.jl`. Four decisions:

**Per-lane coefficient clones, not a serialized shared `H`.** The S6 data race
was that every lane sweeps one `TiledHamiltonian` while `set_coefficients!`
rewrites it in place. `_coefficient_clone(H)` shares every structural array by
reference and copies exactly what `set_coefficients!` writes — `terms`,
`progs.term_coef`, `progs.sent_w` — so the cost is O(model), never O(supercell)
(the program streams are per (template, member slot)), and each lane owns its
coefficient state outright. The clone deliberately shares the flatness verdict
(`comp_free`): the schedule certified flatness for the whole interpolated family
at conversion (S5), so re-measuring per clone could only disagree by roundoff.
The CALLER's `H` never enters a lane — the driver installs the reference into it
up front (the fingerprint identity, the same order-of-capture rule as S7) and it
stays there for the whole run; no end-of-run restore exists because nothing ever
moves it.

**The Hamiltonian reference is exchange payload.** An accepted swap moves
`(config, zrows, energy, disps, com_removed, strain)` between lanes AND swaps
`lane.H` — the installed coefficients describe the payload's scale, so they
travel with it, and the (H, chain) contract of S4 holds through every exchange
with no reinstall (no Horner pass at a swap, nothing to forget on the reject
path). On a fixed-cell run every lane holds the same object and the swap is a
same-reference no-op, which is why the fixed-cell trajectory is byte-neutral
(pinned: a pre-wiring `run_pt` trajectory captured at `67d9363`, asserted `==`).

**The swap rule keeps only β-conjugate content.** For bundles at a common
pressure the exchange ratio is `(β_a−β_b)(W_a−W_b)` with
`W = E_config + n_cells·j0(s) + P·V(s)`. The `V^{N_mob}` measure factor of §8's
marginal does NOT enter: it is a β-independent property of the bundle itself and
cancels exactly between the swapped and unswapped assignments — the general
argument, not a per-channel re-derivation. Two implementation notes. `W_a − W_b`
is formed as the sum of the three DIFFERENCES (`_swap_dweight`, same association
as `strain_delta_energy`), never per-lane totals differenced — `n_cells·j0` and
`P·V` are extensive while `ΔW` is not, so totals would compute the difference at
`ulp(|W|)` granularity, a conditioning loss growing linearly with `n_cells`; the
bracket gate's hand-derived logw mirrors the same association on purpose. And the
configurational half is the incrementally tracked `st.energy` — it carries the
chain's accumulated drift where the `j0`/`P·V` halves are exact, the same mix the
fixed-cell NVT rule has always used; the drift is bounded by `renorm_interval`,
surfaced in `max_drift`, and NOT worth a from-scratch `_total_energy` per lane
per boundary (which would also move the fixed-cell byte-neutrality pin). The
rule is pinned EXACTLY by a
bracket gate (hand-built lanes, the hand-derived logw, uniforms one ulp on each
side of `exp(logw)`; the dropped-j0 / dropped-P·V / NVT mutations shift logw by
1.1 / 1.2 / 2.3 on the fixture, far above a ulp) because the statistical gate
cannot resolve it: patching the NVT rule in moved the rung marginals ≤ 0.6σ at
test cost — the same "replay exists because the marginal is blind" scoping as
S2. The statistical gate (strained 2-rung PT vs independent NPT `run_mc` chains
per rung on the Einstein-well fixture, 4σ) is the end-to-end ensemble check —
a lane sweeping another scale's coefficients or a broken affine rescale at a
swap wrecks it. Determinism is P3 unchanged: strain moves draw only from the
lane RNG inside `_lane_segment!` (same order as `_run_temperature!`), and the
swap weight is a pure function of chain state — `ntasks = 1 ≡ ntasks = R` is
gated with the strain channel live.

**Checkpoint schema v5.** The FILE layout did not change — `chain/strain` per
lane and `plan/strain_*` were already written by v4 — but v4's PT lane strains
were never live: a v4-era READER handed a strained-PT file would pass every
handshake and silently continue the run as fixed-cell at the reference
(`_pt_run!` had no strain support). The version bump turns that silent wrong
physics into a named refusal in both directions, the same refuse-by-name
precedent as v3 → v4. `resume` reuses the mc-kind strain handshake verbatim
(grid fingerprint, reference reinstall before the model fingerprint, pairing
check), then rebuilds per-lane clones at each lane's checkpointed scale;
resume ≡ uninterrupted is gated bit-identically.

Scope notes: one pressure for the whole ladder (the swap-cancellation argument
above needs a common `P`; per-rung pressures are a different method — replica
exchange in the (T, P) plane — not a missing keyword). `pressure_diagnostics`
stays `run_mc`-only, per S9 — refused at `run_pt` ENTRY by observable name, so
the late per-view identity refusal never burns a thermalization phase.
`PTResult.final_strains` records the per-lane end scales since S12 (the
"unrecorded per-lane scale" caveat that used to live here), and `strain_init`
warm-starts a ladder from them.
The lane-owned escape statistics see payloads at whichever scale the exchanges
deliver, so a strained lane's `disp_rms` series mixes scales — diagnostic-only,
and PT already breaks that series at every swap. Memory: R clones of
`sent_w`/`term_coef`/`terms` per run — O(model) each; revisit only if a
production model's program streams make it matter.

## S11 — the strain-aware `W` observable (landed 2026-07-29)

`npt_observables(sch, H; pressure_GPa XOR pressure)` (`src/strain.jl`): raw
`:enthalpy = W = E_config + n_cells·j0(s) + P·V(s)` and `:enthalpy2 = W²`, plus
the jackknifed `:npt_specific_heat = var(W)/(n_active·kT²)` (`scope = :energy`),
closing S8's configurational-only caveat. `W` is exactly `_swap_dweight`'s
content evaluated on one lane — the NPT target's β-conjugate state energy.

**Why `var(W)/(k_BT)²` is the isobaric `C/k_B`.** The sampled measure is
`p ∝ V^{N_mob}·e^{−βW}` on the schedule's bounded volume domain. Both the
Jacobian factor and the domain truncation are β-independent, so
`−d⟨W⟩/dβ = var(W)` holds **exactly** (equivalently
`var(W)/(k_BT)² = d⟨W⟩/d(k_BT)` — note the conjugate variable is `k_BT`, not
`T`) — no boundary terms, unlike the S9 identity, which integrates by parts in
`V`. Two interpretation limits the identity does NOT remove: the truncated
measure is a volume-constrained system, so the number reads as the physical
`C_P` only while `p(V)` is confined well inside the grid (a broad `p(V)` near a
transition is exactly when the wall bites); and v0 samples the isotropic scale
only, so it is the `C` of the hydrostatic fixed-shape ensemble (the five frozen
shear DOFs would classically add `O(1) k_B` per supercell). It is
configurational: no momenta are sampled, so the classical kinetic `(3/2)k_B`
per mobile atom (identical at constant `V` and `P`) is absent — per ACTIVE
site, the analytic add-back is `(3/2)·n_disp_active/n_active`, the two counts
differing exactly on a model with spin-only sites.

One recorded numerical remark: unlike `:energy`, `W` carries the absolute
offset `n_cells·j0 + P·V`, so `⟨W²⟩ − ⟨W⟩²` cancels `~(W/σ_W)²` digits. At
production scale (`n_cells·j0 ~ 10⁶ eV`, `σ_W ~ 30 eV`) the binned variance
keeps a relative error `~ eps·√bin·(W/σ_W)² ≈ 10⁻⁵–10⁻⁴`, growing `∝ n_cells`
— acceptable, and deliberately NOT "fixed" by referencing `j0`/`V` to `s = 1`
inside `W`, which would change `:enthalpy`'s reported zero point.

**Gate scoping** (`test_strainschedule.jl` "npt_observables"). The FORMULA is
owned by an exact anchor: the zeta fixture's `j0(s) = 40η²` (a quadratic the
3-node interpolant reproduces to roundoff everywhere in the domain) and
`V(s) = n_cells·27·s³` from the a = 3 Å cell — machine-precision agreement,
so dropped-`j0` / dropped-`P·V` / misplaced-`n_cells` / wrong-unit mutations
all die there. The FDT cross-gate (central FD of `⟨W⟩` over kT = 0.04…0.06 vs
the evaluable at the midpoint, 4σ; measured over 4 seed sets: −0.2σ … 2.7σ)
checks the ensemble-level consistency of the `⟨W⟩`/`var(W)` pair; its
resolution (σ ≈ 0.6–1.0 k_B against C ≈ 10–12) can NOT see the config-only
mutation — `var(E_config)` sits ≈ 0.3–0.6 k_B ≡ 3–6 % below `var(W)` across
the gate's three kT on this fixture — which is the same
statistics-cannot-carry-the-rule split as S10's bracket-vs-marginal gates.
(Nor can the fixture fence `scope = :energy`: every site carries both channels,
so `n_active ≡ n_disp_active` there.) The strained-PT rung-marginal testset
carries `:enthalpy` in its 4σ PT ≡ run_mc comparison and evaluates the
per-rung isobaric `C` through the lane clones.

**PT-safety by construction.** Unlike `pressure_diagnostics` (per-instance
scratch, `v.H === H` identity, run_mc-only), the closures are pure functions of
the view, the immutable schedule, and the resolved pressure — each measurement
costs two `j0` Horner passes (`:enthalpy2` re-evaluates `W`). The per-view
guard is IDENTITY on a shared structural array (`inst_term`): a lane's
coefficient clone shares it by reference and passes, while a same-shape
Hamiltonian built from another grid point's model — which every count-based
check accepts — is refused instead of silently getting this run's `j0`/`P·V`
applied to its energy. Corollary: after `resume`, rebuild the observables
against the `H` handed to it (closures are not serialized, so this is the
natural usage anyway). A FIXED-CELL run refuses `:enthalpy`/`:enthalpy2` by
name at entry in both drivers (`_refuse_npt_observables`) — the per-view
`strain(v)` throw would otherwise fire only after a spent thermalization
phase, the same argument as S10's `pressure_diagnostics` entry refusal. The
residual seam is the user's contract: the factory's `sch` and `P` must be the
run's own — the view carries neither, and a pairing-compatible second schedule
or a different pressure is undetectable at measurement time (docstring warns,
S9 precedent; the drivers hold both, so a `strain_observables = true` run
keyword would close the class if it ever bites).

## S12 — the strained warm start (landed 2026-07-30)

`MCResult.final_strain` / `PTResult.final_strains` record the end-of-run cell
scales (`nothing` on a fixed-cell run — the MCView "nothing is not 1.0"
discipline, so a fixed-cell result cannot be misread as "ended at the
reference"), and `strain_init` on both drivers starts the chain there: scalar
on `run_mc`, scalar-broadcast or per-lane vector on `run_pt`. The recipe is
`run_mc(H; strain = sch, strain_init = r.final_strain, init = r.final_config,
disps = r.final_disps, ...)` — `final_disps`' absolute lengths are expressed at
`final_strain`, so the triple is self-consistent by construction. A warm start
is a NEW chain (fresh RNG streams, adaptation restarted), never a bit-identical
continuation — that stays `resume`'s job.

**Ordering, twice load-bearing.** The warm-start install happens AFTER the
checkpointer captured the reference-scale model fingerprint (the S7 identity —
swap the order and every warm-started run's file refuses its own resume) and
BEFORE the chain computes its initial energy (the (H, chain) contract holds
from sweep one). On `run_pt` the install goes into each LANE's coefficient
clone before that lane's chain is built — the caller's `H`, and hence the
checkpointer, stay at the reference throughout, exactly as in S10. On a
multi-temperature `run_mc` ladder, `carryover = false`'s independent restarts
return to the reference (they discard the caller's `init` the same way); only
the first temperature starts at `strain_init`.

**What self-heals, and the gate scoping** (`test_strainschedule.jl` "strained
warm start"). A missing entry install (chain at `s0`, `H` still at the
reference) has its COEFFICIENT half repaired by the FIRST strain move — accept
installs the new scale's coefficients and reject restores at `st.strain` by
the deterministic Horner pass — so the mis-sampled footprint is
`strain_interval` sweeps at ~0.5 %-off coefficients. The ENERGY half heals
more slowly on the reject branch: `st.energy` keeps the offset
`E_ref(x) − E_s0(x)` (measured −1.06e-3 on the fixture ≈ 0.05σ of the run's
`:energy` error bar) as carried drift until the next renormalization
re-anchors it — visible in `max_drift`, invisible to every statistical gate by
construction. The gates therefore pin what is pinnable:
`final_strain` equals the strain of the last measurement view (an exact public
witness via a trace observable); the chain STARTS at `s0` (microscopic
`strain_step` freezes the walk within ~1e-6, four decades below the
ignored-`strain_init` failure); the (H, chain) contract holds at every
measurement (a contract observable recomputes the schedule's coefficients at
the view's own strain against the live `v.H`); per-lane `strain_init` lands on
its own lane (exchange-free run — an accepted swap trades the strain payload
between rungs, measured 6e-3 marginal mixing, so the pin would otherwise see
the mixture); a therm-free warm start from an equilibrated parent reproduces
the parent's marginals (0.7–1.5σ measured over 3 seed sets, 4σ gate); and the
checkpoint interplay is gated on a GENUINELY interrupted run — a poison
observable throws mid-measure, leaving the file at the last periodic write, and
`resume` must both accept the file (the reference-fingerprint ordering above)
and reproduce the uninterrupted run bit for bit, per-lane scales included.

That last gate exists because of the two kinds' different checkpoint shapes
(measured 2026-07-30, every pre-existing resume gate's file inspected).
`run_mc` writes an UNCONDITIONAL end-of-temperature boundary checkpoint after
every temperature (run.jl's `_mc_loop!`), so a completed mc file always ends
at `temp_index = n + 1` and `resume`'s early return hands back the stored
result without re-running a sweep — a resume-equals-uninterrupted assertion
built on such a file compares the file's own stored results with themselves,
STRUCTURALLY, whatever the interval. The pre-existing **mc** gates
(`test_checkpoint.jl`'s three MC testsets plus the joint one, and S7's v5
strained-MC testset) had exactly that shape — their "last tick lands
mid-measure" comments were false — and were REPAIRED with the
interrupted-writer pattern on 2026-07-30 (each now asserts its file's mid-run
position by phase and sweep). `run_pt` is the
opposite: it has NO end-of-run write at all — the file ends wherever the
interval arithmetic last fired — so the pre-existing PT resume gates
(`test_checkpoint.jl`'s two, `test_pt.jl`'s async one, S10's v5) genuinely
land mid-measure and re-run 30–200 sweeps; they are NOT vacuous, but their
non-vacuity is an accident of the chosen intervals and must be asserted
(`0 < progress/done < total`), as the S12 gate does.

## S13 — the volume exponent, settled in the SAMPLED regime (2026-07-31)

S3 measured the exponent with `u ≡ 0`, which pins the acceptance rule's literal
constant `3·N_mob + c` and nothing more. That left the *physics* of the rule
untested, and a 2026-07-31 audit measurement on a lattice-only model duly came
back looking wrong: sampled `⟨s⟩` sat 2–3σ above the analytic marginal
`π(s) ∝ s^(3·N_mob − d_dim)·e^(−β n_cells j0)` at three temperatures and three
seeds, an "effective exponent" of 14–16 against that model's 12.

**The sampler is exact; the analytic model was short by `|dV/ds| ∝ s²`.** With
`u` sampled, the affine rescale's Jacobian on the sampled displacement space
cancels `d_dim` powers of `s` against the displacement integral, and what
survives is the gauge factor plus the volume element:

    π(s) ∝ s^(count(comp_free) + 2) · Z_u(s) · e^(−β(n_cells·j0(s) + P·V(s)))

For that fixture `count(comp_free) = 12`, so the prediction is **14 exactly** —
inside the reported range, and 132σ from the 12 the observer compared against.
Three independent lines agree: a first-principles derivation, an implementation
audit, and a purpose-built measurement.

### Why the existing fixtures could not settle it

Every earlier check was blind in one of two ways. The `u ≡ 0` toy measures the
constant, not the physics. The joint fixture used for the 400 k-move quadrature
has the two candidate conventions differing by only 3-of-3N — below its
resolution. And the anomalous fixture had **four disjoint displacement
components**, an artifact of a cutoff that does not reach across the training
cell, which confounds `count(comp_free)` with `n_disp_comps`.

### The fixture that does settle it

A 2-atom cell has ONE minimum-image pair orbit, so an `(n,1,1)` supercell always
splits into disjoint dimers — which is why no fixture in this package was
single-component. Three atoms at asymmetric fractional `x` (`0, 0.30, 0.62`) give
three distinct pair orbits and hence **one connected component**. The
coefficients are not random: the bond graph is read off the Hessian's sparsity,
the target `K = Σ_bonds |u_i − u_j|²` is built by hand, and a least squares over
the SALC space reproduces it to ~1e-16. So `E(u)` is exactly quadratic and
exactly `s`-independent, `Z_u(s)` is exactly constant, and with `j0 ≡ 0`, `P = 0`
the target is a truncated power law — arithmetic, not a second Monte Carlo.

`S` and `S0` are the same crystal at the same `N_mob` on the same grid; `S0` only
adds an on-site well, which pins every rigid shift. `count(comp_free)` goes
3 → 0 while `d_dim` goes 24 → 27, so **the two candidate countings move in
opposite directions** and the pair separates them with no absolute normalization.

Measured (4–8 independent chains; the spread of chains is the error — a
per-chain binning error understates it by ~1.7×):

| fixture | comps | `N_mob` | `F = count(comp_free)` | `d_dim` | regime | predicted | measured |
|---|---|---|---|---|---|---|---|
| S  | 1 | 9  | 3  | 24 | `u` sampled | `F+2 = 5`  | 5.0000 ± 0.0058 (0.0σ) |
| S  | 1 | 9  | 3  | 24 | `u ≡ 0`     | `3N+2 = 29`| 29.026 ± 0.041 (0.6σ) |
| S0 | 1 | 9  | 0  | 27 | `u` sampled | `F+2 = 2`  | 2.011 ± 0.010 (1.0σ) |
| D2 | 2 | 4  | 6  | 6  | `u` sampled | 8          | 8.021 ± 0.018 |
| C4 | 4 | 12 | 12 | 24 | `u` sampled | 14         | 14.017 ± 0.017 |
| D4 | 4 | 8  | 12 | 12 | `u` sampled | 14         | 14.035 ± 0.024 |
| D8 | 8 | 16 | 24 | 24 | `u` sampled | 26         | 26.013 ± 0.034 |

C4 and D4 share `F` while differing in both `N_mob` and `d_dim`, and land on the
same exponent. The rival countings are 345σ and 517σ away on S.

### What it does not depend on — each varied one at a time

Proposal arm (`:logvolume` 14.010 ± 0.045 vs `:scale` 14.022 ± 0.041 — the arm's
`c` is cancelled by `dy/ds`); temperature; the displacement/strain cadence ratio;
`n_disp_comps`; and **the re-centring cadence**, which gives *bitwise identical*
`s` trajectories at `recenter_every ∈ {1, 4, 16, 64, 256, never}` even though
`max|com_removed|` differs by 6.7. That is structural, not luck: `E` is exactly
invariant along the flat directions and the affine rescale commutes with a flat
shift, so `ΔE` is unchanged and the `s`-chain is exactly decoupled from the gauge
coordinates. An independent probe on `_ss_grid` through `run_mc` reproduced the
same bitwise identity.

### The one assumption that remains

A measurement cannot decide whether the flat directions *should* carry a `∝ s`
measure factor — that is a statement about a gauge direction the quotient sampler
never represents. The argument for it is the one already in §8: a rigid shift of
a displacement component by a supercell lattice vector maps every site onto its
own periodic image, i.e. the same microstate, so the gauge orbit is a torus of
edge `∝ s` and contributes one factor of `s` per free (direction, component)
pair. The data establish that the implementation is exactly self-consistent with
that choice; they do not establish the choice itself.

### Gates

`test_strainschedule.jl` "the sampled volume marginal is s^(count(comp_free) + 2)"
carries S and S0 with their fixture invariants asserted as hand predictions
(one component, equal `N_mob`, `F` = 3 vs 0, `d_dim` = 24 vs 27, fit residual
< 1e-12, bounded spectrum). **The pair is load-bearing, not decorative**: under
the `d_dim` mutation the S arm fails at 23σ and the differential at 3 vs −0.065,
while S0 alone passes at 0.5σ — a single-fixture version of this gate would be
half blind.

### Trap to avoid next time

When writing an analytic target for the `s`-marginal, the proposal-variable
Jacobian `|dV/ds|` is easy to omit — it is the whole of the `+2`, and omitting it
looks exactly like a two-power bias in the sampler. Check the exponent against
the frozen-`u` regime first: it must come out at `3·N_mob + 2`, and if it does
the acceptance rule is fine and any remaining discrepancy is in the model of the
displacement integral.
