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

    ln A = 3·power·ln(s′/s) − β·[ΔE_config + n_cells·Δj0 + P·ΔV]

where `power = D/3 + 1` (`:logvolume` proposals) or `D/3 + 2/3` (`:scale`), and
`D = 3·n_disp_active − count(comp_free)` is the dimension of the sampled
displacement space. It never runs inside the color-parallel sweep layer — one
serial move per `strain_interval` compound sweeps.

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
(`_strain_y` / `_strain_s_of_y` / `_strain_power`, adjacent in `strain.jl`),
because §8(β)'s live trap is drawing in one arm's variable while weighting with
the other's — off by `(V′/V)^{2/3}`, invisible on a production cell and
percent-level on the fixtures. Excluded twice: hand-derived closed forms per arm
(`(D+3)`/`(D+2)` powers of `ln(s′/s)`), and a white-box replay in which every
accept decision over both arms must match a hand-paired reconstruction from a
cloned RNG. The statistical marginal cannot resolve the 1/3-power difference at
test cost, which is why the replay exists.

## S3 — the exponent is measured, not asserted

On the §8(γ) toy — constant coefficients across the grid, `j0 ≡ 0`, `P = 0`,
`u ≡ 0` — the stationary volume marginal is the bare Jacobian `p(V) ∝ V^{D/3}`.
The gate histograms the chain and compares the empirical mean volume against the
analytic mean for the fixture's own `D`, and REJECTS the neighbouring `D ± 3`
(the two fixtures — a translation-flat pair model and an on-site-pinned one —
differ by the full `count(comp_free)`, isolating the per-(direction, component)
COM bookkeeping; `D` is not `3(N−1)` and need not be a multiple of 3).

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
  absolute lengths in the same frame) and calls `_reset_escape!` — an accepted
  rescale is a phase boundary for the escape detector, whose radius statistics
  are absolute lengths against a cell that just changed (§8(θ));
- the chain's scale is replica-exchange **payload** (swapped with the state),
  though in v0 PT never strains (S6).

## S5 — ASR hard-error at conversion, not per proposal

`StrainSchedule(sm, H)` on a translation-invariant `H` builds every grid node at
1×1×1 and lets the constructor's flatness measurement refuse a non-flat node,
wrapping the error with the grid context. Flatness is linear in the
coefficients, so flat at every node ⇒ flat for every interpolated model — which
is what licenses `strain_move!`'s `recheck_translation = false` per proposal. A
pinned `H` (`fixed_reference`) declared an absolute frame; its family's pattern
is the caller's contract and the node check is skipped.

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

`run_pt` refuses a schedule by name: every lane sweeps ONE shared
`TiledHamiltonian` by reference while `set_coefficients!` mutates it in place
(a data race before it is a physics question), and `_attempt_swap!` is the NVT
rule where NPT needs `(β_a−β_b)[(E_b+PV_b)−(E_a+PV_a)]` with the strain in the
payload. PT + strain is its own future slice.

GPU: `sync_coefficients!(gH)` re-uploads the ONE device array a host coefficient
swap moves (`sent_w`; every structural table is coefficient-independent because
the site-program skip tests `folded`, never `coef·folded`). Gated bitwise
against the keyed reference on the new weights. The per-move device cost has NOT
been measured on real hardware yet — measure on the next kugui A100 session
before production NPT+GPU runs; the refusal is held in reserve if it disappoints.

Fixed-cell byte-neutrality is pinned, not assumed: a short fixed-cell `run_mc`
trajectory captured at commit `d038b86` (pre-wiring) is asserted bit-identical
in the driver testset. If an intentional sampler change moves it, recapture; any
other movement is the regression the pin exists to catch.

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
- **The chain starts at `s = 1`** (must be in the domain); warm-starting a
  strained chain from a previous run's final scale is not wired (`MCResult`
  carries no final strain) — resume from a checkpoint instead.
- **Mechanical-equilibrium identity observable** (§8(ζ):
  `P = (D/3)·kT/⟨V⟩ + ⟨−∂E/∂V⟩` via `grid_strain_derivative`) — the one
  production-scale check; belongs in the observable set, not yet implemented.
- **PT + strain**, per S6.
