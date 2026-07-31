# Decision record — checkpoint schema (v6; landed as v1) and bit-identical resume

Status: landed (M6; v4 with M5-4, v5 with PT + strain, v6 with the volume-grid boundary screen). Owner: `src/checkpoint.jl`;
gates in `test/unit/test_checkpoint.jl` and (strain / PT + strain)
`test/unit/test_strainschedule.jl`.

## C1 — format: JLD2, plain data only

A checkpoint is a JLD2 file of **plain data** — `Bool`/`Int`/`Float64`/`UInt64`/
`String` scalars and arrays in named groups. No Julia struct is ever serialized
for reconstruction, so a package refactor cannot silently break the format (the
SpinClusterMC hand-rolled positional-serialization failure mode). Writes are
atomic: PID-suffixed temp file + `mv` (one writer per checkpoint path assumed —
concurrent runs must use distinct paths). Checkpoint writing consumes no RNG
(gated: a checkpointed run equals an uncheckpointed one bit-for-bit).

Rejected: `Serialization` stdlib (positional, Julia-version-fragile), TOML/JSON
(no bit-exact `Float64` round-trip without hex-float contortions, huge configs).

## C2 — schema v6

v2 (2026-07-15, colored sweeps): adds `plan/sweep_tasks` and the per-site RNG
streams `chain/site_rngs` (a `words × n_sites` UInt64 matrix — one Xoshiro per
site).

v3 (2026-07-27, M4 slice 3c/3, the displacement channel): the chain's second
coordinate and its second proposal width join the file. Written unconditionally,
including on a pure-spin model — `disps` is then all-zero and `com_removed` is a
3×0 array, which is the honest encoding of "this model has no displacement
components", not a special case to branch on. Older files are rejected by the
version check (pre-release breaking change).

v4 (2026-07-29, M5-4, the NPT strain channel): the chain's cell scale
`chain/strain`, its acceptance counters (the `counters` vector grows 6 → 8), the
plan's strain fields, the per-point `acceptance_strain`, and `grid_fingerprint` —
the volume grid's own FNV-1a identity (scales, abscissa, centred polynomials,
`v_train`, `n_cells`, `n_mobile`, `d_dim`; `0` on a fixed-cell run). All written
unconditionally; a fixed-cell chain stores `strain = 1.0` and zero counters. On a
**strained** run `model_fingerprint` is the model's identity **at the reference
scale `s = 1`**: the fingerprint mixes coefficient values (a hot-swap must refuse
to resume) and a strained chain's coefficients move with its volume, so both the
writer (checkpointer built while `H` carries the reference) and `resume`
(reinstalls the reference from the supplied schedule before comparing) pin the
same state. v3 files are rejected by the version check, with the strain channel
named as the reason.

v5 (2026-07-29, PT + strain, strain-move.md S10): **no layout change** — every
group v5 writes, v4 already wrote (`chain/strain` and the strain counters were
serialized per PT lane from the start). What changed is the semantics: v4's PT
lane strains were never live, so a v4-era reader handed a strained-PT file would
pass every handshake and silently continue the run as **fixed-cell** at the
reference (`_pt_run!` had no strain support). The version bump turns that silent
wrong physics into a named refusal in both directions — the same refuse-by-name
precedent as v3 → v4.

v6 (2026-07-31, the volume-grid boundary screen): `chain/strain_range` (the
phase's `[min, max]` sampled cell scale) and a ninth entry in `chain/counters`
(`att_strain_out`, the strain attempts the volume grid refused for landing
outside its domain), plus the matching per-point `points/i/strain_range` — three
Float64: `strain_min`, `strain_max`, `strain_outside`. Like the escape
accumulators these steer no random decision, so a resume is bit-identical with or
without them; they are serialized because dropping them would restart the
interval and the rate at every checkpoint, leaving a resumed run's boundary screen
able to see only its final segment — and under-reporting a truncated volume
marginal is exactly the failure the screen exists to catch. v5 files are rejected
by the version check with that named as the reason.

```
schema_version    Int     == 6, hard-checked on load
kind              String  "mc" | "pt"
julia_version, package_version   String (informational)
model_fingerprint UInt64  stable FNV-1a over (n_cell_atoms, dims, the row layout
                          on a joint model, every term's coef/atoms/shifts/ls/
                          slots/folded) — NOT Base.hash (which is
                          Julia-version-dependent); mismatch on resume ⇒ error;
                          on a strained run, taken at the reference scale
grid_fingerprint  UInt64  the strain grid's identity (0 = fixed cell); resume of
                          a strained run must supply the matching StrainSchedule
checkpoint_interval, exchange_interval   Int
plan/*            every UpdatePlan field (kts, sweeps, intervals,
                  or_per_metropolis, disp_per_metropolis, step0, step_u0,
                  adapt_*, renorm_interval, nbins, carryover, sweep_tasks, seed,
                  strain_interval, strain_proposal, strain_step, pressure)
plan/observable_names, plan/observable_ncomps   resume-compatibility check
-- kind == "mc":
progress/{temp_index, phase ("therm"|"measure"), sweep}
npoints; points/<i>/{kT, acceptance_* (incl. acceptance_strain),
                     strain_range (3: strain_min, strain_max, strain_outside),
                     final_step, final_step_u, max_drift, disp_rms, disp_max,
                     disp_checks, escaped, stat_names,
                     stats/<name>/{mean, err, tau_int, count}}
chain/{config (3×n), disps (3×n), com_removed (3×n_disp_comps), energy,
       rng (UInt64 words), site_rngs (words × n_sites), step, step_u, strain,
       strain_range (2: the phase's sampled [min, max] scale),
       frozen, counters (9: metro/or/disp/strain × acc/att, + strain-outside),
       max_drift, escape_f (6 Float64), escape_i (4 Int), escape_warned}
has_accs; accs/<obs>/{binner/{count, sums, sums2, pending, pending_full, n},
                      store/{bin_size, means, nfull, acc, nacc}}
-- kind == "pt":
progress/{phase, done, parity}
exchange_rng; swap_att; swap_acc; nlanes
lane/<r>/{chain fields...}; lane/<r>/accs/<obs>/... (measure phase only)
```

`escape_f`/`escape_i` are the escape detector's block accumulators
(`updates-stationarity.md` U8). They steer no random decision, so a resume is
bit-identical with or without them — but dropping them would restart the block
ladder at every write, and a chain checkpointed often enough would then never
accumulate the consecutive strikes that report an escape.

## C3 — what makes resume bit-identical

- `config` and `disps` are stored exactly; `zrows` are **rebuilt** from them
  (`Zlm_unsafe` and the solid-harmonic recursion are pure functions — same bits),
  while `energy` is restored **verbatim** (recomputing would erase the incremental
  value the trajectory depends on).
- `com_removed` is part of the state, not a diagnostic: the re-centred frame a
  chain has been running in is the frame its displacements are expressed in, and a
  resume that dropped it would silently rebase every subsequent absolute-frame
  report.
- Xoshiro state is captured generically over `fieldnames(Xoshiro)` (5 words on
  Julia 1.12) and rebuilt with `Xoshiro(words...)`; a word-count mismatch (another
  Julia's layout) errors instead of silently reseeding. Draw-stream equality is
  gated over 100 draws.
- All counters (acceptance windows, `since`-last-write, phase sweep counts, PT
  parity and swap tallies) are stored; every schedule (adapt/renorm/measure/
  checkpoint/exchange) is deterministic in them.
- LogBinner cascades and BinStore partial bins are stored in full, so error bars
  continue exactly (restore-path inner constructors on both types).
- Write points: MC — every `checkpoint_interval` sweeps + every temperature
  boundary; PT — at segment boundaries once `checkpoint_interval` sweeps have
  accumulated + the thermalization→measurement boundary. `interval = 0` ⇒
  boundaries only.
- Resume boundary semantics match the uninterrupted control flow exactly: a
  temperature-boundary checkpoint re-runs the `carryover = false` restart draw on
  the restored RNG (as the uninterrupted run would); a mid-phase checkpoint skips
  straight into the loop at `sweep + 1`.

## C4 — resume contract

`resume(path, H; observables, evaluables, checkpoint = path, strain)` — the
caller re-supplies the Hamiltonian and the observable/evaluable *functions*
(closures are not serializable); the file's `model_fingerprint` and observable
names/ncomps are checked and mismatches error. The returned result covers the
**whole** run (completed `TempResult`s are stored in the file as plain data and
re-emitted). By default the resumed run keeps checkpointing to the same path with
the stored cadence.

A strained run's file (`plan/strain_interval > 0`) requires the matching
[`StrainSchedule`] as `strain` — the schedule itself is not serialized, only its
fingerprint. The handshake order is load-bearing: grid fingerprint first, then
the structural pairing check, then the **reference-coefficient reinstall into
`H`**, and only then the `model_fingerprint` comparison (which is defined at the
reference); after the state is read, the checkpointed scale's coefficients are
installed so the `(H, chain)` contract holds when the loop continues — for the
"mc" kind into `H` itself, for the "pt" kind into a fresh per-lane coefficient
clone at each lane's own checkpointed scale (strain-move.md S10) — and on
return `H` is handed back at the reference (`run_mc` restores it; `run_pt` and
a "pt" resume never move it off the reference in the first place). The caller's
coefficient state on entry is irrelevant and overwritten — including on a failed
resume, which leaves `H` at the reference. A fixed-cell file refuses a supplied
schedule, and vice versa, both by name.
