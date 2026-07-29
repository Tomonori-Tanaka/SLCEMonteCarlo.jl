# Checkpointing and restart

```@meta
CurrentModule = SLCEMonteCarlo
```

Long runs checkpoint to a JLD2 file and restart **bit-identically** — the resumed
trajectory, error bars, and final configurations equal the uninterrupted run's
exactly (this is tested with `==`, not `≈`).

```julia
run_mc(H; temperature = [900, 600, 300], sweeps_measure = 10^6,
       checkpoint = "run.jld2", checkpoint_interval = 50_000, seed = 1)

# …after a crash / walltime kill:
result = resume("run.jld2", H)      # returns the FULL run's MCResult
```

The same keywords work on [`run_pt`](@ref) (writes at segment boundaries).
`checkpoint_interval = 0` writes only at natural boundaries (MC: each
temperature; PT: the thermalization→measurement boundary).

## The resume contract

- The caller re-supplies `H` and any custom `observables` / `evaluables` —
  function objects are not serialized. The file stores the **model fingerprint**
  and the observable names/component counts, and errors on any mismatch (a resume
  against different physics never silently continues).
- The returned result covers the whole run: completed temperatures are stored in
  the file as plain data and re-emitted.
- By default the resumed run keeps checkpointing to the same path with the stored
  cadence (`checkpoint = nothing` disables, `checkpoint_interval` overrides).
- Resuming a run that already **finished** simply returns its result unchanged —
  `resume` is idempotent, so a blind "resubmit and `resume` if the file exists"
  retry loop (see [parallelism](parallelism.md)) is safe.

## Why it is bit-identical

The file captures the whole sampled state — configurations, displacements and the
frame they have been re-centred to, the cell scale on an NPT run — plus the
incremental energy (restored verbatim), Xoshiro RNG words, both proposal widths,
every schedule counter, and the full binning-accumulator state; and every schedule
in the package is deterministic in those counters. Writes are
atomic (temp file + `mv`) and consume no RNG, so checkpointing never perturbs the
run it protects. One writer per checkpoint path — two concurrent runs must not
share one. Schema and rationale: `docs/specs/checkpoint-schema.md`.

## Strained (NPT) runs

A strained run's checkpoint additionally stores the volume grid's fingerprint, and
resuming it needs the same grid:

```julia
result = resume("run.jld2", H; strain = sch)
```

The schedule is not serialized (it is derived data, and the check is what makes a
wrong one impossible), so the handshake is by fingerprint. Passing a schedule to a
fixed-cell checkpoint — or omitting one on a strained checkpoint — is an error,
and the model-fingerprint check runs *after* `resume` has reinstalled the
reference-scale coefficients, because that fingerprint mixes coefficient values
and a strained chain has moved them. Files written by an older schema version are
refused by name rather than continued under different physics.
