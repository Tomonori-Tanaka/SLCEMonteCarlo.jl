# Decision record — parallel tempering over threads, and its determinism

Status: landed (M5). Owner: `src/pt.jl`; gates in `test/unit/test_pt.jl`.
P6 (the scope of the promise) is package-wide and authoritative for every
"bit-reproducible" claim in the README, docstrings, and guides.

## P1 — lanes own everything except the payload

Lane `r` = rung `r` of a strictly monotone temperature ladder, owning its
`ChainState`, `SweepScratch`, RNG, adaptive step, and measurement accumulators.
An accepted exchange swaps only the **payload** — `config`, `zrows`, `energy`,
and on a joint model `disps`/`com_removed`, on a strained run the cell scale and
the lane's Hamiltonian reference (each lane sweeps its own coefficient clone;
the installed coefficients travel with the scale they describe — strain-move.md
S10) — all O(1) reference swaps between adjacent lanes. RNG/step/accumulators
staying with the lane is what makes each lane's measurement stream a
*fixed-temperature marginal* (so `points[r]` is directly the physics at
`kts[r]`) and keeps the adapted step per-temperature. Strain moves are
lane-local (drawn from the lane RNG inside the lane's own sweeps) and the
strained swap weight `E + n_cells·j0(s) + P·V(s)` is a pure function of chain
state, so P2/P3 below apply unchanged with the strain channel live (gated).

## P2 — segment schedule

All lanes sweep `exchange_interval` compound sweeps per segment; between segments
the coordinator attempts adjacent-pair swaps with
`min(1, exp((βᵢ−βⱼ)(Wᵢ−Wⱼ)))` — `W` the configurational energy on a fixed cell,
`E + n_cells·j0(s) + P·V(s)` on a strained run — alternating even/odd pair
parity per exchange step
(parity carries across the thermalization→measurement boundary). Exchanges run in
**both** phases — the point of PT is that cold rungs keep escaping metastable
basins during measurement too. Step adaptation is thermalization-only per lane
(as in `run_mc`); at the boundary every lane renormalizes, freezes, and resets its
counters/accumulators.

## P3 — determinism (the load-bearing part)

Bit-identical results for a fixed seed **regardless of `ntasks`, `sweep_tasks`, and
`JULIA_NUM_THREADS`** (gated by `ntasks = 1` vs `ntasks = R` equality on every
stat, config, and swap rate):

- `master = Xoshiro(seed)` → four `UInt64` draws per lane RNG in lane order, then
  the exchange RNG; initial configs from each lane's own RNG.
- Lane RNGs are consumed only inside that lane's sweeps (thread-confined); the
  exchange RNG is consumed **serially, pre-drawing per async block** — one uniform
  **unconditionally** per attempted pair, boundary-major in ascending pair order,
  exactly the serial schedule's sequence (an accept-dependent draw would leak the
  decision history into the stream). The async schedule only *reads* the pre-drawn
  values, so the stream never sees thread timing; a boundary's swap decision is a
  pure function of the pre-attributed uniform and the two chains' energies, which
  are Markov-chain-determined.
- Accumulators are lane-owned; the per-pair swap counters have a single writer
  (the pair's lower lane) — no shared mutable state race, nothing depends on
  thread timing.

## P4 — thread layout

`ntasks = 1` is the serial reference schedule. Any `ntasks ≥ 2` (default when
threads are available) runs **one task per lane** for each async block — the
sweeps between two global sync points (checkpoint writes and phase ends; without
periodic checkpoints, a whole phase is one block). At an exchange boundary only
the two lanes of an attempted pair handshake (the lower lane waits for its
partner's arrival, applies the swap on the parked partner's payload, and releases
it — per-lane `Threads.Condition`s, no spinning, oversubscription-safe), so a
straggling lane (an E-core lane, a renormalization) stalls its neighbors instead
of the whole ladder; the alternating parity chain still rate-limits everything to
the slowest lane on average, so the win is straggler absorption (measured ~5–13 %,
largest at `exchange_interval = 1` on mixed P/E cores — bench_log #7). A dying
lane task poisons the block (`_PairSync.failed`) so partners exit their waits and
`@sync` surfaces the original exception (wrapped in the usual
`CompositeException`/`TaskFailedException`) instead of livelocking. Julia's
scheduler multiplexes `R` tasks over the available threads, so `R > nthreads()`
needs no chunking. `run_mc` stays strictly serial (parallel independent chains at
one temperature are a possible future extension; the lane machinery already fits
it).

## P5 — ladder guidance (heuristic, revisit after real-model use)

Adjacent swap acceptance is the diagnostic: aim for O(0.2–0.5); a collapsed pair
partitions the ladder. Geometric spacing in `kT` is the usual starting point;
tighten where `C(T)` peaks. The frozen-fixture gate demonstrates the payoff: at
`kT = 0.03` the anisotropic test model traps independent chains in different
basins, while a 4-rung ladder to `kT = 0.45` recovers the low basin.

## P6 — scope of the promise: a testing discipline, not an eternal guarantee

Bit-reproducibility here is primarily a **testing and debugging instrument**, not a
user-facing feature — MC physics must be seed-robust anyway; the currency of results
is the error bar, never the last bit. The discipline is kept because it is nearly
free at runtime (deterministic color-ordered scan, draw-only-when-needed, per-site
and lane-owned RNGs are design choices, not overhead) and it buys exact `==` gates
that statistics cannot: resume ≡ uninterrupted run (checkpoint correctness),
`ntasks = 1` ≡ `ntasks = R` and `sweep_tasks = 1` ≡ `sweep_tasks = N` (data-race
detectors — a race that shifts results within error bars is otherwise undetectable),
non-flaky fixed-seed CI gates, and bisectable divergences.

**Guaranteed** (and gated): for a fixed seed, with the *same package version and the
same Julia version*, runs are deterministic and independent of `ntasks` /
`sweep_tasks` / `JULIA_NUM_THREADS`; a resumed run equals an uninterrupted one bit-for-bit.

**Explicitly not guaranteed:**

- **Across Julia versions.** Julia does not promise `rand`/`randn` stream stability
  between releases, so fixed-seed trajectories may change on a Julia upgrade —
  nothing this package can control.
- **Across package versions.** A change that alters the RNG-consumption stream (a
  new update scheme, a site-skip rule, a proposal tweak) is allowed and is simply
  recorded as **breaking** in the CHANGELOG (precedent: the inactive-site skip).
  Determinism never holds veto power over a better algorithm — e.g. a future
  checkerboard-parallel sweep would change the stream and costs one CHANGELOG line.
- **The last bits of derived observables across refactors.** The promise covers the
  **Markov-chain trajectory** (RNG stream, spin updates, acceptance counters) plus
  the checkpoint/resume and `ntasks` equalities above. Floating-point summation
  order inside observable *measurement* (e.g. pairwise vs sequential `sum` in
  `:m`) is an implementation detail; ULP-level shifts there are acceptable when the
  trajectory is untouched — though same-version gates that compare stats (`==` in
  the resume/`ntasks` tests) of course still pass, since both sides run the same
  code.
