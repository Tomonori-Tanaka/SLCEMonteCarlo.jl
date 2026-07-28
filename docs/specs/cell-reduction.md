# Decision record — cell reduction (`reduce_cell`)

> **Naming note (2026-07-28).** This is a dated decision record and is kept as
> written; the names below are the ones the decision was taken under. Renamed
> since, in the family-wide naming batch: `multipole_terms` → `spin_multipole_terms`. The current spelling is what
> the code, `SPEC.md` and the API reference use.


Status: landed (post-v0). Owner: `src/reduce.jl`; gates in `test/unit/test_reduce.jl`.

## R1 — why

`TiledHamiltonian` tiles diagonal integer multiples of the cell its terms are
expressed in. A model fitted on a large supercell (e.g. a 4×4×4 bcc conventional
cell, 128 atoms) therefore only offered MC sizes in ×4 jumps — too coarse for
finite-size scaling. `reduce_cell` re-expresses the fitted Hamiltonian in a
**user-chosen smaller cell** (e.g. the 2-atom conventional cube) after verifying the
choice, so `dims` counts multiples of that cell instead.

Rejected: general non-diagonal supercell support in `TiledHamiltonian` itself
(HNF-coordinate wrapping). Reduction subsumes the use case — the user names the fine
cell once, and every downstream mechanism (tiling, updates, PT, checkpointing,
`supercell_crystal`) is reused unchanged, including non-diagonal `M` between the two
cells.

## R2 — the invariant that makes it exact bookkeeping

Let `A_train = A_sub · M` with `M` integer, `nc = |det M|`. Training atom `a`
decomposes as `(b_a, o_a)` — reduced-cell atom and integer sub-lattice offset — via
`f_sub = M f_train`. Member `i` of a training term (atom `a_i`, training-lattice
shift `s_i`) then carries the reduced-lattice shift `σᵢ = o_{aᵢ} + M sᵢ`, and the
**canonical anchored** reduced form — sites sorted by `(reduced atom, σ)`, then
`σᵢ − σ₁` with `shifts[1] = 0` restored, `ls`/`folded` carried through the sort
permutation — is *invariant under the `nc` coset translations*: translating the
whole cluster by `t` adds `t` to every `σᵢ` and cancels in the differences, and
the sort undoes the anchor-role swap the translation induces. (SLCE's
canonical members carry one term per physical instance, so two translation
copies are generally anchored at *different* member sites — without the joint
sort + tensor-axis alignment they would not land on one key.) Consequences:

- one translation orbit of training terms ↦ exactly one canonical anchored
  reduced term;
- an orbit has exactly **one** training member per coset per summand (the anchor's
  coset determines the translation uniquely); a raw list carrying `q` identical
  summands per instance (hand-built directed pairs) shows `q` per coset and reduces to
  `q` copies. The coset of a term is read off its **absolute** anchor `σ₁` with the
  integer adjugate: `σ ↦ mod.(adj(M)·σ, nc)` is a *complete* invariant of `σ + Mℤ³`
  (forward because `adj·M = det·I`, backwards because `adj·v ≡ 0 (mod nc)` forces
  `v ∈ Mℤ³`), so the classification is exact integer arithmetic like everything else
  here;
- pure translations do not rotate spins, so orbit members share `coef` and the
  **aligned** `folded` (same SALC orbit ⇒ the same fitted `jϕ`; the axis
  permutation is exactly compensated by `permutedims`).

So reduction = map every term to its canonical anchored form, group, keep one
representative per group — and **the group census is the verification** (R3).
Coefficients stay raw; the `(4π)` scale still happens exactly once, in
`TiledHamiltonian`.

## R2b — channels: what an axis is (M4 slice 3e)

A pure-spin term's axis `i` *is* its site `i`, so one permutation does both jobs. A
`SLCE.DecoratedTerm` breaks that identity: `slots[v]` names a member position **and**
a `(channel, k, l)` factor, several slots may share one site, and `folded`'s axes are
slots. The canonical anchored form therefore carries **two** permutations:

1. the site sort of R2, applied to `atoms`/`shifts` — and, through `invperm`, to every
   slot's `site` field;
2. the induced re-sort of the slots into SLCE's canonical `(channel, site, k, l)` axis
   order, applied to `folded` with `permutedims`. The original slot index breaks ties,
   so a term repeating one factor on one site still orders deterministically.

Invariance survives unchanged: a lattice translation moves no displacement vector and
rotates no spin, so orbit members share `coef`, `scale` and the aligned `folded`
exactly. `ReducedCell` is parameterized on the term type and emits the list in the form
it was given — never converting — and a decorated reduction additionally carries the
model's `SLCE.RowLayout`, because a slot's `(channel, k, l)` can only be placed by the
numbering the model itself defines. `reduce_cell(model, …)` picks the arm exactly as
`TiledHamiltonian(model)` does (`isempty(row_layout(model).disp_factors)`, via
`restrict(model, :spin)`), so a pure-spin model still takes the frozen path byte for
byte.

**`scale` is carried verbatim.** It is the `(4π)^(n_spin_slots/2)` field of
`DecoratedTerm`, and re-deriving it from `length(atoms)` is exactly the silent
mis-scaling the field exists to prevent (design record §13 risk 3). Its consistency
across one key is checked in R3; its *value* is not second-guessed, for the reason
given there.

**Why a ≤ 2-body fixture cannot gate this.** A 2-body cluster's site permutation is
always an involution, so `invperm(perm) == perm` and the relabel direction is
unfalsifiable — as is the whole pure-spin suite, which is 2-body throughout. The gate
is a 3-body mixed chain whose translation copies come back as genuine 3-cycles
(`test_reduce.jl`); reversing the direction in `src/reduce.jl` breaks *only* that
testset, which is exactly why it exists.

## R3 — verified, never assumed

`reduce_cell` hard-errors (no silent symmetrization) unless all of:

1. **Lattice**: `A_sub \ A_train` is integer (`pos_tol`-scaled residual). Any
   integer `M` is accepted — non-diagonal (primitive ↔ conventional) and
   `|det M| = 1` re-basings included.
2. **Structure**: grouping atoms by (species, fractional residual mod 1 within
   `2·pos_tol`) yields groups of exactly `nc` atoms with `nc` distinct offsets, and
   `nc` divides `n_atoms`.
3. **Hamiltonian**: every canonical anchored group, sub-partitioned by
   (`coef`, aligned `folded`) within `coef_rtol` (distinct SALCs on one cluster stay
   distinct), has the **same** member count `q` in **each** of the `nc` cosets
   (emitting `q` representative copies; `q = 1` for canonical model terms). A fit on a
   distorted structure, or couplings that break the pseudo-translation (e.g. one
   perturbed coefficient), fails here with the offending term named.

   *Per coset, not in total.* The weaker `count % nc == 0` — which is what this check
   was until M4 slice 3e — is satisfied by a class that lives in **one** coset: four
   copies of a term all anchored in the same coset of a `nc = 4` reduction pass it and
   are emitted as one representative, i.e. as a term sitting in *every* reduced cell.
   That is a different Hamiltonian, silently. Unreachable from `decorated_terms` /
   `multipole_terms` output (two SALCs on one key with bit-identical `folded` would be
   linearly dependent), reachable by stitching two term lists together — and the whole
   claim of this file is "verified, never assumed". Gated.

   `scale` is settled separately, for a whole key at once: every copy must declare the
   identical value (reported as a *scale disagreement*, since two spellings of one
   number — `sqrt(4π)` vs `(4π)^0.5`, 1 ulp apart — would otherwise surface as a
   physics failure). Deliberately **not** checked: whether that value is
   `(4π)^(n_spin_slots/2)` at all. `TiledHamiltonian`, the surface that applies the
   scale, treats the field as declared data too, and the package's own
   kernel-arithmetic fixtures use it as a free multiplier; enforcing the rule here
   alone would be an inconsistency, not a safeguard.

   A subtlety found while gating: for **multi-channel** clusters (anisotropic
   `l ≥ 2`), equal coefficients on every SALC do *not* make a `NoSymmetry`-fitted
   model periodic — each per-bond orbit picks its own (arbitrary, orthogonally
   mixed) SALC tensor basis, so translation-partner bonds carry different summed
   tensors. Translation-closed orbits (e.g. a `SpglibBackend` fit on the true
   structure) are what guarantee check 3; `reduce_cell` refusing the former is a
   physics refusal, not a tolerance artifact (both are gated).

Averaging near-miss orbits into a symmetrized Hamiltonian was rejected: it would
silently change the model. The representative's `coef`/`folded` are taken verbatim
(orbit members are bit-identical in practice — same SALC orbit).

Reduction along a **non-periodic** direction (`pbc = false`) is not specially
guarded: a structure never repeats along it, so check 2 rejects any `M` that shrinks
that axis (identity along it passes harmlessly). `pbc` flags are carried onto the
reduced `Crystal` unchanged.

## R4 — determinism and downstream contracts

- Output ordering is deterministic: groups by first occurrence in atom /
  term order, so repeated `reduce_cell` calls build identical `TiledHamiltonian`s
  (checkpoint fingerprints match).
- `ReducedCell.crystal` orders atoms by reduced index (group representatives), so
  `supercell_crystal(red.crystal, dims)` matches `TiledHamiltonian(red; dims)` site
  order — the same pairing contract as the training-cell path.
- `:sublattice_m` / `:sublattice_u2` components index *reduced*-cell atoms;
  `parent_atoms` / `atom_map` translate to training-cell atoms.
- The existing per-term site-distinctness check in the `TiledHamiltonian` ctor is
  the guard against too-small `dims` of the reduced cell (self-image folding).
- `TiledHamiltonian(red; dims, fixed_reference)` forwards `fixed_reference`: whether a
  model pins its displacement reference is a property of the Hamiltonian, and
  reduction does not change it.

## Gates (`test/unit/test_reduce.jl`)

Hand-unfolded diagonal (2×2×1) and non-diagonal (`det M = 2`) supercells reduce back
to the small-cell +x-form representative **exactly** (`==` on every field; the
hand-built ±x directed pair folds onto one canonical key and comes back as two
copies of the +x form); random-config
energy identity through the site permutation at 1e-13; fitted models reduced 2×
(isotropic and anisotropic `l ≤ 2` — the latter exercising the (`coef`, `folded`)
sub-partition with several SALC channels per cluster) agree with
`predict_energy − intercept` at 1e-12; a fitted non-diagonal (`det M = 2`
checkerboard) reduction passes a **non-uniform** coset-painted energy identity;
identity (`|det M| = 1`) and left-handed (`det M < 0`) reductions reproduce the
training Hamiltonian; each verification failure mode (broken coefficient, distorted
structure, species mismatch across cosets, non-integer lattice, indivisible atom
count, coincident fold, empty terms, model/crystal mismatch) throws.

**Channel arm (R2b).** A hand-built mixed-channel `DecoratedTerm` chain (a pure-spin
pair `(4π)^1`, a spin × displacement pair `(4π)^(1/2)`, a lattice-only well `(4π)^0`)
unfolds onto a 2× cell and reduces back to every field verbatim — `scale` included,
with two of the three terms having a `scale` that `(4π)^(body/2)` would get wrong;
its energy identity runs through the joint `total_energy(H, config, disps)`. A
**3-body** mixed chain, canonicalized after unfolding (a helper validated against the
evaluator, not against the code it mirrors), produces two genuine 3-cycles and is the
only gate that can see the relabel direction. A non-diagonal `M` with a displacement
axis pins that a change of cell basis re-expresses shifts and never touches `folded`.
A fitted joint model (`_stacked_joint_model`, spglib-symmetric so the SALC orbits are
translation-closed) reduces 2× with the `predict_energy` gate, a non-commensurate
tiling, and an **odd** reduced-cell count — and the reduction is checked to be
non-vacuous: exactly 29 of its 62 terms need a non-identity site permutation *and* a
non-identity slot re-sort (pinned, not `> 0`, so the count in this file cannot go
stale). A broken coefficient is refused on this arm too; so are lopsided cosets, a
scale disagreement on one key, a body-0 term, a `folded` extent that disagrees with
its slot's `l`, and a slot naming a member position outside the cluster.
