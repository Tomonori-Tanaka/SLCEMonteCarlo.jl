# Decision record — cell reduction (`reduce_cell`)

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
- an orbit has exactly `nc` distinct training members per summand (the anchor's
  coset determines the translation uniquely); a raw list carrying `q` identical
  summands per instance (hand-built directed pairs) shows `q·nc` and reduces to
  `q` copies;
- pure translations do not rotate spins, so orbit members share `coef` and the
  **aligned** `folded` (same SALC orbit ⇒ the same fitted `jϕ`; the axis
  permutation is exactly compensated by `permutedims`).

So reduction = map every term to its canonical anchored form, group, keep one
representative per group — and **the group census is the verification** (R3).
Coefficients stay raw; the `(4π)^(body/2)` scale still happens exactly once, in
`TiledHamiltonian`.

## R3 — verified, never assumed

`reduce_cell` hard-errors (no silent symmetrization) unless all of:

1. **Lattice**: `A_sub \ A_train` is integer (`pos_tol`-scaled residual). Any
   integer `M` is accepted — non-diagonal (primitive ↔ conventional) and
   `|det M| = 1` re-basings included.
2. **Structure**: grouping atoms by (species, fractional residual mod 1 within
   `2·pos_tol`) yields groups of exactly `nc` atoms with `nc` distinct offsets, and
   `nc` divides `n_atoms`.
3. **Hamiltonian**: every canonical anchored group, sub-partitioned by
   (`coef`, aligned `folded`) within `coef_rtol` (distinct SALCs on one cluster
   stay distinct), has a member count that is a multiple `q·nc` (emitting `q`
   representative copies; `q = 1` for canonical model terms). A fit on a
   distorted structure, or couplings that break the pseudo-translation (e.g. one
   perturbed coefficient), fails here with the offending term named.

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
- `:sublattice_m` components index *reduced*-cell atoms; `parent_atoms` / `atom_map`
  translate to training-cell atoms.
- The existing per-term site-distinctness check in the `TiledHamiltonian` ctor is
  the guard against too-small `dims` of the reduced cell (self-image folding).

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
