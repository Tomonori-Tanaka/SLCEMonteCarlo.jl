# Decision record — supercell tiling and the Hamiltonian memory layout

Status: landed (M1). Owner: `src/hamiltonian.jl`, `src/energy.jl`;
gates in `test/unit/test_hamiltonian.jl`, `test/unit/test_energy.jl`.

## T1 — unfold `MultipoleTerm.shifts`, one instance per (term, cell)

`multipole_terms(model)` emits every directed cluster member once, with per-site
integer **training-cell** lattice translations `shifts` (`shifts[1] = 0` anchored —
verified in SLCE: orbit members retain the anchored candidate-cluster shifts).
Tiling onto `dims = (N₁, N₂, N₃)` is pure integer bookkeeping:

```
for every supercell cell t and every term:
    member i sits at site_index(atoms[i], mod.(t + shifts[i], dims))
```

Toroidal (periodic) boundary conditions; no ½ / 1/N factors (the introspection
contract already makes terms plain summands, `Σ terms = predict_energy − intercept`).
Site indexing is atom-fastest, cells column-major:
`site = atom + n_cell_atoms·(c₁ + N₁(c₂ + N₂c₃))`, so `site_atom(s) = mod1(s,
n_cell_atoms)` is the sublattice id.

Consequences pinned by machine-precision gates:

- `dims = (1,1,1)` degenerates to the training cell exactly:
  `total_energy == predict_energy − intercept` (1e-12).
- A periodically replicated configuration has exactly `prod(dims) ×` the cell energy.
- Tiling **replicates** the fitted finite-range couplings; it does not manufacture
  longer-range physics the training cell could not resolve.

## T2 — distinct member *sites*, not distinct atoms

The exact single-spin ΔE (`ΔE = c_s·ΔZ`) needs each instance to touch a site at most
once, so `site_coeffs!`'s leave-one-out vector is independent of that site's spin.
The invariant is checked per term at the **site level under the actual `dims`**:
`(atomᵢ, mod.(shiftsᵢ, dims))` pairwise distinct (the wrapped relative pattern is
cell-independent, so cell 0 covers all instances).

- Minimum-image fitted models (the default) have distinct atoms per cluster outright
  (SLCE drops reused-atom clusters in enumeration) → valid for any `dims`.
- `AllImages` (spin-spiral) models may legitimately couple an atom to its own
  periodic image (`atoms = [a, a]`, `shifts = [0, R]`). These tile fine when `dims`
  keeps the images distinct sites, and are **rejected with a clear error** when the
  wrap folds them together (e.g. `dims = (1,1,1)`) — enlarging `dims` is the fix.
  Gate: the hand-built ±x self-image chain (`test_hamiltonian.jl`).

## T3 — memory: templates once + compact CSR instances

`ScaledTerm` templates hold each fitted term's `folded` payload **once** (with the
`(4π)^(body/2)` scale applied there — the package's single application site);
instances are integer CSR lists (`inst_term`, `inst_ptr`/`inst_sites`) plus a
per-site adjacency (`site_ptr`/`site_inst`/`site_slot`, and a `site_has_l1` mask for
the overrelaxation axis). This is the SpinClusterMC lesson: duplicating per-instance
payloads is what blew that package to multi-GB caches; the index-only layout costs
≈ 13 MB for the Nd₂Fe₁₄B l02 case (4692 terms, 4×4×4, 4352 sites, 300k instances).

Rejected alternative — fully on-the-fly instance reconstruction (no instance list):
saves the MB but puts mod-arithmetic and shift resolution inside the innermost ΔE
loop, and is much harder to test in isolation. The CSR adjacency also precomputes
each site's member slot, replacing the `findfirst` of the SLCETools single-cell
kernel.

## T4 — the 4-function energy contract

Everything above the Hamiltonian touches energy only through
`total_energy` / `site_coeffs!` / `delta_energy` / `site_gradient` — the seam a
future kernel optimization (e.g. body-grouped instance batches) must preserve.
`site_gradient` uses `Harmonics.grad_Zlm_unsafe` (on-sphere, tangent-projected;
`e·∇E = 0`) and is diagnostics/tests only.

## T5 — precompiled sparse contraction programs (the hot-kernel form)

The rank-generic contraction — `CartesianIndices` over the rank-erased
`ScaledTerm.folded` behind a rank-specialized function barrier — costs a **dynamic
dispatch and 2–3 heap allocations per instance per visit**, which made `site_coeffs!`
~85–90 % of every sweep on the Nd₂Fe₁₄B bench fixture (bench_log baseline,
2026-07-14). The constructor therefore flattens each template once into
`_ContractionPrograms`: per (template, member slot) a *site program* — the nonzero
`folded` entries as flat arrays of premultiplied weight `coef·folded[idx]`, target
row `lm_index(ls[slot], μ_slot)`, and factor (row, member-slot) pairs — and per
template an *energy program* (every slot a factor, raw `folded[idx]` weights, the
coef applied to the per-instance entry sum). The hot kernels only index plain
`Int32`/`Int8`/`Float64` arrays: no dispatch, no allocation, no zero-entry
scanning, no `lm_index` recomputation.

**Bitwise contract**: the programs are flattened in the reference kernels' exact
loop order — `CartesianIndices` column-major entries, ascending member slots, the
kernels' own zero-skip predicates (`coef·folded == 0` site / `folded == 0` energy),
and the same operation order (`(coef·folded)·p` site / `coef·Σ w·p` energy) — so
the program kernels reproduce the rank-generic reference kernels (kept at the
bottom of `energy.jl` as the readable spec) **bit for bit**. Trajectories, fixed-seed
tests, and checkpoints are unaffected — this is a pure-speed change, not a
P6-breaking one. Gate: `test_energy.jl` "program kernels ≡ reference kernels
(bitwise)" (body 1/2/3, isotropic + anisotropic + self-image shift + sparse
tensors, `==` on `_total_energy` and on `site_coeffs!` for every site).

**Pair/triplet fast paths** (`site_col`(2)/`pent_row`(2), 2026-07-15). A body-2
(body-3) template's site program has exactly one (two) factors per entry and they
always reference the same member slots — the ones other than `v`, ascending — so
the neighbor columns are constant across the program. All remaining indirections
are precomputed: `site_col[j]` holds the hoisted neighbor column per adjacency
entry with the **sign as the path tag** (`> 0` pair, `< 0` −col₁ of a triplet
whose col₂ sits in `site_col2[j]`, `0` general — the sign trick keeps the pair
path from ever loading `site_col2`, which measurably cost ~6 % on pair-only
models), and `pent_row[e]`/`pent_row2[e]` the factor rows per entry. On these
paths `site_coeffs!` walks purely sequential streams plus the `zrows` gathers.
This stays inside the bitwise contract (`(1.0·z₁)·z₂… ≡ z₁·z₂…` in IEEE 754, same
zero-skip, same accumulation order; run-level fingerprints matched HEAD
byte-for-byte) and cut `site_coeffs!` roughly in half on both the pair-heavy and
the triplet-heavy Nd₂Fe₁₄B fixtures (bench_log #5, #6 — the `nbody = 3` fixture
mirrors the production l044/l064/l066 regime, where ~98 % of the walked entries
are body-3). An adjacency *locality sort* (program-id or neighbor-site order) was
measured first and does nothing (≤2 % — the program arrays fit in L2; the cost is
the per-entry indirection chain, not capacity misses). Body ≥ 4 stays on the
general factor loop (no production model needs it; the same hoisting generalizes
if one ever does).

Memory: programs are per *template* (not per instance) — `Σ_terms body·nnz(folded)`
site entries plus `nnz` energy entries, a few MB even for the Nd₂Fe₁₄B case —
consistent with T3's templates-once rule. The fast-path tables add two `Int32` per
adjacency entry (`site_col`/`site_col2` — the same asymptotics as the CSR
adjacency itself) and two per site-program entry (`pent_row`/`pent_row2`). The
templates themselves stay stored (introspection, `_site_energy_scale`, the
checkpoint fingerprint, the reference kernels).

## T6 — channels: a tensor axis is a *slot*, not a site (M4 slice 3b)

A joint spin–lattice term reads two different kinds of site factor —
`Z_{l,m}(ê_a)` on a spin axis and `|u_a|^{2k} R_{l,m}(u_a)` on a displacement one —
and **one site may carry one of each**. So the object indexed by a tensor axis is
not the site: `ScaledTerm` carries one `TermSlot` per axis (`site` = a position in
the term's `atoms`, `row0`/`l` = the row block it gathers from, `spin` = which
channel), replacing the pre-M4 per-site `ls`. Everything T1–T5 says about tiling,
CSR adjacency, memory and the bitwise program contract carries over verbatim with
"member slot" read as "member **site position**"; the two coincide exactly on a
pure-spin term, which is why nothing a pure-spin model produces moved.

**The row numbering is upstream, not local.** Which row an axis reads is
`SLCE.row_layout(model)` — the sampler-row contract of `SLCE.jl/src/slce/rowlayout.jl`
— whose `SPIN` block is `Harmonics.lm_index` at offset 0, followed by one `2l + 1`
block per `(k, l)` displacement factor. Two consequences are load-bearing:

- A pure-spin model's rows are *the same integers* as before the displacement
  channel existed, so `H.lmax`/`H.nlm` (the SPIN block alone) still describe every
  spin-only consumer's scratch, and `H.nrows` is what a joint one allocates.
- The layout must come from the **model**, not from the surviving terms: it covers
  every factor the basis can build, so a consumer's row tables survive a
  coefficient hot swap. The frozen `MultipoleTerm` constructor deliberately keeps
  deriving its layout from the terms (`_spin_row_layout`) — the frozen surface keeps
  the frozen layout, and `TiledHamiltonian(model)` routes a model with no
  displacement rows down it byte for byte.

**The scale is `(4π)^(n_spin_slots/2)`** — one `√(4π)` per *spin slot*, taken from
`DecoratedTerm.scale` and never re-derived from the cluster shape. It agrees with the
pre-M4 `(4π)^(body/2)` exactly when every site holds one spin factor and nothing
else; on a force-constant term (sites with no spin factor at all) the old shortcut
invents a factor from nothing. Still applied exactly once, in the constructor.

**Site programs merge the axes of one site.** `site_coeffs!` builds the coefficient of
each of site `s`'s rows, so the program of a member site position concatenates the
entry lists of *every* axis sitting there, in ascending axis order (one axis ⇒ the
pre-M4 program byte for byte). Consequences:

- `c` is exact for a move that changes **one channel** of the site: the other
  channel's row is then a constant factor already folded into `c`, and the rows that
  did not move contribute `c_k · 0`. A *simultaneous* spin+displacement move on one
  site misses the cross term `Δz·Δr` — the documented boundary, which no update
  crosses. This is the channel analogue of T2's distinct-sites invariant.
- The pair/triplet fast paths hoist columns only when every axis of the site
  produces the *same* ascending column tuple. Two axes on one site drop different
  axes from their factor lists, and in canonical axis order (all `SPIN`, then all
  `DISP`, each by site) a third axis can separate them — so `_hoisted_columns`
  computes the tuples and compares rather than reasoning about the order. A
  disagreement selects the general path, which is always correct.

**At most one axis per `(site, channel)`** (upstream's `SiteDecor` rule) is a ctor
invariant, not an assumption. Two axes of the *same* channel on one site would make even
a single-channel move drop a cross term (`Δz·Δz′`) — and such a term has a pure-spin
layout, so `_require_spin_only` would not catch it: it would enter the sweeps with wrong
Boltzmann weights (measured 21 % ΔE error on a two-spin-axis onsite term). Conversely,
that the invariant holds is what makes the single-channel exactness above *complete*:
the residual of a simultaneous move is exactly `Δzᵀ M Δr` with no `Δz·Δz` or `Δr·Δr`
remainder, which `test_joint.jl` pins by value rather than by non-vanishing.

**Two per-site activity notions**, because on a joint model they differ. A site
referenced only by displacement axes — a force-constant-only ligand, e.g. boron at
`lmax = 0` alongside a displacement sector — is active for the coloring and the sweep
schedule (`site_active`/`n_active`) yet carries no moment at all
(`site_has_spin`/`n_spin_active`). The spin sweeps, `_renormalize!`, the spin
observables and their per-site normalizations read the *spin* predicate; dividing
`m = Σ_s e_s / n` by a count that includes a frozen random direction is exactly the
constant bias the inactive-site convention exists to prevent. The two coincide on any
pure-spin model, so the switch was a bitwise no-op there.

**The checkpoint fingerprint is unchanged for pure spin.** `_fingerprint` mixes each
axis's degree (which for the identity slot layout *is* the pre-M4 `ls`), and mixes the
slot layout itself only when the term is not that identity — so every checkpoint
written before this slice still identifies its Hamiltonian, while a joint model is
still separated. It must ALSO mix `layout.disp_factors`, because `TermSlot.row0` is a
layout-relative block start and `(k, l) → row0` is not injective across layouts: a
`degree = 3:5` sector's `(1,1),(2,1)` blocks begin exactly where a `1:3` sector's
`(0,1),(1,1)` do, so two models differing only by a factor `|u|²` would otherwise share
a fingerprint and disagree on every energy. Gates: `test_joint.jl` against an in-test
copy of the pre-M4 formula, plus that `|u|²` pair.

**Scope.** Slice 3b is the ingest and the energy. The sweeps, the descent, the
observables and the GPU path are still spin-only and *refuse* a joint Hamiltonian
(`_require_spin_only`) rather than sample it at an implied `u = 0`; displacement moves
and pass scheduling are slice 3c.
