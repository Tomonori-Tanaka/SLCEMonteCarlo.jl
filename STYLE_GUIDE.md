# Style guide

> The shared Julia style lives in `~/Packages/CLAUDE.md` ("Julia style"):
> official Julia + DFTK guides, naming, `for i = 1:n` vs `for x in xs`,
> explicit named tuples, ≤ 92 cols, hot-path `SVector`/`MVector`/`@views`/
> `@inbounds`. List only **package-specific additions or deviations** here.

## Argument order

Energy/update kernels take `(output-or-state, hamiltonian, site, …)` — the mutated
argument first (Julia convention), the `TiledHamiltonian` second.

## Package-specific naming

- `H` is the conventional local name for a `TiledHamiltonian`; `st` for a
  `ChainState`; `sc` for a `SweepScratch`.
- "site" always means a supercell site index in `1:n_sites(H)`; "atom" always means
  a training-cell atom index in `1:H.n_cell_atoms`. Never mix the two words.
- Internal helpers carry a leading underscore; the public-but-unexported tier
  (`SLCEMonteCarlo.site_coeffs!`, …) does not.
