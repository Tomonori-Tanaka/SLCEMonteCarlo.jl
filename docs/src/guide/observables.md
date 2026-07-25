# Observables and evaluables

```@meta
CurrentModule = SLCEMonteCarlo
```

Nothing is hard-coded into the sweep: a run measures a vector of
[`Observable`](@ref)s (raw, binned with autocorrelation-aware errors) and derives
a vector of [`Evaluable`](@ref)s (nonlinear functions of the raw means,
jackknifed over stored bins). The conventions below are stated authoritatively in
`docs/specs/binning-observables.md`.

## The standard set

| name | what it is |
|---|---|
| `:energy`, `:energy2` | total SCE energy (model units, `j0` excluded) and its square |
| `:m` | the magnetization vector `Σₛ eₛ / n_active` over the **active** sites (3 components) |
| `:absm`, `:m2`, `:m4` | `|m|` and its powers |
| `:sublattice_m` | per training-cell atom: the cell-averaged spin vector, flattened (`3·n_cell_atoms` components); inactive sublattices report exactly zero |

Spin **directions** only — magnetic-moment magnitudes (μ_B) are not part of the
fitted model; attach them downstream if needed. Inactive (non-magnetic) sites — no
cluster instance touches them, e.g. a species with `lmax = 0` — are excluded
throughout and per-site normalizations use `n_active` (see
[`TiledHamiltonian`](@ref)); mask custom observables the same way via
`H.site_active`.

Derived (`standard_evaluables()`):

- `:specific_heat` — per site, in units of ``k_B``:
  ``C/k_B = (⟨E²⟩ − ⟨E⟩²)/(n_{\mathrm{sites}}(k_BT)²)``.
- `:susceptibility` — |m|-connected, per site:
  ``χ = n_{\mathrm{sites}}(⟨m²⟩ − ⟨|m|⟩²)/k_BT``. On a finite system with
  continuous symmetry ``⟨\boldsymbol m⟩ = 0`` exactly, so the textbook connected
  form degenerates and grows with system size below the transition; this form
  peaks at it (the finite-size-scaling standard).
- `:binder` — ``U = ⟨m⁴⟩/⟨m²⟩²`` (→ 1 ordered, → 5/3 disordered for 3-component
  spins); `U(T)` crossings between system sizes locate ``T_c``.

## Composing your own

```julia
# a raw observable: f(config, energy, H) -> Real or an ncomp-vector
corr12 = Observable(:corr12, 1, (cfg, E, H) -> dot(cfg[1], cfg[2]))

# a derived quantity: f(means::NamedTuple, kT, n_active) -> Real,
# over *scalar* raw observables named in `inputs`
uovere = Evaluable(:u_over_e, [:m4, :m2], (m, kT, n) -> m.m4 / m.m2^2)

r = run_mc(H; kT = 0.02, observables = vcat(standard_observables(H), corr12),
           evaluables = vcat(standard_evaluables(), uovere))
```

## A ferrimagnet order parameter

For an exchange-only (rotation-invariant) model the *absolute* ordering axis is
arbitrary — compare sublattices through rotation-invariant projections instead of
raw components:

```julia
sub = r.points[1].stats[:sublattice_m].mean
subv = [SVector(sub[3a - 2], sub[3a - 1], sub[3a]) for a = 1:H.n_cell_atoms]
axis = normalize(sum(subv[a] for a in fe_atoms))          # the majority axis
projs = [dot(subv[a], axis) for a = 1:H.n_cell_atoms]     # ferri: signs differ
```

(In the Nd₂Fe₁₄B smoke test this gives Nd ≈ −0.5 vs Fe ≈ +0.7 at 250 K.)
