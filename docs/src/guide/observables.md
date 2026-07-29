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
| `:energy`, `:energy2` | total SLCE energy (model units, `j0` excluded) and its square |
| `:m` | the magnetization vector `Σₛ eₛ / n_spin_active` over the **spin-active** sites (3 components) |
| `:absm`, `:m2`, `:m4` | `|m|` and its powers |
| `:sublattice_m` | per training-cell atom: the cell-averaged spin vector, flattened (`3·n_cell_atoms` components); inactive sublattices report exactly zero |

and, on a **joint spin–lattice** model only:

| name | what it is |
|---|---|
| `:u2` | the mean square displacement `Σₛ \|uₛ − ūc\|² / n_disp_active` |
| `:u4` | its fourth-moment partner, `\|uₛ − ūc\|⁴` |
| `:sublattice_u2`, `:sublattice_u4` | the same two per training-cell atom (`n_cell_atoms` components). `:sublattice_u2` is the isotropic Debye–Waller input, `B_a = 8π²⟨u²⟩_a/3` |

Spin **directions** only — magnetic-moment magnitudes (μ_B) are not part of the
fitted model; attach them downstream if needed. Sites the spin channel does not
touch — a species with `lmax = 0`, or (on a joint model) a purely
displacement-coupled one — are excluded throughout; mask custom **spin**
observables the same way, via `H.site_has_spin`. `H.site_active` is the wider
predicate (touched by an instance in *either* channel) and is what drives the
sweep schedule and the energy-derived normalizations — using it as a magnetization
mask would average a frozen random direction into `⟨m⟩` on a joint model.

**Two site counts.** Once displacements exist, `n_active` (active in *either*
channel) and `n_spin_active` (magnetic) differ, and per-site normalizations must
use the one that carries the quantity: energy-derived by `n_active`,
magnetization-derived by `n_spin_active`. An `Evaluable` declares which through its
`scope` field. On a model with **no** spin-active site the magnetization observables
are omitted rather than reported as `NaN`.

Derived (`standard_evaluables(H)`):

- `:specific_heat` — per active site, in units of ``k_B``:
  ``C/k_B = (⟨E²⟩ − ⟨E⟩²)/(n_{\mathrm{active}}(k_BT)²)``. On a joint model this is
  the **spin + lattice** heat capacity — the classical harmonic limit alone
  contributes 3/2 per displacement-active site. On a **strained (NPT)** run it is
  configurational-only and is **neither** ``C_V`` **nor the NPT** ``C_P``: the
  β-conjugate state energy there is ``W = E + n_{cells}·j_0(s) + P·V(s)``, whose
  elastic and pressure halves fluctuate with the sampled volume and are not in
  ``E`` (`:energy` omits the varying ``j_0(s)`` for the same reason). Measured
  3.4 % low on the test fixture; the gap is ``C_P − C_V = TVα²B``-sized on real
  solids. On a strained run, append [`npt_observables`](@ref) (both vectors —
  the evaluable needs its raw inputs measured) and read `:enthalpy` /
  `:npt_specific_heat` instead: the raw ``W`` with error bars, and the isobaric
  ``C_P/k_B = (⟨W²⟩ − ⟨W⟩²)/(n_{\mathrm{active}}(k_BT)²)`` — an exact
  derivative ``d⟨W⟩/d(k_BT)`` in the sampled ensemble, since the volume
  Jacobian and the grid-truncated domain are both β-independent. Build it with
  the run's own schedule and pressure; it works under `run_pt` too (pure
  closures, unlike `pressure_diagnostics`). Two caveats: trust the ``C_P``
  reading only while the sampled volume distribution sits well inside
  [`strain_domain`](@ref) (the truncated measure is a volume-constrained
  system), and the cell **shape** is frozen in v0 (hydrostatic-only). No
  momenta are sampled, so the classical kinetic term is absent from every heat
  capacity here — for an absolute value add
  ``(3/2)\,n_{\mathrm{disp}}/n_{\mathrm{active}}`` to the reported
  per-active-site number (``3/2\,k_B`` per *mobile* atom). The run-side story —
  setting the move up, and the pressure check — is in
  [NPT and strain moves](npt.md).
- `:susceptibility` — |m|-connected, per spin-active site:
  ``χ = n_{\mathrm{spin}}(⟨m²⟩ − ⟨|m|⟩²)/k_BT``. On a finite system with
  continuous symmetry ``⟨\boldsymbol m⟩ = 0`` exactly, so the textbook connected
  form degenerates and grows with system size below the transition; this form
  peaks at it (the finite-size-scaling standard).
- `:binder` — ``U = ⟨m⁴⟩/⟨m²⟩²`` (→ 1 ordered, → 5/3 disordered for 3-component
  spins); `U(T)` crossings between system sizes locate ``T_c``.
- `:u_moment_ratio` (joint models) — ``⟨u⁴⟩/⟨u²⟩²``, the anharmonicity screen.
  **Read it against temperature, not against 5/3.** For a harmonic model every
  ``σ_s² ∝ T``, so the ratio is *temperature-independent* whatever the crystal — that
  is the signature. Its level is
  ``(5/3)·\mathrm{mean}(σ⁴)/(\mathrm{mean}\,σ²)² ≥ 5/3`` by Jensen, equal to 5/3
  only when every displacement-active site samples the same isotropic Gaussian; a
  two-sublattice Einstein model with a 4× stiffness contrast measures 2.27 while being
  exactly harmonic. The clean 5/3 test is **per sublattice** (translation-equivalent
  sites share a covariance): `stats[:sublattice_u4].mean[a] /
  stats[:sublattice_u2].mean[a]^2`.

## Composing your own

An observable receives **one** argument, an [`MCView`](@ref) of the sampled state:
`v.config`, `v.disps`, `v.energy`, `v.H`, and — on an NPT run — the cell scale
through [`strain(v)`](@ref strain) / [`has_strain(v)`](@ref has_strain). One
argument rather than a widening positional list, because the state grows with the
model — displacements arrived in M4, the cell scale in M5 — and a positional
contract would break every observable ever written each time it did.

```julia
# a raw observable: f(v::MCView) -> Real or an ncomp-vector
corr12 = Observable(:corr12, 1, v -> dot(v.config[1], v.config[2]))

# a derived quantity: f(means::NamedTuple, kT, n) -> Real, over *scalar* raw
# observables named in `inputs`; `scope` picks which site count `n` is
uovere = Evaluable(:u_over_e, [:m4, :m2], (m, kT, n) -> m.m4 / m.m2^2)

r = run_mc(H; kT = 0.02, observables = vcat(standard_observables(H), corr12),
           evaluables = vcat(standard_evaluables(H), uovere))
```

`v.disps` holds the displacements in whatever frame the last renormalization left,
and is **empty** on a Hamiltonian with no displacement channel — so a displacement
observable run against a pure-spin model throws instead of reporting a confident
zero. It must be a gauge-invariant function of those displacements, which means
centring them itself; see the displacement section below and
`docs/specs/updates-stationarity.md` U7.

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

## Displacement observables

On a joint model the displacements arrive in `v.disps` — in the frame the **last
renormalization** left, which is not the centre-of-mass-free one at an arbitrary
measurement (the sampler re-centres every `renorm_interval` sweeps, measurements
fire every `measure_interval`). Along a component's flat directions the centre of
mass random-walks freely between those points, and it has no stationary
distribution: a custom displacement observable must therefore be a
**gauge-invariant** function of `v.disps`, and the standard set achieves that by
subtracting the component mean *inside the observable*. Do the same:

```julia
# `v.disps` is not pre-centred: subtract each displacement-coupling component's
# centre of mass along its FLAT directions — the same quantity `:u2`/`:u4` remove.
# Skipping this adds the free walk's ⟨|ū|²⟩, which is positive, grows with
# `renorm_interval`, and shrinks with system size (so it reads as a finite-size
# effect rather than as a bug).
function centred(v)
    H, u = v.H, v.disps
    c = copy(u)
    for k = 1:H.n_disp_comps
        lo, hi = Int(H.disp_comp_ptr[k]), Int(H.disp_comp_ptr[k + 1]) - 1
        sites = [Int(H.disp_comp_sites[j]) for j = lo:hi]
        free = SVector{3,Float64}(H.comp_free[1, k], H.comp_free[2, k],
                                  H.comp_free[3, k])
        ū = free .* (sum(u[s] for s in sites) / length(sites))
        for s in sites
            c[s] = u[s] - ū
        end
    end
    return c
end

# per-sublattice anisotropic MSD tensor ⟨u_i u_j⟩ for atom 1 (the full
# Debye-Waller input, of which :sublattice_u2 is the isotropic trace/3)
msd1 = Observable(:msd1, 9, v -> begin
    u = centred(v)
    acc = zeros(3, 3)
    n = 0
    for s in eachindex(u)
        v.H.site_has_disp[s] && SLCEMonteCarlo.site_atom(v.H, s) == 1 || continue
        acc .+= u[s] * u[s]'
        n += 1
    end
    vec(acc ./ max(n, 1))
end)

# a spin-lattice cross-correlator: gauge-invariant in u, rotation-invariant in e
sl = Observable(:sl, 1, v -> begin
    u = centred(v)
    sum(dot(v.config[s], u[s])^2 for s in eachindex(u)) / length(u)
end)
```

(`comp_free` marks each component's flat directions: a `fixed_reference = true`
model's pinned directions carry a physical absolute frame and are left alone.)

For the closed-form checks: an isotropic harmonic well of stiffness `a` gives
`⟨u²⟩ = 3kT/(2a)` and `⟨u⁴⟩ = 15(kT/2a)²`, so the **per-site** ratio is `5/3` at every
temperature. Mind the order of averaging — `:u2` and `:u4` average over sites *before*
the ratio is taken, so on a crystal with inequivalent sites the global ratio sits above
5/3 by Jensen even when the model is exactly harmonic (see `:u_moment_ratio` above).

!!! warning "Screen the harmonic part first"
    A truncated expansion need not be bounded below in `u`, and a chain sampling an
    unbounded well produces displacement numbers that mean nothing.
    [`harmonic_stability`](@ref) reports the displacement Hessian's spectrum at a
    given spin configuration **before** the run; an eigenvalue below `−tol` is a
    proof of failure. A clean spectrum proves nothing (deciding global non-negativity
    of a quartic form is NP-hard) — pair it with `:u_moment_ratio` and
    `TempResult.escaped`. The tolerance is not cosmetic: a translation-invariant model
    has `3·n_disp_comps` exact zero eigenvalues, and finite differences scatter each
    of them across zero.
