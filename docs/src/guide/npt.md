# NPT: sampling the cell volume

```@meta
CurrentModule = SLCEMonteCarlo
```

Passing a [`StrainSchedule`](@ref) turns a run isothermal–isobaric: an outer
Metropolis move rescales the whole cell over a `SLCE.StrainedModels` volume grid,
installing the grid's interpolated coefficients in place
([`set_coefficients!`](@ref)) and rescaling the displacements affinely. Without
it, a run samples the **constant-strain (fixed cell) ensemble** — a different
ensemble (``F(T, ε)``: no volume Jacobian, no ``P\cdot V`` term), which is what a
fixed-geometry magnetostriction calculation wants; the two specific heats differ.

v0 is **hydrostatic**: one linear cell scale ``s``, the cell *shape* frozen. The
move's algebra and the conventions behind it are `docs/specs/strain-move.md`; the
acceptance rule itself is in [theory: the update schemes](../theory/updates.md).

## Setting up a run

```julia
sm  = SLCE.StrainedModels(models, scales)   # fitted at, e.g., s ∈ [0.98, 1.02]
iref = findfirst(≈(1.0), sm.scales)         # any grid point works; the reference reads best
H   = TiledHamiltonian(sm.models[iref]; dims = (8, 8, 8), keep_zero_terms = true)
sch = SLCEMonteCarlo.StrainSchedule(sm, H)

r = run_mc(H; temperature = 300, strain = sch, pressure_GPa = 0.0,
           observables = [standard_observables(H);
                          Observable(:scale, 1, v -> SLCEMonteCarlo.strain(v))])
r.points[1].acceptance_strain     # the outer move's acceptance
```

`keep_zero_terms = true` is required, not cosmetic: the upstream introspection
surface prunes exactly-zero coefficients by default, which makes its index → SALC
map a function of the *fit* rather than of the basis — two grid points whose
sparse fits zero different keys would otherwise produce term lists of equal length
with shifted maps. The schedule constructor checks the pairing term by term and
refuses a mismatched Hamiltonian on the spot.

The pressure follows the `temperature` XOR `kT` discipline: **exactly one** of
`pressure_GPa` (GPa) or `pressure` (model units, eV/Å³), explicitly — `0.0` is a
physical choice, not a default — converted once at resolution and never again
downstream. `strain_interval` (default: one attempt per compound sweep),
`strain_proposal` (`:logvolume` or `:scale`) and `strain_step` (default: a tenth
of the grid's domain; fixed for the run) tune the move; a poor width shows up in
`acceptance_strain`, never in the sampled ensemble.

Inside an NPT run every measurement's [`MCView`](@ref) carries the cell scale
([`strain(v)`](@ref strain), with [`has_strain(v)`](@ref has_strain) false on a
fixed-cell run — a confident `1.0` would let a magnetostriction observable average
a constant), so volume statistics are ordinary observables, as in the `:scale`
example above.

## Reading the thermodynamics: `W`, not `E`

One reading caveat for every strained run: `:energy` and `:specific_heat` are
**configurational-only** — they omit the fluctuating ``n_{cells}\,j_0(s) + P\,V(s)``
half of the NPT state energy, so the reported ``C`` is neither ``C_V`` nor the
NPT ``C_P``. Append [`npt_observables`](@ref) — **both** vectors, built with the
run's own schedule and pressure — and read `:enthalpy` / `:npt_specific_heat`
instead: the β-conjugate ``W = E + n_{cells}\,j_0(s) + P\,V(s)`` itself and the
isobaric heat capacity. Being pure closures, they ride along under `run_pt`'s
per-lane clones too.

```julia
nw = SLCEMonteCarlo.npt_observables(sch, H; pressure_GPa = 1.0)
r  = run_mc(H; temperature = 300, strain = sch, pressure_GPa = 1.0,
            observables = [standard_observables(H); nw.observables],
            evaluables  = [standard_evaluables(H); nw.evaluables])
r.points[1].stats[:npt_specific_heat]   # C_P per active site, units of k_B
```

``C_P/k_B = (⟨W²⟩ − ⟨W⟩²)/(n_{\mathrm{active}}(k_BT)²)`` is exact in the sampled
ensemble — ``\mathrm{var}(W) = −d⟨W⟩/dβ`` holds with no boundary term, because the
volume Jacobian and the grid-truncated domain are both β-independent (the reported
number is that derivative *per active site*). Four things to keep in mind when
reading it:

- it is trustworthy only while the sampled volume distribution sits well inside
  [`strain_domain`](@ref) — the truncated measure is a volume-constrained system;
- v0 samples the isotropic scale only, so this is the isobaric heat capacity of
  the **hydrostatic, fixed-shape** ensemble: the five shear degrees of freedom are
  frozen, which they would not be in a real solid;
- no momenta are sampled, so it is configurational; for an absolute value add
  ``(3/2)\,n_{\mathrm{disp\,active}}/n_{\mathrm{active}}``;
- ``W`` carries the absolute offset ``n_{cells} j_0 + P V``, so the variance
  cancels ``\sim(W/σ_W)^2`` digits — a relative error of ``10^{-5}…10^{-4}`` at
  production scale, growing with `n_cells`. That is inherent to the estimator, not
  a bug to re-reference away: the offset is what makes ``W`` the β-conjugate
  variable in the first place.

## Checking the run: the sampled pressure

[`pressure_diagnostics`](@ref) packages the mechanical-equilibrium identity as
observables: on an equilibrated NPT chain the jackknifed `:pressure` evaluable
(``N_{mob}\,kT\,⟨1/V⟩ − ⟨dE_{tot}/dV⟩``, in eV/Å³) must equal the applied
pressure within its statistical uncertainty (a few error bars — a 1–2σ deviation
is ordinary fluctuation). A persistent disagreement means the elastic ``j_0``
bookkeeping, the virial, the coefficient interpolation or the ``P\cdot V`` term is
wrong. The ``dE/dV`` estimator ([`energy_volume_derivative`](@ref)) is exact, not
a finite difference: the coefficient drift is the schedule's differentiated
interpolant, and the displacement response is Euler's theorem on each factor's
homogeneity degree.

```julia
pd = SLCEMonteCarlo.pressure_diagnostics(sch, H)
r  = run_mc(H; temperature = 300, strain = sch, pressure_GPa = 1.0,
            observables = [standard_observables(H); pd.observables],
            evaluables  = [standard_evaluables(H); pd.evaluables])
r.points[1].stats[:pressure]      # ×160.2176634 (GPA_PER_EV_A3) for GPa
```

Trust it only when the sampled volume distribution sits well inside
[`strain_domain`](@ref) — the identity holds up to boundary terms of the bounded
grid, and a chain pressed against a grid edge is answering a different question.
Unlike `npt_observables`, this one is a **`run_mc`-only** check: its scratch
assumes one serial chain, so under `run_pt` it refuses loudly.

## Parallel tempering under pressure

[`run_pt`](@ref) takes the same NPT keywords: every lane becomes an
isothermal–isobaric chain at the **same** pressure, sweeping its own coefficient
clone of `H` (the lanes sit at different volumes concurrently), and the exchange
rule generalizes to ``(β_i−β_j)(W_i−W_j)`` — the cell scale travels with the
swapped payload, together with the lane's Hamiltonian reference, so the installed
coefficients stay paired with the scale they describe. The volume power
``V^{N_{mob}}`` cancels exactly between the two lanes and never enters the rule.

`strain_init` accepts a scalar (broadcast to every lane) or one value per rung.
A per-rung ladder warm start is only *partly* self-consistent, though: `init` and
`disps` are not per-lane keywords, so at most one rung can be handed the
configuration and displacements that belong with its scale. The self-consistent
per-lane continuation is [`resume`](@ref).

## Warm-starting the next run

A finished strained run warm-starts the next one: `MCResult.final_strain`
(per-lane `PTResult.final_strains`) records the end scale, and

```julia
r2 = run_mc(H; temperature = 350, strain = sch, pressure_GPa = 1.0,
            strain_init = r.final_strain, init = r.final_config,
            disps = r.final_disps, step = r.points[end].final_step,
            step_u = r.points[end].final_step_u)
```

continues at the previous cell without re-thermalizing from the reference. The
`final_disps` are expressed at `final_strain`, so the triple is self-consistent;
forwarding the tuned proposal widths matters exactly when `sweeps_therm` is cut
down, since adaptation is thermalization-only. `final_strain` is `nothing` after a
fixed-cell run — that is the "no confident 1.0" discipline again — and passing
`nothing` back is a no-op: the chain simply starts at the schedule's reference
scale, as it does without the keyword. What selects a fixed cell is omitting
`strain`, nothing else.

A warm start is a **new chain**, not a continuation: bit-identical continuation is
[`resume`](@ref)'s job.

## Checkpointing a strained run

Checkpoints work as usual, with three additions (schema v5,
`docs/specs/checkpoint-schema.md`):

- the chain's cell scale is part of the state, and the file stores the volume
  grid's fingerprint;
- resuming requires the same grid — `resume(path, H; strain = sch)`, checked
  against that fingerprint first; the **model** fingerprint is then checked after
  `resume` has reinstalled the reference-scale coefficients, because that
  fingerprint mixes coefficient *values* and a strained chain has moved them;
- passing a schedule to a fixed-cell checkpoint (or omitting one on a strained
  checkpoint) is an error, and v4 files are refused by name rather than silently
  continued as fixed-cell.

## The GPU path does not move the cell

There is no device strain move: the sweep tier samples spins and displacements at
whatever coefficients the device tables hold. A host-side strain move therefore
has to be followed by [`sync_coefficients!`](@ref) on every live
[`GPUTiledHamiltonian`](@ref) wrapper — a stale device weight table is not
detectable from the device side, since the sweep stays type-correct and
bit-stable against it. See the [GPU guide](gpu.md).
