# Composable observables (see `docs/specs/binning-observables.md` for the C/χ/U
# conventions — that file is authoritative).
#
# An `Observable` is measured every measurement sweep and accumulated in a
# `LogBinner` (mean/error/τ_int) and a `BinStore` (bin means for jackknife). An
# `Evaluable` is a derived quantity — a function of the means of scalar observables —
# jackknifed over the stored bins. Nothing here knows about Markov chains; the run
# drivers hand in an `MCView` of the sampled state.

"""
    MCView

A read-only view of the sampled state at one measurement, and the single argument
every [`Observable`](@ref) receives:

- `H` — the [`TiledHamiltonian`](@ref), for the site masks (`site_has_spin`,
  `site_has_disp`), the counts, and `site_atom`;
- `config` — the spin configuration;
- `disps` — the Cartesian displacements, model length units, one per site. **Empty**
  on a Hamiltonian with no displacement channel, whatever the producer passed
  (`SLCEDynamics`'s spin dynamics passes an empty vector for the same reason), so an
  observable that indexes `v.disps[s]` throws there instead of reading a zero. Note
  the failure is loud only for an observable that *indexes*: one that masks on
  `H.site_has_disp` or walks `H.disp_comp_ptr` will find nothing to do and return
  zero, which is why the standard set is gated on `n_disp_active > 0` rather than
  measured and discarded. On a joint Hamiltonian the length is validated;
- `energy` — the configuration's current total SLCE energy (model units, `j0`
  excluded), so `:energy` costs an observable nothing extra;
- `strain` — the chain's current linear scale `s = 1 + η` on a K(ε) volume grid, or
  `nothing` on a fixed-cell run. **`nothing` is not `1.0`**: a fixed-cell run has no
  strain degree of freedom at all, and reporting `s = 1` would let a magnetostriction
  observable average a constant and report zero response rather than refusing. Reach it
  through [`strain`](@ref), which throws on a fixed-cell view, or guard with
  [`has_strain`](@ref) — the same discipline the empty `disps` uses.

One argument rather than a widening positional list: the sampled state grows with
the model (spin, then displacements, then whatever a later channel adds), and an
`f(config, energy, H, disps, …)` contract would break every observable ever written
each time it did.

!!! danger "Read-only means *by contract*, not by construction"
    `config` and `disps` are the sampler's **live arrays**, aliased rather than
    copied — a measurement runs once per stored sweep on lattices of ``10^5`` sites,
    and copying them there would dominate it. Nothing prevents an observable from
    writing through them, and one that does (an in-place normalization, a `sort!`, a
    reused scratch buffer) corrupts the chain itself for every subsequent sweep,
    silently and with no drift warning — the incremental energy stays consistent
    with rows that no longer describe the configuration. Treat both as immutable;
    copy first if you need to transform.

!!! warning "Displacements are gauge-relative"
    `disps` is expressed in the centre-of-mass-free frame of each
    displacement-coupling component, which is the *only* frame in which they are a
    stationary quantity (`docs/specs/updates-stationarity.md` U7). An observable must
    therefore be a **gauge-invariant** function of them — a function of `u_s − ū_c`,
    which the view already gives, and never of an absolute position. The uncentred
    displacement is recoverable as `disps[s] + com_removed[c]`, but a quantity built
    from it has no stationary distribution to average over.
"""
struct MCView
    H::TiledHamiltonian
    config::SpinConfig
    disps::Vector{SVector{3,Float64}}
    energy::Float64
    strain::Union{Nothing,Float64}

    # `disps` is emptied on a Hamiltonian with no displacement channel, whatever the
    # producer passed. A pure-spin chain still CARRIES an all-zero displacement vector
    # (it is simpler to keep the field than to branch on it everywhere), and handing
    # that through would make `sum(abs2, v.disps[s])` return a confident 0.0 on a model
    # that does not describe displacements at all — the silent-wrong-answer the
    # single-argument view exists to prevent. Emptied, an indexing observable throws.
    #
    # The POSITIVE direction is validated rather than assumed: a joint Hamiltonian
    # handed a short or empty `disps` would otherwise be read out of bounds by every
    # `@inbounds` accumulator downstream. The check costs one comparison per
    # measurement, not per site.
    function MCView(H::TiledHamiltonian, config::SpinConfig,
                    disps::Vector{SVector{3,Float64}}, energy::Real,
                    strain::Union{Nothing,Real} = nothing)
        length(config) == H.n_sites || throw(DimensionMismatch(
            "MCView config has $(length(config)) sites; the Hamiltonian has " *
            "$(H.n_sites)"))
        st = strain === nothing ? nothing : Float64(strain)
        st === nothing || st > 0 ||
            throw(ArgumentError("MCView strain (a linear scale s = 1 + η) must be " *
                                "positive; got $st"))
        has_disp(H) || return new(H, config, _NO_DISPS, Float64(energy), st)
        length(disps) == H.n_sites || throw(DimensionMismatch(
            "MCView disps has $(length(disps)) entries; this Hamiltonian carries a " *
            "displacement channel and needs one per site ($(H.n_sites))"))
        return new(H, config, disps, Float64(energy), st)
    end
end

# Shared empty displacement list of every pure-spin view (`MCView.disps` is read-only
# by contract, so one instance is enough and the view stays allocation-free).
const _NO_DISPS = SVector{3,Float64}[]

Base.show(io::IO, v::MCView) =
    print(io, "MCView(", length(v.config), " sites",
          isempty(v.disps) ? "" : " + displacements",
          v.strain === nothing ? "" : " + strain", ", E=",
          @sprintf("%.6g", v.energy), ")")

"""
    has_strain(v::MCView) -> Bool

Whether this view comes from a chain with a strain degree of freedom. Guard a
magnetostriction-type observable with it; a fixed-cell chain has no `s` to report and
[`strain`](@ref) throws rather than returning `1.0`.
"""
has_strain(v::MCView)::Bool = v.strain !== nothing

"""
    strain(v::MCView) -> Float64

The chain's current linear scale `s = 1 + η`. **Throws on a fixed-cell view** — the
same reason `disps` is emptied rather than zeroed there: a confident `1.0` would let an
observable average a constant and report "no volume response" for a run that never had
the degree of freedom.

Accessors exist for the fields the sampled state grows by, so that the next channel is
not another struct break (design record §8).
"""
function strain(v::MCView)::Float64
    v.strain === nothing && throw(ArgumentError(
        "this view has no strain: the chain was run at a fixed cell, so there is no " *
        "linear scale to report (and 1.0 would be a fabricated one). Guard with " *
        "`has_strain(v)`, or run with a `StrainSchedule`."))
    return v.strain
end

"""
    Observable(name::Symbol, ncomp::Integer, f)

A raw observable measured on every stored sweep: `f(view::MCView)` returns a `Real`
(`ncomp == 1`) or an `ncomp`-component vector. Accumulated with
autocorrelation-aware errors ([`LogBinner`](@ref)) and bin means for derived
[`Evaluable`](@ref)s.

```julia
corr12 = Observable(:corr12, 1, v -> dot(v.config[1], v.config[2]))
u2     = Observable(:u2, 1, v -> sum(abs2, v.disps[1]))
```
"""
struct Observable
    name::Symbol
    ncomp::Int
    f::Function

    function Observable(name::Symbol, ncomp::Integer, f)
        ncomp >= 1 || throw(ArgumentError("ncomp must be ≥ 1; got $ncomp"))
        return new(name, ncomp, f)
    end
end

"""
    Evaluable(name::Symbol, inputs::Vector{Symbol}, f; scope = :spin)

A derived quantity `f(means::NamedTuple, kT, n) -> Real` of the means of
**scalar** raw observables named in `inputs` (e.g. specific heat from `:energy` and
`:energy2`). Estimated by leave-one-bin-out [`jackknife`](@ref) over the stored bin
means, which propagates the nonlinearity correctly.

`scope` selects **which site count** `n` is, and it is not cosmetic: a per-site
quantity is only intensive when it is divided by the sites that carry it.

- `:spin` (default) — `n = H.n_spin_active`, the magnetic sites. For magnetization
  quantities (`χ`, Binder), where a non-magnetic or displacement-only site
  contributes nothing.
- `:energy` — `n = H.n_active`, every site active in **either** channel. For
  quantities built from the total energy (`C`), which on a joint spin–lattice model
  carries a lattice contribution of ≈ 1.5 `k_B` per displacement-active site.
  Normalizing that by the *spin* count is wrong by the ratio of the two, and on a
  displacement-only model it is a division by zero.

The two counts coincide on every pure-spin model, so the choice only becomes visible
once displacements are in play.
"""
struct Evaluable
    name::Symbol
    inputs::Vector{Symbol}
    f::Function
    scope::Symbol

    function Evaluable(name::Symbol, inputs::Vector{Symbol}, f; scope::Symbol = :spin)
        isempty(inputs) && throw(ArgumentError("an Evaluable needs ≥ 1 input"))
        scope in (:spin, :energy) || throw(ArgumentError(
            "Evaluable scope must be :spin or :energy; got :$scope"))
        return new(name, copy(inputs), f, scope)
    end
end

"""
    ObservableStat

One observable's finalized statistics at one temperature: component-wise `mean`,
standard `err`, and integrated autocorrelation time `tau_int` (`NaN` for jackknifed
evaluables), plus the measurement `count` (raw) or bin count (evaluables).
"""
struct ObservableStat
    name::Symbol
    mean::Vector{Float64}
    err::Vector{Float64}
    tau_int::Vector{Float64}
    count::Int
end

Base.show(io::IO, s::ObservableStat) =
    length(s.mean) == 1 ?
    print(io, "ObservableStat(", s.name, " = ", @sprintf("%.6g ± %.2g", s.mean[1],
          s.err[1]), ")") :
    print(io, "ObservableStat(", s.name, ", ", length(s.mean), " comps)")

"""
    standard_observables(H::TiledHamiltonian) -> Vector{Observable}

The standard set: `:energy`, `:energy2` (total, model units), the magnetization
vector `:m = Σ_s e_s / n_spin_active` (**spin-active sites only** — a non-magnetic
site's frozen direction is not a magnetic moment), `:absm = |m|`, its powers `:m2`,
`:m4`, and the per-sublattice magnetization `:sublattice_m` (training-cell atom `a`'s
cell-averaged vector, flattened `(x₁,y₁,z₁, x₂,…)`, `3·n_cell_atoms` components;
inactive sublattices report exactly zero). Spin directions only — magnetic-moment
magnitudes are not part of the fitted model.

On a model with **no spin-active site at all** (a displacement-only Hamiltonian) the
magnetization observables are omitted rather than measured: every one of them is
`0/0`, and a table of `NaN`s is a worse answer than an absent column.

On a model with at least one **displacement-active** site four displacement
observables join, every one of them evaluated on `u_s − ū_c` — the site's
displacement minus the centre of mass of its displacement-coupling component along
that component's flat directions:

- `:u2` — the mean square displacement `Σ_s |u_s − ū_c|² / n_disp_active` (model
  length units squared). Unlike `TempResult.disp_rms` it is a *binned* observable
  and comes with an autocorrelation-aware error bar;
- `:u4` — the mean **fourth** moment, so the ratio `⟨u⁴⟩/⟨u²⟩²` is available;
- `:sublattice_u2`, `:sublattice_u4` — the same two resolved per training-cell atom
  (`n_cell_atoms` components; sublattices with no displacement axis report exactly
  zero). `:sublattice_u2` is the isotropic Debye–Waller input, `B_a = 8π²⟨u²⟩_a/3`,
  and the per-sublattice **ratio** is the one that carries the clean harmonic
  reference — see [`standard_evaluables`](@ref).

Subtracting `ū_c` inside the observable is what makes them **gauge-invariant**, which
is what makes them stationary at all (`docs/specs/updates-stationarity.md` U7). It is
not a repeat of the sampler's `_recenter!`: that runs at renormalization points while
measurements fire far more often, so at almost every measurement the frame has drifted
since it was last removed. A displacement *vector* mean is deliberately absent:
`⟨u − ū⟩ ≡ 0` identically, so it would measure nothing, and any absolute-frame
quantity has no stationary distribution to average.
"""
function standard_observables(H::TiledHamiltonian)::Vector{Observable}
    obs = [Observable(:energy, 1, v -> v.energy),
           Observable(:energy2, 1, v -> v.energy^2)]
    if H.n_spin_active > 0
        append!(obs,
                [Observable(:m, 3, _mean_spin),
                 Observable(:absm, 1, v -> norm(_mean_spin(v))),
                 Observable(:m2, 1, v -> sum(abs2, _mean_spin(v))),
                 Observable(:m4, 1, v -> sum(abs2, _mean_spin(v))^2),
                 Observable(:sublattice_m, 3 * H.n_cell_atoms, _sublattice_m)])
    end
    # `n_disp_active`, not `has_disp`: the latter is a property of the row LAYOUT, and
    # a joint basis whose displacement SALCs all fitted to zero has displacement rows
    # and no displacement-active site. Measuring there would report the confident zero
    # (and the `:u_moment_ratio` NaN) that the spin branch above deliberately avoids.
    if H.n_disp_active > 0
        append!(obs,
                [Observable(:u2, 1, v -> _mean_u_moment(v, 1)),
                 Observable(:u4, 1, v -> _mean_u_moment(v, 2)),
                 Observable(:sublattice_u2, H.n_cell_atoms,
                            v -> _sublattice_u_moment(v, 1)),
                 Observable(:sublattice_u4, H.n_cell_atoms,
                            v -> _sublattice_u_moment(v, 2))])
    end
    return obs
end

# Mean of `|u_s − ū_c|^{2p}` over the displacement-active sites, `ū_c` being the
# centre of mass of the site's own displacement-coupling component along that
# component's FLAT directions.
#
# THE SUBTRACTION IS THE OBSERVABLE, not a repeat of `_recenter!`'s work. The sampler
# re-centres at renormalization points (default: every 1000 sweeps) while measurements
# fire every `measure_interval` (default: 1), so at almost every measurement the frame
# has drifted since it was last removed, and `mean_s|u_s|² = mean_s|u_s − ū|² + |ū|²`
# picks up the free random walk `_recenter!` exists to delete. Measured on a
# translation-invariant fixture the excess `⟨|ū|²⟩` is exactly linear in
# `renorm_interval` (0.0041 / 0.081 / 0.400 at 10 / 200 / 1000), i.e. ≈
# `renorm_interval / (12·n_sites_per_component)` relative — over 100 % at the defaults
# on a small cell, always positive, and shrinking with system size, so it would read as
# a finite-size effect rather than as a bug. Subtracting here is what makes
# "gauge-invariant" a property of the observable instead of a property of when it
# happens to be called; it also restores `⟨u2⟩ ≡ TempResult.disp_rms²`, since the
# diagnostic is recorded immediately after `_recenter!`.
#
# Only the FLAT directions are removed (`H.comp_free`), for the same reason
# `_recenter!` projects: a pinned direction's absolute frame is physical, and
# subtracting its mean would delete a real displacement (a substrate-clamped slab's
# standoff from the ideal lattice).
function _mean_u_moment(v::MCView, p::Int)::Float64
    H, u = v.H, v.disps
    q = 0.0
    n = 0
    @inbounds for c = 1:H.n_disp_comps
        lo = Int(H.disp_comp_ptr[c])
        hi = Int(H.disp_comp_ptr[c + 1]) - 1
        ū = _component_mean(H, u, c, lo, hi)
        for k = lo:hi
            x = u[Int(H.disp_comp_sites[k])] - ū
            q += dot(x, x)^p
            n += 1
        end
    end
    return n == 0 ? 0.0 : q / n
end

# The centre of mass of component `c`, projected onto its flat directions — the same
# quantity and the same projection `_recenter!` removes, so the two cannot drift apart
# in what they call "the frame".
@inline function _component_mean(H::TiledHamiltonian, u::Vector{SVector{3,Float64}},
                                 c::Int, lo::Int, hi::Int)::SVector{3,Float64}
    any(view(H.comp_free, :, c)) || return zero(SVector{3,Float64})
    acc = zero(SVector{3,Float64})
    @inbounds for k = lo:hi
        acc += u[Int(H.disp_comp_sites[k])]
    end
    return (acc / (hi - lo + 1)) .*
           SVector{3,Float64}(H.comp_free[1, c], H.comp_free[2, c], H.comp_free[3, c])
end

# Per training-cell atom mean of `|u_s − ū_c|^{2p}` — `p = 1` is the isotropic
# Debye–Waller input, `p = 2` its fourth-moment partner. A sublattice with no
# displacement axis reports exactly zero (the same convention `_sublattice_m` uses for
# a non-magnetic one), never a frozen `u = 0` averaged in as if it were sampled.
#
# Resolved per sublattice, the ratio `⟨u⁴⟩_a/⟨u²⟩_a²` IS the 5/3 harmonic test: all
# sites of one training-cell atom are translation-equivalent, so they share a
# covariance, and the Jensen inequality that spoils the global ratio (see
# `standard_evaluables`) collapses to equality.
function _sublattice_u_moment(v::MCView, p::Int)::Vector{Float64}
    H, u = v.H, v.disps
    out = zeros(H.n_cell_atoms)
    cnt = zeros(Int, H.n_cell_atoms)
    @inbounds for c = 1:H.n_disp_comps
        lo = Int(H.disp_comp_ptr[c])
        hi = Int(H.disp_comp_ptr[c + 1]) - 1
        ū = _component_mean(H, u, c, lo, hi)
        for k = lo:hi
            s = Int(H.disp_comp_sites[k])
            a = site_atom(H, s)
            x = u[s] - ū
            out[a] += dot(x, x)^p
            cnt[a] += 1
        end
    end
    @inbounds for a = 1:H.n_cell_atoms
        cnt[a] > 0 && (out[a] /= cnt[a])
    end
    return out
end

function _mean_spin(v::MCView)::SVector{3,Float64}
    config, H = v.config, v.H
    # All-active fast path: pairwise `sum` — byte-identical to the pre-inactive-site
    # convention (a sequential loop differs by ULPs on large lattices) and slightly
    # more accurate than sequential accumulation.
    H.n_spin_active == H.n_sites && return sum(config) / H.n_spin_active
    m = zero(SVector{3,Float64})
    @inbounds for s in eachindex(config)
        H.site_has_spin[s] && (m += config[s])
    end
    return m / H.n_spin_active
end

function _sublattice_m(v::MCView)::Vector{Float64}
    config, H = v.config, v.H
    out = zeros(3, H.n_cell_atoms)
    for s in eachindex(config)
        H.site_has_spin[s] || continue   # non-magnetic sublattices stay exactly zero
        a = mod1(s, H.n_cell_atoms)
        e = config[s]
        out[1, a] += e[1]
        out[2, a] += e[2]
        out[3, a] += e[3]
    end
    ncells = length(config) ÷ H.n_cell_atoms
    return vec(out) ./ ncells
end

"""
    standard_evaluables() -> Vector{Evaluable}
    standard_evaluables(H::TiledHamiltonian) -> Vector{Evaluable}

The standard derived quantities (conventions: `docs/specs/binning-observables.md`).
The `H` form drops the magnetization-derived ones on a model with no spin-active
site, matching what [`standard_observables`](@ref) measures there; it is what the run
drivers default to.

- `:specific_heat` — per active site, in units of ``k_B``:
  ``C/k_B = (⟨E²⟩ − ⟨E⟩²) / (n_{active}\\, (k_BT)²)`` (intensive — comparable across
  supercell sizes). `scope = :energy`: on a joint spin–lattice model the variance
  includes the lattice heat capacity, so the normalization is by every active site,
  not by the magnetic ones.
- `:susceptibility` — |m|-connected, per active site:
  ``χ = n_{active} (⟨m²⟩ − ⟨|m|⟩²) / k_BT``. On a finite system with continuous
  symmetry ``⟨\\boldsymbol m⟩ = 0`` exactly, so the naive connected form degenerates
  and grows ∝ N below the transition; the |m|-connected form peaks at it (the
  finite-size-scaling standard).
- `:binder` — the plain ratio ``U = ⟨m⁴⟩/⟨m²⟩²`` (→ 1 ordered, → 5/3 disordered for
  3-component spins; U(T) crossings locate ``T_c``).
- `:u_moment_ratio` (models with a displacement-active site) — ``⟨u⁴⟩/⟨u²⟩²``, the
  anharmonicity screen. **Read it as a function of temperature, not against 5/3.**

  For a harmonic model every site's covariance scales as ``σ_s² ∝ T``, so the ratio
  is **temperature-independent** whatever the crystal — that is the signature, and it
  needs no reference value. Its *level* does: because `:u2` and `:u4` are each a mean
  over sites before the ratio is taken, the harmonic value is

  ```
  ⟨u⁴⟩/⟨u²⟩² = (5/3) · mean_s(σ_s⁴) / (mean_s σ_s²)²  ≥  5/3
  ```

  by Jensen, with equality **only** when every displacement-active site samples the
  same isotropic Gaussian. A two-sublattice Einstein model with stiffnesses `(2.5,
  10)` — mild next to a real magnet — measures 2.268 while being exactly harmonic.
  Site anisotropy shifts it too: a general harmonic site gives
  ``1 + 2\\,\\mathrm{tr}(Σ²)/(\\mathrm{tr}\\,Σ)²``, which is 5/3 only for cubic site
  symmetry.

  The clean 5/3 test is **per sublattice**: all sites of one training-cell atom are
  translation-equivalent, so they share a covariance and Jensen collapses to equality.
  Build it from the raw observables — `stats[:sublattice_u4].mean[a] /
  stats[:sublattice_u2].mean[a]^2` — which is why both are measured.
"""
function standard_evaluables()::Vector{Evaluable}
    return [_specific_heat_evaluable(),
            Evaluable(:susceptibility, [:m2, :absm],
                      (m, kT, n) -> n * (m.m2 - m.absm^2) / kT),
            Evaluable(:binder, [:m2, :m4], (m, kT, n) -> m.m4 / m.m2^2)]
end

function standard_evaluables(H::TiledHamiltonian)::Vector{Evaluable}
    evals = H.n_spin_active == 0 ? [_specific_heat_evaluable()] :
            standard_evaluables()
    H.n_disp_active > 0 && push!(evals, Evaluable(:u_moment_ratio, [:u2, :u4],
                                                  (m, kT, n) -> m.u4 / m.u2^2))
    return evals
end

_specific_heat_evaluable() =
    Evaluable(:specific_heat, [:energy, :energy2],
              (m, kT, n) -> (m.energy2 - m.energy^2) / (n * kT^2); scope = :energy)

# --- accumulation ------------------------------------------------------------------

# One observable's accumulators for one temperature / lane.
struct ObsAccumulator
    obs::Observable
    binner::LogBinner
    store::BinStore
    val::Vector{Float64}    # scratch: the current measurement
end

# `planned` = number of measurements this accumulator will receive; the bin size is
# fixed up front so every bin is equal-weight (a trailing remainder is dropped).
function ObsAccumulator(obs::Observable, planned::Integer, nbins::Integer)
    bin_size = max(1, fld(planned, nbins))
    return ObsAccumulator(obs, LogBinner(obs.ncomp),
                          BinStore(obs.ncomp, bin_size, nbins), zeros(obs.ncomp))
end

function _measure!(acc::ObsAccumulator, view::MCView)
    v = acc.obs.f(view)
    if v isa Real
        acc.obs.ncomp == 1 || throw(DimensionMismatch(
            "observable $(acc.obs.name) returned a scalar but declares " *
            "$(acc.obs.ncomp) components"))
        acc.val[1] = Float64(v)
    else
        length(v) == acc.obs.ncomp || throw(DimensionMismatch(
            "observable $(acc.obs.name) returned $(length(v)) components but " *
            "declares $(acc.obs.ncomp)"))
        copyto!(acc.val, v)
    end
    push!(acc.binner, acc.val)
    push!(acc.store, acc.val)
    return acc
end

# Finalize one temperature: raw stats from the binners, evaluables jackknifed over
# the stored bins of their (scalar) inputs.
function _finalize_stats(accs::Vector{ObsAccumulator}, evals::Vector{Evaluable},
                         kT::Float64, n_spin_active::Int,
                         n_active::Int)::Dict{Symbol,ObservableStat}
    stats = Dict{Symbol,ObservableStat}()
    byname = Dict(acc.obs.name => acc for acc in accs)
    for acc in accs
        stats[acc.obs.name] = ObservableStat(acc.obs.name, mean(acc.binner),
                                             std_error(acc.binner),
                                             tau_int(acc.binner), acc.binner.n)
    end
    for ev in evals
        cols = Vector{Vector{Float64}}(undef, length(ev.inputs))
        nb = typemax(Int)
        ok = true
        for (q, name) in enumerate(ev.inputs)
            acc = get(byname, name, nothing)
            acc === nothing && throw(ArgumentError(
                "evaluable $(ev.name) needs observable :$name, which is not measured"))
            acc.obs.ncomp == 1 || throw(ArgumentError(
                "evaluable $(ev.name) input :$name is not a scalar observable"))
            cols[q] = vec(bin_means(acc.store))
            nb = min(nb, length(cols[q]))
            ok &= nb >= 2
        end
        if !ok
            stats[ev.name] = ObservableStat(ev.name, [NaN], [NaN], [NaN], 0)
            continue
        end
        # One temperature's accumulators are built alike and receive the same
        # `_measure!` calls, so the input columns are equal-length by construction —
        # and `jackknife` requires it. Assert rather than truncate to `nb`: a ragged
        # set here means a new accumulation path broke that lockstep, and silent
        # truncation would hide it behind a slightly-wrong error bar.
        all(length(c) == nb for c in cols) || throw(ArgumentError(
            "evaluable $(ev.name): input bin columns have unequal lengths " *
            "$(map(length, cols)) — the accumulators' lockstep is broken"))
        keys_tuple = Tuple(ev.inputs)
        # The count the evaluable declared it needs — the two coincide on every
        # pure-spin model and diverge exactly where a joint model's energy carries a
        # lattice contribution that no magnetic site accounts for.
        n = ev.scope === :energy ? n_active : n_spin_active
        f = (ms...) -> ev.f(NamedTuple{keys_tuple}(ms), kT, n)
        estimator, err = jackknife(f, cols)
        stats[ev.name] = ObservableStat(ev.name, [estimator], [err], [NaN], nb)
    end
    return stats
end
