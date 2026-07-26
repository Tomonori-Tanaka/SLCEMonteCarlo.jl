# Composable observables (see `docs/specs/binning-observables.md` for the C/χ/U
# conventions — that file is authoritative).
#
# An `Observable` is measured every measurement sweep and accumulated in a
# `LogBinner` (mean/error/τ_int) and a `BinStore` (bin means for jackknife). An
# `Evaluable` is a derived quantity — a function of the means of scalar observables —
# jackknifed over the stored bins. Nothing here knows about Markov chains; the run
# drivers feed `(config, energy, H)` in.

"""
    Observable(name::Symbol, ncomp::Integer, f)

A raw observable measured on every stored sweep: `f(config, energy, H)` returns a
`Real` (`ncomp == 1`) or an `ncomp`-component vector. `energy` is the configuration's
current total SCE energy (model units, `j0` excluded) — so `:energy` costs nothing
extra. Accumulated with autocorrelation-aware errors ([`LogBinner`](@ref)) and bin
means for derived [`Evaluable`](@ref)s.
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
"""
function standard_observables(H::TiledHamiltonian)::Vector{Observable}
    energetic = [Observable(:energy, 1, (cfg, E, H) -> E),
                 Observable(:energy2, 1, (cfg, E, H) -> E^2)]
    H.n_spin_active == 0 && return energetic
    return vcat(energetic,
                [Observable(:m, 3, _mean_spin),
                 Observable(:absm, 1, (cfg, E, H) -> norm(_mean_spin(cfg, E, H))),
                 Observable(:m2, 1, (cfg, E, H) -> sum(abs2, _mean_spin(cfg, E, H))),
                 Observable(:m4, 1, (cfg, E, H) -> sum(abs2, _mean_spin(cfg, E, H))^2),
                 Observable(:sublattice_m, 3 * H.n_cell_atoms, _sublattice_m)])
end

function _mean_spin(config::SpinConfig, E, H::TiledHamiltonian)::SVector{3,Float64}
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

function _sublattice_m(config::SpinConfig, E, H::TiledHamiltonian)::Vector{Float64}
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
"""
function standard_evaluables()::Vector{Evaluable}
    return [_specific_heat_evaluable(),
            Evaluable(:susceptibility, [:m2, :absm],
                      (m, kT, n) -> n * (m.m2 - m.absm^2) / kT),
            Evaluable(:binder, [:m2, :m4], (m, kT, n) -> m.m4 / m.m2^2)]
end

standard_evaluables(H::TiledHamiltonian)::Vector{Evaluable} =
    H.n_spin_active == 0 ? [_specific_heat_evaluable()] : standard_evaluables()

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

function _measure!(acc::ObsAccumulator, config::SpinConfig, energy::Float64,
                   H::TiledHamiltonian)
    v = acc.obs.f(config, energy, H)
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
        keys_tuple = Tuple(ev.inputs)
        # The count the evaluable declared it needs — the two coincide on every
        # pure-spin model and diverge exactly where a joint model's energy carries a
        # lattice contribution that no magnetic site accounts for.
        n = ev.scope === :energy ? n_active : n_spin_active
        f = (ms...) -> ev.f(NamedTuple{keys_tuple}(ms), kT, n)
        est, err = jackknife(f, cols)
        stats[ev.name] = ObservableStat(ev.name, [est], [err], [NaN], nb)
    end
    return stats
end
