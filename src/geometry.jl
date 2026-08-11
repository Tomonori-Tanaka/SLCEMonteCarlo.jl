# Geometry helpers at the I/O boundary. The MC core itself is geometry-free
# (integer site topology only); these exist to export sampled configurations with
# real positions. The training `Crystal` is passed in explicitly — this package
# never reads it off the model (`model.basis.crystal` is not public tier).

"""
    supercell_crystal(crystal::Crystal, dims::NTuple{3,Integer}) -> Crystal

The `N₁ × N₂ × N₃` supercell of the training-cell `crystal`, with atoms ordered
**exactly like the sites of a same-`dims` [`TiledHamiltonian`](@ref)** (atom
fastest, cells column-major — `site_index`): supercell atom `s` is training atom
`site_atom(H, s)` in the matching cell. Use it to pair sampled configurations with
positions for DFT input generation (e.g. `SLCETools.VASP`).
"""
function supercell_crystal(crystal::Crystal, dims::NTuple{3,Integer})::Crystal
    all(d -> d >= 1, dims) || throw(ArgumentError("dims must be ≥ 1; got $dims"))
    d = SVector{3,Int}(dims)
    nat = n_atoms(crystal)
    ncells = prod(d)
    frac = Matrix{Float64}(undef, 3, nat * ncells)
    species = Vector{Int}(undef, nat * ncells)
    s = 0
    for c3 = 0:(d[3] - 1), c2 = 0:(d[2] - 1), c1 = 0:(d[1] - 1), a = 1:nat
        s += 1
        for row = 1:3
            frac[row, s] = (crystal.frac_positions[row, a] + (c1, c2, c3)[row]) /
                           d[row]
        end
        species[s] = crystal.species[a]
    end
    vecs = Matrix(crystal.lattice.vectors) * Diagonal(Vector{Float64}(d))
    lat = Lattice(vecs; pbc = Tuple(crystal.lattice.pbc))
    return Crystal(lat, frac, species, copy(crystal.species_labels))
end

"""
    to_matrix(config::SpinConfig) -> Matrix{Float64}

The `3 × n_sites` matrix view (rows x, y, z; columns sites) of a configuration —
the layout of the sibling packages (`SLCE.predict_energy`,
`SLCETools.VASP.write_inputs`).
"""
to_matrix(config::SpinConfig)::Matrix{Float64} = _config_matrix(config)

"""
    from_matrix(m::AbstractMatrix{<:Real}) -> SpinConfig

A [`SpinConfig`](@ref) from a `3 × n` matrix of unit spin directions. Each column
passes the family's unit-direction door (`SLCE.UnitVector3`): it must be finite
and within `1e-6` of unit norm, and is then projected exactly onto the sphere —
float noise is repaired, but a column that is not a direction (a moment-scaled
vector, `‖e‖ = 1.7`) is refused rather than silently normalized. Normalize
deliberately in your own code if that is what you mean.
"""
function from_matrix(m::AbstractMatrix{<:Real})::SpinConfig
    size(m, 1) == 3 || throw(DimensionMismatch(
        "expected a 3 × n matrix; got $(size(m, 1)) × $(size(m, 2))"))
    return SpinConfig([_unit_column(SVector{3,Float64}(m[1, s], m[2, s], m[3, s]), s)
                       for s = 1:size(m, 2)])
end
