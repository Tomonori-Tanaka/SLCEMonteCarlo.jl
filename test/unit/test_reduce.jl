# Cell reduction (`reduce_cell`): exact recovery of hand-built small-cell terms from
# their unfolded supercell form (diagonal and non-diagonal M), energy identities via
# the site permutation, fitted-model paths, and the verification error cases.

# Unfold small-cell terms onto a diagonal N₁×N₂×N₃ supercell: the training-cell term
# list a model fitted on that supercell would expose (atom ordering matches
# `supercell_crystal` / `site_index` — atom fastest, cells column-major).
function _unfold_diag(sub_terms::Vector{T}, nsub_atoms, dims::NTuple{3,Int}) where {T}
    d = SVector{3,Int}(dims)
    out = T[]
    for c3 = 0:(d[3] - 1), c2 = 0:(d[2] - 1), c1 = 0:(d[1] - 1)
        t = SVector(c1, c2, c3)
        for mt in sub_terms
            atoms = Int[]
            shifts = SVector{3,Int}[]
            for (a, sh) in zip(mt.atoms, mt.shifts)
                σ = t + sh
                cw = mod.(σ, d)
                push!(atoms, a + nsub_atoms * (cw[1] + d[1] * (cw[2] + d[2] * cw[3])))
                push!(shifts, fld.(σ, d))
            end
            push!(out, _resite(mt, atoms, shifts))
        end
    end
    return out
end

# The same term on new member sites — member ORDER is preserved, so a decorated term's
# slot → member-position map carries over untouched.
_resite(mt::SpinMultipoleTerm, atoms, shifts) =
    SpinMultipoleTerm(mt.coef, length(atoms), atoms, shifts, copy(mt.ls), copy(mt.folded))
_resite(dt::DecoratedTerm, atoms, shifts) =
    DecoratedTerm(dt.coef, dt.scale, length(atoms), atoms, shifts, copy(dt.slots),
                  copy(dt.folded))

# For a *diagonal* reduction matrix M: the permutation taking training-tiled site s
# (of H_tr, dims D) to the equivalent reduced-tiled site (of H_red, dims |M|·D; the
# wrap makes negative-diagonal — left-handed — M work too).
function _reduce_perm(red, H_tr, H_red)
    md = SVector(red.M[1, 1], red.M[2, 2], red.M[3, 3])
    perm = zeros(Int, H_tr.n_sites)
    for s = 1:H_tr.n_sites
        a = MC.site_atom(H_tr, s)
        idx = (s - a) ÷ H_tr.n_cell_atoms
        c = SVector(idx % H_tr.dims[1], (idx ÷ H_tr.dims[1]) % H_tr.dims[2],
                    idx ÷ (H_tr.dims[1] * H_tr.dims[2]))
        b, o = red.atom_map[a]
        perm[s] = MC.site_index(H_red, b, mod.(o + md .* c, H_red.dims))
    end
    return perm
end

# General M: which training atom's coset the reduced cell `c` belongs to
# (c ≡ o_a mod M·ℤ³, decided exactly with the integer adjugate).
function _coset_atom(red, c::SVector{3,Int})
    m = red.M
    dt = MC._det3(m)
    adj = SMatrix{3,3,Int}(m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2],
                           m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3],
                           m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1],
                           m[1, 3] * m[3, 2] - m[1, 2] * m[3, 3],
                           m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1],
                           m[1, 2] * m[3, 1] - m[1, 1] * m[3, 2],
                           m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2],
                           m[1, 3] * m[2, 1] - m[1, 1] * m[2, 3],
                           m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
    for (a, (_, o)) in enumerate(red.atom_map)
        v = adj * (c - o)
        all(x -> x % dt == 0, v) && return a
    end
    error("reduced cell $c matches no coset of M = $(red.M)")
end

_permute_config(cfg, perm) = begin
    out = MC.SpinConfig(undef, length(cfg))
    for s = 1:length(cfg)
        out[perm[s]] = cfg[s]
    end
    out
end

_permute_disps(u, perm) = begin
    out = Vector{SVector{3,Float64}}(undef, length(u))
    for s = 1:length(u)
        out[perm[s]] = u[s]
    end
    out
end

# The `(4π)^(n_spin_slots/2)` rule, re-derived here from the term's OWN slot list —
# never from the cluster shape, and never from the field being checked.
_slot_scale(dt::DecoratedTerm) =
    (4π)^(count(s -> s.factor.channel == SLCE.SPIN, dt.slots) / 2)

# SLCE's member canonicalization, mirrored: sort members by `(atom, shift)`, relabel
# every slot's site through `invperm`, re-sort the slots into `_slotkey` order, carry
# `folded`'s axes along. `_unfold_diag` preserves member order, so a term list built
# only from it always presents an ascending reduced site list and the reduction's
# permutation machinery never fires — a fitted model's terms do NOT look like that.
# This mirrors the code under test, so it is validated independently below: re-ordering
# a term's members together with its tensor axes must leave `total_energy` alone.
function _canonicalize(dt::DecoratedTerm)
    p = sortperm(1:length(dt.atoms); by = i -> (dt.atoms[i], Tuple(dt.shifts[i])))
    back = invperm(p)
    moved = [SLCE.SlotRef(back[s.site], s.factor) for s in dt.slots]
    q = sortperm(1:length(moved);
                 by = v -> (moved[v].factor.channel, moved[v].site,
                            moved[v].factor.k, moved[v].factor.l, v))
    # re-anchor at the new member 1 (`shifts[1] == 0` is the ingest contract): a lattice
    # translation of the whole cluster, which leaves the summed energy alone
    sh = dt.shifts[p]
    return DecoratedTerm(dt.coef, dt.scale, dt.body, dt.atoms[p], [s - sh[1] for s in sh],
                         moved[q], permutedims(dt.folded, q))
end

@testset "cell reduction" begin
    sub_lat = Matrix(1.0 * I(3))
    sub_cr = Crystal(Lattice(sub_lat), reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
    sub_terms = _chain_terms(-0.03)

    @testset "diagonal supercell: exact recovery and energy identity" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3}(Diagonal([2, 2, 1]))
        @test red.parent_atoms == [1]
        # the ±x directed pair aligns onto one canonical key → two copies of the
        # (0, +x)-form representative
        @test length(red.terms) == 2
        for rt in red.terms
            st = sub_terms[1]
            @test rt.coef == st.coef
            @test rt.atoms == st.atoms
            @test rt.shifts == st.shifts
            @test rt.ls == st.ls
            @test rt.folded == st.folded
        end
        @test cartesian_positions(red.crystal) == zeros(3, 1)

        H_tr = MC.TiledHamiltonian(4, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        rng = MersenneTwister(11)
        cfg = _rand_config(rng, H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        @test total_energy(H_red, cfg_red) ≈ total_energy(H_tr, cfg) atol = 1e-13
        # equal to tiling the original small-cell terms directly (the reduced list
        # regroups the ±x pair, so equality is to summation order)
        H_sub = MC.TiledHamiltonian(1, sub_terms; dims = (2, 2, 1))
        @test total_energy(H_sub, cfg_red) ≈ total_energy(H_red, cfg_red) atol = 1e-13

        # non-training-multiple sizes are the point: 3×2×2 of the 1-atom cell
        H_tr2 = MC.TiledHamiltonian(4, tr_terms; dims = (3, 1, 2))
        H_red2 = TiledHamiltonian(red; dims = (6, 2, 2))
        cfg2 = _rand_config(rng, H_tr2)
        cfg_red2 = _permute_config(cfg2, _reduce_perm(red, H_tr2, H_red2))
        @test total_energy(H_red2, cfg_red2) ≈ total_energy(H_tr2, cfg2) atol = 1e-12
        H_odd = TiledHamiltonian(red; dims = (3, 3, 1))   # not a multiple of (2,2,1)
        @test MC.n_sites(H_odd) == 9
    end

    @testset "non-diagonal M (det 2): exact recovery" begin
        # A_train = A_sub · M with columns v₁ = (1,-1,0), v₂ = (1,1,0), v₃ = ẑ;
        # the two cosets sit at cart (0,0,0) and (1,0,0). Hand-unfolded ±x chain.
        mB = [1 1 0; -1 1 0; 0 0 1]
        tr_cr = Crystal(Lattice(Float64.(mB)), [0 0.5; 0 0.5; 0.0 0.0], [1, 1],
                        ["Fe"])
        raw = sub_terms[1].coef
        folded = sub_terms[1].folded
        z = SVector(0, 0, 0)
        tr_terms = [SpinMultipoleTerm(raw, 2, [1, 2], [z, z], [1, 1], copy(folded)),
                    SpinMultipoleTerm(raw, 2, [2, 1], [z, SVector(1, 1, 0)], [1, 1],
                                  copy(folded)),
                    SpinMultipoleTerm(raw, 2, [1, 2], [z, SVector(-1, -1, 0)], [1, 1],
                                  copy(folded)),
                    SpinMultipoleTerm(raw, 2, [2, 1], [z, z], [1, 1], copy(folded))]
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3,Int}(mB)
        @test length(red.terms) == 2
        for rt in red.terms                     # two copies of the +x-form rep
            st = sub_terms[1]
            @test rt.coef == st.coef
            @test rt.atoms == st.atoms
            @test rt.shifts == st.shifts
            @test rt.folded == st.folded
        end
        # per-site energy identity on a sub-periodic (here: uniform) configuration
        e = _rand_spin(MersenneTwister(3))
        H_tr = MC.TiledHamiltonian(2, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (3, 1, 1))
        E_tr = total_energy(H_tr, MC.SpinConfig([e, e]))
        E_red = total_energy(H_red, MC.SpinConfig([e, e, e]))
        @test E_red / 3 ≈ E_tr / 2 atol = 1e-14
    end

    @testset "fitted model, identity reduction (|det M| = 1)" begin
        model = _dimer_model()
        cr = _dimer_crystal()
        red = reduce_cell(model, cr, Matrix(cr.lattice.vectors))
        @test n_atoms(red) == 4
        @test length(red.terms) == length(spin_multipole_terms(model))
        H_a = TiledHamiltonian(model; dims = (2, 1, 2))
        H_b = TiledHamiltonian(red; dims = (2, 1, 2))
        cfg = _rand_config(MersenneTwister(5), H_a)
        @test total_energy(H_b, cfg) ≈ total_energy(H_a, cfg) atol = 1e-13
    end

    @testset "fitted model, genuine 2× reduction + predict_energy gate" begin
        model, cr = _stacked_chain_model()
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test n_atoms(red) == 1
        @test length(red.terms) == length(spin_multipole_terms(model)) ÷ 2
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (1, 1, 2))
        rng = MersenneTwister(17)
        cfg = _rand_config(rng, H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        E = total_energy(H_red, cfg_red)
        @test E ≈ total_energy(H_tr, cfg) atol = 1e-13
        @test E ≈ predict_energy(model, _config_matrix(cfg)) - intercept(model) atol =
            1e-12
        # a larger, non-commensurate tiling stays consistent with a training tiling
        H_tr2 = TiledHamiltonian(model; dims = (2, 2, 1))
        H_red2 = TiledHamiltonian(red; dims = (2, 2, 2))
        cfg2 = _rand_config(rng, H_tr2)
        cfg_red2 = _permute_config(cfg2, _reduce_perm(red, H_tr2, H_red2))
        @test total_energy(H_red2, cfg_red2) ≈ total_energy(H_tr2, cfg2) atol = 1e-12
    end

    @testset "two SALC channels on one cluster stay distinct" begin
        # same pair cluster, second channel with a different tensor and coefficient:
        # the (coef, folded) sub-partition must keep both, each with nc copies.
        ch1 = _chain_terms(-0.03)
        ising = zeros(3, 3)
        ising[3, 3] = 1.0                       # a second, distinct coupling tensor
        ch2 = [SpinMultipoleTerm(0.007, 2, copy(t.atoms), copy(t.shifts), copy(t.ls),
                             copy(ising)) for t in ch1]
        sub2 = vcat(ch1, ch2)
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub2, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, sub_lat)
        @test length(red.terms) == 4
        # one canonical key; reps in encounter order, each emitted twice (the ±x
        # directed pair folds onto the +x form): [ch1, ch1, ch2, ch2]
        expected = [sub2[1], sub2[1], sub2[3], sub2[3]]
        for (rt, st) in zip(red.terms, expected)
            @test rt.coef == st.coef
            @test rt.shifts == st.shifts
            @test rt.folded == st.folded
        end
        H_sub2 = MC.TiledHamiltonian(1, sub2; dims = (2, 2, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        cfg = _rand_config(MersenneTwister(19), H_red)
        @test total_energy(H_red, cfg) ≈ total_energy(H_sub2, cfg) atol = 1e-13
    end

    @testset "fitted anisotropic model: channels survive a 2× reduction" begin
        model, cr = _stacked_anisotropic_model(SpglibBackend())
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test n_atoms(red) == 1
        @test length(red.terms) == length(spin_multipole_terms(model)) ÷ 2
        # the sub-partition branch is genuinely exercised: some reduced cluster
        # carries several SALC channels (same anchored key, different folded)
        keys = [(t.atoms, t.shifts, t.ls) for t in red.terms]
        @test length(unique(keys)) < length(keys)
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (1, 1, 2))
        cfg = _rand_config(MersenneTwister(23), H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        E = total_energy(H_red, cfg_red)
        @test E ≈ total_energy(H_tr, cfg) atol = 1e-12
        @test E ≈ predict_energy(model, _config_matrix(cfg)) - intercept(model) atol =
            1e-12

        # NoSymmetry per-bond orbits do NOT align their SALC tensor bases across
        # translation partners, so even equal-fill coefficients genuinely break the
        # half-cell periodicity — reduce_cell must refuse (a physics refusal, not a
        # tolerance artifact).
        model_ns, cr_ns = _stacked_anisotropic_model(NoSymmetry(); fill_coefs = true)
        @test_throws ArgumentError reduce_cell(model_ns, cr_ns,
                                               [4.0 0 0; 0 4.0 0; 0 0 2.0])
    end

    @testset "fitted model, non-diagonal M: non-uniform energy identity" begin
        model, cr = _checkerboard_model()
        red = reduce_cell(model, cr, [1.0 0 0; 0 1.0 0; 0 0 4.0])
        @test n_atoms(red) == 1
        @test red.M == SMatrix{3,3,Int}([1 1 0; -1 1 0; 0 0 1])
        @test length(red.terms) == length(spin_multipole_terms(model)) ÷ 2
        # a training-periodic (not uniform!) configuration: paint each reduced cell
        # with its coset's spin — diag(2,2,1) = M·[1 -1 0; 1 1 0; 0 0 1] wraps a
        # sublattice of M·ℤ³ and covers two training cells.
        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        rng = MersenneTwister(29)
        cfg_tr = MC.SpinConfig([_rand_spin(rng), _rand_spin(rng)])
        cfg_red = MC.SpinConfig(undef, MC.n_sites(H_red))
        for s = 1:MC.n_sites(H_red)
            idx = s - 1                          # one atom per reduced cell
            c = SVector(idx % 2, (idx ÷ 2) % 2, idx ÷ 4)
            cfg_red[s] = cfg_tr[_coset_atom(red, c)]
        end
        E_tr = total_energy(H_tr, cfg_tr)
        @test total_energy(H_red, cfg_red) ≈ 2 * E_tr atol = 1e-13
        @test E_tr ≈ predict_energy(model, _config_matrix(cfg_tr)) - intercept(model) atol =
            1e-12
    end

    @testset "left-handed reduced cell (det M < 0)" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))
        red = reduce_cell(tr_cr, tr_terms, Matrix(Diagonal([1.0, -1.0, 1.0])))
        @test MC._det3(red.M) == -4              # M = diag(2, -2, 1)
        @test n_atoms(red) == 1
        @test length(red.terms) == 2
        for rt in red.terms                     # ±x untouched by the y flip;
            st = sub_terms[1]                   # two copies of the +x-form rep
            @test rt.coef == st.coef
            @test rt.shifts == st.shifts
        end
        H_tr = MC.TiledHamiltonian(4, tr_terms; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (2, 2, 1))
        cfg = _rand_config(MersenneTwister(31), H_tr)
        cfg_red = _permute_config(cfg, _reduce_perm(red, H_tr, H_red))
        @test total_energy(H_red, cfg_red) ≈ total_energy(H_tr, cfg) atol = 1e-13
    end

    @testset "hand-built mixed-channel terms: exact recovery and the (4π) pin" begin
        sub_mixed, L = _mixed_chain_terms()
        tr_cr = supercell_crystal(sub_cr, (2, 1, 1))
        tr_terms = _unfold_diag(sub_mixed, 1, (2, 1, 1))
        @test length(tr_terms) == 6
        red = reduce_cell(tr_cr, tr_terms, L, sub_lat)
        @test red isa ReducedCell{DecoratedTerm}
        @test red.layout === L
        @test n_atoms(red) == 1
        @test length(red.terms) == 3
        # every field verbatim, including `scale` — which the reduction carries from the
        # representative and must NEVER re-derive from the cluster shape
        for (rt, st) in zip(red.terms, sub_mixed)
            @test rt.coef == st.coef
            @test rt.scale == st.scale
            @test rt.body == st.body
            @test rt.atoms == st.atoms
            @test rt.shifts == st.shifts
            @test rt.slots == st.slots
            @test rt.folded == st.folded
        end
        # the pin: the surviving scale is the per-SPIN-slot rule, and the fixture really
        # does contain terms where `(4π)^(body/2)` would be a different number
        for rt in red.terms
            @test rt.scale == _slot_scale(rt)
        end
        @test count(rt -> rt.scale != (4π)^(rt.body / 2), red.terms) == 2

        # energy identity: the reduced list, the training (unfolded) list, and the
        # hand-built sub-cell list all tile to the same number
        H_tr = MC.TiledHamiltonian(2, tr_terms, L; dims = (1, 1, 1),
                                   fixed_reference = true)
        H_red = TiledHamiltonian(red; dims = (2, 1, 1), fixed_reference = true)
        H_sub = MC.TiledHamiltonian(1, sub_mixed, L; dims = (2, 1, 1),
                                    fixed_reference = true)
        rng = MersenneTwister(37)
        cfg = _rand_config(rng, H_tr)
        disps = _rand_disps(rng, H_tr)
        perm = _reduce_perm(red, H_tr, H_red)
        cfg_red = _permute_config(cfg, perm)
        dis_red = _permute_disps(disps, perm)
        E = total_energy(H_red, cfg_red, dis_red)
        @test E ≈ total_energy(H_tr, cfg, disps) atol = 1e-13
        @test E ≈ total_energy(H_sub, cfg_red, dis_red) atol = 1e-13

        # slot bookkeeping errors of a hand-built list
        bad_body = [DecoratedTerm(0.1, 4π, 3, [1, 1], [SVector(0, 0, 0),
                                                       SVector(1, 0, 0)],
                                  copy(sub_mixed[1].slots), copy(sub_mixed[1].folded))]
        @test_throws ArgumentError reduce_cell(sub_cr, bad_body, L, sub_lat)
        bad_rank = [DecoratedTerm(0.1, 4π, 2, [1, 1], [SVector(0, 0, 0),
                                                       SVector(1, 0, 0)],
                                  copy(sub_mixed[1].slots), zeros(3))]
        @test_throws ArgumentError reduce_cell(sub_cr, bad_rank, L, sub_lat)
        bad_site = [DecoratedTerm(0.1, (4π)^0.5, 1, [1], [SVector(0, 0, 0)],
                                  [SLCE.SlotRef(2, SLCE.SiteFactor(SLCE.SPIN, 0, 1))],
                                  zeros(3))]
        @test_throws ArgumentError reduce_cell(sub_cr, bad_site, L, sub_lat)
        @test_throws ArgumentError reduce_cell(sub_cr, DecoratedTerm[], L, sub_lat)
    end

    @testset "non-diagonal M with a displacement axis" begin
        # A change of cell basis re-expresses the SHIFTS; it cannot touch `folded`,
        # because a DISP factor is `|u|^{2k} R_{lm}(u)` in CARTESIAN `u` and a SPIN one
        # reads a Cartesian direction. Nothing states that anywhere else, and no other
        # decorated case here uses a non-diagonal M.
        mB = [1 1 0; -1 1 0; 0 0 1]
        tr_cr = Crystal(Lattice(Float64.(mB)), [0 0.5; 0 0.5; 0.0 0.0], [1, 1], ["Fe"])
        L = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])
        sp(site) = SLCE.SlotRef(site, SLCE.SiteFactor(SLCE.SPIN, 0, 1))
        dp(site) = SLCE.SlotRef(site, SLCE.SiteFactor(SLCE.DISP, 0, 1))
        z = SVector(0, 0, 0)
        f = [0.3 -0.1 0.0; 0.0 0.2 0.5; -0.4 0.0 0.1]
        # the two training images of ONE mixed +x pair of the 1-atom cell
        tr_terms = [DecoratedTerm(0.09, (4π)^0.5, 2, [1, 2], [z, z], [sp(1), dp(2)],
                                  copy(f)),
                    DecoratedTerm(0.09, (4π)^0.5, 2, [2, 1], [z, SVector(1, 1, 0)],
                                  [sp(1), dp(2)], copy(f))]
        red = reduce_cell(tr_cr, tr_terms, L, sub_lat)
        @test red.M == SMatrix{3,3,Int}(mB)
        @test n_atoms(red) == 1
        @test length(red.terms) == 1
        rt = red.terms[1]
        @test rt.atoms == [1, 1]
        @test rt.shifts == [z, SVector(1, 0, 0)]
        @test rt.slots == [sp(1), dp(2)]
        @test rt.scale == (4π)^0.5
        @test rt.folded == f                     # verbatim: the basis change never
                                                 # reaches the tensor
        # per-site energy identity on a uniform (hence sub-periodic) state
        H_tr = MC.TiledHamiltonian(2, tr_terms, L; fixed_reference = true)
        H_red = TiledHamiltonian(red; dims = (3, 1, 1), fixed_reference = true)
        rng = MersenneTwister(59)
        e = _rand_spin(rng)
        u = 0.07 .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
        E_tr = total_energy(H_tr, MC.SpinConfig([e, e]), [u, u])
        E_red = total_energy(H_red, MC.SpinConfig([e, e, e]), [u, u, u])
        @test E_red / 3 ≈ E_tr / 2 atol = 1e-14
    end

    @testset "3-body mixed terms: a genuine 3-cycle site permutation" begin
        # The one case that separates `back = invperm(perm)` from `perm`: every other
        # fixture here is ≤ 2-body, and a 2-body site permutation is an involution, so
        # reversing the relabel direction is a bitwise no-op on all of them.
        sub3, L3 = _threebody_mixed_terms()
        tr_cr = supercell_crystal(sub_cr, (3, 1, 1))
        raw = _unfold_diag(sub3, 1, (3, 1, 1))
        tr_terms = _canonicalize.(raw)

        # the mirror helper, validated against the evaluator rather than against the
        # code it mirrors
        H_raw = MC.TiledHamiltonian(3, raw, L3; fixed_reference = true)
        H_can = MC.TiledHamiltonian(3, tr_terms, L3; fixed_reference = true)
        rng = MersenneTwister(53)
        cfg0 = _rand_config(rng, H_raw)
        dis0 = _rand_disps(rng, H_raw)
        @test total_energy(H_can, cfg0, dis0) ≈ total_energy(H_raw, cfg0, dis0) atol =
            1e-14

        red = reduce_cell(tr_cr, tr_terms, L3, sub_lat)
        # two of the three copies really are 3-cycles (the third is the identity)
        perms = map(t -> MC._reduced_sites(t, red.atom_map, red.M, 3, 1)[3],
                    tr_terms)
        @test count(p -> p != invperm(p), perms) == 2
        @test length(red.terms) == 1
        rt, st = red.terms[1], sub3[1]
        @test rt.coef == st.coef
        @test rt.scale == st.scale
        @test rt.body == st.body
        @test rt.atoms == st.atoms
        @test rt.shifts == st.shifts
        @test rt.slots == st.slots
        @test rt.folded == st.folded

        H_red = TiledHamiltonian(red; dims = (3, 1, 1), fixed_reference = true)
        H_sub = MC.TiledHamiltonian(1, sub3, L3; dims = (3, 1, 1),
                                    fixed_reference = true)
        perm = _reduce_perm(red, H_can, H_red)
        cfg = _permute_config(cfg0, perm)
        dis = _permute_disps(dis0, perm)
        E = total_energy(H_red, cfg, dis)
        @test E ≈ total_energy(H_can, cfg0, dis0) atol = 1e-13
        @test E ≈ total_energy(H_sub, cfg, dis) atol = 1e-13
    end

    @testset "fitted joint model: DecoratedTerm reduction survives 2×" begin
        model, cr = _stacked_joint_model()
        layout = SLCE.row_layout(model)
        dts = SLCE.decorated_terms(model)
        red = reduce_cell(model, cr, [4.0 0 0; 0 4.0 0; 0 0 2.0])
        @test red isa ReducedCell{DecoratedTerm}
        @test red.layout == layout
        @test n_atoms(red) == 1
        @test length(red.terms) == length(dts) ÷ 2
        for rt in red.terms
            @test rt.scale == _slot_scale(rt)
        end
        @test any(rt -> rt.scale != (4π)^(rt.body / 2), red.terms)
        # the fixture is genuinely mixed: a site carrying two axes, and a term with no
        # spin content at all
        @test any(rt -> any(q -> count(s -> s.site == q, rt.slots) == 2, 1:rt.body),
                  red.terms)
        @test any(rt -> !any(s -> s.factor.channel == SLCE.SPIN, rt.slots), red.terms)
        # …and the alignment really fires: a reduction whose site permutation were the
        # identity everywhere would gate none of the slot relabel / `folded` axis
        # bookkeeping this arm exists for. Count the training terms whose translation
        # partner is anchored at a DIFFERENT member site AND whose slots then need
        # re-sorting (29 of 62 here).
        slotperm(dt, back) =
            sortperm(1:length(dt.slots);
                     by = v -> (dt.slots[v].factor.channel, back[dt.slots[v].site],
                                dt.slots[v].factor.k, dt.slots[v].factor.l, v))
        @test count(dts) do dt
            _, _, p, _ = MC._reduced_sites(dt, red.atom_map, red.M, n_atoms(cr), 1)
            p != 1:length(p) && slotperm(dt, invperm(p)) != 1:length(dt.slots)
        end == 29

        H_tr = TiledHamiltonian(model; dims = (1, 1, 1))
        H_red = TiledHamiltonian(red; dims = (1, 1, 2))
        rng = MersenneTwister(43)
        cfg = _rand_config(rng, H_tr)
        disps = _rand_disps(rng, H_tr)
        perm = _reduce_perm(red, H_tr, H_red)
        E = total_energy(H_red, _permute_config(cfg, perm), _permute_disps(disps, perm))
        @test E ≈ total_energy(H_tr, cfg, disps) atol = 1e-13
        @test E ≈ predict_energy(model, _config_matrix(cfg), _disp_matrix(disps)) -
                  intercept(model) atol = 1e-12

        # a larger tiling, and the point of the whole feature: an ODD number of reduced
        # cells along the reduced axis, which no training-cell multiple can reach
        H_tr2 = TiledHamiltonian(model; dims = (2, 1, 1))
        H_red2 = TiledHamiltonian(red; dims = (2, 1, 2))
        cfg2 = _rand_config(rng, H_tr2)
        dis2 = _rand_disps(rng, H_tr2)
        perm2 = _reduce_perm(red, H_tr2, H_red2)
        @test total_energy(H_red2, _permute_config(cfg2, perm2),
                           _permute_disps(dis2, perm2)) ≈
              total_energy(H_tr2, cfg2, dis2) atol = 1e-12
        @test MC.n_sites(TiledHamiltonian(red; dims = (1, 1, 3))) == 3

        # a coefficient that breaks the half-cell periodicity is refused on the
        # decorated path too
        bad = copy(dts)
        k = findfirst(dt -> dt.coef != 0, bad)
        bad[k] = DecoratedTerm(bad[k].coef * 1.001, bad[k].scale, bad[k].body,
                               copy(bad[k].atoms), copy(bad[k].shifts),
                               copy(bad[k].slots), copy(bad[k].folded))
        @test_throws ArgumentError reduce_cell(cr, bad, layout,
                                               [4.0 0 0; 0 4.0 0; 0 0 2.0])
    end

    @testset "verification errors" begin
        tr_cr = supercell_crystal(sub_cr, (2, 2, 1))
        tr_terms = _unfold_diag(sub_terms, 1, (2, 2, 1))

        # a coefficient that breaks the translation symmetry of the Hamiltonian
        bad = copy(tr_terms)
        bad[3] = SpinMultipoleTerm(bad[3].coef * 1.001, 2, copy(bad[3].atoms),
                               copy(bad[3].shifts), copy(bad[3].ls),
                               copy(bad[3].folded))
        @test_throws ArgumentError reduce_cell(tr_cr, bad, sub_lat)

        # a distorted structure (atom off its translation image)
        frac = copy(tr_cr.frac_positions)
        frac[1, 2] += 0.02
        cr_bad = Crystal(tr_cr.lattice, frac, tr_cr.species, tr_cr.species_labels)
        @test_throws ArgumentError reduce_cell(cr_bad, tr_terms, sub_lat)

        # lattice not an integer relation
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms, Matrix(0.9 * I(3)))
        # |det M| does not divide n_atoms (A_sub = diag(2/3, 1, 1) ⇒ M = diag(3,2,1))
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms,
                                               Matrix(Diagonal([2 / 3, 1.0, 1.0])))
        # model/crystal mismatch
        @test_throws ArgumentError reduce_cell(_dimer_model(), sub_cr, sub_lat)
        # wrong sub_lattice shape
        @test_throws ArgumentError reduce_cell(tr_cr, tr_terms, ones(2, 2))
        # empty term list
        @test_throws ArgumentError reduce_cell(tr_cr, SpinMultipoleTerm[], sub_lat)

        # geometrically periodic but chemically not: species differ across cosets
        tr2 = supercell_crystal(sub_cr, (2, 1, 1))
        cr_species = Crystal(tr2.lattice, tr2.frac_positions, [1, 2], ["Fe", "Co"])
        terms2 = _unfold_diag(sub_terms, 1, (2, 1, 1))
        @test_throws ArgumentError reduce_cell(cr_species, terms2, sub_lat)

        # coincident atoms folding onto one reduced site
        lat2 = Lattice(Matrix(Diagonal([2.0, 1.0, 1.0])))
        cr_dup = Crystal(lat2, [0.25 0.25; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        @test_throws ArgumentError reduce_cell(cr_dup, sub_terms, Matrix(1.0 * I(3)))

        # LOPSIDED COSETS. The census is per coset, not in total. Four copies of ONE
        # summand, all anchored in the same coset and none in the other three, sum to
        # exactly nc = 4 — so the weaker `count % nc == 0` test accepts them and emits
        # the term as if it sat in every reduced cell, which is not the Hamiltonian
        # that was fitted. Unreachable from a fitted model, reachable by stitching two
        # term lists together, and precisely what "verified, never assumed" is for.
        copy_of(t) = SpinMultipoleTerm(t.coef, length(t.atoms), copy(t.atoms),
                                   copy(t.shifts), copy(t.ls), copy(t.folded))
        one_sided = [copy_of(tr_terms[1]) for _ = 1:4]
        @test length(one_sided) % 4 == 0          # …the weaker test would pass
        @test allequal(MC._reduced_sites(t, [(1, SVector(0, 0, 0)) for _ = 1:4],
                                         SMatrix{3,3,Int}(Diagonal([2, 2, 1])), 4,
                                         1)[4] for t in one_sided)   # …one coset only
        @test_throws ArgumentError reduce_cell(tr_cr, one_sided, sub_lat)

        # degenerate hand-built shapes reach an ArgumentError, not a BoundsError /
        # DimensionMismatch out of the middle of the census
        @test_throws ArgumentError reduce_cell(tr_cr, [SpinMultipoleTerm(0.1, 0, Int[],
                                                                    SVector{3,Int}[],
                                                                    Int[], fill(1.0))],
                                               sub_lat)
        L1 = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])
        wrong_extent = [DecoratedTerm(0.1, (4π)^0.5, 1, [1], [SVector(0, 0, 0)],
                                      [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0,
                                                                       1))],
                                      zeros(5))]
        @test_throws ArgumentError reduce_cell(tr_cr, wrong_extent, L1, sub_lat)

        # one key, two spellings of the scale: reported as a scale disagreement, not as
        # a physics failure (`sqrt(4π)` and `(4π)^0.5` differ by 1 ulp)
        sub_mixed2, L2 = _mixed_chain_terms()
        scale_split = _unfold_diag(sub_mixed2, 1, (2, 1, 1))
        j = findlast(t -> t.slots == sub_mixed2[1].slots, scale_split)
        scale_split[j] = DecoratedTerm(scale_split[j].coef, sqrt(4π) * (1 + 1e-9), 2,
                                       copy(scale_split[j].atoms),
                                       copy(scale_split[j].shifts),
                                       copy(scale_split[j].slots),
                                       copy(scale_split[j].folded))
        @test_throws ArgumentError reduce_cell(tr_cr, scale_split, L2, sub_lat)

        # ReducedCell is exported, so its own invariants have to hold for a
        # hand-built one too: a term type it cannot tile, and a pure-spin reduction
        # handed a RowLayout it must not carry (the decorated arm is the only one
        # whose slots need the model's row numbering).
        eye = SMatrix{3,3,Int}(I(3))
        z = SVector(0, 0, 0)
        @test_throws ArgumentError MC.ReducedCell(1, [1.0], sub_cr, eye, [1],
                                                  [(1, z)], nothing)
        @test_throws ArgumentError MC.ReducedCell(1, sub_terms, sub_cr, eye, [1],
                                                  [(1, z)],
                                                  SLCE.RowLayout(4, 1, 4,
                                                                 Tuple{Int,Int}[],
                                                                 Int[]))
    end
end
