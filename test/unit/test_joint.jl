# The joint (spin + displacement) ingest — M4 slice 3b.
#
# Two claims are under test and they pull in opposite directions, which is why they are
# gated together:
#
#   1. A joint model's tiled energy is the model's own energy. The fence is the same one
#      the pure-spin tiling has always used — `dims = (1,1,1)` against
#      `predict_energy − intercept`, and periodic replication against `prod(dims) ×`
#      that — now with displacements in play, plus a bitwise check that this package's
#      row filler agrees with the upstream reference `SLCE.site_rows!`.
#   2. NOTHING a pure-spin model produces moved. The strongest available form of that is
#      not "the energies agree" but "the precompiled program arrays are byte-identical":
#      the same terms fed through the old `MultipoleTerm` surface and the new
#      `DecoratedTerm` one must flatten to the very same integers and floats, which
#      makes the channel generalization a pure relabeling on that path. The checkpoint
#      fingerprint is pinned the same way, against an in-test copy of the pre-M4 formula.

using Test

@testset "joint spin-lattice ingest" begin
    model, cr = _joint_model()
    L = SLCE.row_layout(model)
    H = MC.TiledHamiltonian(model)
    rng = MersenneTwister(11)

    # One site's full basis-row column, the way a consumer builds it.
    function rowcol(H, e, u)
        col = zeros(H.nrows)
        MC._zlm_row!(view(col, 1:H.nlm), e, H.lmax)
        MC.has_disp(H) &&
            MC._disp_rows!(col, H, u, zeros(max(0, (H.disp_lmax + 1)^2)))
        return col
    end

    @testset "the fixture actually exercises the channel cases" begin
        @test MC.has_disp(H)
        @test H.nrows > H.nlm
        @test H.layout == L
        @test H.nlm == L.disp_offset == 4          # (lmax_spin + 1)² with lmax = 1
        @test H.lmax == 1
        @test H.disp_lmax == maximum(l for (_, l) in L.disp_factors)
        # a site carrying two axes (spin × displacement), and a site carrying none of
        # the spin ones (the lattice-only sector) — the two cases the scale rule and the
        # merged site programs exist for
        twoaxis = any(t -> any(q -> count(s -> s.site == q, t.slots) == 2,
                               1:length(t.atoms)), H.terms)
        spinfree = any(t -> any(q -> !any(s -> s.site == q && s.spin, t.slots),
                                1:length(t.atoms)), H.terms)
        @test twoaxis
        @test spinfree
        # the scale is applied exactly once and taken from the FIELD: check the ingested
        # coefficient against `coef · scale` bitwise, that the field itself is the
        # per-spin-slot rule, and — the teeth — that the fixture contains a term where
        # the pure-spin-era `(4π)^(body/2)` shortcut would give a different number
        dts = SLCE.decorated_terms(model)
        @test length(dts) == length(H.terms)
        for (dt, t) in zip(dts, H.terms)
            @test t.coef == dt.coef * dt.scale
            @test dt.scale == (4π)^(count(s -> s.factor.channel == SLCE.SPIN,
                                          dt.slots) / 2)
        end
        @test any(dt -> dt.scale != (4π)^(length(dt.atoms) / 2), dts)
        @test any(!=(0), H.progs.site_col)
    end

    @testset "_hoisted_columns: the fast path declines exactly when it must" begin
        # A bare `any(==(0), site_col)` would be satisfied by the fixture's rank-1
        # single-ion terms, which have no factors at all — it does not pin the DECLINE.
        # Test the predicate directly instead, on the layout that motivates it.
        sp(site, l) = MC.TermSlot(site, 0, l, true)
        dp(site, l) = MC.TermSlot(site, 4, l, false)
        cols(slots, q) = MC._hoisted_columns(slots, Int8[Int8(s.site) for s in slots],
                                             findall(s -> s.site == q, slots))
        # one axis per site: always hoistable (the pure-spin case, every rank)
        @test cols([sp(1, 1), sp(2, 1)], 1) == (Int8(2), Int8(0))
        @test cols([sp(1, 1), sp(2, 1)], 2) == (Int8(1), Int8(0))
        @test cols([sp(1, 1), sp(2, 1), sp(3, 1)], 2) == (Int8(1), Int8(3))
        # two axes on one site, ADJACENT in axis order (rank 2, body 1): both drop an
        # axis sitting on the same site, so the column tuples agree and hoisting is fine
        @test cols([sp(1, 1), dp(1, 2)], 1) == (Int8(1), Int8(0))
        # the fixture's shape — two same-site axes SEPARATED by a third: axis 1 sees
        # (2, 1), axis 3 sees (1, 2). Different order ⇒ must decline.
        sep = [sp(1, 1), sp(2, 1), dp(1, 2)]
        @test cols(sep, 1) == (Int8(0), Int8(0))          # the decline
        @test cols(sep, 2) == (Int8(1), Int8(1))          # site 2 has one axis: hoists
        # rank ≥ 4 has no fast path at all
        @test cols([sp(1, 1), sp(2, 1), dp(1, 1), dp(2, 1)], 1) == (Int8(0), Int8(0))
        # and the fixture really does contain that declining shape
        @test any(H.terms) do t
            length(t.slots) == 3 &&
                any(q -> MC._hoisted_columns(t.slots,
                                             Int8[Int8(s.site) for s in t.slots],
                                             findall(s -> s.site == q, t.slots)) ==
                         (Int8(0), Int8(0)), 1:length(t.atoms))
        end
    end

    @testset "row filler ≡ SLCE.site_rows! (bitwise)" begin
        # the coupled site: this package tabulates the rows itself (one solid-harmonic
        # batch per site, reusing it for every (k, l) block), the upstream reference
        # calls the convenience accessor per (k, l, m). Same numbers or the whole
        # gather is reading a different basis than the model was fitted in.
        for _ = 1:6
            e = _rand_spin(rng)
            u = 0.08 .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
            ref = zeros(L.nrows)
            SLCE.site_rows!(ref, L, e, u)
            @test rowcol(H, e, u) == ref
        end
        # u = 0 zeroes every displacement row and leaves the spin rows alone
        e = _rand_spin(rng)
        z = rowcol(H, e, zero(SVector{3,Float64}))
        @test all(iszero, z[(H.nlm + 1):H.nrows])
        @test z[1:H.nlm] == rowcol(H, e, 0.05 .* SVector(1.0, 2.0, -1.0))[1:H.nlm]
    end

    @testset "the solid-harmonic batch is prefix-stable" begin
        # `_disp_rows!` batches ONCE per site up to `disp_lmax` and indexes every (k, l)
        # block out of it; the upstream reference `SLCE.site_rows!` calls the convenience
        # `Rlm` accessor, which re-batches up to that `l` alone. Their bitwise agreement
        # rests on a property of the recurrences — a higher upper limit cannot change a
        # lower entry — so pin the property directly instead of inferring it from one
        # fixture happening to agree.
        SH = SLCE.SolidHarmonics
        for u in [SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) for _ = 1:12]
            for lo = 0:4, hi = lo:5
                a = SH.solid_harmonics(lo, u)
                @test a == SH.solid_harmonics(hi, u)[1:length(a)]
            end
        end
        # the degenerate arguments the accessor path would also hit
        for u in (zero(SVector{3,Float64}), SVector(0.0, 0.0, 1e-9),
                  SVector(1e8, -1e8, 0.0), SVector(0.0, 0.0, -2.0))
            for lo = 0:3
                a = SH.solid_harmonics(lo, u)
                @test a == SH.solid_harmonics(5, u)[1:length(a)]
            end
        end
    end

    @testset "dims=(1,1,1) ≡ predict_energy − intercept" begin
        for _ = 1:5
            config = _rand_config(rng, H)
            disps = _rand_disps(rng, H)
            E = total_energy(H, config, disps)
            ref = SLCE.predict_energy(model, _config_matrix(config),
                                      _disp_matrix(disps)) - intercept(model)
            @test E ≈ ref atol = 1e-12 rtol = 1e-12
            @test abs(ref) > 1e-6                      # not a trivial zero
        end
    end

    @testset "periodic replication: 2×2×2 is 8× the cell" begin
        H8 = MC.TiledHamiltonian(model; dims = (2, 2, 2))
        @test H8.layout == L && H8.nrows == H.nrows
        for _ = 1:3
            cell_cfg = _rand_config(rng, H)
            cell_u = _rand_disps(rng, H)
            E1 = total_energy(H, cell_cfg, cell_u)
            E8 = total_energy(H8, _tile_config(H8, cell_cfg), _tile_disps(H8, cell_u))
            @test E8 ≈ 8 * E1 atol = 1e-10 rtol = 1e-12
        end
    end

    @testset "program kernels ≡ reference kernels (bitwise)" begin
        # the rank-generic reference kernels are the spec of the μ-mapping and of the
        # entry/axis/accumulation order; the merged site programs must reproduce them
        for _ = 1:3
            config = _rand_config(rng, H)
            disps = _rand_disps(rng, H)
            zrows = MC._zrows(H, config, disps)
            @test MC._total_energy(H, zrows) == MC._total_energy_ref(H, zrows)
            ok = true
            for s = 1:n_sites(H)
                ok &= MC.site_coeffs!(zeros(H.nrows), H, s, zrows) ==
                      MC._site_coeffs_ref!(zeros(H.nrows), H, s, zrows)
            end
            @test ok
        end
    end

    @testset "single-channel ΔE is exact; a simultaneous move is not" begin
        config = _rand_config(rng, H)
        disps = _rand_disps(rng, H)
        zrows = MC._zrows(H, config, disps)
        for s = 1:n_sites(H)
            c = MC.site_coeffs!(zeros(H.nrows), H, s, zrows)
            old = collect(view(zrows, :, s))
            e2 = _rand_spin(rng)
            u2 = 0.08 .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
            E0 = total_energy(H, config, disps)

            # spin only
            cfg2 = copy(config); cfg2[s] = e2
            @test MC.delta_energy(c, old, rowcol(H, e2, disps[s])) ≈
                  total_energy(H, cfg2, disps) - E0 atol = 1e-12
            # displacement only
            u_2 = copy(disps); u_2[s] = u2
            @test MC.delta_energy(c, old, rowcol(H, config[s], u2)) ≈
                  total_energy(H, config, u_2) - E0 atol = 1e-12

            # BOTH at once: the leave-one-out vector cannot see the cross term of the
            # site's own two axes. Pin the VALUE of the miss, not merely that it is
            # nonzero — with `c′` recomputed at (old spin, new u), `c′ − c` is exactly
            # `(MΔr, 0)`, so the residual is `delta_energy(c′ − c, old, new)`. That this
            # closes the gap exactly is also what proves there is no `Δz·Δz` or `Δr·Δr`
            # remainder, i.e. that the one-axis-per-(site, channel) invariant holds.
            true_dE = total_energy(H, cfg2, u_2) - E0
            new = rowcol(H, e2, u2)
            approx = MC.delta_energy(c, old, new)
            cp = MC.site_coeffs!(zeros(H.nrows), H, s, MC._zrows(H, config, u_2))
            @test true_dE - approx ≈ MC.delta_energy(cp .- c, old, new) atol = 1e-13
            @test abs(true_dE - approx) > 1e-9        # the miss is real, not vacuous
        end
    end

    @testset "site_coeffs! is independent of the site's own rows" begin
        # per channel: moving the spin of s cannot change the coefficients targeting s
        config = _rand_config(rng, H)
        disps = _rand_disps(rng, H)
        for s = 1:n_sites(H)
            c1 = MC.site_coeffs!(zeros(H.nrows), H, s, MC._zrows(H, config, disps))
            cfg2 = copy(config); cfg2[s] = _rand_spin(rng)
            u2 = copy(disps); u2[s] = _rand_disps(rng, H)[s]
            # only the SPIN target rows are guaranteed invariant under a spin change of
            # s (the displacement targets multiply the spin row of s), and vice versa
            c_spin = MC.site_coeffs!(zeros(H.nrows), H, s, MC._zrows(H, cfg2, disps))
            @test c1[(H.nlm + 1):H.nrows] != c_spin[(H.nlm + 1):H.nrows] ||
                  all(iszero, c1[(H.nlm + 1):H.nrows])
            @test c1[1:H.nlm] == c_spin[1:H.nlm]
            c_disp = MC.site_coeffs!(zeros(H.nrows), H, s, MC._zrows(H, config, u2))
            @test c1[(H.nlm + 1):H.nrows] == c_disp[(H.nlm + 1):H.nrows]
        end
    end

    @testset "refusals" begin
        @test_throws ArgumentError total_energy(H, _rand_config(rng, H))
        @test_throws DimensionMismatch total_energy(H, _rand_config(rng, H),
                                                   _rand_disps(rng, H)[1:1])
        @test_throws ArgumentError MC.ChainState(H, _rand_config(rng, H),
                                                 Xoshiro(1), 0.5)
        @test_throws ArgumentError minimize_energy(H)
        @test_throws ArgumentError find_ground_state(H; kT = 0.01)
        @test_throws ArgumentError MC.energy_gradient(H, _rand_config(rng, H))
        @test_throws ArgumentError MC.site_gradient(H, 1, _rand_config(rng, H))
        # a member site with no tensor axis at all would lie to the inactive-site
        # convention (active, yet contributing nothing)
        bad = [DecoratedTerm(0.1, 1.0, 2, [1, 2], [SVector(0, 0, 0), SVector(0, 0, 0)],
                             [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1))],
                             randn(3))]
        @test_throws ArgumentError MC.TiledHamiltonian(2, bad, MC._spin_row_layout(1))
        # TWO AXES OF ONE CHANNEL on a site. This one is the dangerous case: such a term
        # has a pure-spin layout, so `has_disp` is false and `_require_spin_only` would
        # NOT catch it — it would go into the sweeps with a `delta_energy` that silently
        # drops the site's own Δz·Δz′ cross term (measured 21 % off).
        two = [DecoratedTerm(0.1, 4π, 1, [1], [SVector(0, 0, 0)],
                             [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1)),
                              SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1))],
                             randn(3, 3))]
        @test_throws ArgumentError MC.TiledHamiltonian(1, two, MC._spin_row_layout(1))
        # a layout the upstream row_index itself rejects (l beyond the spin block) …
        wide = [DecoratedTerm(0.1, 1.0, 1, [1], [SVector(0, 0, 0)],
                              [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 2))],
                              randn(5))]
        @test_throws ArgumentError MC.TiledHamiltonian(1, wide, MC._spin_row_layout(1))
        # … and one it accepts while the block still runs off the end of `nrows`, which
        # is the ctor's own bound check (unreachable through `_spin_row_layout`, so it
        # takes a deliberately malformed layout to exercise)
        short = SLCE.RowLayout(2, 1, 2, Tuple{Int,Int}[], Int[])
        @test SLCE.row_index(short, SLCE.SiteFactor(SLCE.SPIN, 0, 1), 1) == 4
        ok1 = [DecoratedTerm(0.1, 4π, 1, [1], [SVector(0, 0, 0)],
                             [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1))],
                             randn(3))]
        @test_throws ArgumentError MC.TiledHamiltonian(1, ok1, short)
    end

    @testset "site activity: scheduling vs magnetism" begin
        # A site referenced only by displacement axes is active for the coloring but is
        # NOT magnetic — conflating the two would divide `m = Σ_s e_s / n` by a count
        # that includes a frozen random direction.
        z3 = SVector(0, 0, 0)
        mixed = [DecoratedTerm(0.2, 4π, 2, [1, 2], [z3, z3],
                               [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1)),
                                SLCE.SlotRef(2, SLCE.SiteFactor(SLCE.DISP, 0, 1))],
                               randn(3, 3))]
        L2 = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])
        Hm = MC.TiledHamiltonian(2, mixed, L2)
        @test Hm.site_active == [true, true] && Hm.n_active == 2
        @test Hm.site_has_spin == [true, false] && Hm.n_spin_active == 1
        @test Hm.site_has_l1 == [true, false]
        # on every pure-spin model the two notions coincide exactly
        for Hs in (TiledHamiltonian(_dimer_model()),
                   TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1)))
            @test Hs.site_has_spin == Hs.site_active
            @test Hs.n_spin_active == Hs.n_active
        end
    end

    @testset "fingerprint separates layouts that differ only in the radial power" begin
        # `TermSlot.row0` is a layout-relative block start, so `(k, l) → row0` is not
        # injective across layouts: a `degree = 3:5` sector's (1,1),(2,1) blocks start
        # where a `1:3` sector's (0,1),(1,1) do. Two models differing only by `k` — hence
        # by a factor |u|² — must not share a fingerprint.
        z3 = SVector(0, 0, 0)
        mk(k) = [DecoratedTerm(0.3, 4π, 2, [1, 2], [z3, z3],
                               [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1)),
                                SLCE.SlotRef(2, SLCE.SiteFactor(SLCE.DISP, k, 1))],
                               fill(0.5, 3, 3))]
        HA = MC.TiledHamiltonian(2, mk(0), SLCE.RowLayout(7, 1, 4, [(0, 1)], [4]))
        HB = MC.TiledHamiltonian(2, mk(1), SLCE.RowLayout(7, 1, 4, [(1, 1)], [4]))
        @test MC.model_fingerprint(HA) != MC.model_fingerprint(HB)
        cfg = MC.SpinConfig([SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 0.0)])
        u = [SVector(0.1, -0.2, 0.3), SVector(-0.05, 0.15, 0.2)]
        @test total_energy(HA, cfg, u) != total_energy(HB, cfg, u)
    end

    @testset "a pure-spin model still takes the frozen path" begin
        Hs = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        @test !MC.has_disp(Hs)
        @test Hs.nrows == Hs.nlm == SLCE.Harmonics.num_lm(Hs.lmax)
        @test isempty(Hs.layout.disp_factors)
        @test all(t -> MC._is_spin_identity(t.slots), Hs.terms)
        # and `disps` may be omitted there, as it always could
        @test total_energy(Hs, _rand_config(rng, Hs)) isa Float64
        # all-zero displacements are the state such a model describes, so they are fine;
        # nonzero ones would be silently ignored, so they throw
        cfg = _rand_config(rng, Hs)
        zu = [zero(SVector{3,Float64}) for _ = 1:n_sites(Hs)]
        @test total_energy(Hs, cfg, zu) == total_energy(Hs, cfg)
        @test_throws ArgumentError total_energy(Hs, cfg, _rand_disps(rng, Hs))
    end
end

@testset "pure-spin ingest: MultipoleTerm ≡ DecoratedTerm (bitwise)" begin
    # The same terms through the frozen surface and the general one must flatten to
    # byte-identical program arrays — that is what makes the channel generalization a
    # pure relabeling for every pure-spin consumer, rather than "numerically the same".
    z3 = SVector(0, 0, 0)
    x3 = SVector(1, 0, 0)
    rng = MersenneTwister(23)
    sparse_folded(dims...) = begin
        f = randn(rng, dims...)
        f[rand(rng, length(f)) .< 0.5] .= 0.0
        f
    end
    cases = [(2, (2, 2, 1), [MultipoleTerm(0.3, 1, [1], [z3], [2], sparse_folded(5)),
                             MultipoleTerm(-0.2, 2, [1, 2], [z3, z3], [1, 1],
                                           sparse_folded(3, 3)),
                             MultipoleTerm(0.1, 3, [1, 2, 1], [z3, z3, x3], [1, 1, 2],
                                           sparse_folded(3, 3, 5))]),
             (1, (4, 1, 1), _chain_terms(0.05)),
             (1, (4, 1, 1), _threebody_terms(0.05)),
             (1, (4, 1, 1), _fourbody_terms(0.05))]

    for (nat, dims, mterms) in cases
        lmax = maximum(maximum(t.ls) for t in mterms)
        dterms = [DecoratedTerm(mt.coef, (4π)^(mt.body / 2), mt.body, mt.atoms,
                                mt.shifts,
                                [SLCE.SlotRef(i, SLCE.SiteFactor(SLCE.SPIN, 0, mt.ls[i]))
                                 for i in eachindex(mt.ls)], mt.folded)
                  for mt in mterms]
        Ha = MC.TiledHamiltonian(nat, mterms; dims = dims)
        Hb = MC.TiledHamiltonian(nat, dterms, MC._spin_row_layout(lmax); dims = dims)

        @test Ha.nrows == Hb.nrows && Ha.nlm == Hb.nlm && Ha.lmax == Hb.lmax
        @test Ha.layout == Hb.layout
        @test [t.coef for t in Ha.terms] == [t.coef for t in Hb.terms]
        @test [t.slots for t in Ha.terms] == [t.slots for t in Hb.terms]
        @test Ha.site_has_l1 == Hb.site_has_l1
        @test Ha.site_active == Hb.site_active && Ha.n_colors == Hb.n_colors
        # every precompiled program array, byte for byte
        for f in fieldnames(MC._ContractionPrograms)
            @test getfield(Ha.progs, f) == getfield(Hb.progs, f)
        end
        @test MC.model_fingerprint(Ha) == MC.model_fingerprint(Hb)

        cfg = _rand_config(rng, Ha)
        za, zb = MC._zrows(Ha, cfg), MC._zrows(Hb, cfg)
        @test za == zb
        @test MC._total_energy(Ha, za) == MC._total_energy(Hb, zb)
        for s = 1:n_sites(Ha)
            @test MC.site_coeffs!(zeros(Ha.nrows), Ha, s, za) ==
                  MC.site_coeffs!(zeros(Hb.nrows), Hb, s, zb)
        end
    end
end

@testset "checkpoint fingerprint: unchanged for pure spin, live for joint" begin
    # An in-test copy of the PRE-M4 formula (it mixed a per-site `ls`, which for the
    # identity slot layout is exactly the per-axis degree). Every checkpoint written
    # before the displacement channel existed must still identify its Hamiltonian.
    function fp_pre_m4(H)
        mix(h, x::UInt64) = (h ⊻ x) * 0x00000100000001b3
        mixi(h, x) = mix(h, reinterpret(UInt64, Int64(x)))
        mixf(h, x) = mix(h, reinterpret(UInt64, x))
        h = 0xcbf29ce484222325
        h = mixi(h, H.n_cell_atoms)
        for d in H.dims
            h = mixi(h, d)
        end
        for t in H.terms
            h = mixf(h, t.coef)
            for a in t.atoms
                h = mixi(h, a)
            end
            for s in t.shifts, i = 1:3
                h = mixi(h, s[i])
            end
            for sl in t.slots
                h = mixi(h, sl.l)          # ≡ the pre-M4 `t.ls`
            end
            for v in t.folded
                h = mixf(h, v)
            end
        end
        return h
    end

    for H in (TiledHamiltonian(_dimer_model()),
              TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1)),
              MC.TiledHamiltonian(1, _chain_terms(0.05); dims = (4, 1, 1)))
        @test MC.model_fingerprint(H) == fp_pre_m4(H)
    end
    # a joint model's slot layout is NOT the identity, so its layout is mixed in and
    # the fingerprint separates it from a bare degree list
    model, _ = _joint_model()
    Hj = MC.TiledHamiltonian(model)
    @test any(t -> !MC._is_spin_identity(t.slots), Hj.terms)
    @test MC.model_fingerprint(Hj) != fp_pre_m4(Hj)
end
