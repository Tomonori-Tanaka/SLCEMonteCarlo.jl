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
        # `ChainState` itself accepts a joint model since slice 3c/2 — what still
        # refuses is the drivers, which schedule no displacement pass yet and would
        # otherwise sample a *different* ensemble (u frozen at the clamped-ion state)
        # without saying so.
        @test_throws ArgumentError run_mc(H; kT = 0.1, sweeps_therm = 1,
                                          sweeps_measure = 2, nbins = 2)
        @test_throws ArgumentError run_pt(H; kT = [0.1, 0.2], sweeps_therm = 1,
                                          sweeps_measure = 2, nbins = 2)
        # `resume` reaches `_read_chain` directly, bypassing the driver guards, and
        # schema v2 stores no displacement state — so the reader refuses on its own.
        # The guard is its first statement, hence the unopened file argument.
        @test_throws ArgumentError MC._read_chain(nothing, "chain", H)
        # …and the writer refuses too, so a run that could never be resumed fails
        # while it is still cheap rather than at the resume attempt
        stj = MC.ChainState(H, _rand_config(rng, H), Xoshiro(1), 0.5;
                            disps = _rand_disps(rng, H))
        @test_throws ArgumentError MC._write_chain(nothing, "chain", stj)
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

        # A LYING CHANNEL FLAG. `spin` and `row0` are independent fields of the public
        # `TermSlot`, and the two partitions of the row table must agree: the activity
        # predicates and the one-axis-per-(site, channel) rule go by the FLAG, while
        # the range-limited ΔE goes by the BLOCK. Two slots on one site flagged
        # (spin, disp) but BOTH sitting in the displacement block pass the channel-count
        # rule, yet a displacement move moves both rows and the ΔE drops the quadratic
        # remainder (measured 0.25 on |E| ≈ 1).
        z3 = SVector(0, 0, 0)
        L3 = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])   # spin 1:4, displacement 5:7
        liar = [MC.ScaledTerm(0.7, [1], [z3],
                              [MC.TermSlot(1, 4, 1, true), MC.TermSlot(1, 4, 1, false)],
                              randn(3, 3))]
        @test_throws ArgumentError MC.TiledHamiltonian(1, liar, L3;
                                                       fixed_reference = true)
        # …and the mirror image: a displacement-flagged slot inside the SPIN block
        liar2 = [MC.ScaledTerm(0.7, [1], [z3], [MC.TermSlot(1, 0, 1, false)], randn(3))]
        @test_throws ArgumentError MC.TiledHamiltonian(1, liar2, L3;
                                                       fixed_reference = true)
        # the honest versions of both are accepted
        ok2 = [MC.ScaledTerm(0.7, [1], [z3],
                             [MC.TermSlot(1, 0, 1, true), MC.TermSlot(1, 4, 1, false)],
                             randn(3, 3))]
        @test MC.TiledHamiltonian(1, ok2, L3; fixed_reference = true) isa
              MC.TiledHamiltonian
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
        # hand-built and not translation-invariant; this test is about the activity
        # predicates, not the flat direction
        Hm = MC.TiledHamiltonian(2, mixed, L2; fixed_reference = true)
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
        HA = MC.TiledHamiltonian(2, mk(0), SLCE.RowLayout(7, 1, 4, [(0, 1)], [4]);
                                 fixed_reference = true)
        HB = MC.TiledHamiltonian(2, mk(1), SLCE.RowLayout(7, 1, 4, [(1, 1)], [4]);
                                 fixed_reference = true)
        @test MC.model_fingerprint(HA) != MC.model_fingerprint(HB)
        cfg = MC.SpinConfig([SVector(0.0, 0.0, 1.0), SVector(1.0, 0.0, 0.0)])
        u = [SVector(0.1, -0.2, 0.3), SVector(-0.05, 0.15, 0.2)]
        @test total_energy(HA, cfg, u) != total_energy(HB, cfg, u)
    end

    @testset "the uniform-shift direction must be flat, or be declared physical" begin
        # The displacement sampler re-centres along the uniform direction and the fit's
        # trust region is centre-of-mass-free, so a model whose rigid shift costs energy
        # is refused — measured on the Hamiltonian itself, not inferred from the fit.
        @test H.translation_invariant
        @test H.translation_residual < 1e-10
        @test SLCE.asr_residual(model) < 1e-10
        # the components: with no inter-cell displacement coupling in this fixture, each
        # supercell cell's dimer is its OWN flat direction — a single global centre of
        # mass would leave the other seven drifting
        @test H.n_disp_comps == 1
        H8 = MC.TiledHamiltonian(model; dims = (2, 2, 2))
        @test H8.n_disp_comps == 8
        @test H8.translation_residual < 1e-10
        @test sum(H8.disp_comp_ptr[c + 1] - H8.disp_comp_ptr[c]
                  for c = 1:H8.n_disp_comps) == H8.n_disp_active
        @test issorted(H8.disp_comp_sites[1:2])            # ascending within a component

        # an unconstrained fit of the SAME basis is refused, and the message names both
        # ways forward
        bad, _ = _joint_model(5; asr = false)
        @test SLCE.asr_residual(bad) > 1e-3
        err = try
            MC.TiledHamiltonian(bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("asr = true", sprint(showerror, err))
        @test occursin("fixed_reference = true", sprint(showerror, err))
        # …and accepted when the absolute frame is declared physical
        Hf = MC.TiledHamiltonian(bad; fixed_reference = true)
        @test !Hf.translation_invariant && Hf.translation_residual > 1e-3
        # a pure-spin model has no uniform direction to be flat, and is never probed
        @test TiledHamiltonian(_dimer_model()).translation_residual == 0.0
        @test TiledHamiltonian(_dimer_model()).translation_invariant
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

# The displacement sampler — M4 slice 3c/2.
#
# Three claims:
#   1. the displacement move's incremental energy is EXACT (it is the same
#      leave-one-out contraction the spin move uses, restricted to the site's
#      displacement rows), and its row bookkeeping reproduces a from-scratch fill;
#   2. it is bit-deterministic for any `sweep_tasks`, like every other sweep (P6);
#   3. re-centring is a projection along an exactly flat direction — it removes each
#      displacement-coupling component's mean without moving the energy, and it is a
#      no-op on a model whose absolute frame is declared physical.
@testset "joint displacement sampling" begin
    model, _ = _joint_model()
    H = MC.TiledHamiltonian(model; dims = (2, 2, 2))
    rng = MersenneTwister(11)

    # Per-component mean displacement — what `_recenter!` is supposed to zero. A
    # GLOBAL mean would leave these eight numbers individually nonzero.
    comp_means(H, disps) =
        [sum(disps[Int(H.disp_comp_sites[q])]
             for q = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)) /
         (H.disp_comp_ptr[c + 1] - H.disp_comp_ptr[c]) for c = 1:H.n_disp_comps]

    @testset "the chain carries displacements" begin
        cfg = _rand_config(rng, H)
        st = MC.ChainState(H, cfg, Xoshiro(7), 0.5; step_u = 0.03)
        @test length(st.disps) == H.n_sites
        @test all(iszero, st.disps)                     # clamped-ion start
        @test length(st.com_removed) == H.n_disp_comps == 8
        @test all(iszero, st.com_removed)
        @test st.step_u == 0.03
        # the cached rows and the incremental energy are the from-scratch ones
        @test st.zrows == MC._zrows(H, cfg, st.disps)
        @test st.energy === total_energy(H, cfg, st.disps)
        @test_throws ArgumentError MC.ChainState(H, cfg, Xoshiro(7), 0.5; step_u = 0)
        # an explicit start, in either I/O layout
        u0 = _rand_disps(rng, H)
        @test MC.ChainState(H, cfg, Xoshiro(7), 0.5; disps = u0).disps == u0
        @test MC.ChainState(H, cfg, Xoshiro(7), 0.5;
                            disps = _disp_matrix(u0)).disps == u0
        @test_throws DimensionMismatch MC.ChainState(H, cfg, Xoshiro(7), 0.5;
                                                     disps = u0[1:3])
        # a pure-spin chain refuses a nonzero start rather than sampling u = 0
        Hs = TiledHamiltonian(_dimer_model())
        cs = _rand_config(rng, Hs)
        @test all(iszero, MC.ChainState(Hs, cs, Xoshiro(7), 0.5).disps)
        @test isempty(MC.ChainState(Hs, cs, Xoshiro(7), 0.5).com_removed)
        @test_throws ArgumentError MC.ChainState(Hs, cs, Xoshiro(7), 0.5;
                                                 disps = _rand_disps(rng, Hs))
    end

    @testset "a displacement move's ΔE is exact" begin
        st = MC.ChainState(H, _rand_config(rng, H), Xoshiro(7), 0.5; step_u = 0.03)
        sc = MC.SweepScratch(H)
        for _ = 1:40
            MC.displacement_sweep!(st, H, 2.0, sc)
        end
        @test st.acc_disp > 0 && any(!iszero, st.disps)
        @test st.att_disp == 40 * H.n_disp_active
        # 40 sweeps of purely incremental bookkeeping, against a full recomputation
        E = total_energy(H, st.config, st.disps)
        @test isapprox(st.energy, E; rtol = 1e-12)
        # and the cached rows are still the from-scratch ones, bitwise
        @test st.zrows == MC._zrows(H, st.config, st.disps)

        # one attempt in isolation: the staged ΔE IS the total-energy difference
        s = findfirst(H.site_has_disp)
        for trial = 1:12
            E0 = total_energy(H, st.config, st.disps)
            fill!(sc.dE, 0.0)
            MC._attempt_disp!(st.disps, st.zrows, H, 2.0, 0.05, s, sc,
                              st.site_rngs[s], sc.dE)
            E1 = total_energy(H, st.config, st.disps)
            @test isapprox(E1 - E0, sc.dE[s]; atol = 1e-12 * max(1.0, abs(E0)))
        end
        @test st.zrows == MC._zrows(H, st.config, st.disps)

        # the spin move is unaffected by the widened row table: its ΔE still ignores
        # every displacement row, so a joint chain's spin sweep stays exact too
        stm = MC.ChainState(H, _rand_config(rng, H), Xoshiro(3), 0.5; step_u = 0.03)
        scm = MC.SweepScratch(H)
        for _ = 1:10
            MC.displacement_sweep!(stm, H, 2.0, scm)
            MC.metropolis_sweep!(stm, H, 2.0, scm)
        end
        @test isapprox(stm.energy, total_energy(H, stm.config, stm.disps);
                       rtol = 1e-12)
        @test stm.zrows == MC._zrows(H, stm.config, stm.disps)
    end

    @testset "displacement sweeps are bit-deterministic across tasks" begin
        cfg = _rand_config(rng, H)
        function chain(ntasks)
            st = MC.ChainState(H, copy(cfg), Xoshiro(99), 0.5; step_u = 0.04)
            scs = [MC.SweepScratch(H) for _ = 1:ntasks]
            for _ = 1:6
                MC.displacement_sweep!(st, H, 2.0, scs)
                MC.metropolis_sweep!(st, H, 2.0, scs)
            end
            MC._renormalize!(st, H, scs[1])
            return st
        end
        ref = chain(1)
        for ntasks in (2, 3, 5)
            got = chain(ntasks)
            @test got.disps == ref.disps
            @test got.config == ref.config
            @test got.zrows == ref.zrows
            @test got.energy === ref.energy
            @test got.com_removed == ref.com_removed
            @test (got.acc_disp, got.att_disp) == (ref.acc_disp, ref.att_disp)
        end
    end

    @testset "re-centring projects along the flat direction" begin
        st = MC.ChainState(H, _rand_config(rng, H), Xoshiro(5), 0.5; step_u = 0.04)
        sc = MC.SweepScratch(H)
        for _ = 1:40
            MC.displacement_sweep!(st, H, 2.0, sc)
        end
        before = comp_means(H, st.disps)
        # the walk really has drifted — otherwise the test below is vacuous
        @test maximum(norm, before) > 1e-3
        E0 = total_energy(H, st.config, st.disps)
        snapshot = copy(st.disps)
        MC._renormalize!(st, H, sc)
        after = comp_means(H, st.disps)
        # EVERY component's mean is gone, not just the global one
        @test maximum(norm, after) < 1e-14
        # the removed shift is recorded, and it is exactly what was subtracted
        for c = 1:H.n_disp_comps, q = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)
            s = Int(H.disp_comp_sites[q])
            @test st.disps[s] ≈ snapshot[s] - before[c] atol = 1e-15
        end
        @test all(c -> isapprox(st.com_removed[c], before[c]; atol = 1e-15),
                  1:H.n_disp_comps)
        # the direction is flat, so the energy did not move
        @test isapprox(st.energy, E0; rtol = 1e-10)
        # …and a second pass has nothing left to remove
        com1 = copy(st.com_removed)
        MC._recenter!(st, H)
        @test maximum(norm, comp_means(H, st.disps)) < 1e-14
        @test all(c -> norm(st.com_removed[c] - com1[c]) < 1e-15, 1:H.n_disp_comps)
    end

    @testset "a physical absolute frame is left alone" begin
        # site 1 carries the spin axis, site 2 the displacement one: disp-inactive
        # sites must stay bitwise frozen, and `fixed_reference` disables re-centring.
        z3 = SVector(0, 0, 0)
        mixed = [DecoratedTerm(0.2, 4π, 2, [1, 2], [z3, z3],
                               [SLCE.SlotRef(1, SLCE.SiteFactor(SLCE.SPIN, 0, 1)),
                                SLCE.SlotRef(2, SLCE.SiteFactor(SLCE.DISP, 0, 1))],
                               randn(MersenneTwister(2), 3, 3))]
        L2 = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])
        Hm = MC.TiledHamiltonian(2, mixed, L2; dims = (2, 1, 1),
                                 fixed_reference = true)
        @test !Hm.translation_invariant
        @test Hm.n_disp_active == 2 && count(Hm.site_has_disp) == 2
        st = MC.ChainState(Hm, _rand_config(MersenneTwister(21), Hm), Xoshiro(8), 0.5; step_u = 0.05)
        sc = MC.SweepScratch(Hm)
        for _ = 1:25
            MC.displacement_sweep!(st, Hm, 2.0, sc)
        end
        @test st.att_disp == 25 * 2
        # the spin-only sites never moved
        for s = 1:Hm.n_sites
            Hm.site_has_disp[s] && continue
            @test iszero(st.disps[s])
        end
        # re-centring is skipped: the mean survives a renormalization untouched
        m0 = comp_means(Hm, st.disps)
        @test maximum(norm, m0) > 1e-3
        MC._renormalize!(st, Hm, sc)
        @test comp_means(Hm, st.disps) == m0
        @test all(iszero, st.com_removed)
        @test isapprox(st.energy, total_energy(Hm, st.config, st.disps); rtol = 1e-12)
    end

    @testset "a pure-spin chain is untouched by the new machinery" begin
        Hs = TiledHamiltonian(_biquadratic_model(0); dims = (2, 2, 1))
        cfg = _rand_config(rng, Hs)
        @test Hs.n_disp_active == 0 && Hs.n_disp_comps == 0
        # the sweep is a no-op that consumes NO randomness: interposing it cannot
        # change what the following Metropolis sweep does
        function chain(with_disp::Bool)
            st = MC.ChainState(Hs, copy(cfg), Xoshiro(21), 0.5)
            sc = MC.SweepScratch(Hs)
            for _ = 1:4
                with_disp && @test MC.displacement_sweep!(st, Hs, 3.0, sc) == 0
                MC.metropolis_sweep!(st, Hs, 3.0, sc)
            end
            return st
        end
        a, b = chain(false), chain(true)
        @test a.config == b.config
        @test a.zrows == b.zrows
        @test a.energy === b.energy
        @test a.acc_metro == b.acc_metro
        @test b.att_disp == 0
        # the widened scratch is still exactly the spin block on this model
        sc = MC.SweepScratch(Hs)
        @test length(sc.c) == length(sc.znew) == Hs.nrows == Hs.nlm
        @test isempty(sc.rbuf)
    end
end

# The displacement sampler's two external gates — M4 slice 3c/2.
#
# Everything else in the displacement suite compares the sampler with itself. These two
# do not:
#   1. an Einstein oscillator has `⟨|u|²⟩ = 3kT/(2a)` in closed form, so the sampler is
#      checked against an analytic Gaussian rather than against its own bookkeeping;
#   2. the truncated cluster expansion is a finite polynomial in `u`, so `exp(−βE)` need
#      not be normalizable at all — and when it is not, the chain simply runs downhill
#      with a perfectly exact ΔE (drift 1e-14) and an acceptance near 1. The escape
#      detector is the only thing in the package that measures RECURRENCE, and the two
#      testsets together pin both of its error rates: it must fire on the unbounded
#      model and stay silent on the equilibrated one.
@testset "displacement sampler: external gates" begin
    @testset "an Einstein oscillator reproduces its analytic Gaussian" begin
        a = 2.5
        terms, L = _einstein_terms(a)
        H = MC.TiledHamiltonian(1, terms, L; dims = (2, 2, 2), fixed_reference = true)
        # each atom is pinned in its own well: one component per site, none of them flat
        @test H.n_disp_comps == H.n_sites == 8
        @test !any(H.comp_free) && !H.translation_invariant
        @test size(H.comp_residual) == (3, H.n_disp_comps)
        @test H.translation_residual == maximum(H.comp_residual)
        # the energy really is a|u|², bitwise
        u = [SVector(0.3, -0.2, 0.5) for _ = 1:H.n_sites]
        cfg = MC.SpinConfig([SVector(0.0, 0.0, 1.0) for _ = 1:H.n_sites])
        @test total_energy(H, cfg, u) ≈ H.n_sites * a * 0.38 rtol = 1e-14

        kT = 0.5
        st = MC.ChainState(H, cfg, Xoshiro(4), 0.5; step_u = 0.3)
        sc = MC.SweepScratch(H)
        acc = Float64[]
        for sw = 1:12_000
            MC.displacement_sweep!(st, H, 1 / kT, sc)
            sw % 100 == 0 && MC._renormalize!(st, H, sc)
            sw > 2_000 && sw % 10 == 0 &&
                push!(acc, sum(dot(x, x) for x in st.disps) / H.n_sites)
        end
        # the gate: an external, analytic target
        @test isapprox(sum(acc) / length(acc), 3kT / (2a); rtol = 0.03)
        # a well-tuned chain, not one that accepts everything (which is what the
        # escaping fixture does) — the proposal width is doing real work here
        @test 0.3 < st.acc_disp / st.att_disp < 0.7
        # …and re-centring stayed off, because every component's frame is physical
        @test all(iszero, st.com_removed)
        # NO FALSE POSITIVE: an equilibrated chain must not trip the escape detector,
        # over a long run with many checks
        @test st.disp_checks == 120
        @test st.escape_strikes == 0
        @test st.disp_rms > 0
    end

    @testset "the escape detector fires on an unbounded model" begin
        # `_joint_model()` has max displacement degree 2 (even) and one imaginary
        # branch, so `Z = ∞` exactly: no barrier, no metastable basin. The chain runs
        # downhill while the ΔE bookkeeping stays exact and the acceptance sits near 1 —
        # every pre-existing diagnostic is silent by construction.
        model, _ = _joint_model()
        He = MC.TiledHamiltonian(model; dims = (2, 2, 2))
        st = MC.ChainState(He, _rand_config(MersenneTwister(12), He), Xoshiro(12), 0.5;
                           step_u = 0.02)
        sc = MC.SweepScratch(He)
        run! = () -> for sw = 1:4_000
            MC.displacement_sweep!(st, He, 2.0, sc)
            sw % 100 == 0 && MC._renormalize!(st, He, sc)
        end
        @test_logs (:warn, r"r\.m\.s\. displacement has grown") match_mode = :any run!()
        # the verdict, not the route: on a fast escape the absolute guard (an
        # order-of-magnitude growth since the start of the phase) fires before the
        # block-to-block strike test can accumulate its three strikes
        @test st.escape_warned
        @test st.disp_rms > MC._ESCAPE_ABSOLUTE * st.disp_rms0
        @test st.escape_strikes > 0            # the block test sees it too
        # the run really did escape — the detector is not firing on noise
        @test st.disp_max > 10.0
        # and the incremental energy stayed exact the whole way, which is exactly why
        # the drift warning cannot be the diagnostic for this
        @test st.max_drift < 1e-9
        @test st.acc_disp / st.att_disp > 0.8
    end

    @testset "a mixed model re-centres only its flat components" begin
        # Atoms 1–2 carry the joint fixture's translation-invariant terms; atom 3 is an
        # Einstein oscillator sharing the fixture's own `(k, l) = (1, 0)` row block. The
        # flatness verdict must therefore be per component — a single global `false`
        # would let atoms 1–2's frame drift merely because atom 3 is pinned.
        model, _ = _joint_model()
        Hj = MC.TiledHamiltonian(model)
        L = Hj.layout
        i10 = findfirst(==((1, 0)), L.disp_factors)
        @test i10 !== nothing
        pin = MC.ScaledTerm(2.5, [3], [SVector(0, 0, 0)],
                            [MC.TermSlot(1, L.disp_starts[i10], 0, false)], [1.0])
        Hm = MC.TiledHamiltonian(3, vcat(Hj.terms, [pin]), L; dims = (2, 1, 1),
                                 fixed_reference = true)
        @test !Hm.translation_invariant                 # atom 3 is pinned
        @test any(Hm.comp_free)                         # …but not everything is
        @test !all(Hm.comp_free)
        freec = [all(view(Hm.comp_free, :, c)) for c = 1:Hm.n_disp_comps]
        pinned = findall(!, freec)
        flat = findall(freec)
        @test !isempty(pinned) && !isempty(flat)
        st = MC.ChainState(Hm, _rand_config(MersenneTwister(21), Hm), Xoshiro(6), 0.5; step_u = 0.03)
        sc = MC.SweepScratch(Hm)
        for _ = 1:60
            MC.displacement_sweep!(st, Hm, 2.0, sc)
        end
        mean_of(c) = sum(st.disps[Int(Hm.disp_comp_sites[q])]
                         for q = Hm.disp_comp_ptr[c]:(Hm.disp_comp_ptr[c + 1] - 1)) /
                     (Hm.disp_comp_ptr[c + 1] - Hm.disp_comp_ptr[c])
        before = [norm(mean_of(c)) for c = 1:Hm.n_disp_comps]
        @test maximum(before[flat]) > 1e-4              # the flat frames did drift
        MC._renormalize!(st, Hm, sc)
        after = [norm(mean_of(c)) for c = 1:Hm.n_disp_comps]
        @test maximum(after[flat]) < 1e-14              # …and were re-centred
        @test after[pinned] == before[pinned]           # the pinned ones, untouched
        @test all(iszero, st.com_removed[pinned])
    end

    @testset "a component pinned along one axis stays free in the other two" begin
        # The substrate-clamped slab. A 1-body `l = 1` displacement axis makes the energy
        # LINEAR in one Cartesian direction and independent of the other two, so the
        # component's flat directions are a proper 2-D subspace. A single-probe-direction
        # verdict would mark the whole component pinned, `_recenter!` would skip it
        # entirely, and the in-plane centre of mass would then random-walk without bound
        # — below the escape detector's threshold by construction, so nothing would catch
        # it. Deliberately the linear term ALONE: adding an isotropic `|u|²` well would
        # pin all three directions and destroy the very case under test.
        L = SLCE.RowLayout(4, 0, 1, [(0, 1)], [1])
        for slot = 1:3
            fold = zeros(3)
            fold[slot] = 1.0
            lin = MC.ScaledTerm(1.0, [1], [SVector(0, 0, 0)],
                                [MC.TermSlot(1, 1, 1, false)], fold)
            H = MC.TiledHamiltonian(1, [lin], L; dims = (2, 1, 1),
                                    fixed_reference = true)
            # exactly one direction pinned, two free — for every choice of tesseral slot
            @test count(!, view(H.comp_free, :, 1)) == 1
            @test count(view(H.comp_free, :, 1)) == 2
            @test !H.translation_invariant
            # …and the free directions really are re-centred while the pinned one is not
            st = MC.ChainState(H, MC.SpinConfig([SVector(0.0, 0.0, 1.0)
                                                 for _ = 1:H.n_sites]),
                               Xoshiro(9), 0.5; step_u = 0.05)
            sc = MC.SweepScratch(H)
            for _ = 1:80
                MC.displacement_sweep!(st, H, 2.0, sc)
            end
            ū = sum(st.disps) / length(st.disps)
            @test norm(ū) > 1e-4                       # it did drift
            MC._renormalize!(st, H, sc)
            after = sum(st.disps) / length(st.disps)
            for d = 1:3
                if H.comp_free[d, 1]
                    @test abs(after[d]) < 1e-14        # projected out
                else
                    @test after[d] == ū[d]             # left alone: the frame is physical
                end
            end
        end
    end

    @testset "a flat one-site component is refused as pure gauge" begin
        # A `(k, l) = (0, 0)` displacement axis has factor `|u|^0 R_{0,0}(u) ≡ 1`, so the
        # energy does not depend on that site's displacement at all: sampling it would
        # spend randomness on always-accepted moves and write a meaningless displacement
        # into every reported configuration.
        z3 = SVector(0, 0, 0)
        L = SLCE.RowLayout(5, 1, 4, [(0, 0)], [4])
        gauge = [MC.ScaledTerm(0.3, [1, 2], [z3, z3],
                               [MC.TermSlot(1, 0, 1, true), MC.TermSlot(2, 4, 0, false)],
                               reshape([0.4, -0.2, 0.7], 3, 1))]
        @test_throws ArgumentError MC.TiledHamiltonian(2, gauge, L;
                                                       fixed_reference = true)
    end
end
