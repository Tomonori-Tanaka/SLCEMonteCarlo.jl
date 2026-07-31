# GPU-path gates, all on the KernelAbstractions CPU backend (CI needs no GPU).
# Decision record: docs/specs/gpu-prototype.md — G2 (keyed Philox layout), G3
# (determinism contract: kernel ≡ keyed reference bitwise on one backend), G4
# (kernel shape, bitwise device zlm), G5 (this gate list).

using KernelAbstractions: KernelAbstractions, CPU, @kernel, @index

# Test-local KA kernel exercising the device zlm row through the backend compiler
# (the direct host call below is the sharper bitwise gate; this one pins that the
# same code also runs AS a kernel).
@kernel function _test_zlm_kernel!(out, dirs, ::Val{LMAX}) where {LMAX}
    i = @index(Global, Linear)
    @inbounds MC._zlm_row_device!(view(out, :, i), dirs[i], Val(LMAX))
end

@kernel function _test_grad_kernel!(out, dirs, ::Val{LMAX}) where {LMAX}
    i = @index(Global, Linear)
    @inbounds MC._grad_zlm_row_device!(view(out, :, i), dirs[i], Val(LMAX))
end

@kernel function _test_solid_kernel!(out, us, ::Val{LMAX}) where {LMAX}
    i = @index(Global, Linear)
    @inbounds MC._solid_row_device!(view(out, :, i), us[i], Val(LMAX))
end

@kernel function _test_disprow_kernel!(out, buf, us, @Const(fk), @Const(fl),
                                       @Const(fs), ::Val{DLMAX}) where {DLMAX}
    i = @index(Global, Linear)
    @inbounds MC._disp_rows_device!(view(out, :, i), view(buf, :, i), us[i], fk, fl,
                                    fs, Val(DLMAX))
end

# Fresh (H, ChainState, GPU pair) on the CPU backend with a seeded random config —
# and, on a joint Hamiltonian, seeded random displacements (`with_disps`; the
# clamped-ion start would leave every `|u|^{2k}` factor at 0 and hide the channel).
function _gpu_setup(H; seed_cfg = 7, seed_dev = UInt64(0xc0ffee), step = 0.6,
                    with_disps = false)
    rng = Xoshiro(seed_cfg)
    cfg = _rand_config(rng, H)
    st = with_disps ? MC.ChainState(H, cfg, rng, step; disps = _rand_disps(rng, H)) :
         MC.ChainState(H, cfg, rng, step)
    gH = MC.GPUTiledHamiltonian(CPU(), H)
    gst = MC.GPUChainState(gH, st; seed = seed_dev)
    return st, gH, gst
end

# The joint Hamiltonians the M4 device path is gated on: the hand-built channel split
# (atom 1 in both channels, atom 2 displacement-only — the only fixture where the two
# sweep schedules differ) and a fitted joint model where every site carries both.
function _joint_gpu_cases()
    tms, L = _channel_split_terms()
    return [("channel split", MC.TiledHamiltonian(2, tms, L; dims = (4, 2, 1),
                                                  fixed_reference = true)),
            ("fitted joint", TiledHamiltonian(_joint_model(5)[1]; dims = (2, 2, 2)))]
end

# Reference-side sweep loop: accumulates the energy in the driver's exact
# association order (`E0 += reduce` per sweep) so the comparison is bitwise.
function _ref_sweeps!(cfg, zr, dE, acc, H, β, step, seed, E0, nsweeps, ws)
    E = E0
    naccs = Int[]
    for sw = 1:nsweeps
        push!(naccs, MC._metropolis_sweep_keyed_ref!(cfg, zr, dE, acc, H, β, step,
                                                     seed, Int32(sw), ws))
        E += MC._reduce_dE(H, dE)
    end
    return E, naccs
end

@testset "gpu: philox4x32-10" begin
    # Random123 kat_vectors known answers (tests/kat_vectors, philox4x32 10).
    @test MC._philox4x32((0x00000000, 0x00000000, 0x00000000, 0x00000000),
                         (0x00000000, 0x00000000)) ==
          (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)
    @test MC._philox4x32((0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),
                         (0xffffffff, 0xffffffff)) ==
          (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)
    @test MC._philox4x32((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344),
                         (0xa4093822, 0x299f31d0)) ==
          (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)

    # public facade ≡ internals (the dependent-package contract)
    seed = 0x299f31d0a4093822
    ctr = (0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344)
    @test MC.philox_block(seed, ctr) ==
          MC._philox4x32(ctr, (seed % UInt32, (seed >>> 32) % UInt32))
    @test MC.philox_normal2(MC.philox_block(seed, ctr)) ==
          MC._philox_normal2(MC.philox_block(seed, ctr))

    # uniform bit convention: strictly open (0, 1) even on the edge words
    @test 0.0 < MC._philox_uniform(0x00000000, 0x00000000)
    @test MC._philox_uniform(0xffffffff, 0xffffffff) < 1.0

    # keyed streams: any coordinate change changes the block
    blk = MC._philox_block(UInt64(1), Int32(3), Int32(5), UInt32(0))
    @test blk != MC._philox_block(UInt64(2), Int32(3), Int32(5), UInt32(0))
    @test blk != MC._philox_block(UInt64(1), Int32(4), Int32(5), UInt32(0))
    @test blk != MC._philox_block(UInt64(1), Int32(3), Int32(6), UInt32(0))
    @test blk != MC._philox_block(UInt64(1), Int32(3), Int32(5), UInt32(1))
    @test blk == MC._philox_block(UInt64(1), Int32(3), Int32(5), UInt32(0))

    # normal pairs are finite (open-interval uniforms keep log() off the edge)
    n1, n2 = MC._philox_normal2(blk)
    @test isfinite(n1) && isfinite(n2)
end

@testset "gpu: device zlm row ≡ host _zlm_row! (bitwise)" begin
    rng = Xoshiro(2026)
    dirs = SVector{3,Float64}[SVector(0, 0, 1.0), SVector(0, 0, -1.0),
                              SVector(1, 0, 0.0), SVector(-1, 0, 0.0),
                              SVector(0, 1, 0.0), SVector(0, -1, 0.0)]
    for k = 0:11                              # equatorial ring
        push!(dirs, SVector(cos(k * π / 6), sin(k * π / 6), 0.0))
    end
    for _ = 1:2000
        push!(dirs, _rand_spin(rng))
    end

    # complex integer power replica (the Base.power_by_squaring value path;
    # n = 0 is the gradient row's `zxy^(n−1)` at n = 1 — must be exactly one)
    for _ = 1:2000
        z = ComplexF64(randn(rng), randn(rng))
        for n = 0:6
            @test MC._zlm_cpow(z, n) === z^n
        end
    end
    @test MC._zlm_cpow(ComplexF64(0.0, 0.0), 0) === ComplexF64(1.0, 0.0)

    for lmax = 0:6
        nlm = (lmax + 1)^2
        zh = zeros(nlm)
        plm = Vector{Float64}(undef, lmax + 1)
        zd = zeros(nlm)
        ok_direct = true
        for u in dirs
            MC._zlm_row!(zh, u, lmax, plm)
            MC._zlm_row_device_dyn!(zd, u, lmax)
            ok_direct &= zd == zh
        end
        @test ok_direct

        # …and through an actual KA-CPU kernel
        out = zeros(nlm, length(dirs))
        kern = _test_zlm_kernel!(CPU())
        kern(out, dirs, Val(lmax); ndrange = length(dirs))
        KernelAbstractions.synchronize(CPU())
        ok_kernel = true
        for (i, u) in enumerate(dirs)
            MC._zlm_row!(zh, u, lmax, plm)
            ok_kernel &= view(out, :, i) == zh
        end
        @test ok_kernel
    end
end

@testset "gpu: device solid-harmonic row ≡ SolidHarmonics (bitwise)" begin
    # Displacements, NOT directions: the solid harmonics are homogeneous polynomials
    # with no normalization, so magnitude is part of the input space. u = 0 is in the
    # set because that is where the whole polynomial form earns its keep (a `Z_lm`-style
    # kernel would divide by |u| there).
    rng = Xoshiro(4242)
    us = SVector{3,Float64}[SVector(0.0, 0.0, 0.0),
                            SVector(0.0, 0.0, 0.3), SVector(0.0, 0.0, -0.3),
                            SVector(0.25, 0.0, 0.0), SVector(0.0, -0.25, 0.0),
                            SVector(1e-9, 0.0, 0.0), SVector(0.0, 0.0, 12.0)]
    for _ = 1:2000
        amp = exp(4 * randn(rng) - 2)          # magnitudes over ~8 decades
        push!(us, amp .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
    end

    for lmax = 0:6
        n = (lmax + 1)^2
        ref = zeros(n)
        got = zeros(n)
        ok_direct = true
        for u in us
            SLCE.SolidHarmonics.solid_harmonics!(ref, lmax, u)
            MC._solid_row_device_dyn!(got, u, lmax)
            ok_direct &= got == ref
        end
        @test ok_direct

        # …and through an actual KA-CPU kernel
        out = zeros(n, length(us))
        kern = _test_solid_kernel!(CPU())
        kern(out, us, Val(lmax); ndrange = length(us))
        KernelAbstractions.synchronize(CPU())
        ok_kernel = true
        for (i, u) in enumerate(us)
            SLCE.SolidHarmonics.solid_harmonics!(ref, lmax, u)
            ok_kernel &= view(out, :, i) == ref
        end
        @test ok_kernel
    end
    @test_throws ArgumentError MC._solid_row_device_dyn!(zeros(64), us[2], 7)

    # u = 0: R_00 = 1 and every l ≥ 1 row exactly zero (the polynomial form's point)
    z0 = zeros(9)
    MC._solid_row_device_dyn!(z0, SVector(0.0, 0.0, 0.0), 2)
    @test z0[1] == 1.0 && all(iszero, view(z0, 2:9))
end

@testset "gpu: device displacement rows ≡ host _disp_rows! (bitwise)" begin
    # Three layouts spanning the k/l block structure: the k = 0 dipole alone, the
    # `pmax = 2` set (a radial trace channel next to l = 1, 2), and a k = 2 block —
    # the one whose `r2^k` prefactor is a genuine libm `pow` (the determinism-scope
    # boundary documented in disp_device.jl).
    layouts = [SLCE.RowLayout(7, 1, 4, [(0, 1)], [4]),
               SLCE.RowLayout(13, 1, 4, [(0, 1), (0, 2), (1, 0)], [4, 7, 12]),
               SLCE.RowLayout(13, 1, 4, [(0, 1), (1, 2), (2, 0)], [4, 7, 12])]
    rng = Xoshiro(909)
    us = SVector{3,Float64}[SVector(0.0, 0.0, 0.0), SVector(0.4, 0.0, 0.0)]
    for _ = 1:500
        push!(us, 0.5 .* SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
    end

    for L in layouts
        dlmax = maximum(l for (_, l) in L.disp_factors)
        fk, fl, fs = MC._disp_layout_tables(L)
        @test fk == Int32[k for (k, _) in L.disp_factors]
        @test fl == Int32[l for (_, l) in L.disp_factors]
        @test fs == Int32.(L.disp_starts)
        # the host reference reads its layout off a TiledHamiltonian, so borrow one
        # whose layout is exactly `L` (a single well-formed term is enough — the row
        # filler only ever reads `H.layout` and `H.disp_lmax`)
        H = MC.TiledHamiltonian(1, [MC.ScaledTerm(1.0, [1], [SVector(0, 0, 0)],
                                                  [MC.TermSlot(1, Int(fs[1]),
                                                               Int(fl[1]), false)],
                                                  ones(2 * Int(fl[1]) + 1))], L;
                                fixed_reference = true)
        @test H.layout == L && H.disp_lmax == dlmax

        # Rows of a k = 0 block carry no `|u|^{2k}` prefactor and are bitwise; a
        # k ≥ 1 block's prefactor is `r2^k`, and `r2` is where the two sides can
        # disagree in the last ulp — see the DETERMINISM SCOPE note in
        # disp_device.jl: the host computes it in a function where the harmonic
        # batch is a separate call, the device inlines the batch, and LLVM's
        # FP-contraction/CSE decisions then differ. Measured on this machine:
        # ~22 % of random `u` differ by exactly 1 ulp in those rows.
        k0rows = Int[]
        for (i, (k, l)) in pairs(L.disp_factors), m = -l:l
            k == 0 && push!(k0rows, L.disp_starts[i] + m + l + 1)
        end
        ref = zeros(L.nrows)
        got = zeros(L.nrows)
        rbuf = zeros((dlmax + 1)^2)
        gbuf = zeros((dlmax + 1)^2)
        ok_exact = true
        worst = 0.0
        for u in us
            fill!(ref, 0.0)
            fill!(got, 0.0)
            MC._disp_rows!(ref, H, u, rbuf)
            MC._disp_rows_device_dyn!(got, gbuf, u, fk, fl, fs, dlmax)
            ok_exact &= view(got, k0rows) == view(ref, k0rows)
            for r = 1:L.nrows
                d = abs(got[r] - ref[r])
                d == 0.0 && continue
                worst = max(worst, d / eps(abs(ref[r])))
            end
        end
        @test ok_exact                          # every k = 0 row, bitwise
        # k ≥ 1 rows: one ulp in `r2`, amplified by the exponent (`r2^k` multiplies
        # the relative error by k), plus the product's own rounding
        kmax = maximum(k for (k, _) in L.disp_factors)
        @test worst <= kmax + 1
        @test any(!iszero, got)                 # not vacuously all-zero

        out = zeros(L.nrows, length(us))
        buf = zeros((dlmax + 1)^2, length(us))
        kern = _test_disprow_kernel!(CPU())
        kern(out, buf, us, fk, fl, fs, Val(dlmax); ndrange = length(us))
        KernelAbstractions.synchronize(CPU())
        ok_kernel = true
        for (i, u) in enumerate(us)
            fill!(ref, 0.0)
            MC._disp_rows!(ref, H, u, rbuf)
            ok_kernel &= view(out, k0rows, i) == view(ref, k0rows)
        end
        @test ok_kernel
    end

    # A pure-spin layout has no displacement blocks and the filler is a no-op.
    fk0, fl0, fs0 = MC._disp_layout_tables(MC._spin_row_layout(2))
    @test isempty(fk0) && isempty(fl0) && isempty(fs0)
    rows0 = fill(7.0, 9)
    MC._disp_rows_device_dyn!(rows0, zeros(1), SVector(0.1, 0.2, 0.3), fk0, fl0, fs0, 0)
    @test all(==(7.0), rows0)
end

@testset "gpu: direct-ΔE entry walk vs site_coeffs!+delta_energy" begin
    models = [("biquadratic l≤2", TiledHamiltonian(_biquadratic_model(3);
                                                   dims = (2, 2, 2))),
              ("3-body chain", MC.TiledHamiltonian(1, _threebody_terms(0.05);
                                                   dims = (4, 2, 2))),
              ("4-body chain", MC.TiledHamiltonian(1, _fourbody_terms(0.03);
                                                   dims = (4, 2, 2)))]
    # the 3-/4-body fixtures must exercise the triplet fast path and the
    # general (site_col == 0) branch respectively
    @test any(<(Int32(0)), models[2][2].progs.site_col)
    @test any(==(Int32(0)), models[3][2].progs.site_col)

    rng = Xoshiro(31)
    for (name, H) in models
        st = MC.ChainState(H, _rand_config(rng, H), rng, 0.6)
        tb = MC._host_tables(H)
        c = zeros(H.nlm)
        znew = zeros(H.nlm)
        plm = Vector{Float64}(undef, H.lmax + 1)
        for s in Int.(H.color_sites[1:min(end, 32)])
            MC._zlm_row!(znew, _rand_spin(rng), H.lmax, plm)
            fill!(c, 0.0)
            MC.site_coeffs!(c, H, s, st.zrows)
            ΔE_cpu = MC.delta_energy(c, view(st.zrows, :, s), znew)
            scale = sum(abs, c) * (sum(abs, znew) + sum(abs, view(st.zrows, :, s)))
            for ws in (4, 32)
                ΔE_walk = 0.0
                for lane = 1:ws
                    ΔE_walk += MC._entry_walk_partial(tb, st.zrows, znew, s, lane, ws,
                                                      1, H.nlm)
                end
                @test abs(ΔE_walk - ΔE_cpu) <= 1e-12 * max(scale, 1e-30)
            end
        end
    end
end

@testset "gpu: full sweep ≡ keyed reference (bitwise)" begin
    cases = [TiledHamiltonian(_dimer_model()),
             TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2)),
             TiledHamiltonian(_biquadratic_model(4); dims = (3, 2, 1)),
             MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (4, 2, 2)),
             MC.TiledHamiltonian(1, _fourbody_terms(0.03); dims = (4, 2, 2))]
    for H in cases, ws in (4, 32)
        st, gH, gst = _gpu_setup(H)
        β = 1 / 0.05
        cfg2 = copy(st.config)
        zr2 = copy(st.zrows)
        dE2 = zeros(H.n_sites)
        acc2 = zeros(Int32, H.n_sites)
        E0 = gst.energy
        naccs = Int[]
        for _ = 1:5
            push!(naccs, MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = ws))
        end
        E_ref, naccs_ref = _ref_sweeps!(cfg2, zr2, dE2, acc2, H, β, st.step,
                                        gst.seed, E0, 5, ws)
        MC.to_host!(st, gst)
        @test st.config == cfg2
        @test st.zrows == zr2
        @test gst.energy == E_ref
        @test naccs == naccs_ref
        @test gst.acc_metro == sum(naccs_ref)
        @test gst.att_metro == 5 * H.n_active
    end
end

@testset "gpu: sync_coefficients! re-uploads a host coefficient swap (bitwise)" begin
    # A host `set_coefficients!` mutates `H.progs.sent_w` in place; the device copy
    # is from construction time and goes stale silently. The sync is a `sent_w`-only
    # copy because every other table is coefficient-independent by design.
    m = _biquadratic_model(3)
    H = TiledHamiltonian(m; dims = (2, 2, 2))
    st, gH, _ = _gpu_setup(H)
    coefs = [1.17 * t.coef for t in SLCE.spin_multipole_terms(m)]
    MC.set_coefficients!(H, coefs)
    @test gH.dev.sent_w != gH.host.progs.sent_w      # stale, and detectably so
    @test MC.sync_coefficients!(gH) === gH
    @test gH.dev.sent_w == gH.host.progs.sent_w      # the one array a swap moves
    # ...and the synced device sweeps bitwise against the keyed reference, which
    # reads the host tables fresh and therefore carries the NEW weights
    β = 1 / 0.05
    gst = MC.GPUChainState(gH, st; seed = UInt64(0xc0ffee))
    cfg2 = copy(st.config)
    zr2 = copy(st.zrows)
    dE2 = zeros(H.n_sites)
    acc2 = zeros(Int32, H.n_sites)
    E0 = gst.energy
    naccs = [MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = 32) for _ = 1:3]
    E_ref, naccs_ref = _ref_sweeps!(cfg2, zr2, dE2, acc2, H, β, st.step, gst.seed,
                                    E0, 3, 32)
    MC.to_host!(st, gst)
    @test st.config == cfg2
    @test st.zrows == zr2
    @test gst.energy == E_ref
    @test naccs == naccs_ref

    # the joint fixture goes through `decorated_terms`; the same one-array contract
    mj, _ = _joint_model(5)
    Hj = TiledHamiltonian(mj; dims = (2, 2, 2))
    gHj = MC.GPUTiledHamiltonian(CPU(), Hj)
    MC.set_coefficients!(Hj, [0.9 * t.coef for t in SLCE.decorated_terms(mj)];
                         recheck_translation = false)
    @test gHj.dev.sent_w != gHj.host.progs.sent_w
    MC.sync_coefficients!(gHj)
    @test gHj.dev.sent_w == gHj.host.progs.sent_w
end

@testset "gpu: repeated-run identity" begin
    H = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
    β = 1 / 0.05
    st1, gH, gst1 = _gpu_setup(H)
    # both replicas (and the different-seed control) start from st1's INITIAL state
    gst2 = MC.GPUChainState(gH, st1; seed = gst1.seed)
    gst3 = MC.GPUChainState(gH, st1; seed = gst1.seed + 1)
    for _ = 1:10
        MC.gpu_metropolis_sweep!(gst1, gH, β)
        MC.gpu_metropolis_sweep!(gst2, gH, β)
        MC.gpu_metropolis_sweep!(gst3, gH, β)
    end
    st2 = MC.ChainState(H, copy(st1.config), Xoshiro(0), st1.step)
    st3 = MC.ChainState(H, copy(st1.config), Xoshiro(0), st1.step)
    MC.to_host!(st1, gst1)
    MC.to_host!(st2, gst2)
    MC.to_host!(st3, gst3)
    @test st1.config == st2.config
    @test st1.zrows == st2.zrows
    @test gst1.energy == gst2.energy
    @test gst1.acc_metro == gst2.acc_metro
    # a different seed must give a different trajectory
    @test st3.config != st1.config
end

@testset "gpu: inactive sites bitwise frozen" begin
    H = TiledHamiltonian(_dimer_model())          # atoms 3–4 have no instance
    @test H.n_active < H.n_sites
    st, gH, gst = _gpu_setup(H)
    frozen = [s for s = 1:H.n_sites if !H.site_active[s]]
    cfg0 = copy(st.config)
    zr0 = copy(st.zrows)
    MC.gpu_run_sweeps!(gst, gH, st, 1 / 0.05, 50; renorm_interval = 20)
    @test all(st.config[s] === cfg0[s] for s in frozen)
    @test all(view(st.zrows, :, s) == view(zr0, :, s) for s in frozen)
end

@testset "gpu: incremental-energy drift gate" begin
    H = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
    st, gH, gst = _gpu_setup(H)
    MC.gpu_run_sweeps!(gst, gH, st, 1 / 0.05, 200; renorm_interval = 0)
    E = total_energy(H, st.config)
    @test abs(gst.energy - E) <= 1e-8 * max(1.0, abs(E))
end

# --- the joint (spin + displacement) device path, G8 phase 2 --------------------

@testset "gpu: channel color schedules" begin
    # On a pure-spin Hamiltonian the spin schedule IS the coloring, verbatim — which is
    # what leaves every pre-M4 launch, and with it every bitwise gate above, unchanged.
    for H in (TiledHamiltonian(_dimer_model()),
              TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2)))
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        @test gH.spin_ptr == H.color_ptr
        @test gH.dev.spin_sites == H.color_sites
        @test isempty(gH.dev.disp_sites)
        @test all(==(Int32(1)), gH.disp_ptr)
        @test isempty(gH.dev.fac_k) && isempty(gH.dev.fac_start)
    end
    for (name, H) in _joint_gpu_cases()
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        spin = Int.(gH.dev.spin_sites)
        disp = Int.(gH.dev.disp_sites)
        @test spin == [s for s in Int.(H.color_sites) if H.site_has_spin[s]]
        @test disp == [s for s in Int.(H.color_sites) if H.site_has_disp[s]]
        @test length(spin) == H.n_spin_active
        @test length(disp) == H.n_disp_active
        @test Int(gH.spin_ptr[end]) == length(spin) + 1
        @test Int(gH.disp_ptr[end]) == length(disp) + 1
        for c = 1:H.n_colors                      # ascending within each class
            @test issorted(spin[Int(gH.spin_ptr[c]):(Int(gH.spin_ptr[c + 1]) - 1)])
            @test issorted(disp[Int(gH.disp_ptr[c]):(Int(gH.disp_ptr[c + 1]) - 1)])
        end
        @test Int.(gH.dev.fac_k) == [k for (k, _) in H.layout.disp_factors]
        @test Int.(gH.dev.fac_l) == [l for (_, l) in H.layout.disp_factors]
        @test Int.(gH.dev.fac_start) == H.layout.disp_starts
    end
    # Non-vacuity: without a Hamiltonian whose two schedules genuinely differ, the whole
    # split is untested — a sweep over the full coloring would pass everything above.
    Hsplit = _joint_gpu_cases()[1][2]
    @test Hsplit.n_spin_active < Hsplit.n_active
    @test Hsplit.n_disp_active == Hsplit.n_active
end

@testset "gpu: joint ΔE row range vs site_coeffs!+delta_energy" begin
    rng = Xoshiro(97)
    for (name, H) in _joint_gpu_cases()
        st = MC.ChainState(H, _rand_config(rng, H), rng, 0.6;
                           disps = _rand_disps(rng, H))
        tb = MC._host_tables(H)
        # Non-vacuity, part 1: a spin site's program must target displacement rows too,
        # or the row range never skips anything and this testset is trivial.
        outside = 0
        for s in Int.(tb.spin_sites), j = Int(H.site_ptr[s]):(Int(H.site_ptr[s + 1]) - 1)
            pid = Int(H.progs.site_prog[j])
            for e = Int(H.progs.sprog_ptr[pid]):(Int(H.progs.sprog_ptr[pid + 1]) - 1)
                Int(H.progs.sent_tgt[e]) > H.nlm && (outside += 1)
            end
        end
        @test outside > 0
        c = zeros(H.nrows)
        znew = zeros(H.nlm)
        plm = Vector{Float64}(undef, H.lmax + 1)
        nz_disp = 0
        for s in Int.(tb.spin_sites[1:min(end, 32)])
            MC._zlm_row!(znew, _rand_spin(rng), H.lmax, plm)
            fill!(c, 0.0)
            MC.site_coeffs!(c, H, s, st.zrows)
            # Non-vacuity, part 2: the skipped rows carry NONZERO coefficients, so an
            # unrestricted walk would give a different (and wrong) ΔE.
            any(!iszero, view(c, (H.nlm + 1):H.nrows)) && (nz_disp += 1)
            ΔE_cpu = MC.delta_energy(c, view(st.zrows, :, s), znew, 1, H.nlm)
            scale = sum(abs, c) * (sum(abs, znew) + sum(abs, view(st.zrows, :, s)))
            for ws in (4, 32)
                ΔE_walk = 0.0
                for lane = 1:ws
                    ΔE_walk += MC._entry_walk_partial(tb, st.zrows, znew, s, lane, ws,
                                                      1, H.nlm)
                end
                @test abs(ΔE_walk - ΔE_cpu) <= 1e-12 * max(scale, 1e-30)
            end
        end
        @test nz_disp > 0
    end
end

@testset "gpu: joint spin sweep ≡ keyed reference (bitwise)" begin
    for (name, H) in _joint_gpu_cases(), ws in (4, 32)
        st, gH, gst = _gpu_setup(H; with_disps = true)
        β = 1 / 0.05
        cfg0 = copy(st.config)
        zr0 = copy(st.zrows)
        u0 = copy(st.disps)
        cfg2 = copy(st.config)
        zr2 = copy(st.zrows)
        dE2 = zeros(H.n_sites)
        acc2 = zeros(Int32, H.n_sites)
        E0 = gst.energy
        naccs = Int[]
        for _ = 1:5
            push!(naccs, MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = ws))
        end
        E_ref, naccs_ref = _ref_sweeps!(cfg2, zr2, dE2, acc2, H, β, st.step,
                                        gst.seed, E0, 5, ws)
        MC.to_host!(st, gst)
        @test st.config == cfg2
        @test st.zrows == zr2
        @test gst.energy == E_ref
        @test naccs == naccs_ref
        @test gst.acc_metro == sum(naccs_ref)
        @test gst.att_metro == 5 * H.n_spin_active
        @test sum(naccs) > 0                      # the chain actually moved
        # A SPIN sweep writes the spin block and nothing else: the displacements and
        # their rows come back exactly as uploaded, and a site with no spin axis keeps
        # its (frozen, meaningless) direction bit for bit.
        @test st.disps == u0
        @test view(st.zrows, (H.nlm + 1):H.nrows, :) ==
              view(zr0, (H.nlm + 1):H.nrows, :)
        @test all(st.config[s] === cfg0[s] for s = 1:H.n_sites if !H.site_has_spin[s])
    end
end

@testset "gpu: displacement proposal slots are disjoint from the spin ones" begin
    # A compound sweep attempts both moves at the same (seed, site, sweep). A shared
    # slot would tie the displacement accept uniform to the spin one — the two moves
    # would accept and reject together, which is not the chain either sweep's
    # stationarity argument is about.
    slots = (MC._SLOT_FLIP_ACC, MC._SLOT_AXIS12, MC._SLOT_AXIS3_ANGLE,
             MC._SLOT_DISP_ACC, MC._SLOT_DISP12, MC._SLOT_DISP3)
    @test length(unique(slots)) == 6
    seed = UInt64(0xbeef)
    e = SVector(0.0, 0.0, 1.0)
    u = SVector(0.1, -0.2, 0.05)
    for site = Int32(1):Int32(8), sw = Int32(1):Int32(3)
        _, ua_spin = MC._keyed_proposal(seed, site, sw, e, 0.6)
        u2, ua_disp = MC._keyed_disp_proposal(seed, site, sw, u, 0.02)
        @test ua_spin != ua_disp
        @test u2 != u                             # a nonzero shift was drawn
    end
    # the shift is exactly `u + step_u · (g1, g2, g3)` off slots 4–5
    g1, g2 = MC._philox_normal2(MC._philox_block(seed, Int32(3), Int32(2),
                                                 MC._SLOT_DISP12))
    g3, _ = MC._philox_normal2(MC._philox_block(seed, Int32(3), Int32(2),
                                                MC._SLOT_DISP3))
    u2, _ = MC._keyed_disp_proposal(seed, Int32(3), Int32(2), u, 0.02)
    @test u2 === u + 0.02 * SVector(g1, g2, g3)
end

@testset "gpu: the spin proposal reads its named slots, value for value" begin
    # The mirror of the displacement value-pin above, and it is not redundant with the
    # bitwise sweep gates: kernel and keyed reference both call `_keyed_proposal`, so
    # collapsing two of its slots onto one moves both sides together and every bitwise
    # comparison still passes. Physically that collapse is fatal — tie the rotation axis
    # to the angle and the proposal loses the θ → −θ symmetry its detailed balance rests
    # on — and the only thing that can see it is a value pin computed from the slot
    # constants OUTSIDE the function. (Verified: mutating `_SLOT_AXIS3_ANGLE` to
    # `_SLOT_AXIS12` inside `_keyed_proposal` left the whole GPU suite green before this.)
    seed = UInt64(0xbeef)
    e = SVector(0.0, 0.0, 1.0)
    step = 0.6
    nflip = 0
    for site = Int32(1):Int32(24), sw = Int32(1):Int32(4)
        u_flip, u_acc = MC._philox_uniform2(MC._philox_block(seed, site, sw,
                                                             MC._SLOT_FLIP_ACC))
        e2, ua = MC._keyed_proposal(seed, site, sw, e, step)
        @test ua === u_acc                       # the accept uniform is slot 0's second
        if u_flip < MC._FLIP_FRACTION
            nflip += 1
            @test e2 === -e                      # the flip branch takes no rotation draw
        else
            n1, n2 = MC._philox_normal2(MC._philox_block(seed, site, sw,
                                                         MC._SLOT_AXIS12))
            n3, n4 = MC._philox_normal2(MC._philox_block(seed, site, sw,
                                                        MC._SLOT_AXIS3_ANGLE))
            @test e2 === MC._rotate(e, normalize(SVector(n1, n2, n3)), step * n4)
        end
    end
    # both branches were actually exercised (a pin that only ever saw one is half a pin)
    @test 0 < nflip < 96
end

@testset "gpu: displacement sweep ≡ keyed reference (bitwise)" begin
    for (name, H) in _joint_gpu_cases(), ws in (4, 32)
        st, gH, gst = _gpu_setup(H; with_disps = true)
        β = 1 / 0.05
        cfg0 = copy(st.config)
        zr0 = copy(st.zrows)
        u2 = copy(st.disps)
        zr2 = copy(st.zrows)
        dE2 = zeros(H.n_sites)
        acc2 = zeros(Int32, H.n_sites)
        E = gst.energy
        naccs = Int[]
        naccs_ref = Int[]
        for sw = 1:5
            push!(naccs, MC.gpu_displacement_sweep!(gst, gH, β; workgroupsize = ws))
            push!(naccs_ref,
                  MC._displacement_sweep_keyed_ref!(u2, zr2, dE2, acc2, H, β,
                                                    st.step_u, gst.seed, Int32(sw), ws))
            E += MC._reduce_dE(H, dE2)
        end
        MC.to_host!(st, gst)
        @test st.disps == u2
        @test st.zrows == zr2
        @test gst.energy == E
        @test naccs == naccs_ref
        @test sum(naccs) > 0                      # the chain actually moved
        @test gst.acc_disp == sum(naccs_ref)
        @test gst.att_disp == 5 * H.n_disp_active
        @test gst.disp_index == 5 && gst.sweep_index == 0
        # A DISPLACEMENT sweep writes the displacement block and nothing else.
        @test st.config == cfg0
        @test view(st.zrows, 1:H.nlm, :) == view(zr0, 1:H.nlm, :)
        @test all(st.disps[s] === u2[s] for s = 1:H.n_sites if !H.site_has_disp[s])
    end
end

@testset "gpu: displacement sweep vs the host displacement_sweep! (statistics)" begin
    # Independent of the reference above: the device chain and a CPU chain are
    # different Markov chains (keyed Philox vs per-site Xoshiro), so they can only
    # agree in distribution. Compared here on the exactly-known Einstein oscillator.
    a = 2.5
    kT = 0.04
    tms, L = _einstein_terms(a)
    H = MC.TiledHamiltonian(1, tms, L; dims = (3, 3, 3), fixed_reference = true)
    rng = Xoshiro(11)
    # From the CLAMPED-ION start (`disps = nothing` ⇒ all zero), so `⟨|u|²⟩` has to be
    # built up by the sampler: a random start at amplitude 0.08 already sits within
    # 20 % of the answer and would forgive under-thermalization.
    st = MC.ChainState(H, _rand_config(rng, H), rng, 0.6; step_u = 0.12)
    @test all(iszero, st.disps)
    gH = MC.GPUTiledHamiltonian(CPU(), H)
    gst = MC.GPUChainState(gH, st; seed = UInt64(4242))
    β = 1 / kT
    MC.gpu_run_sweeps!(gst, gH, st, β, 2_000; renorm_interval = 0)   # thermalize
    acc = 0.0
    nmeas = 4_000
    for _ = 1:nmeas
        MC.gpu_run_sweeps!(gst, gH, st, β, 1; renorm_interval = 0)
        MC.to_host!(st, gst)
        acc += sum(u -> dot(u, u), st.disps) / H.n_sites
    end
    # ⟨|u|²⟩ = 3kT/(2a) exactly for E = a|u|² (an isotropic Gaussian, σ² = kT/2a).
    # Known limits of this gate, recorded rather than papered over: the stationary law
    # of a symmetric-proposal Metropolis is step-independent, so it is insensitive to
    # `step_u` and to a degenerate proposal covariance (that is the slot-map test's
    # job); and the fixture's one displacement row is `(k, l) = (1, 0)`, so `R₀₀ ≡ 1`
    # and `r2^1` is exact — no `l ≥ 1` solid harmonic and no libm-`pow` path here.
    @test acc / nmeas ≈ 3 * kT / (2a) rtol = 0.04
    @test 0.1 < gst.acc_disp / gst.att_disp < 0.95      # a live, non-degenerate chain
end

@testset "gpu: joint incremental-energy drift gate" begin
    for (name, H) in _joint_gpu_cases()
        # both renormalization modes: off (pure incremental bookkeeping) and on (the
        # host round-trip, which on a joint model also re-centres the displacements)
        # 30 does not divide 200: the run must END on unrenormalized incremental
        # bookkeeping, or the last action before the comparison is a `_renormalize!`
        # that re-anchors `gst.energy` on a fresh `_total_energy` and the gate is
        # near-vacuous.
        for interval in (0, 30)
            st, gH, gst = _gpu_setup(H; with_disps = true)
            u0 = copy(st.disps)
            MC.gpu_run_sweeps!(gst, gH, st, 1 / 0.05, 200; renorm_interval = interval)
            E = total_energy(H, st.config, st.disps)
            @test abs(gst.energy - E) <= 1e-8 * max(1.0, abs(E))
            @test gst.att_disp == 200 * H.n_disp_active     # one pass per sweep
            @test st.disps != u0                            # the lattice really moved

            # The frozen-lattice conditional, asked for explicitly. With
            # renormalization OFF the displacements are then untouched bit for bit;
            # with it ON `_renormalize!` re-centres each component, so `st.disps`
            # moves by a rigid per-component shift — a GAUGE change (the energy stays
            # exact), recorded in `com_removed`, not a different lattice.
            st2, gH2, gst2 = _gpu_setup(H; with_disps = true)
            MC.gpu_run_sweeps!(gst2, gH2, st2, 1 / 0.05, 200;
                               renorm_interval = interval, disp_per_metropolis = 0)
            E2 = total_energy(H, st2.config, st2.disps)
            @test abs(gst2.energy - E2) <= 1e-8 * max(1.0, abs(E2))
            @test gst2.att_disp == 0
            if interval == 0
                @test st2.disps == u0
                @test all(iszero, st2.com_removed)
            else
                @test any(H.comp_free) == any(!iszero, st2.com_removed)
                for c = 1:H.n_disp_comps,
                    q = H.disp_comp_ptr[c]:(H.disp_comp_ptr[c + 1] - 1)

                    s = Int(H.disp_comp_sites[q])
                    @test st2.disps[s] + st2.com_removed[c] ≈ u0[s] atol = 1e-12
                end
            end
        end
    end
end

@testset "gpu: compound sweep scheduling and the empty-sweep trap" begin
    Hj = _joint_gpu_cases()[1][2]
    st, gH, gst = _gpu_setup(Hj; with_disps = true)
    # The default is what protects: `nothing` means "whatever this Hamiltonian
    # needs", so a joint model gets its displacement pass without being asked. A
    # plain default of 0 would be the silent-wrong-ensemble trap (run.jl's
    # `_resolve_disp_passes`, shared verbatim).
    @test MC._resolve_disp_passes(Hj, nothing) == 1
    @test MC._resolve_disp_passes(TiledHamiltonian(_dimer_model()), nothing) == 0
    MC.gpu_run_sweeps!(gst, gH, st, 1 / 0.05, 5)
    @test gst.sweep_index == 5 && gst.disp_index == 5
    @test gst.att_metro == 5 * Hj.n_spin_active
    @test gst.att_disp == 5 * Hj.n_disp_active
    # several passes per Metropolis step advance the displacement counter alone
    MC.gpu_run_sweeps!(gst, gH, st, 1 / 0.05, 4; disp_per_metropolis = 3)
    @test gst.sweep_index == 9 && gst.disp_index == 5 + 12
    # a pure-spin Hamiltonian asks for no displacement pass, and refuses one
    Hs = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
    sts, gHs, gsts = _gpu_setup(Hs)
    @test MC.gpu_run_sweeps!(gsts, gHs, sts, 1 / 0.05, 5) === gsts
    @test gsts.att_disp == 0
    @test_throws ArgumentError MC.gpu_run_sweeps!(gsts, gHs, sts, 1 / 0.05, 5;
                                                  disp_per_metropolis = 1)
    @test_throws ArgumentError MC.gpu_displacement_sweep!(gsts, gHs, 1 / 0.05)

    # `_resolve_disp_passes` gates on the row LAYOUT (`has_disp`), the sweep on the
    # SITES (`n_disp_active`), and a joint basis whose displacement couplings all
    # fitted to zero separates them: layout rows, no site whose energy depends on `u`.
    # The driver must run — the host does, as a no-op sweep — not throw out of its own
    # default. (Zero passes is not a wrong-ensemble risk here: with nothing depending
    # on `u`, the frozen-`u` conditional IS the joint distribution.)
    sp1(site) = SLCE.Slot(site, SLCE.SiteFactor(SLCE.SPIN, 0, 1))
    z1 = SVector(0, 0, 0)
    xx = SVector(1, 0, 0)
    pr = zeros(3, 3)
    pr[1, 1] = pr[2, 2] = pr[3, 3] = -0.5
    Lz = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])          # displacement rows exist ...
    Hz = MC.TiledHamiltonian(1, [DecoratedTerm(-0.03, (4π)^1, 2, [1, 1], [z1, xx],
                                               [sp1(1), sp1(2)], pr)], Lz;
                             dims = (4, 1, 1))            # ... but no term uses them
    @test MC.has_disp(Hz) && Hz.n_disp_active == 0
    @test MC._resolve_disp_passes(Hz, nothing) == 1       # the layout says yes ...
    @test_throws ArgumentError MC.gpu_displacement_sweep!(
        MC.GPUChainState(MC.GPUTiledHamiltonian(CPU(), Hz),
                         MC.ChainState(Hz, _rand_config(Xoshiro(2), Hz), Xoshiro(2),
                                       0.6; disps = _rand_disps(Xoshiro(2), Hz))),
        MC.GPUTiledHamiltonian(CPU(), Hz), 1 / 0.05)      # ... the sites say no
    rngz = Xoshiro(8)
    stz = MC.ChainState(Hz, _rand_config(rngz, Hz), rngz, 0.6;
                        disps = _rand_disps(rngz, Hz))
    gHz = MC.GPUTiledHamiltonian(CPU(), Hz)
    gstz = MC.GPUChainState(gHz, stz; seed = UInt64(6))
    uz = copy(stz.disps)
    MC.gpu_run_sweeps!(gstz, gHz, stz, 1 / 0.05, 5; renorm_interval = 0)
    @test gstz.sweep_index == 5 && gstz.disp_index == 0
    @test stz.disps == uz                                # nothing to move, nothing moved

    # A lattice-only model: the spin PRIMITIVE throws (a sweep there attempts nothing
    # and would report a 0/0 acceptance over a bit-for-bit unchanged state), while the
    # driver simply skips the Metropolis pass — the displacement channel is the whole
    # model. With both channels silenced there is nothing to run at all.
    tms, L = _einstein_terms()
    Hd = MC.TiledHamiltonian(1, tms, L; dims = (2, 2, 2), fixed_reference = true)
    @test Hd.n_spin_active == 0
    rng = Xoshiro(3)
    std = MC.ChainState(Hd, _rand_config(rng, Hd), rng, 0.6;
                        disps = _rand_disps(rng, Hd))
    u0 = copy(std.disps)
    gHd = MC.GPUTiledHamiltonian(CPU(), Hd)
    gstd = MC.GPUChainState(gHd, std; seed = UInt64(1))
    @test_throws ArgumentError MC.gpu_metropolis_sweep!(gstd, gHd, 1 / 0.05)
    @test_throws ArgumentError MC.gpu_run_sweeps!(gstd, gHd, std, 1 / 0.05, 5;
                                                  disp_per_metropolis = 0)
    MC.gpu_run_sweeps!(gstd, gHd, std, 1 / 0.05, 20; renorm_interval = 0)
    @test gstd.sweep_index == 0 && gstd.disp_index == 20
    @test std.disps != u0
end

@testset "gpu: the device kernel rejects a padded spin block" begin
    # The kernel derives the SPIN block width from `lmax` (`@localmem` trial row, ΔE
    # row range, write-back extent) where the host reads `H.nlm`. `row_layout` makes
    # them equal, but `TiledHamiltonian` takes a caller-supplied `RowLayout` and only
    # checks `disp_offset` against the slot rows — a padded one would truncate the
    # device ΔE and write partially into the displacement block.
    sp(site) = SLCE.Slot(site, SLCE.SiteFactor(SLCE.SPIN, 0, 1))
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    pair = zeros(3, 3)
    pair[1, 1] = pair[2, 2] = pair[3, 3] = -0.5
    # lmax = 1 needs 4 spin rows; this layout reserves 6 and starts the (0,1) block at 6
    padded = SLCE.RowLayout(9, 1, 6, [(0, 1)], [6])
    terms = [DecoratedTerm(-0.03, (4π)^1, 2, [1, 1], [z, x], [sp(1), sp(2)], pair)]
    Hp = MC.TiledHamiltonian(1, terms, padded; dims = (4, 1, 1), fixed_reference = true)
    @test Hp.nlm == 6 && Hp.lmax == 1          # the host is fine with it
    @test_throws ArgumentError MC.GPUTiledHamiltonian(CPU(), Hp)

    # The mirror on the other side: the displacement sweep writes `nlm+1:nrows` back
    # wholesale while `_disp_rows_device!` fills only the declared `(k, l)` blocks, so
    # a layout whose blocks do not TILE that range would copy uninitialized local
    # memory into the row table. Row 8 here belongs to no block.
    gappy = SLCE.RowLayout(8, 1, 4, [(0, 1)], [4])
    Hg = MC.TiledHamiltonian(1, terms, gappy; dims = (4, 1, 1), fixed_reference = true)
    @test Hg.nrows == 8 && Hg.layout.disp_starts[end] + 3 == 7   # ... 8 is unclaimed
    @test_throws ArgumentError MC.GPUTiledHamiltonian(CPU(), Hg)
    # and the well-formed layout of the same shape is accepted
    tight = SLCE.RowLayout(7, 1, 4, [(0, 1)], [4])
    Ht = MC.TiledHamiltonian(1, terms, tight; dims = (4, 1, 1), fixed_reference = true)
    @test MC.GPUTiledHamiltonian(CPU(), Ht) isa MC.GPUTiledHamiltonian
end

@testset "gpu: the gradient path rejects a joint Hamiltonian" begin
    H = _joint_gpu_cases()[1][2]
    st, gH, gst = _gpu_setup(H; with_disps = true)
    dG = Vector{SVector{3,Float64}}(undef, H.n_sites)
    # `GPUTiledHamiltonian` now accepts joint models for the SWEEP path, so each
    # gradient entry point carries the spin-only guard itself.
    @test_throws ArgumentError MC.GPUGradientScratch(gH)
    @test_throws ArgumentError MC._gpu_gradient_rows!(dG, gH, gst.config, gst.zrows)
    @test_throws ArgumentError MC._gradient_lane_ref!(dG, H, st.config, st.zrows, 4)
    # ... and through both `gpu_energy_gradient!` overloads and `gpu_zlm_rows!`. None of
    # them can inherit the guard from its `gsc`: the SPIN RESTRICTION of the very same
    # model gives a scratch of the matching `(nlm, n_sites)` shape, so the dimension
    # check passes and the scratch is no evidence at all about the Hamiltonian it is
    # handed with.
    model = _joint_model(5)[1]
    Hj = TiledHamiltonian(model; dims = (2, 2, 2))
    Hs = TiledHamiltonian(SLCE.restrict(model, :spin); dims = (2, 2, 2))
    gsc = MC.GPUGradientScratch(MC.GPUTiledHamiltonian(CPU(), Hs))
    @test size(gsc.zrows) == (Hj.nlm, Hj.n_sites)        # the shape check cannot help
    stj, gHj, gstj = _gpu_setup(Hj; with_disps = true)
    dGj = Vector{SVector{3,Float64}}(undef, Hj.n_sites)
    @test_throws ArgumentError MC.gpu_energy_gradient!(dGj, gstj, gHj, gsc)
    for refresh in (true, false)
        # `refresh_zrows = false` is a documented fast path that skips `gpu_zlm_rows!`,
        # so a guard placed only there would leave this walking the joint program table
        # — whose `sent_tgt` reaches past `nlm` — against a spin-sized gradient row, an
        # `@inbounds` out-of-bounds read that returns garbage instead of throwing.
        @test_throws ArgumentError MC.gpu_energy_gradient!(dGj, gHj, gstj.config, gsc;
                                                           refresh_zrows = refresh)
    end
    @test_throws ArgumentError MC.gpu_zlm_rows!(gsc, gHj, gstj.config)
end

@testset "gpu: dimer statistics ⟨e₁·e₂⟩ = L(β|J|)" begin
    H = TiledHamiltonian(_dimer_model())
    J = _dimer_J(H)                               # < 0 (ferro)
    βJ = 1.5
    β = βJ / abs(J)
    st, gH, gst = _gpu_setup(H; seed_dev = UInt64(2026), step = 1.2)
    MC.gpu_run_sweeps!(gst, gH, st, β, 500; renorm_interval = 0)   # thermalize
    acc = 0.0
    nmeas = 20_000
    for _ = 1:nmeas
        MC.gpu_metropolis_sweep!(gst, gH, β)
        MC.to_host!(st, gst)
        acc += dot(st.config[1], st.config[2])
    end
    @test acc / nmeas ≈ _langevin(βJ) atol = 0.03
end

# ---------------------------------------------------------------------------
# Phase-2 gradient gates (decision record G7): device gradient row, rows
# rebuild, the all-site gradient kernel vs its lane reference, and the scaled
# tolerance vs the host energy_gradient!.
# ---------------------------------------------------------------------------

@testset "gpu: device grad row ≡ host grad_Zlm_unsafe (bitwise)" begin
    rng = Xoshiro(2027)
    dirs = SVector{3,Float64}[SVector(0, 0, 1.0), SVector(0, 0, -1.0),
                              SVector(1, 0, 0.0), SVector(-1, 0, 0.0),
                              SVector(0, 1, 0.0), SVector(0, -1, 0.0)]
    for k = 0:11
        push!(dirs, SVector(cos(k * π / 6), sin(k * π / 6), 0.0))
    end
    for _ = 1:2000
        push!(dirs, _rand_spin(rng))
    end
    H = SLCE.Harmonics
    for lmax = 0:6
        nlm = (lmax + 1)^2
        cache = Vector{Float64}(undef, lmax + 2)
        grow = Vector{Float64}(undef, 3 * nlm)
        ok = true
        for u in dirs
            MC._grad_zlm_row_device_dyn!(grow, u, lmax)
            k = 0
            for l = 0:lmax, m = -l:l
                gh = H.grad_Zlm_unsafe(l, m, u, cache)
                # === per component — signed zeros included (the dnPl l < n
                # trivial-zero branch feeds parity·norm·(+0.0) → −0.0 for odd l)
                ok &= grow[3k + 1] === gh[1] && grow[3k + 2] === gh[2] &&
                      grow[3k + 3] === gh[3]
                k += 1
            end
        end
        @test ok

        # …and through an actual KA-CPU kernel
        out = zeros(3 * nlm, length(dirs))
        kern = _test_grad_kernel!(CPU())
        kern(out, dirs, Val(lmax); ndrange = length(dirs))
        KernelAbstractions.synchronize(CPU())
        ok_kernel = true
        for (i, u) in enumerate(dirs)
            MC._grad_zlm_row_device_dyn!(grow, u, lmax)
            ok_kernel &= view(out, :, i) == grow
        end
        @test ok_kernel
    end
end

@testset "gpu: rows rebuild kernel ≡ host _zrows (bitwise)" begin
    for H in (TiledHamiltonian(_dimer_model()),
              TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2)))
        rng = Xoshiro(11)
        config = _rand_config(rng, H)
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        gsc = MC.GPUGradientScratch(gH)
        dconfig = KernelAbstractions.allocate(CPU(), SVector{3,Float64},
                                              H.n_sites)
        copyto!(dconfig, config)
        MC.gpu_zlm_rows!(gsc, gH, dconfig)
        @test Matrix(gsc.zrows) == MC._zrows(H, config)
    end
end

@testset "gpu: gradient kernel ≡ lane reference (bitwise)" begin
    cases = [TiledHamiltonian(_dimer_model()),                 # inactive sites
             TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2)),
             MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (4, 2, 2)),
             MC.TiledHamiltonian(1, _fourbody_terms(0.03); dims = (4, 2, 2))]
    for H in cases, ws in (4, 32)
        rng = Xoshiro(23)
        config = _rand_config(rng, H)
        zrows = MC._zrows(H, config)
        ref = Vector{SVector{3,Float64}}(undef, H.n_sites)
        MC._gradient_lane_ref!(ref, H, config, zrows, ws)
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        gsc = MC.GPUGradientScratch(gH)
        dconfig = KernelAbstractions.allocate(CPU(), SVector{3,Float64},
                                              H.n_sites)
        copyto!(dconfig, config)
        dG = KernelAbstractions.allocate(CPU(), SVector{3,Float64}, H.n_sites)
        MC.gpu_energy_gradient!(dG, gH, dconfig, gsc; workgroupsize = ws)
        G = Vector(dG)
        @test G == ref
        # inactive sites are exactly zero (empty adjacency → fold of +0.0s)
        for s = 1:H.n_sites
            H.site_active[s] && continue
            @test G[s] === SVector(0.0, 0.0, 0.0)
        end
        # repeated-run identity
        MC.gpu_energy_gradient!(dG, gH, dconfig, gsc; workgroupsize = ws)
        @test Vector(dG) == G
    end
end

@testset "gpu: gradient vs host energy_gradient! (tolerance) + tangency" begin
    for H in (TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2)),
              MC.TiledHamiltonian(1, _threebody_terms(0.05); dims = (4, 2, 2)))
        rng = Xoshiro(29)
        config = _rand_config(rng, H)
        Ghost = MC.energy_gradient(H, config)
        gH = MC.GPUTiledHamiltonian(CPU(), H)
        gsc = MC.GPUGradientScratch(gH)
        dconfig = KernelAbstractions.allocate(CPU(), SVector{3,Float64},
                                              H.n_sites)
        copyto!(dconfig, config)
        dG = KernelAbstractions.allocate(CPU(), SVector{3,Float64}, H.n_sites)
        MC.gpu_energy_gradient!(dG, gH, dconfig, gsc)
        G = Vector(dG)
        scale = max(1.0, maximum(norm, Ghost))
        @test maximum(norm.(G .- Ghost)) <= 1e-12 * scale
        @test maximum(abs(dot(config[s], G[s])) / max(1.0, norm(G[s]))
                      for s = 1:H.n_sites) <= 1e-13
    end
end

@testset "gpu: GPUChainState gradient overload (rows current)" begin
    H = TiledHamiltonian(_biquadratic_model(3); dims = (2, 2, 2))
    st, gH, gst = _gpu_setup(H)
    gsc = MC.GPUGradientScratch(gH)
    dG = KernelAbstractions.allocate(CPU(), SVector{3,Float64}, H.n_sites)
    MC.gpu_energy_gradient!(dG, gst, gH, gsc)      # reads gst.zrows, no rebuild
    ref = Vector{SVector{3,Float64}}(undef, H.n_sites)
    MC._gradient_lane_ref!(ref, H, st.config, Matrix(gst.zrows), 128)
    @test Vector(dG) == ref
end
