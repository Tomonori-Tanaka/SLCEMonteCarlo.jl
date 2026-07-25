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

# Fresh (H, ChainState, GPU pair) on the CPU backend with a seeded random config.
function _gpu_setup(H; seed_cfg = 7, seed_dev = UInt64(0xc0ffee), step = 0.6)
    rng = Xoshiro(seed_cfg)
    st = MC.ChainState(H, _rand_config(rng, H), rng, step)
    gH = MC.GPUTiledHamiltonian(CPU(), H)
    gst = MC.GPUChainState(gH, st; seed = seed_dev)
    return st, gH, gst
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
        tb = MC._GPUTables(H.site_ptr, H.site_inst, H.inst_ptr, H.inst_sites,
                           H.progs.site_prog, H.progs.sprog_ptr, H.progs.sent_w,
                           H.progs.sent_tgt, H.progs.sfac_ptr, H.progs.sfac_row,
                           H.progs.sfac_slot, H.progs.site_col, H.progs.site_col2,
                           H.progs.pent_row, H.progs.pent_row2, H.color_sites)
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
                    ΔE_walk += MC._entry_walk_partial(tb, st.zrows, znew, s, lane, ws)
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
