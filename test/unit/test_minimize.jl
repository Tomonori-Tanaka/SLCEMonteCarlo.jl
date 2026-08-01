# Ground-state search: the deterministic descent (`minimize_energy`), the
# multi-start anneal + polish (`find_ground_state`), and their determinism /
# validation contracts (`docs/specs/ground-state-search.md`).

# The rugged fixture of this suite: the anisotropic random-coupling pair model on a
# 2×1×1 tile — many distinct local minima (measured spread ≈ 1.9 across seeds), the
# lowest at ≈ −4.884356.
_rugged_H() = MC.TiledHamiltonian(_biquadratic_model(0); dims = (2, 1, 1))

@testset "gradient consistency" begin
    H = _rugged_H()
    config = _rand_config(Xoshiro(3), H)
    G = Vector{SVector{3,Float64}}(undef, H.n_sites)
    gsup = MC._gradient!(G, H, config, MC._zrows(H, config), zeros(H.nlm),
                         Vector{Float64}(undef, H.lmax + 1))
    # bit-identical to the public per-site gradient (the coupled-site gate: both
    # walk the same (l, m) loop in energy.jl / minimize.jl)
    for s = 1:H.n_sites
        @test G[s] == MC.site_gradient(H, s, config)
        @test abs(dot(config[s], G[s])) <= 1e-14 * (1 + norm(G[s]))   # tangency
    end
    @test gsup == maximum(norm(g) for g in G)
    @test gsup > 0
end

@testset "dimer ground state" begin
    H = MC.TiledHamiltonian(_dimer_model())
    J = _dimer_J(H)
    r = minimize_energy(H; seed = 1)
    @test r.converged
    @test r.gradnorm <= 1e-8 * MC._site_energy_scale(H)
    @test r.energy ≈ -abs(J) atol = 1e-10
    @test dot(r.config[1], r.config[2]) ≈ 1 atol = 1e-6
    for s = 1:H.n_sites          # uncoupled sites stay unit vectors
        @test norm(r.config[s]) ≈ 1 atol = 1e-12
    end
    @test r.best == 1 && length(r.energies) == 1
    @test r.config == r.configs[1] && r.energy == r.energies[1]
end

@testset "perturbed known minimum recovers" begin
    H = MC.TiledHamiltonian(_dimer_model())
    J = _dimer_J(H)
    up = SVector(0.0, 0.0, 1.0)
    rng = Xoshiro(5)
    init = MC.SpinConfig([normalize(up + 0.2 * _rand_spin(rng)) for _ = 1:H.n_sites])
    E0 = total_energy(H, init)
    r = minimize_energy(H; init = init)
    @test r.converged
    @test r.energy ≈ -abs(J) atol = 1e-10
    @test abs(dot(r.config[1], r.config[2])) ≈ 1 atol = 1e-6
    @test r.energy < E0        # the nonmonotone window never rises above the start
end

@testset "stationary start returns immediately" begin
    H = MC.TiledHamiltonian(_dimer_model())
    up = SVector(0.0, 0.0, 1.0)
    init = MC.SpinConfig([up for _ = 1:H.n_sites])
    r = minimize_energy(H; init = init, seed = 4)
    @test r.converged && r.iterations == 0
    @test r.config == init
    @test r.energy == total_energy(H, init)
end

@testset "chain Néel ground state" begin
    J = 0.05                                    # J > 0 ⇒ antiferromagnetic
    H = MC.TiledHamiltonian(1, _chain_terms(J); dims = (4, 1, 1))
    fgs = find_ground_state(H; nstarts = 4, seed = 3)
    @test fgs.energy ≈ -4 * J atol = 1e-8
    for s = 1:4                                 # alternating neighbors around the ring
        @test dot(fgs.config[s], fgs.config[mod1(s + 1, 4)]) ≈ -1 atol = 1e-6
    end
end

@testset "checkerboard ferromagnetic ground state" begin
    model, _ = _checkerboard_model()            # equal-fill jphi < 0 ⇒ ferro NN
    H = MC.TiledHamiltonian(model; dims = (2, 2, 1))
    up = SVector(0.0, 0.0, 1.0)
    E_aligned = total_energy(H, MC.SpinConfig([up for _ = 1:H.n_sites]))
    fgs = find_ground_state(H; nstarts = 4, seed = 5)
    @test fgs.energy ≈ E_aligned atol = 1e-8
    @test fgs.energy <= E_aligned + 1e-12
    for s = 2:H.n_sites
        @test dot(fgs.config[1], fgs.config[s]) ≈ 1 atol = 1e-6
    end
end

@testset "rugged landscape: local < global" begin
    H = _rugged_H()
    fgs = find_ground_state(H; nstarts = 16, seed = 7)
    # measured: seed-1 descent lands in the −4.3279 basin, the global one is −4.8844
    loc = minimize_energy(H; seed = 1)
    @test loc.converged
    @test loc.energy > fgs.energy + 1e-6
    @test maximum(fgs.energies) > minimum(fgs.energies) + 1e-6   # basin spread
    @test fgs.energy == minimum(fgs.energies)
    @test fgs.config == fgs.configs[fgs.best]
    @test fgs.gradnorm == fgs.gradnorms[fgs.best]
    @test all(fgs.converged_starts)
end

@testset "thermal cycling escapes a frozen anneal" begin
    H = _rugged_H()
    kts = 0.6 .* 0.3 .^ range(0, 1; length = 5)  # deliberately short & cold ladder
    kw = (; kT = kts, anneal_sweeps = 20, nstarts = 1)
    e1 = find_ground_state(H; kw..., seed = 6, cycles = 1).energy
    e3 = find_ground_state(H; kw..., seed = 6, cycles = 3).energy
    @test e3 < e1 - 1e-6      # measured: −4.884 vs −4.328 (deterministic at seed 6,
                              # re-picked after the colored-sweep RNG change)
    # ...and not merely "better than one frozen seed": the cycled single start
    # reaches the SAME basin the 16-start multistart search finds independently,
    # so the claim survives future RNG-stream changes as "cycling recovers the
    # global minimum here", not "seed 6 happened to improve"
    @test e3 ≈ find_ground_state(H; nstarts = 16, seed = 7).energy atol = 1e-6
end

@testset "bit determinism across ntasks" begin
    H = _rugged_H()
    for cycles in (1, 2)
        kw = (; nstarts = 8, seed = 11, cycles = cycles, kT = [0.5, 0.1],
              anneal_sweeps = 20)
        a = find_ground_state(H; kw..., ntasks = 1)
        b = find_ground_state(H; kw..., ntasks = 4)
        @test a.configs == b.configs
        @test a.energies == b.energies
        @test a.gradnorms == b.gradnorms
        @test a.best == b.best && a.converged == b.converged
    end
    kw = (; nstarts = 2, kT = [0.5, 0.1], anneal_sweeps = 20, ntasks = 1)
    other = find_ground_state(H; kw..., seed = 12)
    base = find_ground_state(H; kw..., seed = 11)
    @test other.configs != base.configs
    # default seed: fresh per call, recorded, and sufficient to replay
    d1 = find_ground_state(H; kw...)
    d2 = find_ground_state(H; kw...)
    @test d1.seed != d2.seed
    replay = find_ground_state(H; kw..., seed = d1.seed)
    @test replay.configs == d1.configs && replay.energies == d1.energies
    m1 = minimize_energy(H)
    m2 = minimize_energy(H)
    @test m1.seed != m2.seed
    @test minimize_energy(H; seed = m1.seed).config == m1.config
end

@testset "non-convergence is reported, not thrown" begin
    H = _rugged_H()
    r = minimize_energy(H; maxiter = 1, seed = 2)
    @test !r.converged && r.iterations <= 1
    @test isfinite(r.energy) && isfinite(r.gradnorm)
    r0 = minimize_energy(H; maxiter = 0, seed = 2)
    @test !r0.converged && r0.iterations == 0
    @test r0.energy == total_energy(H, r0.config)
end

@testset "init passthrough" begin
    H = MC.TiledHamiltonian(_dimer_model())
    config = _rand_config(Xoshiro(8), H)
    rv = minimize_energy(H; init = config, seed = 1)
    rm = minimize_energy(H; init = _config_matrix(config), seed = 2)
    @test rv.config == rm.config && rv.energy == rm.energy   # seed-independent
    @test_throws DimensionMismatch minimize_energy(H; init = zeros(3, H.n_sites + 1))
    @test_throws DimensionMismatch minimize_energy(H; init = config[1:2])
    bad = copy(config)
    bad[1] = SVector(0.0, 0.0, 0.0)
    @test_throws ArgumentError minimize_energy(H; init = bad)
end

@testset "gtol is respected" begin
    H = _rugged_H()
    loose = minimize_energy(H; seed = 6, gtol = 1e-2)
    tight = minimize_energy(H; seed = 6, gtol = 1e-10)
    @test loose.converged && loose.gradnorm <= 1e-2
    @test tight.converged && tight.gradnorm <= 1e-10
    @test tight.energy <= loose.energy + 1e-12
    @test_throws ArgumentError minimize_energy(H; gtol = 0.0)
    @test_throws ArgumentError minimize_energy(H; gtol = -1e-8)
end

@testset "explicit inits and validation" begin
    H = _rugged_H()
    rng = Xoshiro(9)
    cfg1, cfg2 = _rand_config(rng, H), _rand_config(rng, H)
    fgs = find_ground_state(H; inits = [cfg1, cfg2], anneal_sweeps = 0, seed = 1,
                            ntasks = 1)
    @test length(fgs.energies) == 2
    for (r, config) in ((1, cfg1), (2, cfg2))     # RNG-free polish ⇒ exact match
        m = minimize_energy(H; init = config, seed = 42)
        @test fgs.energies[r] == m.energy
        @test fgs.configs[r] == m.config
    end
    @test_throws ArgumentError find_ground_state(H; nstarts = 2, inits = [cfg1])
    @test_throws ArgumentError find_ground_state(H; inits = MC.SpinConfig[])
    @test_throws ArgumentError find_ground_state(H; kT = [0.1, 0.5])   # increasing
    @test_throws ArgumentError find_ground_state(H; anneal_sweeps = -1)
    @test_throws ArgumentError find_ground_state(H; nstarts = 0)
    @test_throws ArgumentError find_ground_state(H; cycles = 0)
    @test_throws ArgumentError find_ground_state(H; reheat = 1.0)
    @test_throws ArgumentError find_ground_state(H; reheat = 0.0)
    @test_throws ArgumentError find_ground_state(H; ntasks = 0)
    @test_throws ArgumentError find_ground_state(H; maxiter = -1)
    @test_throws ArgumentError find_ground_state(H; step = 0.0)
    @test_throws ArgumentError find_ground_state(H; adapt_target = 1.0)
    @test_throws ArgumentError find_ground_state(H; adapt_interval = 0)
end

@testset "printing" begin
    H = _rugged_H()
    r = find_ground_state(H; nstarts = 2, kT = [0.5, 0.1], anneal_sweeps = 10,
                          seed = 1, ntasks = 1)
    @test occursin("GroundStateResult", sprint(show, r))
    plain = sprint(show, MIME"text/plain"(), r)
    @test occursin("E - E_best", plain) && occursin("start", plain)
    @test occursin("converged", plain)
end
