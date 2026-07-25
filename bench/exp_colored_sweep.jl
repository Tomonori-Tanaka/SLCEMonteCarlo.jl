# EXPERIMENT (prototype — not wired into the package): colored parallel Metropolis
# sweeps. Kept for the record behind bench_log "Experiment: colored sweeps".
#
# Sites that share no cluster instance have exactly independent single-spin ΔE
# (site_coeffs! of one is independent of the other's spin), so a color class of
# pairwise non-conflicting sites can be updated concurrently — the update is
# equivalent to SOME serial order of those sites, i.e. a valid (but different)
# Markov chain: trajectory and RNG stream change (P6-breaking if landed).
#
# Measures ms/sweep of the sequential metropolis_sweep! vs the colored threaded
# sweep on the two bench fixtures. Run: julia -t 8 --project=bench <this file>
# from the SLCEMonteCarlo.jl directory.

include(joinpath(@__DIR__, "fixtures.jl"))

using Base.Threads: @threads, nthreads

# ---------------------------------------------------------------- coloring ----
# Conflict = shares at least one instance. Greedy coloring in site order.
function build_coloring(H::MC.TiledHamiltonian)
    n = MC.n_sites(H)
    colors = zeros(Int, n)
    used = Int[]           # colors seen among already-colored conflicting sites
    for s = 1:n
        H.site_active[s] || continue
        empty!(used)
        for j = H.site_ptr[s]:(H.site_ptr[s + 1] - 1)
            i = H.site_inst[j]
            for q = H.inst_ptr[i]:(H.inst_ptr[i + 1] - 1)
                t = H.inst_sites[q]
                c = colors[t]
                c > 0 && !(c in used) && push!(used, c)
            end
        end
        c = 1
        while c in used
            c += 1
        end
        colors[s] = c
    end
    ncol = maximum(colors)
    groups = [Int32[] for _ = 1:ncol]
    for s = 1:n
        colors[s] > 0 && push!(groups[colors[s]], Int32(s))
    end
    return groups
end

# ------------------------------------------------------- colored sweep --------
# Per-thread scratch + RNG (:static scheduling so threadid() is stable).
function colored_metropolis!(config::MC.SpinConfig, zrows::Matrix{Float64},
                             H::MC.TiledHamiltonian, β::Float64, step::Float64,
                             groups::Vector{Vector{Int32}},
                             scs::Vector{MC.SweepScratch},
                             rngs::Vector{Xoshiro})
    nt = length(scs)
    acc = zeros(Int, nt)
    dE = zeros(nt)
    for grp in groups
        @threads :static for k in eachindex(grp)
            tid = Threads.threadid()
            sc = scs[tid]
            rng = rngs[tid]
            s = Int(grp[k])
            fill!(sc.c, 0.0)
            MC.site_coeffs!(sc.c, H, s, zrows)
            e = config[s]
            e2 = if rand(rng) < MC._FLIP_FRACTION
                -e
            else
                MC._rotate(e, MC._random_unit(rng), step * randn(rng))
            end
            MC._zlm_row!(sc.znew, e2, H.lmax, sc.plm)
            ΔE = MC.delta_energy(sc.c, view(zrows, :, s), sc.znew)
            if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
                config[s] = e2
                copyto!(view(zrows, :, s), sc.znew)
                dE[tid] += ΔE
                acc[tid] += 1
            end
        end
    end
    return sum(acc), sum(dE)
end


# ------------------------------------------- v2: one spawn + spin barrier -----
mutable struct SpinBarrier
    const count::Threads.Atomic{Int}
    const gen::Threads.Atomic{Int}
    const n::Int
end
SpinBarrier(n::Int) = SpinBarrier(Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), n)

@inline function wait_barrier!(b::SpinBarrier)
    g = b.gen[]
    if Threads.atomic_add!(b.count, 1) == b.n - 1
        b.count[] = 0
        Threads.atomic_add!(b.gen, 1)
    else
        while b.gen[] == g
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
        end
    end
    return nothing
end

# One task spawn per sweep; colors separated by spin barriers; task t owns the
# deterministic slice k = t:ntasks:length(grp) of every color class.
function colored_metropolis2!(config::MC.SpinConfig, zrows::Matrix{Float64},
                              H::MC.TiledHamiltonian, β::Float64, step::Float64,
                              groups::Vector{Vector{Int32}}, ntasks::Int,
                              scs::Vector{MC.SweepScratch},
                              rngs::Vector{Xoshiro},
                              acc::Vector{Int}, dE::Vector{Float64})
    fill!(acc, 0)
    fill!(dE, 0.0)
    bar = SpinBarrier(ntasks)
    @threads :static for t = 1:ntasks
        sc = scs[t]
        rng = rngs[t]
        a = 0
        d = 0.0
        for grp in groups
            for k = t:ntasks:length(grp)
                s = Int(grp[k])
                fill!(sc.c, 0.0)
                MC.site_coeffs!(sc.c, H, s, zrows)
                e = config[s]
                e2 = if rand(rng) < MC._FLIP_FRACTION
                    -e
                else
                    MC._rotate(e, MC._random_unit(rng), step * randn(rng))
                end
                MC._zlm_row!(sc.znew, e2, H.lmax, sc.plm)
                ΔE = MC.delta_energy(sc.c, view(zrows, :, s), sc.znew)
                if ΔE <= 0.0 || rand(rng) < exp(-β * ΔE)
                    config[s] = e2
                    copyto!(view(zrows, :, s), sc.znew)
                    d += ΔE
                    a += 1
                end
            end
            wait_barrier!(bar)
        end
        acc[t] = a
        dE[t] = d
    end
    return sum(acc), sum(dE)
end

# ------------------------------------------------------------- benchmark ------
function bench_fixture(name::String, H::MC.TiledHamiltonian; nsweeps::Int = 30,
                       kt::Float64 = BENCH_KT)
    β = 1.0 / kt
    groups = build_coloring(H)
    sizes = sort!([length(g) for g in groups]; rev = true)
    println("\n--- $name: ", MC.n_sites(H), " sites (", H.n_active, " active), ",
            length(groups), " colors, class sizes max=", sizes[1], " median=",
            sizes[cld(end, 2)], " min=", sizes[end])

    # sequential reference (the shipped kernel)
    st, sc = chain_state(H; seed = 21)
    MC.metropolis_sweep!(st, H, β, sc)               # warm up
    t0 = time_ns()
    for _ = 1:nsweeps
        MC.metropolis_sweep!(st, H, β, sc)
    end
    tseq = (time_ns() - t0) / 1e6 / nsweeps
    @printf("sequential  %8.3f ms/sweep   acc=%.2f\n", tseq,
            st.acc_metro / st.att_metro)

    # colored threaded sweep
    nt = Threads.maxthreadid()
    st2, _ = chain_state(H; seed = 21)
    scs = [MC.SweepScratch(H) for _ = 1:nt]
    rngs = [Xoshiro(1000 + 7 * t) for t = 1:nt]
    E0 = MC.total_energy(H, st2.config)
    Einc = E0
    a, d = colored_metropolis!(st2.config, st2.zrows, H, β, st2.step, groups,
                               scs, rngs)             # warm up
    Einc += d
    nattempt = H.n_active
    t0 = time_ns()
    atot = 0
    for _ = 1:nsweeps
        a, d = colored_metropolis!(st2.config, st2.zrows, H, β, st2.step, groups,
                                   scs, rngs)
        atot += a
        Einc += d
    end
    tcol = (time_ns() - t0) / 1e6 / nsweeps
    drift = abs(Einc - MC.total_energy(H, st2.config)) / max(1.0, abs(Einc))
    @printf("colored %2dT %8.3f ms/sweep   acc=%.2f   speedup=%.2fx   drift=%.1e\n",
            nt, tcol, atot / (nsweeps * nattempt), tseq / tcol, drift)

    # v2: one spawn per sweep + spin barriers between colors
    ntasks = nthreads()
    st3, _ = chain_state(H; seed = 21)
    scs3 = [MC.SweepScratch(H) for _ = 1:ntasks]
    rngs3 = [Xoshiro(2000 + 7 * t) for t = 1:ntasks]
    accv = zeros(Int, ntasks)
    dEv = zeros(ntasks)
    Einc3 = MC.total_energy(H, st3.config)
    a3, d3 = colored_metropolis2!(st3.config, st3.zrows, H, β, st3.step, groups,
                                  ntasks, scs3, rngs3, accv, dEv)
    Einc3 += d3
    t0 = time_ns()
    atot3 = 0
    for _ = 1:nsweeps
        a3, d3 = colored_metropolis2!(st3.config, st3.zrows, H, β, st3.step,
                                      groups, ntasks, scs3, rngs3, accv, dEv)
        atot3 += a3
        Einc3 += d3
    end
    tcol2 = (time_ns() - t0) / 1e6 / nsweeps
    drift3 = abs(Einc3 - MC.total_energy(H, st3.config)) / max(1.0, abs(Einc3))
    @printf("colored2 %2dT %7.3f ms/sweep   acc=%.2f   speedup=%.2fx   drift=%.1e\n",
            ntasks, tcol2, atot3 / (nsweeps * nattempt), tseq / tcol2, drift3)
    return nothing
end

println("threads = ", nthreads())
bench_fixture("bcc Fe 8³ (l=1 NN)", MC.TiledHamiltonian(bcc_fe_model(); dims = (8, 8, 8)))
bench_fixture("bcc Fe 16³", MC.TiledHamiltonian(bcc_fe_model(); dims = (16, 16, 16));
              nsweeps = 10)
bench_fixture("Nd₂Fe₁₄B 2³", MC.TiledHamiltonian(nd2fe14b_model(); dims = (2, 2, 2)))
bench_fixture("Nd₂Fe₁₄B 4³", MC.TiledHamiltonian(nd2fe14b_model(); dims = (4, 4, 4));
              nsweeps = 10)
