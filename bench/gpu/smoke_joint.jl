# Device-only smoke for the JOINT spin–lattice GPU path (M4 slice 3f, decision record
# docs/specs/gpu-prototype.md G8).
#
#   julia --project=bench/gpu bench/gpu/smoke_joint.jl
#
# Why this exists separately from `bench_gpu.jl`: that one runs pure-spin fixtures and
# so never launches `_disp_kernel!` at all. And why it must run on a real device: the
# KA-CPU gates in `test/unit/test_gpu.jl` cannot see the class of bug the G7 field note
# records — the CUDA backend's `@index` returns `Int32` where the CPU backend's returns
# `Int`, which turns an Int-typed call inside a kernel into a compile-time
# `MethodError`. A KA-CPU suite that is 100 % green says nothing about that. Every
# kernel-adjacent change owes this smoke a run.
#
# REFUSES to fall back to the CPU backend: a smoke that silently ran on the host would
# report success for precisely the thing it exists to check.

using SLCEMonteCarlo
using SLCE
using KernelAbstractions: KernelAbstractions, CPU
using CUDA
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Statistics

const MC = SLCEMonteCarlo

CUDA.functional() || error("no functional CUDA device — this smoke is device-only " *
                           "by design (see the header)")
const BACKEND = CUDABackend()
const WS = 128

@printf("device : %s\n", CUDA.name(CUDA.device()))
@printf("julia  : %s\nCUDA.jl: %s\n", VERSION, pkgversion(CUDA))
flush(stdout)

const FAILURES = String[]
function check(name::AbstractString, ok::Bool, detail::AbstractString = "")
    @printf("[%s] %-52s %s\n", ok ? " ok " : "FAIL", name, detail)
    ok || push!(FAILURES, name)
    flush(stdout)
    return ok
end

# --- fixtures (hand-built: no SLCEBasis, so no symmetry backend is needed here) -------

sp(site) = SLCE.Slot(site, SLCE.SiteFactor(SLCE.SPIN, 0, 1))
dp(site, k, l) = SLCE.Slot(site, SLCE.SiteFactor(SLCE.DISP, k, l))

# A BOUNDED joint chain: spin pair + an onsite spin–displacement coupling (linear in u)
# + a well on EVERY atom, so the displacement energy is a shifted harmonic and the
# chain is stationary. Bounded-ness is the point: the committed test fixtures are not,
# which is fine for implementation-vs-implementation gates but useless for statistics
# (an escaped chain agrees with nothing).
function joint_terms()
    z = SVector(0, 0, 0)
    x = SVector(1, 0, 0)
    pair = zeros(3, 3)
    pair[1, 1] = pair[2, 2] = pair[3, 3] = -0.5
    onsite = zeros(3, 3)
    onsite[1, 2] = 0.35
    onsite[2, 1] = 0.15
    onsite[3, 3] = -0.2
    L = SLCE.RowLayout(8, 1, 4, [(0, 1), (1, 0)], [4, 7])
    return [DecoratedTerm(-0.03, (4π)^1, 2, [1, 1], [z, x], [sp(1), sp(2)], pair),
            DecoratedTerm(0.4, (4π)^0.5, 1, [1], [z], [sp(1), dp(1, 0, 1)], onsite),
            DecoratedTerm(2.0, 1.0, 1, [1], [z], [dp(1, 1, 0)], [1.0]),
            DecoratedTerm(2.0, 1.0, 1, [2], [z], [dp(1, 1, 0)], [1.0])], L
end

# The Einstein oscillator `E = a|u|²`: `⟨|u|²⟩ = 3kT/(2a)` in closed form — the one
# displacement check against an external truth rather than another implementation.
function einstein_terms(a::Float64 = 2.5)
    L = SLCE.RowLayout(2, 0, 1, [(1, 0)], [1])
    return [MC.ScaledTerm(a, [1], [SVector(0, 0, 0)],
                          [MC.TermSlot(1, 1, 0, false)], [1.0])], L
end

rand_config(rng, H) = MC.SpinConfig([normalize(SVector{3,Float64}(randn(rng),
                                                                  randn(rng),
                                                                  randn(rng)))
                                     for _ = 1:H.n_sites])
rand_disps(rng, H; amp = 0.02) =
    [amp * SVector{3,Float64}(randn(rng), randn(rng), randn(rng)) for _ = 1:H.n_sites]

function setup(H; seed = 0xd15b, step = 0.6, step_u = 0.05, amp = 0.02,
               clamped = false)
    rng = Xoshiro(21)
    config = rand_config(rng, H)
    st = clamped ? MC.ChainState(H, config, rng, step; step_u = step_u) :
         MC.ChainState(H, config, rng, step; disps = rand_disps(rng, H; amp = amp),
                       step_u = step_u)
    gH = GPUTiledHamiltonian(BACKEND, H)
    gst = GPUChainState(gH, st; seed = UInt64(seed))
    return st, gH, gst
end

binned(v) = (b = MC.LogBinner(1); for x in v; push!(b, x); end;
             (mean(v), MC.std_error(b)[1], MC.tau_int(b)[1]))

# --- 1. the compile check: does `_disp_kernel!` build and run on CUDA at all? ---------

terms, L = joint_terms()
Hj = MC.TiledHamiltonian(2, terms, L; dims = (4, 4, 4), fixed_reference = true)
@printf("\njoint fixture: %d sites (spin %d, disp %d), nrows %d, disp_lmax %d\n",
        Hj.n_sites, Hj.n_spin_active, Hj.n_disp_active, Hj.nrows, Hj.disp_lmax)
flush(stdout)

st, gH, gst = setup(Hj; seed = 0xd15b)
β = 1 / 0.05
nacc = gpu_displacement_sweep!(gst, gH, β; workgroupsize = WS)
check("_disp_kernel! compiles and runs on CUDA", true, "accepted $(nacc)")
check("displacement sweep moved something", nacc > 0, "$(nacc)/$(Hj.n_disp_active)")
nacc_s = gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS)
check("_metro_kernel! (Val(NROWS) signature) runs on CUDA", true, "accepted $(nacc_s)")

# --- 2. channel isolation, on device -------------------------------------------------

st2, gH2, gst2 = setup(Hj; seed = 0xd15b)
cfg0 = copy(st2.config)
zr0 = copy(st2.zrows)
u0 = copy(st2.disps)
for _ = 1:5
    gpu_displacement_sweep!(gst2, gH2, β; workgroupsize = WS)
end
to_host!(st2, gst2)
check("disp sweep leaves the spins bitwise alone", st2.config == cfg0)
check("disp sweep leaves the SPIN rows bitwise alone",
      view(st2.zrows, 1:Hj.nlm, :) == view(zr0, 1:Hj.nlm, :))
check("disp sweep actually moved the lattice", st2.disps != u0)

st3, gH3, gst3 = setup(Hj; seed = 0xd15b)
u3 = copy(st3.disps)
zr3 = copy(st3.zrows)
for _ = 1:5
    gpu_metropolis_sweep!(gst3, gH3, β; workgroupsize = WS)
end
to_host!(st3, gst3)
check("spin sweep leaves the displacements bitwise alone", st3.disps == u3)
check("spin sweep leaves the DISP rows bitwise alone",
      view(st3.zrows, (Hj.nlm + 1):Hj.nrows, :) ==
      view(zr3, (Hj.nlm + 1):Hj.nrows, :))

# --- 3. determinism on device (G3(a)) ------------------------------------------------

sa, ga, gsa = setup(Hj; seed = 0xbeef)
sb, gb, gsb = setup(Hj; seed = 0xbeef)
sc, gc, gsc = setup(Hj; seed = 0xbef0)
for _ = 1:20
    gpu_run_sweeps!(gsa, ga, sa, β, 1; renorm_interval = 0)
    gpu_run_sweeps!(gsb, gb, sb, β, 1; renorm_interval = 0)
    gpu_run_sweeps!(gsc, gc, sc, β, 1; renorm_interval = 0)
end
check("compound run: same seed ⇒ bitwise-identical trajectory",
      sa.config == sb.config && sa.disps == sb.disps && sa.zrows == sb.zrows &&
      gsa.energy == gsb.energy && gsa.acc_disp == gsb.acc_disp)
check("compound run: a different seed ⇒ a different trajectory",
      sa.config != sc.config && sa.disps != sc.disps)

# --- 4. the incremental-energy drift gate (physics-exact, backend-independent) --------

for interval in (0, 30)
    sd, gd, gsd = setup(Hj; seed = 0x5eed)
    gpu_run_sweeps!(gsd, gd, sd, β, 200; renorm_interval = interval)
    E = total_energy(Hj, sd.config, sd.disps)
    drift = abs(gsd.energy - E)
    check("joint drift gate (renorm_interval = $interval)",
          drift <= 1e-8 * max(1.0, abs(E)),
          @sprintf("drift %.3e vs |E| %.3e", drift, abs(E)))
end

# --- 5. the Einstein oscillator against its closed form ------------------------------

let a = 2.5, kT = 0.04
    et, eL = einstein_terms(a)
    He = MC.TiledHamiltonian(1, et, eL; dims = (4, 4, 4), fixed_reference = true)
    se, ge, gse = setup(He; seed = 0x4242, step_u = 0.12, clamped = true)
    βe = 1 / kT
    gpu_run_sweeps!(gse, ge, se, βe, 2_000; renorm_interval = 0)
    acc = 0.0
    n = 4_000
    for _ = 1:n
        gpu_run_sweeps!(gse, ge, se, βe, 1; renorm_interval = 0)
        to_host!(se, gse)
        acc += sum(u -> dot(u, u), se.disps) / He.n_sites
    end
    got = acc / n
    want = 3 * kT / (2a)
    check("Einstein ⟨|u|²⟩ = 3kT/(2a) on device",
          isapprox(got, want; rtol = 0.05),
          @sprintf("%.6f vs %.6f (%.2f %%), acc %.3f", got, want,
                   100 * (got - want) / want, gse.acc_disp / gse.att_disp))
end

# --- 6. CUDA vs KA-CPU: the same ensemble, statistically -----------------------------
#
# Not bitwise — Box–Muller and the accept test go through backend libm (G3(b)). This is
# the cross-backend gate the determinism contract actually promises.

function chain_stats(backend, seed; nth = 3_000, nme = 5_000)
    rng = Xoshiro(21)
    stx = MC.ChainState(Hj, rand_config(rng, Hj), rng, 0.6;
                        disps = rand_disps(rng, Hj), step_u = 0.05)
    gHx = GPUTiledHamiltonian(backend, Hj)
    gstx = GPUChainState(gHx, stx; seed = UInt64(seed))
    gpu_run_sweeps!(gstx, gHx, stx, β, nth; renorm_interval = 500)
    E = Float64[]
    U = Float64[]
    for _ = 1:nme
        gpu_run_sweeps!(gstx, gHx, stx, β, 1; renorm_interval = 0)
        to_host!(stx, gstx)
        push!(E, gstx.energy / Hj.n_sites)
        push!(U, sum(u -> dot(u, u), stx.disps) / Hj.n_sites)
    end
    return binned(E), binned(U), gstx.acc_metro / gstx.att_metro,
           gstx.acc_disp / gstx.att_disp
end

dev = chain_stats(BACKEND, 0xc0de)
hos = chain_stats(CPU(), 0x99)
for (nm, i) in (("⟨E⟩/site", 1), ("⟨|u|²⟩/site", 2))
    (md, sd, td) = dev[i]
    (mh, sh, th) = hos[i]
    nsig = abs(md - mh) / sqrt(sd^2 + sh^2)
    check("CUDA ≡ KA-CPU in distribution: $nm", nsig <= 4.0,
          @sprintf("%.6g ± %.2g vs %.6g ± %.2g → %.2fσ (τ %.1f/%.1f)",
                   md, sd, mh, sh, nsig, td, th))
end
check("acceptance rates agree across backends",
      isapprox(dev[3], hos[3]; atol = 0.02) && isapprox(dev[4], hos[4]; atol = 0.02),
      @sprintf("CUDA %.3f/%.3f vs KA-CPU %.3f/%.3f", dev[3], dev[4], hos[3], hos[4]))

# --- verdict --------------------------------------------------------------------------

println()
if isempty(FAILURES)
    println("SMOKE PASSED — the joint device path compiles, runs, and agrees.")
else
    println("SMOKE FAILED — ", length(FAILURES), " check(s):")
    for f in FAILURES
        println("  - ", f)
    end
    exit(1)
end
