# GPU Metropolis sweep vs the tuned CPU path — the go/no-go bench of
# docs/specs/gpu-feasibility.md F7 / gpu-prototype.md G6.
#
#   julia --project=bench/gpu bench/bench_gpu.jl [n_2141_max] [nsweeps]
#
# Backend selection: uses CUDA when functional, else the KernelAbstractions CPU
# backend (correctness smoke — its throughput is NOT a GPU number). Reports, per
# supercell size of the nbody=3 Nd₂Fe₁₄B fixture (plus a bcc control): table
# upload time/bytes, device ms/sweep, the SAME-NODE CPU 4-task baseline, the
# acceptance-rate sanity check, and the go/no-go ratio at the 8³ bar (< 5× ⇒
# stop, per the decision record — never compare against another machine's CPU
# numbers). Every line flushes (batch-job logs survive a walltime kill).

using SLCEMonteCarlo
using SLCE
using KernelAbstractions: KernelAbstractions, CPU
using LinearAlgebra
using Printf
using Random
using StaticArrays

include(joinpath(@__DIR__, "fixtures.jl"))

const HAVE_CUDA = try
    using CUDA
    CUDA.functional()
catch
    false
end

backend = HAVE_CUDA ? CUDABackend() : CPU()
backend_name = HAVE_CUDA ? "CUDA ($(CUDA.name(CUDA.device())))" : "KA-CPU (smoke)"

n_2141_max = argn(1, 8)
nsweeps = argn(2, 100)
const WS = 128

bench_header("GPU Metropolis sweep — backend: $backend_name, ws=$WS, " *
             "kT=$(BENCH_KT) eV, $nsweeps sweeps/point")
flush(stdout)

_gib(b) = b / 2^30

function table_bytes(H)
    bytes = 0
    for obj in (H, H.progs)
        for f in propertynames(obj)
            v = getproperty(obj, f)
            v isa Array && (bytes += sizeof(v))
        end
    end
    return bytes
end

function gpu_point(name, H; cpu_baseline::Bool = true)
    β = 1 / BENCH_KT
    println("\n--- $name: ", describe(H))
    flush(stdout)

    # upload
    t_up = @elapsed begin
        gH = MC.GPUTiledHamiltonian(backend, H)
        HAVE_CUDA && CUDA.synchronize()
    end
    rng = Xoshiro(7)
    st = MC.ChainState(H, MC._initial_config(H, nothing, rng), rng, 0.6)
    gst = MC.GPUChainState(gH, st; seed = UInt64(0xbe11c0de))
    @printf("upload: %.2f s   tables %.2f GiB (host-side size)\n", t_up,
            _gib(table_bytes(H)))
    flush(stdout)

    # device sweeps (warmup + timed; synchronize is inside the driver)
    for _ = 1:10
        MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS)
    end
    t_dev = @elapsed for _ = 1:nsweeps
        MC.gpu_metropolis_sweep!(gst, gH, β; workgroupsize = WS)
    end
    dev_ms = 1e3 * t_dev / nsweeps
    acc_dev = gst.acc_metro / max(gst.att_metro, 1)
    @printf("device:  %9.2f ms/sweep   %8.1f ns/attempt   acc=%.2f\n", dev_ms,
            1e9 * t_dev / (nsweeps * H.n_active), acc_dev)
    flush(stdout)

    cpu_ms = NaN
    if cpu_baseline
        # same-node tuned CPU path, 4 sweep tasks (the production configuration)
        ntasks = min(4, Threads.nthreads())
        st2, _ = chain_state(H)                       # step 0.6 frozen — same as gst
        scs = [MC.SweepScratch(H) for _ = 1:ntasks]
        for _ = 1:10                                  # warmup, symmetric with device
            metropolis_sweep!(st2, H, β, ntasks == 1 ? scs[1] : scs)
        end
        t_cpu = @elapsed for _ = 1:nsweeps
            metropolis_sweep!(st2, H, β, ntasks == 1 ? scs[1] : scs)
        end
        cpu_ms = 1e3 * t_cpu / nsweeps
        acc_cpu = st2.acc_metro / max(st2.att_metro, 1)
        @printf("cpu %dT:  %9.2f ms/sweep   %8.1f ns/attempt   acc=%.2f\n", ntasks,
                cpu_ms, 1e9 * t_cpu / (nsweeps * H.n_active), acc_cpu)
        @printf("ratio (cpu/device): %.2fx\n", cpu_ms / dev_ms)
        # acceptance sanity: both samplers at the same kT should land close
        abs(acc_dev - acc_cpu) < 0.1 ||
            println("WARNING: acceptance mismatch device=$acc_dev cpu=$acc_cpu")
    end
    flush(stdout)
    return (; dev_ms, cpu_ms)
end

# On a real device, run the two backend-independent correctness gates first
# (gpu-prototype.md G5): repeated-run bitwise identity and the drift gate.
function device_correctness_smoke()
    H = TiledHamiltonian(nd2fe14b3_model(); dims = (2, 2, 2))
    β = 1 / BENCH_KT
    gH = MC.GPUTiledHamiltonian(backend, H)
    rng = Xoshiro(3)
    st = MC.ChainState(H, MC._initial_config(H, nothing, rng), rng, 0.6)
    g1 = MC.GPUChainState(gH, st; seed = UInt64(11))
    g2 = MC.GPUChainState(gH, st; seed = UInt64(11))
    for _ = 1:50
        MC.gpu_metropolis_sweep!(g1, gH, β; workgroupsize = WS)
        MC.gpu_metropolis_sweep!(g2, gH, β; workgroupsize = WS)
    end
    same = Array(g1.config) == Array(g2.config) && g1.energy == g2.energy
    MC.to_host!(st, g1)
    E = total_energy(H, st.config)
    drift_ok = abs(g1.energy - E) <= 1e-8 * max(1.0, abs(E))
    println("device correctness: repeat-identity=", same, "  drift-gate=", drift_ok)
    (same && drift_ok) || error("device correctness smoke FAILED — do not bench")
    flush(stdout)
end
HAVE_CUDA && device_correctness_smoke()

# bcc control (light kernel — the case the GPU should NOT be expected to win)
gpu_point("bcc Fe 16³ (light-kernel control)",
          TiledHamiltonian(bcc_fe_model(); dims = (16, 16, 16)))

# the target regime: nbody=3 Nd₂Fe₁₄B at growing supercells
results = Dict{Int,Any}()
model3 = nd2fe14b3_model()
for n in (4, 8, 16)
    n <= n_2141_max || continue
    results[n] = gpu_point("Nd2Fe14B nbody=3 $(n)³",
                           TiledHamiltonian(model3; dims = (n, n, n)))
end

if haskey(results, 8) && HAVE_CUDA
    r = results[8]
    ratio = r.cpu_ms / r.dev_ms
    verdict = ratio >= 5 ? "GO (≥ 5x)" : "NO-GO (< 5x — keep the CPU path)"
    @printf("\n=== go/no-go @ 8³ nbody=3: %.2fx vs same-node cpu-4T → %s ===\n",
            ratio, verdict)
elseif !HAVE_CUDA
    println("\n(KA-CPU smoke only — no go/no-go readout without a CUDA device)")
end
flush(stdout)

# ---------------------------------------------------------------------------
# Phase-2 gradient section (G7): T_grad per eval, the grad/sweep ratio, and the
# CUDA-side bitwise claim vs the lane reference (GR9 — fallback: report the
# scaled deviation and record it in the G7 decision record).
# ---------------------------------------------------------------------------
function gradient_point(label::String, H)
    rng = Xoshiro(5)
    config = MC._initial_config(H, nothing, rng)
    gH = MC.GPUTiledHamiltonian(backend, H)
    gsc = MC.GPUGradientScratch(gH)
    dconfig = KernelAbstractions.allocate(backend, SVector{3,Float64}, H.n_sites)
    copyto!(dconfig, config)
    dG = KernelAbstractions.allocate(backend, SVector{3,Float64}, H.n_sites)
    MC.gpu_energy_gradient!(dG, gH, dconfig, gsc; workgroupsize = WS)  # warmup
    t = @elapsed for _ = 1:20
        MC.gpu_energy_gradient!(dG, gH, dconfig, gsc; workgroupsize = WS)
    end
    grad_ms = 1e3 * t / 20
    # GR9: bitwise vs the serial lane reference (host)
    zrows = MC._zrows(H, config)
    ref = Vector{SVector{3,Float64}}(undef, H.n_sites)
    MC._gradient_lane_ref!(ref, H, config, zrows, WS)
    G = Vector(dG)
    if G == ref
        println(label, ": grad ", round(grad_ms; digits = 3),
                " ms/eval — GR9 bitwise OK")
    else
        scale = max(1.0, maximum(norm, ref))
        dev = maximum(norm.(G .- ref)) / scale
        println(label, ": grad ", round(grad_ms; digits = 3),
                " ms/eval — GR9 NOT bitwise (scaled dev ", dev,
                ") — record in G7, gate falls back to 1e-12 tolerance")
        dev <= 1e-12 || error("gradient exceeds even the fallback tolerance")
    end
    return grad_ms
end

if haskey(results, 8)
    gms = gradient_point("Nd2Fe14B nbody=3 8³ gradient",
                         TiledHamiltonian(model3; dims = (8, 8, 8)))
    r = results[8]
    @printf("grad/sweep ratio @ 8³: %.2f (LLG step ≈ 2 grads ≈ %.0f ms)\n",
            gms / r.dev_ms, 2 * gms)
    flush(stdout)
end
