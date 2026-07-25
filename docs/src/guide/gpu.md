# The GPU path

```@meta
CurrentModule = SLCEMonteCarlo
```

The device path runs the single-spin Metropolis sweep — any body order, the exact
``\Delta E`` of the fitted Hamiltonian — on a KernelAbstractions backend:
[`GPUTiledHamiltonian`](@ref) uploads the tiled tables once,
[`GPUChainState`](@ref) holds one chain on the device, [`gpu_run_sweeps!`](@ref)
drives it, and [`to_host!`](@ref) downloads the state for measurement. The API is
exported (since the A100 go/no-go and two production-model validations — see
`docs/specs/gpu-prototype.md`).

## Scope: a chain-level API

`run_mc` / `run_pt` remain **CPU drivers** — the device path is the chain tier
underneath them, and it is deliberately narrow:

- **Metropolis only.** No overrelaxation sweeps, no parallel-tempering rungs, no
  adaptive-step schedule (the proposal `step` is whatever the uploaded
  `ChainState` carries — thermalize/adapt on the host first, or set it
  explicitly).
- **Measurement happens on the host.** [`to_host!`](@ref) downloads the
  configuration (and the running energy) into a `ChainState`; observables,
  binning, and `Evaluable`s then use the ordinary CPU machinery.
- Temperature schedules, annealing ladders, and checkpointing are the caller's
  loop.

```julia
using SLCEMonteCarlo, CUDA

H   = TiledHamiltonian(model; dims = (8, 8, 8))
st  = SLCEMonteCarlo.ChainState(H, config0, Xoshiro(1), 0.6)   # host chain, step 0.6
gH  = GPUTiledHamiltonian(CUDABackend(), H)          # upload tables ONCE, reuse
gst = GPUChainState(gH, st; seed = 0x5ce)            # keys the device Philox

gpu_run_sweeps!(gst, gH, st, 1 / kT, 10_000; renorm_interval = 1_000)
to_host!(st, gst)                                    # st now holds the result
```

The package has **no CUDA dependency** — the caller passes the backend object
(`CUDABackend()`, or `KernelAbstractions.CPU()` to run the same code path on the
host). `gpu_run_sweeps!` renormalizes on the host every `renorm_interval` sweeps
(drift check + energy re-anchor) and downloads the final state before returning.

## Determinism

The device sweep draws keyed counter-based Philox4x32-10 noise — a draw is a pure
function of `(seed, site, sweep)`, with no RNG state on the device. A device
trajectory is bitwise reproducible for a fixed (`seed`, backend,
`workgroupsize`, package + Julia version); `workgroupsize` (pinned default 128)
is part of the contract. A CPU chain and a device chain are **different
realizations** of the same ensemble (the CPU chain uses its own `Xoshiro`
stream) — they are compared statistically, never bitwise. The gate suite runs
the full device code path on the KA-CPU backend against a keyed serial
reference, bitwise.

A small live run on the KA-CPU backend (the cubic-Heisenberg model of the
tutorial):

```@example gpu
using SLCEMonteCarlo, SLCE
import Spglib                      # activates SLCE's SpglibBackend extension
using LinearAlgebra, Random

lat = Lattice(Matrix(1.0 * I(3)))
cell = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
spec = BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [1], isotropy = true)
basis = SLCEBasis(cell, spec; backend = SpglibBackend(), images = AllImages())
model = SLCEModel(basis, 0.0, [-0.01])
H = TiledHamiltonian(model; dims = (4, 4, 4))

config0 = SLCEMonteCarlo.from_matrix(randn(Xoshiro(11), 3, n_sites(H)))

backend = SLCEMonteCarlo.KernelAbstractions.CPU()
gH = GPUTiledHamiltonian(backend, H)

st1 = SLCEMonteCarlo.ChainState(H, config0, Xoshiro(11), 0.6)
gst1 = GPUChainState(gH, st1; seed = UInt64(0xc0ffee))
gpu_run_sweeps!(gst1, gH, st1, 1 / 0.02, 200; renorm_interval = 50)

st2 = SLCEMonteCarlo.ChainState(H, config0, Xoshiro(11), 0.6)
gst2 = GPUChainState(gH, st2; seed = UInt64(0xc0ffee))
gpu_run_sweeps!(gst2, gH, st2, 1 / 0.02, 200; renorm_interval = 50)

(energy_per_site = st1.energy / n_sites(H),
 acceptance = round(gst1.acc_metro / gst1.att_metro; digits = 3),
 repeat_bitwise = st1.config == st2.config)
```

## Measured performance (A100)

From the decision record `docs/specs/gpu-prototype.md` (kugui A100-SXM4-40GB vs
the same-node CPU baseline): **30.1×** at the 8³ go/no-go bar on the nbody = 3
Nd₂Fe₁₄B fixture (bar was ≥ 5×), and on real fitted production models **14.4×**
(l02, isotropic bilinear, 8³) and **38.1×** (l044, nbody = 3, 4672 SALCs, 8³ —
15.11 s → 396 ms per sweep; a 12k-sweep campaign drops from ~50 h to 79 min).
Physics gates: energies and magnetizations agree with independent CPU chains
within statistics at every validated size, drift gates pass, acceptance matches.
Table memory is the size limit for heavy models (l044: 12.7 GiB at 8³; 16³ does
not fit the 40 GB part). No performance claims beyond these measurements.

## The gradient tier (dependent packages)

The device all-site gradient — `SLCEMonteCarlo.gpu_energy_gradient!` with
`SLCEMonteCarlo.GPUGradientScratch` and the row builder
`SLCEMonteCarlo.gpu_zlm_rows!` — is the seam consumed by `SLCEDynamics.jl`'s
GPU dynamics. It stays **public, unexported** (call it qualified): it is an
inter-package contract, not an end-user surface.
