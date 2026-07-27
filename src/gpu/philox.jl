# Philox4x32-10 — the keyed, stateless RNG of the GPU path (decision record
# docs/specs/gpu-prototype.md G2; Salmon, Moraes, Dror & Shaw, SC'11).
#
# Every draw is a pure function of logical coordinates (seed, site, sweep, slot):
# integer-only arithmetic, so the stream is bitwise identical on every backend and
# independent of thread scheduling by construction. This deliberately replaces the
# CPU path's per-site Xoshiro streams (a different chain — P6-breaking, GPU path
# only). The uniform bit convention is the strictly OPEN interval (0, 1) — unlike
# `rand()`'s [0, 1) — so `log(u)` in Box–Muller and the accept comparison never see
# an endpoint.

# Multiplier / Weyl constants of philox4x32 (Random123 reference implementation).
const _PHILOX_M0 = 0xd2511f53
const _PHILOX_M1 = 0xcd9e8d57
const _PHILOX_W0 = 0x9e3779b9
const _PHILOX_W1 = 0xbb67ae85

const _TWO_PI = 6.283185307179586    # Float64(2π), spelled out to stay device-safe

"""
    _philox_round(ctr, key) -> NTuple{4,UInt32}

One philox4x32 S-box round (Feistel-like: two 32×32→64 multiplies, xors with the
odd counter words and the key).
"""
@inline function _philox_round(ctr::NTuple{4,UInt32},
                               key::NTuple{2,UInt32})::NTuple{4,UInt32}
    p0 = UInt64(_PHILOX_M0) * UInt64(ctr[1])
    p1 = UInt64(_PHILOX_M1) * UInt64(ctr[3])
    hi0 = (p0 >>> 32) % UInt32
    lo0 = p0 % UInt32
    hi1 = (p1 >>> 32) % UInt32
    lo1 = p1 % UInt32
    return (hi1 ⊻ ctr[2] ⊻ key[1], lo1, hi0 ⊻ ctr[4] ⊻ key[2], lo0)
end

"""
    _philox4x32(ctr, key) -> NTuple{4,UInt32}

philox4x32-10: ten rounds with the Weyl key schedule (key bumped between rounds).
Verified against the Random123 `kat_vectors` known answers in `test_gpu.jl`.
"""
@inline function _philox4x32(ctr::NTuple{4,UInt32},
                             key::NTuple{2,UInt32})::NTuple{4,UInt32}
    for _ = 1:9
        ctr = _philox_round(ctr, key)
        key = (key[1] + _PHILOX_W0, key[2] + _PHILOX_W1)
    end
    return _philox_round(ctr, key)
end

# Counter layout (G2): ctr = (site, sweep, slot, 0). The fourth word is reserved
# zero — a future replica id / update-kind tag gets its own subspace without moving
# any existing stream. Key = the run seed split into two words.
@inline function _philox_block(seed::UInt64, site::Int32, sweep::Int32,
                               slot::UInt32)::NTuple{4,UInt32}
    ctr = (reinterpret(UInt32, site), reinterpret(UInt32, sweep), slot, 0x00000000)
    key = (seed % UInt32, (seed >>> 32) % UInt32)
    return _philox4x32(ctr, key)
end

# Per-proposal slot map (G2). One proposal consumes exactly three blocks; whether a
# slot is *evaluated* is branch-dependent, but the value each slot would produce is
# not — branch-dependent consumption disappears.
#
# The two move kinds take DISJOINT slots. They must: a compound sweep attempts both at
# the same `(seed, site, sweep)`, and a shared slot would tie the displacement accept
# uniform to the spin one — the moves would accept and reject together, which is not
# the chain either sweep's stationarity argument is about. Slots are cheap (a whole
# 32-bit counter word), so the map wastes them freely rather than packing.
const _SLOT_FLIP_ACC = 0x00000000    # words 1–2 → flip uniform, words 3–4 → accept
const _SLOT_AXIS12 = 0x00000001      # Box–Muller pair → axis normals n1, n2
const _SLOT_AXIS3_ANGLE = 0x00000002 # Box–Muller pair → axis normal n3, angle n4
const _SLOT_DISP_ACC = 0x00000003    # words 3–4 → the displacement accept uniform
const _SLOT_DISP12 = 0x00000004      # Box–Muller pair → shift components g1, g2
const _SLOT_DISP3 = 0x00000005       # Box–Muller pair → shift component g3 (+1 spare)

"""
    _philox_uniform(hi, lo) -> Float64

Two 32-bit words → one Float64 uniform on the strictly open interval (0, 1):
the top 52 of the 64 bits, centered — `(w >>> 12 + 0.5) · 2⁻⁵²`. Both the +0.5 and
the product are exact (52-bit integer + half fits a Float64 mantissa below 2⁵³), so
the endpoints are `2⁻⁵³` and `1 − 2⁻⁵³` — never 0.0 or 1.0. A 53-bit variant would
round its top value to exactly 1.0.
"""
@inline function _philox_uniform(hi::UInt32, lo::UInt32)::Float64
    w = (UInt64(hi) << 32) | UInt64(lo)
    return (Float64(w >>> 12) + 0.5) * 0x1p-52
end

"""
    _philox_uniform2(blk) -> (Float64, Float64)

Both uniforms of one block: words 1–2 and words 3–4.
"""
@inline function _philox_uniform2(blk::NTuple{4,UInt32})::NTuple{2,Float64}
    return (_philox_uniform(blk[1], blk[2]), _philox_uniform(blk[3], blk[4]))
end

"""
    _philox_normal2(blk) -> (Float64, Float64)

Box–Muller pair from one block's two uniforms: `r = √(−2 log u₁)`,
`(r cos 2πu₂, r sin 2πu₂)`. Uses libm `log`/`cos`/`sin` — bitwise identical only
within one backend (determinism contract G3(b)); `u₁ > 0` strictly, so `log` is
finite.
"""
@inline function _philox_normal2(blk::NTuple{4,UInt32})::NTuple{2,Float64}
    u1, u2 = _philox_uniform2(blk)
    r = sqrt(-2.0 * log(u1))
    a = _TWO_PI * u2
    return (r * cos(a), r * sin(a))
end

# --- public facade ----------------------------------------------------------------
#
# The counter-based stream as a stable, public contract for dependent packages
# (SLCEDynamics' thermal noise). Thin wrappers over the gated internals — the
# Random123 known-answer test in test_gpu.jl pins the block function, so any
# consumer's stream is anchored to the reference implementation. Counter-layout
# discipline: this package's GPU sweeps use `ctr[4] == 0`; a dependent package
# must claim a NONZERO `ctr[4]` domain tag so its streams are disjoint from every
# MC stream under a shared seed.

"""
    philox_block(seed::UInt64, ctr::NTuple{4,UInt32}) -> NTuple{4,UInt32}

One philox4x32-10 block: a pure, integer-only function of `(seed, ctr)` —
bit-identical on every backend and free of sequential state. The seed is the
two-word key. See the counter-layout note above for domain separation.
"""
philox_block(seed::UInt64, ctr::NTuple{4,UInt32})::NTuple{4,UInt32} =
    _philox4x32(ctr, (seed % UInt32, (seed >>> 32) % UInt32))

"""
    philox_normal2(blk::NTuple{4,UInt32}) -> NTuple{2,Float64}

Box–Muller pair from one block: two standard normals from the block's two
strictly-open-(0,1) uniforms. Uses libm `log`/`cos`/`sin` — bitwise identical
only within one backend (the P6 / G3(b) determinism scope).
"""
philox_normal2(blk::NTuple{4,UInt32})::NTuple{2,Float64} = _philox_normal2(blk)
