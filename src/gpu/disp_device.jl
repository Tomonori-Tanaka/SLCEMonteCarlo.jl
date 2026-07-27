# Device-safe displacement basis rows — the displacement-channel counterpart of
# `zlm_device.jl` (decision record docs/specs/gpu-prototype.md G4/G8; gate: the
# displacement-row section of test/unit/test_gpu.jl).
#
# Why it is a replication rather than a call: the upstream
# `SLCE.SolidHarmonics.solid_harmonics!` wrapper validates its arguments with throws
# carrying runtime strings, which cannot compile inside a GPU kernel. Its CORE
# (`_solid_harmonics_impl!`) is already pure scalar arithmetic with no allocation and
# no dispatch — unlike the tesseral path's `dnPl`, which needed a genuine
# operation-order transcription — so this file is that core with the gradient half
# (the `grads !== nothing` branch and its `Az`/`Ar` recurrence state, which never
# feed `vals`) dropped and `lmax` lifted to a `Val` for static stack buffers.
#
# COUPLED SITES: `SLCE.SolidHarmonics._solid_harmonics_impl!` (the recurrences and
# their operation order), `SLCE.SolidHarmonics._racah_norm` /
# `_double_factorial_odd` / `solid_harmonic_index`, and this package's
# `_disp_rows!` (energy.jl — the `(k, l)` block loop and the `r2^k` prefactor). Any
# upstream change breaks the bitwise gate loudly; update this file with it.
#
# DETERMINISM SCOPE — read this before assuming "bitwise" the way `zlm_device.jl`
# means it.
#
# `_solid_row_device!` is `+ - * /` and `sqrt` only: IEEE-exact, and gated bitwise
# against `SLCE.SolidHarmonics.solid_harmonics!`.
#
# `_disp_rows_device!` adds the `|u|^{2k}` prefactor, and there the k ≥ 1 blocks are
# bitwise-identical to the host `_disp_rows!` only up to ONE ULP of `r2`, amplified by
# the exponent. Two reasons, in order of size:
#
#  1. `r2 = dot(u, u)` is a mul-add chain, so its last bit depends on whether LLVM
#     CONTRACTS it into an FMA — and that decision is not stable across compilation
#     contexts. The host computes `r2` in a function whose harmonic batch is a
#     separate, non-inlined call; the device inlines the batch (it must, for a
#     kernel), which puts a second `x² + y² + z²` in the same scope, and the two
#     then get canonicalized/CSEd together. Measured here: ~22 % of random `u` give a
#     1-ulp difference in `r2`, and neither reordering the two computations nor
#     transcribing `dot` as an explicit `muladd` chain removes it.
#  2. `r2^k` is `Float64 ^ Int`, which Julia routes through `x^Float64(k)` — a libm
#     `pow`, NOT `power_by_squaring` (they diverge from k = 4). `pow(x, 0) = 1` and
#     `pow(x, 1) = x` are exact, so this one only bites at k ≥ 2.
#
# Consequence for G4's renormalization seam: the host and device DISPLACEMENT rows of
# a k ≥ 1 block may differ in the last bit, so a host round-trip is not bit-preserving
# there the way it is for the tesseral rows. k = 0 blocks — every model whose
# displacement sector is built from `2k + l` with k = 0, and every row of the spin
# block — are unaffected. Gate: `test/unit/test_gpu.jl` checks k = 0 rows bitwise and
# k ≥ 1 rows at ≤ (kmax + 1) ulp.
#
# A stable k ≥ 1 contract needs ONE `r2` shared by the harmonic core and the
# prefactor, which is an upstream (`SLCE.site_rows!`) convention change — deliberately
# NOT taken here.

@inline _sh_index(l::Int, m::Int)::Int = l * l + l + m + 1

# SolidHarmonics._double_factorial_odd, verbatim: (2n − 1)!! with (−1)!! = 1.
@inline function _sh_dfact_odd(n::Int)::Float64
    acc = 1.0
    for k = 1:2:(2 * n - 1)
        acc *= k
    end
    return acc
end

# SolidHarmonics._racah_norm, verbatim: 1 for m = 0, √(2·(l−m)!/(l+m)!) otherwise,
# formed by a division loop rather than factorials.
@inline function _sh_racah_norm(l::Int, m::Int)::Float64
    m == 0 && return 1.0
    acc = 2.0
    for i = (l - m + 1):(l + m)
        acc /= i
    end
    return sqrt(acc)
end

"""
    _solid_row_device!(vals, u, ::Val{LMAX}) -> nothing

Fill `vals[1:(LMAX+1)²]` with the real solid harmonics `R_{l,m}(u)` in
`SLCE.SolidHarmonics.solid_harmonic_index` order — the device replica of
`SLCE.SolidHarmonics.solid_harmonics!(vals, LMAX, u)`, bitwise.

Polynomial in the Cartesian components throughout (no division by `|u|`), so it is
regular and exact at `u = 0`, where every `l ≥ 1` row is `0.0` and `R_{0,0} = 1`.
"""
@inline function _solid_row_device!(vals::AbstractVector{Float64},
                                    u::SVector{3,Float64},
                                    ::Val{LMAX})::Nothing where {LMAX}
    x = u[1]
    y = u[2]
    z = u[3]
    r2 = x * x + y * y + z * z
    # (c_n, s_n) = Re/Im of (x + i y)^n, by the same two-term recurrence upstream uses
    c = 1.0
    s = 0.0
    @inbounds for n = 0:LMAX
        # A_l^n = r^{l−n} P_l^{(n)}(z/r), homogeneous of degree l − n; seeds
        # A_n^n = (2n−1)!! and A_{n+1}^n = (2n+1)!! z. The gradient state (Az, Ar)
        # of the upstream core is dropped — `vals` never reads it.
        Aprev = 0.0
        A = _sh_dfact_odd(n)
        for l = n:LMAX
            K = _sh_racah_norm(l, n)
            if n == 0
                vals[_sh_index(l, 0)] = K * A
            else
                vals[_sh_index(l, n)] = K * A * c
                vals[_sh_index(l, -n)] = K * A * s
            end
            l == LMAX && break
            lnew = l + 1
            if lnew == n + 1
                Anew = (2.0 * n + 1.0) * z * A
            else
                denom = lnew - n
                a1 = 2.0 * lnew - 1.0
                a2 = lnew + n - 1.0
                Anew = (a1 * z * A - a2 * r2 * Aprev) / denom
            end
            Aprev = A
            A = Anew
        end
        cm1 = c
        sm1 = s
        c = x * cm1 - y * sm1
        s = x * sm1 + y * cm1
    end
    return nothing
end

"""
    _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, ::Val{DLMAX}) -> nothing

Fill one site's DISPLACEMENT basis rows of `rows` — the device replica of
`_disp_rows!` (energy.jl), bitwise. `fac_k`/`fac_l`/`fac_start` are the flattened
`RowLayout` displacement blocks (`disp_factors` split into its two integer columns,
plus `disp_starts`), and `rbuf` is scratch of at least `(DLMAX+1)²` entries.

Row `fac_start[i] + m + l + 1` of block `i = (k, l)` carries `|u|^{2k} R_{l,m}(u)`.
The SPIN rows of `rows` are not touched.
"""
@inline function _disp_rows_device!(rows::AbstractVector{Float64},
                                    rbuf::AbstractVector{Float64},
                                    u::SVector{3,Float64},
                                    fac_k::AbstractVector{Int32},
                                    fac_l::AbstractVector{Int32},
                                    fac_start::AbstractVector{Int32},
                                    ::Val{DLMAX})::Nothing where {DLMAX}
    _solid_row_device!(rbuf, u, Val(DLMAX))
    r2 = dot(u, u)
    @inbounds for i = 1:length(fac_k)
        k = Int(fac_k[i])
        l = Int(fac_l[i])
        r2k = r2^k                          # libm `pow` for k ≥ 2 (see the scope note)
        base = Int(fac_start[i])
        for m = -l:l
            rows[base + m + l + 1] = r2k * rbuf[_sh_index(l, m)]
        end
    end
    return nothing
end

# Runtime-lmax dispatch onto the Val-specialized device row (reference/test use), the
# `_zlm_row_device_dyn!` counterpart.
function _solid_row_device_dyn!(vals::AbstractVector{Float64}, u::SVector{3,Float64},
                                lmax::Int)::Nothing
    if lmax == 0
        _solid_row_device!(vals, u, Val(0))
    elseif lmax == 1
        _solid_row_device!(vals, u, Val(1))
    elseif lmax == 2
        _solid_row_device!(vals, u, Val(2))
    elseif lmax == 3
        _solid_row_device!(vals, u, Val(3))
    elseif lmax == 4
        _solid_row_device!(vals, u, Val(4))
    elseif lmax == 5
        _solid_row_device!(vals, u, Val(5))
    elseif lmax == 6
        _solid_row_device!(vals, u, Val(6))
    else
        throw(ArgumentError("disp_lmax = $lmax unsupported on the device path (≤ 6)"))
    end
    return nothing
end

function _disp_rows_device_dyn!(rows::AbstractVector{Float64},
                                rbuf::AbstractVector{Float64}, u::SVector{3,Float64},
                                fac_k::AbstractVector{Int32},
                                fac_l::AbstractVector{Int32},
                                fac_start::AbstractVector{Int32}, dlmax::Int)::Nothing
    if dlmax == 0
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(0))
    elseif dlmax == 1
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(1))
    elseif dlmax == 2
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(2))
    elseif dlmax == 3
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(3))
    elseif dlmax == 4
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(4))
    elseif dlmax == 5
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(5))
    elseif dlmax == 6
        _disp_rows_device!(rows, rbuf, u, fac_k, fac_l, fac_start, Val(6))
    else
        throw(ArgumentError("disp_lmax = $dlmax unsupported on the device path (≤ 6)"))
    end
    return nothing
end

"""
    _disp_layout_tables(L::RowLayout) -> (fac_k, fac_l, fac_start)

The `RowLayout` displacement blocks as three flat `Int32` host vectors, ready for
`_to_device`. Empty on a pure-spin layout.
"""
function _disp_layout_tables(L::RowLayout)
    n = length(L.disp_factors)
    fac_k = Vector{Int32}(undef, n)
    fac_l = Vector{Int32}(undef, n)
    fac_start = Vector{Int32}(undef, n)
    for i = 1:n
        k, l = L.disp_factors[i]
        fac_k[i] = Int32(k)
        fac_l[i] = Int32(l)
        fac_start[i] = Int32(L.disp_starts[i])
    end
    return fac_k, fac_l, fac_start
end
