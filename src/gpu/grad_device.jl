# Device-safe tesseral-harmonic GRADIENT row — the bitwise-faithful device twin
# of the host path `Harmonics.grad_Zlm_unsafe` → `_barP`/`_dbarP` →
# `LegendrePolynomials.dnPl` → `_grad_zlm_assemble` (decision record
# docs/specs/gpu-prototype.md G7; gate: the gradient-row section of
# test/unit/test_gpu.jl). It reuses `zlm_device.jl`'s `_zlm_parity` /
# `_zlm_plm_norm` / `_zlm_dnpl` / `_zlm_cpow` replicas and adds only what the
# gradient needs.
#
# COUPLED SITES (in addition to zlm_device.jl's list): any upstream change to
# `Harmonics.grad_Zlm_unsafe`, `_grad_zlm_assemble`, `_barP`/`_dbarP`, or to
# LegendrePolynomials' `dnPl` trivial-zero branch (`l < n` returns +0.0 BEFORE
# touching the cache — replicated by `_zlm_dnpl_or0`) breaks the bitwise gate
# loudly — update this file with it. The whole pipeline is `+ − * /` and
# correctly-rounded `sqrt` (no libm), so the row is IEEE-exact and expected to
# be bitwise identical across backends; keep `muladd`/`@fastmath` OUT of it.

# dnPl including the upstream trivial-zero branch: `dⁿ⁺ᵏPₗ = 0` for l < n,
# returned as a +0.0 LITERAL before the cache is touched (LegendrePolynomials'
# exact behavior). The host then multiplies parity·norm·(+0.0), which yields a
# −0.0 for odd parity — the sign is produced by the multiply chain in
# `_grad_zlm_device`, never here.
@inline _zlm_dnpl_or0(x::Float64, l::Int, n::Int,
                      cache::MVector{N,Float64}) where {N} =
    l < n ? 0.0 : _zlm_dnpl(x, l, n, cache)

"""
    _grad_zlm_device(l, m, u, cache) -> SVector{3,Float64}

Tangent-projected gradient `∇Zₗₘ(u)` (`u·∇Z = 0` analytically), the
operation-order-faithful device replica of `Harmonics.grad_Zlm_unsafe` (assumes
`|m| ≤ l`; `cache` as in `_zlm_dnpl` — `LMAX + 1` covers every call, including
the order-`n + 1` derivative lift).
"""
@inline function _grad_zlm_device(l::Int, m::Int, u::SVector{3,Float64},
                                  cache::MVector{N,Float64})::SVector{3,Float64} where {N}
    x, y, z = u[1], u[2], u[3]
    n = abs(m)
    # _barP / _dbarP, verbatim (two sequential dnPl calls share the cache)
    plm = _zlm_parity(n) * _zlm_plm_norm(l, n) * _zlm_dnpl(z, l, n, cache)
    dplm = _zlm_parity(n) * _zlm_plm_norm(l, n) * _zlm_dnpl_or0(z, l, n + 1, cache)
    # _grad_zlm_assemble, expression for expression (including the `/ r2` radial
    # removal — dropping it here while the host divides would break the bitwise
    # host ≡ device parity gate, which is exactly what that gate is for)
    r2 = x * x + y * y + z * z
    if m == 0
        zz = z * dplm / r2
        return SVector{3,Float64}(-x * zz, -y * zz, dplm - z * zz)
    end
    c = _zlm_parity(n) * sqrt(2.0)
    zxy = ComplexF64(x, y)
    zpn = _zlm_cpow(zxy, n)
    zpn1 = _zlm_cpow(zxy, n - 1)
    rn, iN = real(zpn), imag(zpn)
    rn1, in1 = real(zpn1), imag(zpn1)
    dZx, dZy, dZz = if m > 0
        (c * n * plm * rn1, -c * n * plm * in1, c * dplm * rn)
    else
        (c * n * plm * in1, c * n * plm * rn1, c * dplm * iN)
    end
    zz = (x * dZx + y * dZy + z * dZz) / r2   # û · ∂Z, the radial part to remove
    return SVector{3,Float64}(dZx - x * zz, dZy - y * zz, dZz - z * zz)
end

"""
    _grad_zlm_row_device!(grow, u, ::Val{LMAX}) -> nothing

Fill `grow[1:3(LMAX+1)²]` with the gradient rows `∇Z_{lm}(u)` in `lm_index`
order, component-fastest (`grow[3(k−1)+d]`, `d = 1:3` — the layout
`_entry_walk_grad` loads as an `SVector` per entry). `grow` may be any writable
`Float64` vector (`@localmem` inside a kernel).
"""
@inline function _grad_zlm_row_device!(grow::AbstractVector{Float64},
                                       u::SVector{3,Float64},
                                       ::Val{LMAX})::Nothing where {LMAX}
    cache = MVector{LMAX + 1,Float64}(undef)
    i = 0
    for l = 0:LMAX, m = -l:l
        g = _grad_zlm_device(l, m, u, cache)
        @inbounds begin
            grow[3 * i + 1] = g[1]
            grow[3 * i + 2] = g[2]
            grow[3 * i + 3] = g[3]
        end
        i += 1
    end
    return nothing
end

# Runtime-lmax dispatch onto the Val-specialized gradient row (reference/test
# use — the mirror of `_zlm_row_device_dyn!`).
function _grad_zlm_row_device_dyn!(grow::AbstractVector{Float64},
                                   u::SVector{3,Float64}, lmax::Int)::Nothing
    if lmax == 0
        _grad_zlm_row_device!(grow, u, Val(0))
    elseif lmax == 1
        _grad_zlm_row_device!(grow, u, Val(1))
    elseif lmax == 2
        _grad_zlm_row_device!(grow, u, Val(2))
    elseif lmax == 3
        _grad_zlm_row_device!(grow, u, Val(3))
    elseif lmax == 4
        _grad_zlm_row_device!(grow, u, Val(4))
    elseif lmax == 5
        _grad_zlm_row_device!(grow, u, Val(5))
    elseif lmax == 6
        _grad_zlm_row_device!(grow, u, Val(6))
    else
        throw(ArgumentError("lmax = $lmax unsupported on the device path (≤ 6)"))
    end
    return nothing
end
