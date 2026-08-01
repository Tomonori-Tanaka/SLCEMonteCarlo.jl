# Streaming error analysis (see `docs/specs/binning-observables.md`).
#
# `LogBinner` — logarithmic binning with O(levels) memory: level `k` accumulates the
# statistics of bin means of size `2^(k-1)`; a completed pair at level `k` cascades
# its mean to level `k+1`. The autocorrelation-aware error is the naive standard
# error at the deepest level that still holds ≥ `_MIN_BINS` bins (the plateau proxy).
# Full time series are never stored — that is what keeps long parallel-tempering
# runs at O(MB).
#
# `BinStore` — a fixed number of equal-size bin means, kept for jackknifing derived
# (nonlinear) quantities. `jackknife` is the standard leave-one-bin-out estimator.

const _MIN_BINS = 32       # plateau: deepest level with at least this many bins
const _MAX_LEVELS = 62     # 2^62 measurements ≫ any run

"""
    LogBinner(ncomp::Integer)

Streaming logarithmic binning accumulator for an `ncomp`-component observable.
`push!(b, x)` (scalar `ncomp == 1`) or `push!(b, xs)` feeds one measurement; level
`k` of the cascade holds the statistics of bin means of size `2^(k-1)`, in O(levels)
memory (no time series is stored). Read out with `Statistics.mean(b)`,
[`std_error`](@ref) and [`tau_int`](@ref).
"""
mutable struct LogBinner
    const ncomp::Int
    const count::Vector{Int}          # completed entries per level
    const sums::Matrix{Float64}       # level × comp: Σ of bin means
    const sums2::Matrix{Float64}      # level × comp: Σ of squared bin means
    const pending::Matrix{Float64}    # level × comp: the waiting half-pair
    const pending_full::Vector{Bool}
    n::Int                            # total measurements pushed

    function LogBinner(ncomp::Integer)
        ncomp >= 1 || throw(ArgumentError("ncomp must be ≥ 1; got $ncomp"))
        return new(ncomp, zeros(Int, _MAX_LEVELS), zeros(_MAX_LEVELS, ncomp),
                   zeros(_MAX_LEVELS, ncomp), zeros(_MAX_LEVELS, ncomp),
                   zeros(Bool, _MAX_LEVELS), 0)
    end

    # Checkpoint-restore path: rebuild from captured cascade state verbatim.
    LogBinner(ncomp::Int, count::Vector{Int}, sums::Matrix{Float64},
              sums2::Matrix{Float64}, pending::Matrix{Float64},
              pending_full::Vector{Bool}, n::Int) =
        new(ncomp, count, sums, sums2, pending, pending_full, n)
end

Base.show(io::IO, b::LogBinner) =
    print(io, "LogBinner(", b.ncomp, " comps, ", b.n, " measurements)")

function Base.push!(b::LogBinner, xs::AbstractVector{<:Real})::LogBinner
    length(xs) == b.ncomp || throw(DimensionMismatch(
        "measurement has $(length(xs)) components; the binner holds $(b.ncomp)"))
    b.n += 1
    @inbounds for k = 1:_MAX_LEVELS
        b.count[k] += 1
        for j = 1:b.ncomp
            x = k == 1 ? Float64(xs[j]) : b.pending[k - 1, j]  # value entering level k
            b.sums[k, j] += x
            b.sums2[k, j] += x * x
        end
        if b.pending_full[k]
            # complete the pair: its mean becomes the value entering level k+1
            for j = 1:b.ncomp
                x = k == 1 ? Float64(xs[j]) : b.pending[k - 1, j]
                b.pending[k, j] = 0.5 * (b.pending[k, j] + x)
            end
            b.pending_full[k] = false
        else
            for j = 1:b.ncomp
                x = k == 1 ? Float64(xs[j]) : b.pending[k - 1, j]
                b.pending[k, j] = x
            end
            b.pending_full[k] = true
            break
        end
    end
    return b
end

Base.push!(b::LogBinner, x::Real)::LogBinner = push!(b, SVector(Float64(x)))

Statistics.mean(b::LogBinner)::Vector{Float64} =
    b.n == 0 ? fill(NaN, b.ncomp) : vec(b.sums[1, :]) ./ b.n

# Naive standard error of the mean at cascade level k (NaN with < 2 bins).
function _level_error(b::LogBinner, k::Int, j::Int)::Float64
    n = b.count[k]
    n >= 2 || return NaN
    var = (b.sums2[k, j] - b.sums[k, j]^2 / n) / (n - 1)
    return sqrt(max(var, 0.0) / n)
end

# The plateau level: the deepest with at least `_MIN_BINS` completed entries.
function _plateau_level(b::LogBinner)::Int
    k = findlast(>=(_MIN_BINS), b.count)
    return k === nothing ? 1 : k
end

"""
    std_error(b::LogBinner) -> Vector{Float64}

Autocorrelation-aware standard error of the mean, per component: the naive standard
error at the deepest binning level that still holds ≥ 32 bins (the log-binning
plateau proxy). `NaN` until enough measurements exist.
"""
function std_error(b::LogBinner)::Vector{Float64}
    b.n >= 2 || return fill(NaN, b.ncomp)
    k = _plateau_level(b)
    return [_level_error(b, k, j) for j = 1:b.ncomp]
end

"""
    tau_int(b::LogBinner) -> Vector{Float64}

Integrated autocorrelation time estimate, per component:
`τ_int = ((err_plateau / err_naive)² − 1) / 2` — the error inflation of the plateau
over the raw (level-1) standard error. `≈ 0` for uncorrelated measurements.
"""
function tau_int(b::LogBinner)::Vector{Float64}
    b.n >= 2 || return fill(NaN, b.ncomp)
    k = _plateau_level(b)
    return [begin
                e1 = _level_error(b, 1, j)
                e1 > 0 ? 0.5 * ((_level_error(b, k, j) / e1)^2 - 1) : NaN
            end
            for j = 1:b.ncomp]
end

"""
    BinStore(ncomp::Integer, bin_size::Integer, nbins::Integer)

Fixed-layout bin-mean store for jackknifing derived quantities: measurements pushed
with `push!` are averaged in blocks of `bin_size`; only the `nbins` completed block
means are kept (a trailing remainder — and anything beyond `nbins` blocks — is
dropped). `bin_means(store)` returns the completed `nfull × ncomp` block.
"""
mutable struct BinStore
    const ncomp::Int
    const bin_size::Int
    const means::Matrix{Float64}      # nbins × ncomp
    nfull::Int
    const acc::Vector{Float64}        # partial-bin accumulator
    nacc::Int

    function BinStore(ncomp::Integer, bin_size::Integer, nbins::Integer)
        ncomp >= 1 || throw(ArgumentError("ncomp must be ≥ 1; got $ncomp"))
        bin_size >= 1 || throw(ArgumentError("bin_size must be ≥ 1; got $bin_size"))
        nbins >= 2 || throw(ArgumentError("nbins must be ≥ 2; got $nbins"))
        return new(ncomp, bin_size, zeros(nbins, ncomp), 0, zeros(ncomp), 0)
    end

    # Checkpoint-restore path: rebuild from captured state verbatim.
    BinStore(ncomp::Int, bin_size::Int, means::Matrix{Float64}, nfull::Int,
             acc::Vector{Float64}, nacc::Int) =
        new(ncomp, bin_size, means, nfull, acc, nacc)
end

Base.show(io::IO, s::BinStore) =
    print(io, "BinStore(", s.nfull, "/", size(s.means, 1), " bins of ", s.bin_size, ")")

function Base.push!(s::BinStore, xs::AbstractVector{<:Real})::BinStore
    length(xs) == s.ncomp || throw(DimensionMismatch(
        "measurement has $(length(xs)) components; the store holds $(s.ncomp)"))
    s.nfull >= size(s.means, 1) && return s          # beyond the layout: dropped
    for j = 1:s.ncomp
        s.acc[j] += Float64(xs[j])
    end
    s.nacc += 1
    if s.nacc == s.bin_size
        s.nfull += 1
        for j = 1:s.ncomp
            s.means[s.nfull, j] = s.acc[j] / s.bin_size
            s.acc[j] = 0.0
        end
        s.nacc = 0
    end
    return s
end

Base.push!(s::BinStore, x::Real)::BinStore = push!(s, SVector(Float64(x)))

"""
    bin_means(s::BinStore) -> Matrix{Float64}

The completed bin means, `nfull × ncomp` (a view-free copy).
"""
bin_means(s::BinStore)::Matrix{Float64} = s.means[1:s.nfull, :]

"""
    jackknife(f, cols::Vector{Vector{Float64}}) -> (estimate, error)

Leave-one-bin-out jackknife of `θ = f(means...)`: `f` maps one mean per input series
(the `cols`, equal-length bin-mean series) to a scalar. Returns the bias-corrected
estimate `n_b·f(m̄) − (n_b−1)·mean(θᵢ)` and the jackknife standard error
`sqrt((n_b−1)/n_b · Σᵢ (θᵢ − θ̄)²)`.
"""
function jackknife(f::F, cols::Vector{Vector{Float64}}) where {F}
    isempty(cols) && throw(ArgumentError("jackknife needs at least one input series"))
    nb = length(cols[1])
    all(c -> length(c) == nb, cols) ||
        throw(DimensionMismatch("jackknife input series have unequal lengths"))
    nb >= 2 || throw(ArgumentError("jackknife needs ≥ 2 bins; got $nb"))
    totals = [sum(c) for c in cols]
    θfull = f((totals ./ nb)...)
    θ = Vector{Float64}(undef, nb)
    loo = Vector{Float64}(undef, length(cols))
    for i = 1:nb
        for (q, c) in enumerate(cols)
            loo[q] = (totals[q] - c[i]) / (nb - 1)
        end
        θ[i] = f(loo...)
    end
    θbar = mean(θ)
    estimator = nb * θfull - (nb - 1) * θbar
    err = sqrt((nb - 1) / nb * sum(abs2, θ .- θbar))
    return estimator, err
end
