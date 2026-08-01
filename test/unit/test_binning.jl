# Binning machinery: log-binning error/τ_int against iid and AR(1) series with known
# answers, BinStore layout semantics, and jackknife identities.

@testset "binning" begin
    @testset "the level-1 error is the Bessel-corrected SEM, exactly" begin
        # The iid / AR(1) arms below compare against asymptotic formulas at n ~ 10⁴ with
        # rtol 0.2-0.5, where an unbiased and a biased variance differ by ~1/n — far
        # under the tolerance. So the Bessel correction itself is unprotected there, and
        # dropping it is a silent ~(1 - 1/n) shrink of every error bar in the package.
        # Pin it where the two are far apart instead: four hand-chosen values, arithmetic
        # done by hand.
        #
        #   x = [1, 2, 3, 4]   Σx = 10   Σx² = 30   n = 4
        #   unbiased  var = (30 - 10²/4)/(4-1) = 5/3   ⇒ SEM = √(5/12) = 0.645497…
        #   biased    var = (30 - 10²/4)/4     = 5/4   ⇒ SEM = √(5/16) = 0.559017…
        b = MC.LogBinner(1)
        for x in (1.0, 2.0, 3.0, 4.0)
            push!(b, x)
        end
        @test b.count[1] == 4
        @test MC._level_error(b, 1, 1) ≈ sqrt(5 / 12) rtol = 1e-14
        @test !isapprox(MC._level_error(b, 1, 1), sqrt(5 / 16); rtol = 1e-3)
        # …and fewer than two entries is NaN, not a zero error bar
        b1 = MC.LogBinner(1)
        push!(b1, 7.0)
        @test isnan(MC._level_error(b1, 1, 1))
    end

    @testset "LogBinner: iid Gaussian" begin
        rng = MersenneTwister(42)
        n = 2^14
        b = MC.LogBinner(1)
        for _ = 1:n
            push!(b, randn(rng))
        end
        @test b.n == n
        @test abs(mean(b)[1]) < 5 / sqrt(n)
        err = MC.std_error(b)[1]
        @test isapprox(err, 1 / sqrt(n); rtol = 0.2)
        @test abs(MC.tau_int(b)[1]) < 0.2
    end

    @testset "LogBinner: AR(1), ρ = 0.9" begin
        rng = MersenneTwister(1)
        ρ = 0.9
        n = 2^16
        b = MC.LogBinner(1)
        x = 0.0
        for _ = 1:n
            x = ρ * x + sqrt(1 - ρ^2) * randn(rng)   # stationary variance 1
            push!(b, x)
        end
        # The deepest-32-bin plateau proxy carries an O(1/√bins) ≈ 13% error on the
        # error itself (≈ 26% on τ, quadratic) — the tolerances reflect that; the
        # fixed seed keeps the test deterministic.
        err = MC.std_error(b)[1]
        err_exact = sqrt((1 + ρ) / (1 - ρ)) / sqrt(n)
        @test isapprox(err, err_exact; rtol = 0.3)
        τ_exact = ρ / (1 - ρ)                        # Σ_{t≥1} ρ^t = 9
        @test isapprox(MC.tau_int(b)[1], τ_exact; rtol = 0.5)
        @test MC.tau_int(b)[1] > 5                   # clearly correlated, not ≈ 0
    end

    @testset "LogBinner: exact mean and vector components" begin
        b = MC.LogBinner(2)
        vals = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0), (4.0, 40.0)]
        for (x, y) in vals
            push!(b, [x, y])
        end
        @test mean(b) ≈ [2.5, 25.0] atol = 1e-14
        # level 2 holds the two pair means (1.5, 3.5) — cascade bookkeeping
        @test b.count[2] == 2
        @test b.sums[2, 1] ≈ 1.5 + 3.5 atol = 1e-14
        @test_throws DimensionMismatch push!(b, [1.0])
        @test all(isnan, mean(MC.LogBinner(1)))
    end

    @testset "BinStore: layout, remainder drop" begin
        s = MC.BinStore(1, 3, 4)          # 4 bins of 3
        for x = 1:13                      # 13 = 4×3 + 1 → the 13th is dropped
            push!(s, Float64(x))
        end
        @test s.nfull == 4
        @test vec(MC.bin_means(s)) ≈ [2.0, 5.0, 8.0, 11.0] atol = 1e-14
        @test_throws ArgumentError MC.BinStore(1, 0, 4)
        @test_throws ArgumentError MC.BinStore(1, 3, 1)
    end

    @testset "jackknife: linear function ≡ plain mean/error" begin
        rng = MersenneTwister(5)
        bins = randn(rng, 64)
        estimator, err = MC.jackknife(m -> 2m + 1, [bins])
        @test estimator ≈ 2 * mean(bins) + 1 atol = 1e-12
        @test err ≈ 2 * std(bins) / sqrt(64) atol = 1e-12
    end

    @testset "jackknife: two-series variance (specific-heat shape)" begin
        # f = ⟨x²⟩ − ⟨x⟩² on iid Gaussian bins: estimate → var within jackknife error
        rng = MersenneTwister(9)
        nb = 256
        xs = randn(rng, nb)
        estimator, err = MC.jackknife((m1, m2) -> m2 - m1^2, [xs, xs .^ 2])
        @test abs(estimator - 1.0) < 4 * err
        @test 0 < err < 0.2
        @test_throws DimensionMismatch MC.jackknife(+, [xs, randn(rng, 8)])
        @test_throws ArgumentError MC.jackknife(identity, [Float64[1.0]])
    end
end
