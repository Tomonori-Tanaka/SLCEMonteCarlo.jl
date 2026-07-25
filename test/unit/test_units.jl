@testset "units / temperature control" begin
    @testset "KB_EV" begin
        @test SLCEMonteCarlo.KB_EV == 1.380649e-23 / 1.602176634e-19
        @test isapprox(KB_EV, 8.617333262e-5; rtol = 1e-9)   # exported
    end

    @testset "resolve_kt: exactly one control" begin
        @test_throws ArgumentError MC.resolve_kt(nothing, nothing)
        @test_throws ArgumentError MC.resolve_kt(300.0, 0.02)
    end

    @testset "resolve_kt: kT passthrough" begin
        @test MC.resolve_kt(nothing, 0.02) == [0.02]
        @test MC.resolve_kt(nothing, [0.03, 0.01]) == [0.03, 0.01]
        @test MC.resolve_kt(nothing, 1) == [1.0]           # integer promotes
        @test_throws ArgumentError MC.resolve_kt(nothing, 0.0)
        @test_throws ArgumentError MC.resolve_kt(nothing, -0.1)
        @test_throws ArgumentError MC.resolve_kt(nothing, Inf)
        @test_throws ArgumentError MC.resolve_kt(nothing, Float64[])
    end

    @testset "resolve_kt: kelvin conversion" begin
        @test MC.resolve_kt(300.0, nothing) == [MC.KB_EV * 300.0]
        @test MC.resolve_kt([1200, 300], nothing) == MC.KB_EV .* [1200.0, 300.0]
        # validated in kelvin first: the error message carries the kelvin value
        err = try
            MC.resolve_kt(-5.0, nothing)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("kelvin", err.msg)
        @test occursin("-5.0", err.msg)
    end
end
