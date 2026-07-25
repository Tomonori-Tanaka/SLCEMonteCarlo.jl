using SLCEMonteCarlo
using JET

@testset "JET" begin
    JET.test_package(SLCEMonteCarlo; target_modules = (SLCEMonteCarlo,))
end
