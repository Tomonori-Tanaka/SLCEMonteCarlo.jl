using Test

const TEST_MODE = get(ENV, "TEST_MODE", "default")

@testset "SLCEMonteCarlo.jl" begin
    if TEST_MODE in ("default", "all", "unit")
        include("unit/fixtures.jl")
        include("unit/test_units.jl")
        include("unit/test_hamiltonian.jl")
        include("unit/test_energy.jl")
        include("unit/test_coefficients.jl")
        include("unit/test_strainschedule.jl")
        include("unit/test_joint.jl")
        include("unit/test_gradient.jl")
        include("unit/test_inactive.jl")
        include("unit/test_binning.jl")
        include("unit/test_observables.jl")
        include("unit/test_metropolis.jl")
        include("unit/test_overrelaxation.jl")
        include("unit/test_parallel.jl")
        include("unit/test_gpu.jl")
        include("unit/test_pt.jl")
        include("unit/test_minimize.jl")
        include("unit/test_checkpoint.jl")
        include("unit/test_geometry.jl")
        include("unit/test_reduce.jl")
    end
    if TEST_MODE in ("default", "all", "aqua")
        include("aqua.jl")
    end
    if TEST_MODE in ("all", "jet")
        include("jet.jl")
    end
end
