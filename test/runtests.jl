using Test
using Ensembles
using DataFrames

@testset "Ensembles.jl" begin
    include("test_forecast_table.jl")
    include("test_simple.jl")
    include("test_interop.jl")
    include("test_distfromq.jl")
    include("test_linear_pool.jl")
end
