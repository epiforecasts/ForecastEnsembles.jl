using Test
using Ensembles
using DataFrames

@testset "Ensembles.jl" begin
    include("test_forecast_table.jl")
    include("test_simple.jl")
    include("test_interop.jl")
end
