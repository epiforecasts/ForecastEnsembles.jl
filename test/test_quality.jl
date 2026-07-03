using Aqua
using JuliaFormatter

@testset "Code quality (Aqua.jl)" begin
    # Ambiguities from dependencies are noisy and not our concern; the rest
    # of the Aqua battery (stale deps, compat bounds, undefined exports,
    # project consistency, piracy) runs.
    Aqua.test_all(ForecastEnsembles; ambiguities = false)
end

@testset "Code formatting (JuliaFormatter)" begin
    @test JuliaFormatter.format(pkgdir(ForecastEnsembles); overwrite = false)
end
