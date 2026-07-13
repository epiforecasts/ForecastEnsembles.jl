using Random: MersenneTwister
using Distributions: Normal, quantile as dquantile
using Statistics: mean

@testset "CRPS (sample)" begin
    # A point mass at the observation scores 0.
    @test ForecastEnsembles._crps_sample(fill(2.0, 50), 2.0) ≈ 0.0 atol = 1e-12
    # Non-negative.
    rng = MersenneTwister(1)
    @test ForecastEnsembles._crps_sample(randn(rng, 200), 0.5) >= 0

    # Empirical CRPS of N(0,1) draws against y = 0 approaches the analytic
    # value (√2 − 1)/√π ≈ 0.2337.
    big = randn(MersenneTwister(2), 100_000)
    analytic = (sqrt(2) - 1) / sqrt(pi)
    @test ForecastEnsembles._crps_sample(big, 0.0) ≈ analytic atol = 3e-3

    # Single sample: reduces to |x − y|.
    @test ForecastEnsembles._crps_sample([3.0], 1.0) ≈ 2.0
end

@testset "WIS (quantile)" begin
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    vals = dquantile.(Normal(0, 1), levels)
    # With only the median, WIS reduces to the absolute error.
    @test ForecastEnsembles._wis_quantile([0.5], [1.0], 3.0) ≈ 2.0
    @test ForecastEnsembles._wis_quantile([0.5], [1.0], 1.0) ≈ 0.0
    # Non-negative; zero when every quantile equals y.
    @test ForecastEnsembles._wis_quantile(levels, vals, 0.3) >= 0
    @test ForecastEnsembles._wis_quantile(levels, fill(2.0, 5), 2.0) ≈ 0.0
    # Sharper (correct) forecast scores better than a shifted one.
    good = ForecastEnsembles._wis_quantile(levels, vals, 0.0)
    shifted = ForecastEnsembles._wis_quantile(levels, vals .+ 5, 0.0)
    @test good < shifted
end

@testset "score / mean_score" begin
    rng = MersenneTwister(3)
    # Two models over two locations, sample forecasts.
    rows = DataFrame[]
    for loc in ["A", "B"], (mid, sd) in (("good", 0.5), ("bad", 3.0))
        push!(
            rows,
            DataFrame(
                model_id = mid,
                output_type = "sample",
                output_type_id = 1:100,
                location = loc,
                value = randn(rng, 100) .* sd,
            ),
        )
    end
    ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:location])
    obs = DataFrame(location = ["A", "B"], observed = [0.0, 0.0])

    per = score(ft, obs)
    @test sort(propertynames(per)) == sort([:model_id, :location, :score])
    @test nrow(per) == 4                 # 2 models × 2 locations
    @test all(per.score .>= 0)

    ms = mean_score(ft, obs)
    @test Set(ms.model_id) == Set(["good", "bad"])
    # The sharper model has the lower (better) mean CRPS.
    good = ms[ms.model_id .== "good", :score][1]
    bad = ms[ms.model_id .== "bad", :score][1]
    @test good < bad

    # explicit rule matches the default for samples
    @test score(ft, obs; rule = CRPS()).score == per.score
end
