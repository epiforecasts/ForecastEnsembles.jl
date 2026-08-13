@testitem "InverseScore (any score function)" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # m_good scores well, m_noisy scores badly; performance weighting should
    # favour m_good without ever looking at how they combine.
    rng = MersenneTwister(2026)
    T = 60
    K = 100
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for (mid, sampler) in (
        ("m_good", (y, r) -> y .+ randn(r, K)),
        ("m_noisy", (y, r) -> 5.0 .* randn(r, K))
    )
        for t in 1:T
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "sample",
                    output_type_id = 1:K,
                    t = t,
                    value = sampler(obs.observed[t], rng)
                )
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])

    fitted = fit(InverseScore(crps), train, obs)
    @test isa(fitted, FittedInverseScore)
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-8
    gw = fitted.weights[fitted.weights.model_id .== "m_good", :weight][1]
    nw = fitted.weights[fitted.weights.model_id .== "m_noisy", :weight][1]
    @test gw > nw
    # scores stored are negatively oriented → m_good has the lower one.
    @test fitted.scores[findfirst(==("m_good"), fitted.models)] <
          fitted.scores[findfirst(==("m_noisy"), fitted.models)]

    # combine → sample table; weights() round-trips.
    @test all(DataFrame(combine(train, fitted; rng = MersenneTwister(1))).output_type .===
              :sample)
    @test weights(fitted) isa EnsembleWeights

    # Temperature: hot → winner-take-all; cold → near-equal.
    hot = fit(InverseScore(crps; temperature = 20.0), train, obs)
    @test hot.weights[hot.weights.model_id .== "m_good", :weight][1] > 0.95
    cold = fit(InverseScore(crps; temperature = 1e-3), train, obs)
    @test all(abs.(cold.weights.weight .- 0.5) .< 0.05)

    # temperature must be positive.
    @test_throws ArgumentError InverseScore(crps; temperature = 0.0)

    # Quantile input is out of scope.
    qtrain = ForecastTable(
        DataFrame(
            model_id = repeat(["m_a", "m_b"], inner = 2),
            output_type = "quantile",
            output_type_id = repeat([0.25, 0.75], 2),
            t = 1,
            value = [1.0, 3.0, 2.0, 4.0]
        );
        task_id_cols = [:t]
    )
    @test_throws ArgumentError fit(InverseScore(crps), qtrain, obs)
end

@testitem "InverseScore scores members without the w keyword" begin
    using Random: MersenneTwister
    using DataFrames
    # A score with NO `w` keyword: InverseScore must call it as score(samples, y),
    # not score(samples, y; w = ...). (Stacking, by contrast, requires the w path.)
    naive(samples, y) = abs(sum(samples) / length(samples) - y)

    rng = MersenneTwister(3)
    T, K = 20, 40
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for (mid, sd) in (("good", 0.5), ("bad", 3.0))
        for t in 1:T
            push!(rows,
                DataFrame(model_id = mid, output_type = "sample",
                    output_type_id = 1:K, t = t,
                    value = obs.observed[t] .+ sd .* randn(rng, K)))
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])

    fitted = fit(InverseScore(naive), train, obs)
    gw = fitted.weights[fitted.weights.model_id .== "good", :weight][1]
    bw = fitted.weights[fitted.weights.model_id .== "bad", :weight][1]
    @test gw > bw
end
