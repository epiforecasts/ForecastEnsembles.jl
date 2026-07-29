@testitem "Stacking{Score} (any score function)" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Two sample models: m_good predicts y with unit noise, m_noisy is wide and
    # uninformative. Score-optimal stacking against CRPS should load onto m_good.
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

    fitted = fit(Stacking(crps), train, obs)
    @test isa(fitted, FittedStacking)
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-6
    good_w = fitted.weights[fitted.weights.model_id .== "m_good", :weight][1]
    @test good_w > 0.7

    # combine on the fitted method → a sample ForecastTable.
    out = combine(train, fitted; rng = MersenneTwister(1))
    @test all(DataFrame(out).output_type .=== :sample)

    # weights() round-trips as EnsembleWeights (composable into LinearPool etc.).
    w = weights(fitted)
    @test w isa EnsembleWeights
    @test DataFrame(w).weight ≈ fitted.weights.weight

    # A strong Dirichlet prior pulls the weights toward the simplex centre.
    fitted_prior = fit(Stacking(crps; dirichlet_alpha = 50.0), train, obs)
    centre = 1.0 / nrow(fitted_prior.weights)
    @test all(abs.(fitted_prior.weights.weight .- centre) .< 0.3)

    # Quantile input is out of scope — QRA is the WIS-optimal quantile stacker.
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
    @test_throws ArgumentError fit(Stacking(crps), qtrain, obs)
end
