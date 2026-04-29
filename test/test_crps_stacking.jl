using Random: MersenneTwister
using Statistics: mean

@testset "CRPSStacking" begin
    rng = MersenneTwister(2026)

    # Setup: T tasks. For each task t, observed y_t ~ N(0, 1).
    # Two models providing samples:
    #   m_good:   X ~ N(y_t, 1)        — actually predictive
    #   m_noisy:  X ~ N(0, 5)           — far worse
    # The optimum should put almost all weight on m_good.
    T = 60
    K = 100
    obs = DataFrame(t = 1:T, observed = randn(rng, T))

    rows = DataFrame[]
    for (mid, sampler) in (
        ("m_good",  (y, rng) -> y .+ randn(rng, K)),
        ("m_noisy", (y, rng) -> 5.0 .* randn(rng, K)),
    )
        for t in 1:T
            samples = sampler(obs.observed[t], rng)
            push!(rows, DataFrame(
                model_id = mid,
                output_type = "sample",
                output_type_id = 1:K,
                t = t,
                value = samples,
            ))
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])

    fitted = fit(CRPSStacking(), train, obs)
    @test isa(fitted, FittedCRPSStacking)
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-8
    good_w = fitted.weights[fitted.weights.model_id .== "m_good", :weight][1]
    @test good_w > 0.7

    # combine: produces a sample ForecastTable.
    out = combine(train, fitted; rng = MersenneTwister(99))
    d = DataFrame(out)
    @test all(d.output_type .=== :sample)
    @test all(d.model_id .== "hub-ensemble")

    # Strong Dirichlet prior pulls weights toward the simplex centre.
    fitted_prior = fit(CRPSStacking(; dirichlet_alpha = 50.0), train, obs)
    centre = 1.0 / nrow(fitted_prior.weights)
    @test all(abs.(fitted_prior.weights.weight .- centre) .< 0.15)
end
