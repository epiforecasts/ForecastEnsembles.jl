using Random: MersenneTwister
using Distributions: Normal, quantile

@testset "weights() accessor + composition" begin
    # ---- CRPSStacking → LinearPool ----------------------------------------
    rng = MersenneTwister(2026)
    T = 40; K = 80
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for (mid, sampler) in (
        ("m_good",  (y, rng) -> y .+ randn(rng, K)),
        ("m_noisy", (y, rng) -> 5.0 .* randn(rng, K)),
    )
        for t in 1:T
            push!(rows, DataFrame(
                model_id = mid, output_type = "sample",
                output_type_id = 1:K, t = t,
                value = sampler(obs.observed[t], rng),
            ))
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    fitted_crps = fit(CRPSStacking(), train, obs)

    # weights() round-trips
    w_df = weights(fitted_crps)
    @test w_df isa DataFrame
    @test sort(propertynames(w_df)) == [:model_id, :weight]
    @test sum(w_df.weight) ≈ 1.0 atol = 1e-8

    # Pass the fitted method directly to LinearPool / SimpleEnsemble.
    lp = LinearPool(weights = fitted_crps, n_samples = 1000)
    @test lp.weights isa DataFrame
    @test lp.weights.weight ≈ w_df.weight

    se = SimpleEnsemble(:mean; weights = fitted_crps)
    @test se.weights.weight ≈ w_df.weight

    # ---- QRA → weights() returns DataFrame only when fit is "simplex-shape"
    n = 100; levels = [0.1, 0.5, 0.9]
    y = randn(rng, n)
    qrows = DataFrame[]
    for (mid, prediction) in (
        ("m_a", y .+ 0.3 .* randn(rng, n)),
        ("m_b", y .+ 0.3 .* randn(rng, n)),
    )
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            push!(qrows, DataFrame(
                model_id = mid, output_type = "quantile",
                output_type_id = τ, t = 1:n, value = prediction .+ zτ,
            ))
        end
    end
    qtrain = ForecastTable(reduce(vcat, qrows); task_id_cols = [:t])
    qobs   = DataFrame(t = 1:n, observed = y)

    # Eligible: simplex + no intercept + joint coefficients.
    fitted_q_ok = fit(QRA(; per_quantile_weights = false,
                           enforce_normalisation = true,
                           intercept = false),
                      qtrain, qobs)
    w_q = weights(fitted_q_ok)
    @test w_q isa DataFrame
    @test sum(w_q.weight) ≈ 1.0 atol = 1e-6

    # Ineligible: per-quantile weights → no single per-model weight vector.
    fitted_q_pq = fit(QRA(; per_quantile_weights = true,
                           enforce_normalisation = true,
                           intercept = false),
                      qtrain, qobs)
    @test weights(fitted_q_pq) === nothing

    # Ineligible: with intercept.
    fitted_q_int = fit(QRA(; per_quantile_weights = false,
                            enforce_normalisation = true,
                            intercept = true),
                       qtrain, qobs)
    @test weights(fitted_q_int) === nothing

    # Ineligible: unconstrained.
    fitted_q_un = fit(QRA(; per_quantile_weights = false,
                           enforce_normalisation = false,
                           intercept = false),
                      qtrain, qobs)
    @test weights(fitted_q_un) === nothing

    # Passing an ineligible fitted QRA to LinearPool must raise (rather
    # than silently ignoring the user's intent).
    @test_throws ArgumentError LinearPool(weights = fitted_q_pq)
end
