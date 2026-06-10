using Random: MersenneTwister
using Distributions: Normal, quantile

@testset "weights() accessor + composition" begin
    # ---- CRPSStacking → LinearPool ----------------------------------------
    rng = MersenneTwister(2026)
    T = 40;
    K = 80
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for (mid, sampler) in (
        ("m_good", (y, rng) -> y .+ randn(rng, K)),
        ("m_noisy", (y, rng) -> 5.0 .* randn(rng, K)),
    )
        for t = 1:T
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "sample",
                    output_type_id = 1:K,
                    t = t,
                    value = sampler(obs.observed[t], rng),
                ),
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    fitted_crps = fit(CRPSStacking(), train, obs)

    # weights() round-trips
    w = weights(fitted_crps)
    @test w isa EnsembleWeights
    @test !Ensembles.is_per_quantile(w)
    w_df = DataFrame(w)
    @test sort(propertynames(w_df)) == [:model_id, :weight]
    @test sum(w_df.weight) ≈ 1.0 atol = 1e-8

    # Pass the fitted method directly to LinearPool / QuantileEnsemble.
    lp = LinearPool(weights = fitted_crps, n_samples = 1000)
    @test lp.weights isa EnsembleWeights
    @test DataFrame(lp.weights).weight ≈ w_df.weight

    qe = QuantileEnsemble(:mean; weights = fitted_crps)
    @test DataFrame(qe.weights).weight ≈ w_df.weight

    # ---- QRA → weights() returns DataFrame only when fit is "simplex-shape"
    n = 100;
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n)
    qrows = DataFrame[]
    for (mid, prediction) in
        (("m_a", y .+ 0.3 .* randn(rng, n)), ("m_b", y .+ 0.3 .* randn(rng, n)))
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            push!(
                qrows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n,
                    value = prediction .+ zτ,
                ),
            )
        end
    end
    qtrain = ForecastTable(reduce(vcat, qrows); task_id_cols = [:t])
    qobs = DataFrame(t = 1:n, observed = y)

    # Eligible: simplex + no intercept + joint coefficients.
    fitted_q_ok = fit(
        QRA(;
            per_quantile_weights = false,
            enforce_normalisation = true,
            intercept = false,
        ),
        qtrain,
        qobs,
    )
    w_q = weights(fitted_q_ok)
    @test w_q isa EnsembleWeights
    @test !Ensembles.is_per_quantile(w_q)
    @test sum(DataFrame(w_q).weight) ≈ 1.0 atol = 1e-6

    # Per-quantile weights → long-format DataFrame; LinearPool dispatches
    # on shape and does direct vertical pooling.
    fitted_q_pq = fit(
        QRA(; per_quantile_weights = true, enforce_normalisation = true, intercept = false),
        qtrain,
        qobs,
    )
    w_pq = weights(fitted_q_pq)
    @test w_pq isa EnsembleWeights
    @test Ensembles.is_per_quantile(w_pq)
    w_pq_df = DataFrame(w_pq)
    @test sort(propertynames(w_pq_df)) == [:model_id, :output_type_id, :weight]
    for τ in unique(w_pq_df.output_type_id)
        @test sum(w_pq_df[w_pq_df.output_type_id .== τ, :weight]) ≈ 1.0 atol = 1e-6
    end

    # End-to-end: per-quantile weights from QRA produce the same predictions
    # as `combine(ft, fitted_qra_perq)` when applied to the same forecasts.
    target = qtrain
    via_qra = DataFrame(combine(target, fitted_q_pq))
    via_qe = DataFrame(combine(target, QuantileEnsemble(:mean; weights = fitted_q_pq)))
    j = innerjoin(
        select(via_qra, :t, :output_type_id, :value),
        rename(select(via_qe, :t, :output_type_id, :value), :value => :v_qe);
        on = [:t, :output_type_id],
    )
    @test maximum(abs.(j.value .- j.v_qe)) < 1e-10

    # MixtureEnsemble refuses per-quantile weights — that path is now
    # explicitly QuantileEnsemble's job.
    @test_throws ArgumentError MixtureEnsemble(weights = fitted_q_pq)

    # Ineligible: with intercept.
    fitted_q_int = fit(
        QRA(; per_quantile_weights = false, enforce_normalisation = true, intercept = true),
        qtrain,
        qobs,
    )
    @test weights(fitted_q_int) === nothing

    # Ineligible: unconstrained.
    fitted_q_un = fit(
        QRA(;
            per_quantile_weights = false,
            enforce_normalisation = false,
            intercept = false,
        ),
        qtrain,
        qobs,
    )
    @test weights(fitted_q_un) === nothing

    # Passing an ineligible (intercepted) fitted QRA to LinearPool raises.
    @test_throws ArgumentError LinearPool(weights = fitted_q_int)
end
