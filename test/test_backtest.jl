using Random: MersenneTwister
using Statistics: mean

# Non-stationary regime: model A is sharp in the first half of the time
# window, model B in the second. A scheme that learns weights from recent
# performance should beat equal weighting out of sample.
function _bt_sample_data(; T = 30, K = 80, seed = 4)
    rng = MersenneTwister(seed)
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for t = 1:T
        y = obs.observed[t]
        a_good = t <= T ÷ 2
        for (mid, sharp) in (("m_a", a_good), ("m_b", !a_good))
            sd = sharp ? 0.4 : 3.0
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "sample",
                    output_type_id = 1:K,
                    t = t,
                    value = y .+ sd .* randn(rng, K),
                ),
            )
        end
    end
    return ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
end

@testset "backtest — shape and columns" begin
    ft, obs = _bt_sample_data(T = 10)
    schemes = Dict("equal" => MixtureEnsemble(; n_samples = 500), "crps" => CRPSStacking())
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 3, rng = MersenneTwister(1))
    @test sort(propertynames(res)) == sort([:scheme, :t, :score])
    # (10 − 3) folds × 2 schemes
    @test nrow(res) == (10 - 3) * 2
    @test Set(res.scheme) == Set(["equal", "crps"])
    @test all(isfinite, res.score)
end

@testset "backtest — recency-weighted CRPS beats equal out of sample" begin
    ft, obs = _bt_sample_data(T = 30)
    schemes = [
        "equal" => MixtureEnsemble(; n_samples = 1000),
        "crps_recency" => CRPSStacking(; lambda = 0.6, time_col = :t),
    ]
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 6, rng = MersenneTwister(7))
    agg =
        DataFrames.combine(DataFrames.groupby(res, :scheme), :score => mean => :mean_score)
    equal = agg[agg.scheme .== "equal", :mean_score][1]
    recency = agg[agg.scheme .== "crps_recency", :mean_score][1]
    @test recency < equal
end

@testset "backtest — quantile schemes (QRA vs equal)" begin
    using Distributions: Normal, quantile as dquantile
    rng = MersenneTwister(11)
    T = 25
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    y = randn(rng, T)
    rows = DataFrame[]
    for (mid, pred) in
        (("m_good", y .+ 0.3 .* randn(rng, T)), ("m_noisy", 2 .* randn(rng, T)))
        for t = 1:T, τ in levels
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = t,
                    value = pred[t] + dquantile(Normal(0, 1), τ),
                ),
            )
        end
    end
    ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:T, observed = y)

    schemes = [
        "equal" => QuantileEnsemble(:mean),
        "qra" => QRA(; enforce_normalisation = true, intercept = false),
    ]
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 8)
    agg =
        DataFrames.combine(DataFrames.groupby(res, :scheme), :score => mean => :mean_score)
    equal = agg[agg.scheme .== "equal", :mean_score][1]
    qra = agg[agg.scheme .== "qra", :mean_score][1]
    # QRA should learn to downweight the noisy model → better WIS.
    @test qra < equal
    @test all(res.score .>= 0)
end

@testset "backtest — guards" begin
    ft, obs = _bt_sample_data(T = 5)
    @test_throws ArgumentError backtest(
        ft,
        obs,
        ["equal" => QuantileEnsemble(:mean)];
        time_col = :nope,
    )
    @test_throws ArgumentError backtest(
        ft,
        obs,
        ["equal" => QuantileEnsemble(:mean)];
        time_col = :t,
        min_train = 10,
    )
end
