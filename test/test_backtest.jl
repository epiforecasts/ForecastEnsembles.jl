@testitem "backtest — shape and columns" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # A CRPS scorer over the fold's tasks, using the local weighted-sample CRPS.
    function sample_crps(fc, o)
        d = DataFrames.innerjoin(DataFrame(fc), o; on = :t)
        per = DataFrames.combine(DataFrames.groupby(d, :t),
            [:value, :observed] => ((v, y) -> crps(Float64.(v), Float64(first(y)))) => :s)
        return mean(per.s)
    end

    ft, obs = _bt_sample_data(T = 10)
    schemes = Dict("equal" => MixtureEnsemble(; n_samples = 500), "crps" => CRPSStacking())
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 3,
        rng = MersenneTwister(1), score_fn = sample_crps)
    @test sort(propertynames(res)) == sort([:scheme, :t, :score])
    # (10 − 3) folds × 2 schemes
    @test nrow(res) == (10 - 3) * 2
    @test Set(res.scheme) == Set(["equal", "crps"])
    @test all(isfinite, res.score)
end

@testitem "backtest — recency-weighted CRPS beats equal out of sample" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    function sample_crps(fc, o)
        d = DataFrames.innerjoin(DataFrame(fc), o; on = :t)
        per = DataFrames.combine(DataFrames.groupby(d, :t),
            [:value, :observed] => ((v, y) -> crps(Float64.(v), Float64(first(y)))) => :s)
        return mean(per.s)
    end

    ft, obs = _bt_sample_data(T = 30)
    schemes = [
        "equal" => MixtureEnsemble(; n_samples = 1000),
        "crps_recency" => CRPSStacking(; lambda = 0.6, time_col = :t)
    ]
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 6,
        rng = MersenneTwister(7), score_fn = sample_crps)
    agg = DataFrames.combine(DataFrames.groupby(res, :scheme), :score =>
        mean => :mean_score)
    equal = agg[agg.scheme .== "equal", :mean_score][1]
    recency = agg[agg.scheme .== "crps_recency", :mean_score][1]
    @test recency < equal
end

@testitem "backtest — quantile schemes (QRA vs equal)" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    using Distributions: Normal, quantile as dquantile
    rng = MersenneTwister(11)
    T = 25
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    y = randn(rng, T)
    rows = DataFrame[]
    for (mid, pred) in (("m_good", y .+ 0.3 .* randn(rng, T)), (
        "m_noisy", 2 .* randn(rng, T)))
        for t in 1:T, τ in levels

            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = t,
                    value = pred[t] + dquantile(Normal(0, 1), τ)
                )
            )
        end
    end
    ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:T, observed = y)

    # A mean-quantile-score (WIS kernel) scorer, using the local quantile score.
    function qscore(fc, o)
        d = DataFrames.innerjoin(DataFrame(fc), o; on = :t)
        per = DataFrames.combine(DataFrames.groupby(d, :t)) do g
            s = sort(g, :output_type_id)
            (;
                s = mean(quantile_score(Float64.(s.output_type_id), Float64.(s.value),
                Float64(first(s.observed)))))
        end
        return mean(per.s)
    end

    schemes = [
        "equal" => QuantileEnsemble(:mean),
        "qra" => QRA(; enforce_normalisation = true, intercept = false)
    ]
    res = backtest(ft, obs, schemes; time_col = :t, min_train = 8, score_fn = qscore)
    agg = DataFrames.combine(DataFrames.groupby(res, :scheme), :score =>
        mean => :mean_score)
    equal = agg[agg.scheme .== "equal", :mean_score][1]
    qra = agg[agg.scheme .== "qra", :mean_score][1]
    # QRA should learn to downweight the noisy model → better WIS.
    @test qra < equal
    @test all(res.score .>= 0)
end

@testitem "backtest — guards" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    dummy(fc, o) = 0.0
    ft, obs = _bt_sample_data(T = 5)
    # Missing time_col throws before scoring; missing score_fn is itself an error.
    @test_throws ArgumentError backtest(
        ft, obs, ["equal" => QuantileEnsemble(:mean)];
        time_col = :nope, min_train = 2, score_fn = dummy)
    @test_throws ArgumentError backtest(
        ft, obs, ["equal" => QuantileEnsemble(:mean)]; time_col = :t, min_train = 2)
    @test_throws ArgumentError backtest(
        ft, obs, ["equal" => QuantileEnsemble(:mean)];
        time_col = :t, min_train = 10, score_fn = dummy)
    # min_train is required (no default) — omitting it is a keyword error.
    @test_throws UndefKeywordError backtest(
        ft, obs, ["equal" => QuantileEnsemble(:mean)]; time_col = :t, score_fn = dummy)
end
