@testitem "BLP beta-transformed linear pool" begin
    using DataFrames
    using Distributions: Normal, quantile
    using Random: MersenneTwister
    using Statistics: mean

    levels = collect(0.05:0.05:0.95)
    # A model whose quantile forecast is Normal(0, σ), across `nt` tasks.
    normrows(mid, σ, ts) = reduce(
        vcat,
        [DataFrame(model_id = mid, output_type = "quantile", output_type_id = levels,
             t = t, value = [quantile(Normal(0.0, σ), τ) for τ in levels]) for t in ts]
    )

    ft = ForecastTable(normrows("m1", 0.5, [1]); task_id_cols = [:t])

    # α = β = 1 is the identity: BLP reduces to the plain linear pool.
    id = sort(DataFrame(combine(ft, FittedBLP(1.0, 1.0, nothing))), :output_type_id)
    lp = sort(DataFrame(combine(ft, MixtureEnsemble())), :output_type_id)
    @test id.value ≈ lp.value

    # Underdispersed pool: forecasts are Normal(0, 0.5) but truth is Normal(0, 1).
    # The PIT piles up in the tails, so the fitted Beta is U-shaped (α, β < 1)
    # and the recalibrated forecast is wider than the raw pool.
    rng = MersenneTwister(7)
    T = 400
    train = ForecastTable(normrows("m1", 0.5, 1:T); task_id_cols = [:t])
    obs = DataFrame(t = 1:T, observed = randn(rng, T))

    fitted = fit(BLP(), train, obs)
    @test fitted isa FittedBLP
    @test fitted.alpha < 1.0
    @test fitted.beta < 1.0
    @test weights(fitted) === nothing

    rec = sort(DataFrame(combine(ft, fitted)), :output_type_id)
    width(df) = df[df.output_type_id .== 0.95, :value][1] -
                df[df.output_type_id .== 0.05, :value][1]
    @test width(rec) > width(lp)
    # Recalibration preserves quantile monotonicity.
    @test issorted(rec.value)

    # A well-calibrated pool leaves the forecast ~unchanged (Beta ≈ uniform).
    cal_train = ForecastTable(normrows("m1", 1.0, 1:T); task_id_cols = [:t])
    cal_fit = fit(BLP(), cal_train, obs)
    cal_ft = ForecastTable(normrows("m1", 1.0, [1]); task_id_cols = [:t])
    raw = sort(DataFrame(combine(cal_ft, MixtureEnsemble())), :output_type_id)
    calrec = sort(DataFrame(combine(cal_ft, cal_fit)), :output_type_id)
    @test maximum(abs.(calrec.value .- raw.value)) < 0.25

    # Sample forecasts are out of scope (fit and combine).
    sft = ForecastTable(
        DataFrame(model_id = repeat(["m1", "m2"], inner = 3), output_type = "sample",
            output_type_id = repeat(1:3, 2), t = 1, value = Float64.(1:6));
        task_id_cols = [:t]
    )
    @test_throws ArgumentError fit(BLP(), sft, obs)
    @test_throws ArgumentError combine(sft, fitted)

    # Per-quantile pool weights are rejected.
    pq = EnsembleWeights(
        DataFrame(model_id = ["m1"], output_type_id = [0.5], weight = [1.0]))
    @test_throws ArgumentError BLP(; weights = pq)
end
