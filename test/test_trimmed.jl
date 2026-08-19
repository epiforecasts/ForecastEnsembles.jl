@testitem "TrimmedMean" begin
    using DataFrames
    using Statistics: mean

    # Six models at one quantile level, with a low and a high outlier.
    vals = [100.0, 0.0, 10.0, 10.0, 11.0, 20.0]   # sorted: 0,10,10,11,20,100
    ft = ForecastTable(
        DataFrame(
            model_id = string.("m", 1:6),
            output_type = "quantile",
            output_type_id = 0.5,
            t = 1,
            value = vals
        );
        task_id_cols = [:t]
    )

    # fraction 0.2, n = 6 → k = round(1.2) = 1: drop 0 and 100.
    o_trim = DataFrame(combine(ft, TrimmedMean(; fraction = 0.2, mode = :trim)))
    @test o_trim.value[1] ≈ mean([10.0, 10.0, 11.0, 20.0])   # 12.75
    @test o_trim.model_id[1] == "hub-ensemble"

    # Winsorise clamps 0→10 and 100→20 instead of dropping them.
    o_wins = DataFrame(combine(ft, TrimmedMean(; fraction = 0.2, mode = :winsorise)))
    @test o_wins.value[1] ≈ mean([10.0, 10.0, 10.0, 11.0, 20.0, 20.0])   # 13.5
    @test o_wins.value[1] != o_trim.value[1]

    # fraction 0 → plain mean.
    o_none = DataFrame(combine(ft, TrimmedMean(; fraction = 0.0)))
    @test o_none.value[1] ≈ mean(vals)

    # k is capped so a value always survives: two models, heavy fraction → mean.
    ft2 = ForecastTable(
        DataFrame(model_id = ["m1", "m2"], output_type = "quantile",
            output_type_id = 0.5, t = 1, value = [1.0, 3.0]);
        task_id_cols = [:t]
    )
    @test DataFrame(combine(ft2, TrimmedMean(; fraction = 0.4))).value[1] ≈ 2.0

    # Each (task, τ) is trimmed independently.
    multi = ForecastTable(
        DataFrame(
            model_id = repeat(string.("m", 1:3), outer = 2),
            output_type = "quantile",
            output_type_id = repeat([0.25, 0.75], inner = 3),
            t = 1,
            value = [1.0, 2.0, 30.0, 5.0, 6.0, 70.0]
        );
        task_id_cols = [:t]
    )
    om = DataFrame(combine(multi, TrimmedMean(; fraction = 0.34)))   # k = 1 of 3 → median
    @test nrow(om) == 2
    @test om[om.output_type_id .== 0.25, :value][1] ≈ 2.0
    @test om[om.output_type_id .== 0.75, :value][1] ≈ 6.0

    # Sample forecasts are out of scope.
    sft = ForecastTable(
        DataFrame(model_id = repeat(["m1", "m2"], inner = 3), output_type = "sample",
            output_type_id = repeat(1:3, 2), t = 1, value = Float64.(1:6));
        task_id_cols = [:t]
    )
    @test_throws ArgumentError combine(sft, TrimmedMean())

    # Constructor guards.
    @test_throws ArgumentError TrimmedMean(; fraction = 0.5)
    @test_throws ArgumentError TrimmedMean(; fraction = -0.1)
    @test_throws ArgumentError TrimmedMean(; mode = :bogus)
end

@testitem "TrimmedMean warns when the fraction trims nothing" begin
    using DataFrames
    # Five models with fraction 0.1: round(0.1 × 5) = 0, so no trimming — the
    # result is the plain mean, which should warn rather than pass silently.
    df = DataFrame(model_id = string.("m", 1:5), output_type = "quantile",
        output_type_id = 0.5, location = "A", value = [1.0, 2, 3, 4, 100])
    ft = ForecastTable(df; task_id_cols = [:location])
    @test_logs (:warn, r"trims nothing") combine(ft, TrimmedMean(; fraction = 0.1))
    # A fraction that does trim (round(0.4 × 5) = 2) emits no such warning.
    @test_logs combine(ft, TrimmedMean(; fraction = 0.4))
end
