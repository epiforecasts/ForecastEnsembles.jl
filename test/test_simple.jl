@testset "SimpleEnsemble" begin
    df = DataFrame(
        model_id = repeat(["m1", "m2", "m3"], inner = 2),
        output_type = "quantile",
        output_type_id = repeat([0.25, 0.75], 3),
        location = "A",
        value = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5],
    )
    ft = ForecastTable(df; task_id_cols = [:location])

    # unweighted mean
    out = combine(ft, SimpleEnsemble(:mean))
    d = DataFrame(out)
    @test sort(unique(d.model_id)) == ["hub-ensemble"]
    sort!(d, :output_type_id)
    @test d.value ≈ [(1.0 + 2.0 + 0.5)/3, (3.0 + 4.0 + 2.5)/3]

    # unweighted median
    out_med = combine(ft, SimpleEnsemble(:median))
    d_med = sort(DataFrame(out_med), :output_type_id)
    @test d_med.value == [1.0, 3.0]

    # weighted mean
    w = DataFrame(model_id = ["m1", "m2", "m3"], weight = [0.5, 0.25, 0.25])
    out_w = combine(ft, SimpleEnsemble(:mean; weights = w))
    d_w = sort(DataFrame(out_w), :output_type_id)
    @test d_w.value ≈ [
        0.5 * 1.0 + 0.25 * 2.0 + 0.25 * 0.5,
        0.5 * 3.0 + 0.25 * 4.0 + 0.25 * 2.5,
    ]

    # weighted median: lower 50% of cumulative weight at 0.25 quantile
    # values sorted: m3 (0.5), m1 (1.0), m2 (2.0); weights 0.25, 0.5, 0.25
    # cum weights: 0.25, 0.75 → median is 1.0
    out_wm = combine(ft, SimpleEnsemble(:median; weights = w))
    d_wm = sort(DataFrame(out_wm), :output_type_id)
    @test d_wm.value[1] == 1.0

    # weights with missing model raise
    w_bad = DataFrame(model_id = ["m1", "m2"], weight = [0.5, 0.5])
    @test_throws ArgumentError combine(ft, SimpleEnsemble(:mean; weights = w_bad))

    # invalid agg
    @test_throws ArgumentError SimpleEnsemble(:max)
end
