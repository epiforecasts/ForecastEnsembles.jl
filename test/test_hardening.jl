@testitem "Hardening: all-zero ensemble weights raise a clear error" begin
    using DataFrames
    ft = ForecastTable(
        DataFrame(
            model_id = ["a", "b"],
            output_type = "quantile",
            output_type_id = 0.5,
            t = 1,
            value = [1.0, 3.0]
        );
        task_id_cols = [:t]
    )
    w = EnsembleWeights(DataFrame(model_id = ["a", "b"], weight = [0.0, 0.0]))
    @test_throws ArgumentError combine(ft, QuantileEnsemble(:mean; weights = w))
end

@testitem "Hygiene: DataFrame(ft) returns a defensive copy" begin
    using DataFrames
    ft = ForecastTable(
        DataFrame(
            model_id = ["a", "b"],
            output_type = "quantile",
            output_type_id = 0.5,
            t = 1,
            value = [1.0, 3.0]
        );
        task_id_cols = [:t]
    )
    d = DataFrame(ft)
    d[!, :value] .= -99.0
    # Mutating the accessor result must not touch the table's backing store.
    @test DataFrame(ft).value == [1.0, 3.0]
end
