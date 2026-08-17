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

@testitem "Hygiene: DataFrame(ft) isolates in-place element writes" begin
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
    d.value[1] = -99.0   # in-place element write, not a column replacement
    @test DataFrame(ft).value == [1.0, 3.0]
end

@testitem "Hardening: all-zero weights raise a clear error for the median too" begin
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
    @test_throws ArgumentError combine(ft, QuantileEnsemble(:median; weights = w))
end

@testitem "Hardening: EnsembleWeights rejects non-finite weights" begin
    using DataFrames
    @test_throws ArgumentError EnsembleWeights(
        DataFrame(model_id = ["a", "b"], weight = [Inf, 1.0]))
    @test_throws ArgumentError EnsembleWeights(
        DataFrame(model_id = ["a", "b"], weight = [NaN, 1.0]))
end

@testitem "Hardening: BLP Beta MLE fallback warns and returns the identity" begin
    using Logging: Warn
    # A constant PIT vector makes the Beta MLE fail and its moments degenerate,
    # exercising both warn branches; the fit falls back to Beta(1, 1) (no transform).
    a, b = @test_logs (:warn, r"MLE fit failed") (:warn, r"moments are degenerate") ForecastEnsembles._fit_beta(fill(
        0.5, 10))
    @test (a, b) == (1.0, 1.0)
end
