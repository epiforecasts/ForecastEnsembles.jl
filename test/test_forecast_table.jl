@testset "ForecastTable" begin
    df = DataFrame(
        model_id = ["m1", "m1", "m2", "m2"],
        output_type = ["quantile", "quantile", "quantile", "quantile"],
        output_type_id = [0.25, 0.75, 0.25, 0.75],
        location = ["A", "A", "A", "A"],
        value = [1.0, 3.0, 2.0, 4.0],
    )

    ft = ForecastTable(df; task_id_cols = [:location])
    @test Ensembles.task_id_cols(ft) == [:location]
    @test Ensembles.model_ids(ft) == ["m1", "m2"]
    @test Ensembles.output_type(ft) === :quantile

    # task_id_cols inferred when omitted
    ft2 = ForecastTable(df)
    @test Ensembles.task_id_cols(ft2) == [:location]

    # missing required column
    bad = select(df, Not(:value))
    @test_throws ArgumentError ForecastTable(bad; task_id_cols = [:location])

    # unknown output_type
    bad2 = copy(df); bad2.output_type .= "weird"
    @test_throws ArgumentError ForecastTable(bad2; task_id_cols = [:location])

    # mixed output_types accepted at construction, rejected by `output_type`
    mixed = vcat(df, DataFrame(
        model_id = ["m3"], output_type = ["mean"],
        output_type_id = [missing], location = ["A"], value = [2.5]))
    ft3 = ForecastTable(mixed; task_id_cols = [:location])
    @test_throws ArgumentError Ensembles.output_type(ft3)
end
