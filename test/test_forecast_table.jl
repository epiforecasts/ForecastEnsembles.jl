@testitem "ForecastTable" begin
    using DataFrames

    df = DataFrame(
        model_id = ["m1", "m1", "m2", "m2"],
        output_type = ["quantile", "quantile", "quantile", "quantile"],
        output_type_id = [0.25, 0.75, 0.25, 0.75],
        location = ["A", "A", "A", "A"],
        value = [1.0, 3.0, 2.0, 4.0]
    )

    ft = ForecastTable(df; task_id_cols = [:location])
    @test ForecastEnsembles.task_id_cols(ft) == [:location]
    @test ForecastEnsembles.model_ids(ft) == ["m1", "m2"]
    @test ForecastEnsembles.output_type(ft) === :quantile

    # task_id_cols inferred when omitted
    ft2 = ForecastTable(df)
    @test ForecastEnsembles.task_id_cols(ft2) == [:location]

    # missing required column
    bad = select(df, Not(:value))
    @test_throws ArgumentError ForecastTable(bad; task_id_cols = [:location])

    # unknown output_type
    bad2 = copy(df);
    bad2.output_type .= "weird"
    @test_throws ArgumentError ForecastTable(bad2; task_id_cols = [:location])

    # mixed output_types accepted at construction, rejected by `output_type`
    mixed = vcat(
        df,
        DataFrame(
            model_id = ["m3"],
            output_type = ["mean"],
            output_type_id = [missing],
            location = ["A"],
            value = [2.5]
        )
    )
    ft3 = ForecastTable(mixed; task_id_cols = [:location])
    @test_throws ArgumentError ForecastEnsembles.output_type(ft3)
end

@testitem "DataFrame(ft) copying is explicit about copycols" begin
    using DataFrames, Tables

    df = DataFrame(
        location = "A",
        model_id = repeat(["m1", "m2"], inner = 2),
        output_type = "quantile",
        output_type_id = repeat([0.25, 0.75], 2),
        value = [1.0, 3.0, 2.0, 4.0]
    )
    ft = ForecastTable(df; task_id_cols = [:location])
    backing = Tables.columns(ft).value

    # The default and an explicit `copycols = true` both isolate the caller: a
    # write through the returned frame must not reach the table's store.
    for d in (DataFrame(ft), DataFrame(ft; copycols = true))
        d.value[1] = -111.0
        @test Tables.columns(ft).value[1] == 1.0
    end

    # `copycols = false` opts into the zero-copy path, so the columns alias and a
    # write does reach the store. This is deliberate rather than a dispatch
    # accident: without the keyword on the method, the call fell through to the
    # generic Tables.jl constructor and aliased with no way to ask for a copy.
    shared = DataFrame(ft; copycols = false)
    @test shared.value === backing
    shared.value[1] = -222.0
    @test Tables.columns(ft).value[1] == -222.0
end
