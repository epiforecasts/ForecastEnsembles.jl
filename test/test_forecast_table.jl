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
    using DataFrames

    df = DataFrame(
        location = "A",
        model_id = repeat(["m1", "m2"], inner = 2),
        output_type = "quantile",
        output_type_id = repeat([0.25, 0.75], 2),
        value = [1.0, 3.0, 2.0, 4.0]
    )
    ft = ForecastTable(df; task_id_cols = [:location])

    # The keyword must be declared on the ForecastTable method itself: every
    # other assertion below passes either way, because falling through to the
    # generic Tables.jl constructor happens to give the same copying behaviour.
    # `hasmethod` cannot see the difference, since the generic constructor also
    # accepts `copycols` and applies here; the matching method's own signature
    # can.
    m = only(methods(DataFrame, Tuple{ForecastTable}))
    @test Base.kwarg_decl(m) == [:copycols]

    # The default and an explicit `copycols = true` both isolate the caller: a
    # write through the returned frame must not reach the table's store.
    for d in (DataFrame(ft), DataFrame(ft; copycols = true))
        d.value[1] = -111.0
        @test DataFrame(ft).value[1] == 1.0
    end

    # `copycols = false` opts into the zero-copy path, so the columns alias and a
    # write does reach the store.
    shared = DataFrame(ft; copycols = false)
    @test DataFrame(ft; copycols = false).value === shared.value
    shared.value[1] = -222.0
    @test DataFrame(ft).value[1] == -222.0
end

@testitem "Constructor and weights errors say what to do about it" begin
    using DataFrames

    # A frame using scoringutils/lopensemble column names should be told which
    # renames to make, not merely which of our names are absent. The hint map is
    # mirrored in r-pkg/ForecastEnsembles/R/utils.R.
    foreign = DataFrame(
        model = "m1", quantile_level = 0.5, predicted = 1.0,
        output_type = "quantile", location = "A"
    )
    err = try
        ForecastTable(foreign; task_id_cols = [:location])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("`model` -> `model_id`", err.msg)
    @test occursin("`quantile_level` -> `output_type_id`", err.msg)
    @test occursin("`predicted` -> `value`", err.msg)

    # A frame with no recognisable aliases gets the plain message, with no
    # misleading rename advice appended.
    bare = DataFrame(thing = 1, output_type = "quantile", location = "A")
    err2 = try
        ForecastTable(bare; task_id_cols = [:location])
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test !occursin("rename", err2.msg)

    # A method with no weight vector must say why, not just that it has none.
    blp = ForecastEnsembles.FittedBLP(1.0, 1.0, nothing)
    err3 = try
        effective_num_models(blp)
        nothing
    catch e
        e
    end
    @test err3 isa ArgumentError
    @test occursin("recalibrates", err3.msg)
end
