@testset "Interop converters" begin
    # scoringutils-shaped → ForecastTable
    su = DataFrame(
        model = ["m1", "m1", "m2", "m2"],
        quantile_level = [0.25, 0.75, 0.25, 0.75],
        predicted = [1.0, 3.0, 2.0, 4.0],
        observed = [2.0, 2.0, 2.0, 2.0],
        location = ["A", "A", "A", "A"],
    )
    ft = ForecastEnsembles.from_scoringutils(su; task_id_cols = [:location])
    d = DataFrame(ft)
    @test :model_id in propertynames(d)
    @test :output_type_id in propertynames(d)
    @test :value in propertynames(d)
    @test all(d.output_type .=== :quantile)
    @test !(:observed in propertynames(d))

    # samples-shaped → ForecastTable
    smp = DataFrame(
        model = repeat(["m1", "m2"], inner = 3),
        sample = repeat([1, 2, 3], 2),
        predicted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        location = "A",
    )
    ft2 = ForecastEnsembles.from_samples(smp; task_id_cols = [:location])
    d2 = DataFrame(ft2)
    @test all(d2.output_type .=== :sample)
    @test Set(unique(d2.model_id)) == Set(["m1", "m2"])
end
