@testitem "LogarithmicPool geometric opinion pool" begin
    using DataFrames
    using Distributions: Normal, quantile
    using Statistics: median

    levels = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.99]
    # A model whose quantile forecast is Normal(μ, σ).
    normrows(mid, μ, σ) = DataFrame(
        model_id = mid, output_type = "quantile", output_type_id = levels,
        location = "A", value = [quantile(Normal(μ, σ), τ) for τ in levels])

    q(df, τ) = df[df.output_type_id .== τ, :value][1]
    width90(df) = q(df, 0.95) - q(df, 0.05)

    # Two identical N(0,1) members: the geometric pool is that same N(0,1).
    same = ForecastTable(
        vcat(normrows("m1", 0.0, 1.0), normrows("m2", 0.0, 1.0));
        task_id_cols = [:location]
    )
    gp = DataFrame(combine(same, LogarithmicPool(; ngrid = 4000)))
    @test isapprox(q(gp, 0.5), 0.0; atol = 0.03)
    @test isapprox(width90(gp), 2 * quantile(Normal(0, 1), 0.95); atol = 0.08)

    # N(-1,1) and N(1,1): the product of experts is exactly N(0,1) — centred at
    # 0 and much sharper than the (bimodal, wide) linear pool.
    split = ForecastTable(
        vcat(normrows("m1", -1.0, 1.0), normrows("m2", 1.0, 1.0));
        task_id_cols = [:location]
    )
    lg = DataFrame(combine(split, LogarithmicPool(; ngrid = 4000)))
    lp = DataFrame(combine(split, QuantileEnsemble(:mean)))
    @test isapprox(q(lg, 0.5), 0.0; atol = 0.05)
    @test isapprox(width90(lg), 2 * quantile(Normal(0, 1), 0.95); atol = 0.15)
    @test width90(lg) < width90(lp)                      # sharper than linear pool

    # Output is a valid, monotone quantile forecast.
    @test all(diff(sort(lg, :output_type_id).value) .>= 0)
    @test all(lg.output_type .=== :quantile)
    @test lg.model_id[1] == "hub-ensemble"

    # Precision weighting: unequal σ pulls the pooled centre toward the sharper
    # (tighter) member.
    prec = ForecastTable(
        vcat(normrows("sharp", -2.0, 0.5), normrows("wide", 2.0, 2.0));
        task_id_cols = [:location]
    )
    pg = DataFrame(combine(prec, LogarithmicPool(; ngrid = 4000)))
    @test q(pg, 0.5) < 0.0                                # nearer the sharp member at -2

    # Weights bias the pool; heavy weight on one member ≈ that member.
    w = EnsembleWeights(DataFrame(model_id = ["m1", "m2"], weight = [0.98, 0.02]))
    biased = DataFrame(combine(split, LogarithmicPool(; weights = w, ngrid = 4000)))
    @test q(biased, 0.5) < -0.5                           # pulled toward N(-1,1)

    # Guards.
    @test_throws ArgumentError LogarithmicPool(; ngrid = 10)
    pq = EnsembleWeights(
        DataFrame(model_id = ["m1"], output_type_id = [0.5], weight = [1.0]))
    @test_throws ArgumentError LogarithmicPool(; weights = pq)
    sft = ForecastTable(
        DataFrame(model_id = repeat(["m1", "m2"], inner = 3), output_type = "sample",
            output_type_id = repeat(1:3, 2), location = "A", value = Float64.(1:6));
        task_id_cols = [:location]
    )
    @test_throws ArgumentError combine(sft, LogarithmicPool())
end

@testitem "LogarithmicPool grid reaches levels more extreme than 1e-4" begin
    using DataFrames
    using Distributions: Normal, quantile as dquantile

    # Request a level (1e-5) more extreme than the fixed 1e-4 grid tail. With the
    # grid keyed to the requested levels it should extend to hold it rather than
    # clamp to the edge, so the 1e-5 quantile sits clearly below the 1% one.
    levels = [1.0e-5, 0.01, 0.5, 0.99, 1 - 1.0e-5]
    rows = DataFrame[]
    for (mid, μ) in (("m1", 0.0), ("m2", 2.0))
        push!(rows,
            DataFrame(model_id = mid, output_type = "quantile",
                output_type_id = levels, location = "A",
                value = [dquantile(Normal(μ, 1), τ) for τ in levels]))
    end
    ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:location])
    out = sort(DataFrame(combine(ft, LogarithmicPool())), :output_type_id)
    q(τ) = out.value[out.output_type_id .== τ][1]

    @test issorted(out.value)          # still monotone in τ
    @test q(1.0e-5) < q(0.01)          # extreme lower level not clamped to the 1% edge
    @test q(1 - 1.0e-5) > q(0.99)      # extreme upper level not clamped either
end
