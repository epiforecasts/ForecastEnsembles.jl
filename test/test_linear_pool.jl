using Random: MersenneTwister
using Statistics: mean, std, quantile

@testset "LinearPool — sample path" begin
    rng = MersenneTwister(7)
    # Two models, very different sample distributions.
    df = DataFrame(
        model_id = vcat(fill("m1", 200), fill("m2", 200)),
        output_type = "sample",
        output_type_id = vcat(1:200, 1:200),
        location = "A",
        value = vcat(randn(rng, 200), randn(rng, 200) .+ 10.0)
    )
    ft = ForecastTable(df; task_id_cols = [:location])

    # Equal weights: pooled mean ≈ 5.0
    out = combine(ft, LinearPool(; n_samples = 5000); rng = MersenneTwister(1))
    pooled = DataFrame(out).value
    @test mean(pooled) ≈ 5.0 atol = 0.5

    # 80/20 weights toward m1: pooled mean ≈ 2.0
    w = DataFrame(model_id = ["m1", "m2"], weight = [0.8, 0.2])
    out_w = combine(ft, LinearPool(; n_samples = 5000, weights = w); rng = MersenneTwister(1))
    pooled_w = DataFrame(out_w).value
    @test mean(pooled_w) ≈ 2.0 atol = 0.5
end

@testset "LinearPool — quantile path" begin
    using Distributions
    rng = MersenneTwister(11)

    # Two normals: N(0, 1) and N(5, 1). With equal weights the mixture mean
    # is 2.5; the median should be roughly there too.
    probs = collect(0.05:0.05:0.95)
    rows = DataFrame[]
    for (mid, dist) in (("m1", Normal(0, 1)), ("m2", Normal(5, 1)))
        push!(
            rows,
            DataFrame(
                model_id = mid,
                output_type = "quantile",
                output_type_id = probs,
                location = "A",
                value = quantile.(Ref(dist), probs)
            )
        )
    end
    df = reduce(vcat, rows)
    ft = ForecastTable(df; task_id_cols = [:location])

    out = combine(ft, LinearPool(; n_samples = 20_000); rng = MersenneTwister(2))
    d = sort(DataFrame(out), :output_type_id)
    # Median of the equal-weighted mixture is somewhere in (0, 5); not too
    # close to either component.
    median_row = d[d.output_type_id .== 0.5, :].value[1]
    @test 1.0 < median_row < 4.0
    # 5th and 95th percentile of mixture should bracket [-3, 7] roughly.
    @test d[d.output_type_id .== 0.05, :].value[1] < 0.0
    @test d[d.output_type_id .== 0.95, :].value[1] > 4.0
end

@testset "LinearPool — cdf path" begin
    df = DataFrame(
        model_id = repeat(["m1", "m2"], inner = 3),
        output_type = "cdf",
        output_type_id = repeat([0.0, 1.0, 2.0], 2),
        location = "A",
        value = [0.1, 0.5, 0.9, 0.3, 0.7, 0.95]
    )
    ft = ForecastTable(df; task_id_cols = [:location])

    # Equal weights → average of the two CDFs.
    out = combine(ft, LinearPool())
    d = sort(DataFrame(out), :output_type_id)
    @test d.value ≈ [(0.1 + 0.3)/2, (0.5 + 0.7)/2, (0.9 + 0.95)/2]

    # 75/25 weights.
    w = DataFrame(model_id = ["m1", "m2"], weight = [0.75, 0.25])
    out_w = combine(ft, LinearPool(; weights = w))
    d_w = sort(DataFrame(out_w), :output_type_id)
    @test d_w.value ≈ [0.75*0.1 + 0.25*0.3, 0.75*0.5 + 0.25*0.7, 0.75*0.9 + 0.25*0.95]
end
