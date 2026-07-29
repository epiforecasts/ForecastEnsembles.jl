@testitem "PartialPooling hierarchical stacking (any score function)" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Two strata with opposite preferences: in loc A, m1 is sharp and m2 wide;
    # in loc B the roles swap. Independent stacking should split them; strong
    # pooling should drag both strata toward the (balanced) global vector.
    function build(; K = 60, T = 30, seed = 1)
        rng = MersenneTwister(seed)
        obs = DataFrame(loc = String[], t = Int[], observed = Float64[])
        rows = DataFrame[]
        for loc in ("A", "B")
            for t in 1:T
                y = randn(rng)
                push!(obs, (loc, t, y))
                sharp, wide = ("m1", "m2")
                loc == "B" && ((sharp, wide) = ("m2", "m1"))
                for (mid, sd) in ((sharp, 1.0), (wide, 5.0))
                    push!(
                        rows,
                        DataFrame(
                            model_id = mid,
                            output_type = "sample",
                            output_type_id = 1:K,
                            loc = loc,
                            t = t,
                            value = y .+ sd .* randn(rng, K)
                        )
                    )
                end
            end
        end
        return ForecastTable(reduce(vcat, rows); task_id_cols = [:loc, :t]), obs
    end

    train, obs = build()

    w_of(df, loc, mid) = df[(df.loc .== loc) .& (df.model_id .== mid), :weight][1]

    # Independent (lambda = 0): each stratum backs its own sharp model.
    indep = fit(PartialPooling(crps; strata = [:loc], lambda = 0.0), train, obs)
    @test isa(indep, FittedPartialPooling)
    @test w_of(indep.weights, "A", "m1") > w_of(indep.weights, "A", "m2")
    @test w_of(indep.weights, "B", "m2") > w_of(indep.weights, "B", "m1")
    # Each stratum is a simplex.
    for g in groupby(indep.weights, :loc)
        @test sum(g.weight) ≈ 1.0 atol = 1e-6
    end

    # Strong pooling drags the strata together: the A–B gap in m1's weight
    # shrinks relative to the independent fit.
    pooled = fit(PartialPooling(crps; strata = [:loc], lambda = 50.0), train, obs)
    gap(f) = abs(w_of(f.weights, "A", "m1") - w_of(f.weights, "B", "m1"))
    @test gap(pooled) < gap(indep)
    # Global (pooled) vector is roughly balanced, since the strata cancel.
    @test sum(pooled.global_weights.weight) ≈ 1.0 atol = 1e-6
    @test abs(pooled.global_weights[pooled.global_weights.model_id .== "m1", :weight][1] -
              0.5) < 0.2

    # combine applies each stratum's own weights and keeps the strata column.
    out = DataFrame(combine(train, indep; rng = MersenneTwister(1)))
    @test all(out.output_type .=== :sample)
    @test Set(out.loc) == Set(["A", "B"])
    @test weights(indep) isa EnsembleWeights

    # An unseen stratum falls back to the global vector (no error).
    unseen = ForecastTable(
        DataFrame(
            model_id = repeat(["m1", "m2"], inner = 4),
            output_type = "sample",
            output_type_id = repeat(1:4, 2),
            loc = "C",
            t = 1,
            value = randn(MersenneTwister(9), 8)
        );
        task_id_cols = [:loc, :t]
    )
    ocomb = DataFrame(combine(unseen, indep; rng = MersenneTwister(2)))
    @test all(ocomb.loc .== "C")

    # Guards.
    @test_throws ArgumentError PartialPooling(crps; strata = [:loc], lambda = -1.0)
    @test_throws ArgumentError PartialPooling(crps; strata = Symbol[])
    @test_throws ArgumentError fit(
        PartialPooling(crps; strata = [:nope]), train, obs)

    # Quantile input is out of scope.
    qtrain = ForecastTable(
        DataFrame(
            model_id = repeat(["m1", "m2"], inner = 2),
            output_type = "quantile",
            output_type_id = repeat([0.25, 0.75], 2),
            loc = "A",
            t = 1,
            value = [1.0, 3.0, 2.0, 4.0]
        );
        task_id_cols = [:loc, :t]
    )
    @test_throws ArgumentError fit(PartialPooling(crps; strata = [:loc]), qtrain, obs)
end
