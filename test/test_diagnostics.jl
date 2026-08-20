@testitem "Weight diagnostics" begin
    using DataFrames

    # effective_num_models: 1 when concentrated, M when uniform.
    @test effective_num_models([1.0, 0.0, 0.0]) ≈ 1.0
    @test effective_num_models([0.25, 0.25, 0.25, 0.25]) ≈ 4.0
    @test effective_num_models([0.5, 0.5]) ≈ 2.0
    # Unnormalised input is normalised first.
    @test effective_num_models([2.0, 2.0]) ≈ 2.0
    @test 1.0 < effective_num_models([0.7, 0.2, 0.1]) < 3.0

    # Accepts an EnsembleWeights and a raw DataFrame.
    ew = EnsembleWeights(DataFrame(model_id = ["a", "b"], weight = [0.5, 0.5]))
    @test effective_num_models(ew) ≈ 2.0
    @test effective_num_models(DataFrame(model_id = ["a", "b", "c"],
        weight = [1 / 3, 1 / 3, 1 / 3])) ≈ 3.0

    # Accepts a fitted method through `weights`.
    fitted = FittedCRPSStacking(
        DataFrame(model_id = ["a", "b"], weight = [0.9, 0.1]), ["a", "b"], 0.0)
    @test effective_num_models(fitted) ≈ 1 / (0.81 + 0.01)

    # Per-quantile weights → one value per level.
    pq = EnsembleWeights(
        DataFrame(
        model_id = ["a", "b", "a", "b"],
        output_type_id = [0.25, 0.25, 0.75, 0.75],
        weight = [1.0, 0.0, 0.5, 0.5]
    )
    )
    res = effective_num_models(pq)
    @test res isa DataFrame
    @test nrow(res) == 2
    @test res[res.output_type_id .== 0.25, :effective_num_models][1] ≈ 1.0
    @test res[res.output_type_id .== 0.75, :effective_num_models][1] ≈ 2.0

    # Zero-sum weights are rejected.
    @test_throws ArgumentError effective_num_models([0.0, 0.0])

    # weight_stability over a Hedge trajectory: a stable model has zero total
    # variation, a swinging one has more.
    traj = DataFrame(
        model_id = repeat(["stable", "swing"], inner = 3),
        weight = [0.5, 0.5, 0.5, 0.5, 0.2, 0.8],
        t = [1, 2, 3, 1, 2, 3]
    )
    fh = FittedHedge(
        DataFrame(model_id = ["stable", "swing"], weight = [0.5, 0.8]),
        ["stable", "swing"],
        traj,
        :t
    )
    ws = weight_stability(fh)
    @test ws isa DataFrame
    @test ws[ws.model_id .== "stable", :total_variation][1] ≈ 0.0
    @test ws[ws.model_id .== "swing", :total_variation][1] ≈ 0.9
    @test ws[ws.model_id .== "swing", :total_variation][1] >
          ws[ws.model_id .== "stable", :total_variation][1]

    # A trajectory carrying extra columns still orders by the stored time_col;
    # the rows are shuffled so the ordering has to come from `time_col`.
    wide = traj[[6, 1, 4, 2, 5, 3], :]
    wide[!, :location] .= "A"
    wide[!, :eta] .= 0.1
    wide_fh = FittedHedge(
        DataFrame(model_id = ["stable", "swing"], weight = [0.5, 0.8]),
        ["stable", "swing"],
        wide,
        :t
    )
    @test sort(weight_stability(wide_fh), :model_id) == sort(ws, :model_id)

    # A time_col that is not in the trajectory names the missing column.
    bad_fh = FittedHedge(
        DataFrame(model_id = ["stable", "swing"], weight = [0.5, 0.8]),
        ["stable", "swing"],
        traj,
        :date
    )
    @test_throws ArgumentError weight_stability(bad_fh)

    # Empty trajectory → empty result, no error.
    empty_fh = FittedHedge(
        DataFrame(model_id = String[], weight = Float64[]), String[], DataFrame(), :t)
    @test nrow(weight_stability(empty_fh)) == 0
end
