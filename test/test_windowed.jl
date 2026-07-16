@testitem "Windowed training" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames

    rng = MersenneTwister(11)
    T = 30
    K = 60
    obs = DataFrame(t = 1:T, observed = randn(rng, T))
    rows = DataFrame[]
    for (mid, s) in (("m_a", (y, r) -> y .+ randn(r, K)), (
        "m_b", (y, r) -> 2.0 .* randn(r, K)))
        for t in 1:T
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "sample",
                    output_type_id = 1:K,
                    t = t,
                    value = s(obs.observed[t], rng)
                )
            )
        end
    end
    full_df = reduce(vcat, rows)
    train = ForecastTable(full_df; task_id_cols = [:t])

    # Windowed fit == fitting the inner method on the last `window` times.
    w = 10
    fw = fit(Windowed(CRPSStacking(), w; time_col = :t), train, obs)
    lastt = Set(sort(unique(full_df.t))[(end - w + 1):end])
    sub = full_df[in.(full_df.t, Ref(lastt)), :]
    sub_obs = obs[in.(obs.t, Ref(lastt)), :]
    fm = fit(CRPSStacking(), ForecastTable(sub; task_id_cols = [:t]), sub_obs)
    @test sort(fw.weights, :model_id).weight ≈ sort(fm.weights, :model_id).weight

    # A window larger than the data uses every time (== a plain fit).
    fw_all = fit(Windowed(CRPSStacking(), 1000; time_col = :t), train, obs)
    fm_all = fit(CRPSStacking(), train, obs)
    @test sort(fw_all.weights, :model_id).weight ≈ sort(fm_all.weights, :model_id).weight

    # Guards.
    @test_throws ArgumentError Windowed(CRPSStacking(), 0; time_col = :t)
    @test_throws ArgumentError fit(
        Windowed(CRPSStacking(), 5; time_col = :nope), train, obs)

    # Composes as a scheme in backtest — rolling vs expanding window.
    using ScoringRules
    res = backtest(
        train,
        obs,
        [
            "expanding" => CRPSStacking(),
            "rolling" => Windowed(CRPSStacking(), 8; time_col = :t)
        ];
        time_col = :t,
        min_train = 12
    )
    @test Set(res.scheme) == Set(["expanding", "rolling"])
    @test all(isfinite, res.score)
end
