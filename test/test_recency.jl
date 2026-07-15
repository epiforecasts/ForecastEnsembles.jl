@testitem "CRPSStacking recency weighting" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames

    # Regime-switch setup shared by the recency tests: model A is sharp in the
    # first half of the training window, model B in the second half. Equal
    # weighting sees a symmetric problem; recency weighting should favour B.
    function _regime_data(; T = 40, K = 80, seed = 17)
        rng = MersenneTwister(seed)
        obs = DataFrame(t = 1:T, observed = randn(rng, T))
        rows = DataFrame[]
        for t in 1:T
            y = obs.observed[t]
            good_a = t <= T ÷ 2
            for (mid, sharp) in (("m_a", good_a), ("m_b", !good_a))
                sd = sharp ? 0.3 : 4.0
                push!(
                    rows,
                    DataFrame(
                        model_id = mid,
                        output_type = "sample",
                        output_type_id = 1:K,
                        t = t,
                        value = y .+ sd .* randn(rng, K)
                    )
                )
            end
        end
        ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
    end

    train, obs = _regime_data()

    w_b(fitted) = fitted.weights[fitted.weights.model_id .== "m_b", :weight][1]

    # Symmetric regimes, equal weighting → roughly balanced weights.
    equal = fit(CRPSStacking(), train, obs)
    @test 0.3 < w_b(equal) < 0.7

    # Strong exponential recency → the recently-good model dominates.
    recent = fit(CRPSStacking(; lambda = 0.7, time_col = :t), train, obs)
    @test w_b(recent) > 0.9

    # φ = 1 is exactly equal weighting.
    phi1 = fit(CRPSStacking(; lambda = 1.0, time_col = :t), train, obs)
    @test phi1.weights.weight ≈ equal.weights.weight atol = 1e-8

    # :equal symbol too.
    eq_sym = fit(CRPSStacking(; lambda = :equal, time_col = :t), train, obs)
    @test eq_sym.weights.weight ≈ equal.weights.weight atol = 1e-8

    # The lopensemble ramp leans recent, but only mildly (oldest ≈ 1,
    # newest 2): between equal and strong decay.
    ramp = fit(CRPSStacking(; lambda = :lopensemble, time_col = :t), train, obs)
    @test w_b(equal) < w_b(ramp) < w_b(recent)

    # Function form: quadratic ramp expressed directly matches :lopensemble.
    fn = fit(CRPSStacking(; lambda = u -> 2 - (1 - u)^2, time_col = :t), train, obs)
    @test fn.weights.weight ≈ ramp.weights.weight atol = 1e-8

    # Vector form: explicit per-time weights (one per unique t).
    T = length(unique(obs.t))
    vec_w = [float(t > T ÷ 2) for t in 1:T]   # second half only
    late_only = fit(CRPSStacking(; lambda = vec_w, time_col = :t), train, obs)
    @test w_b(late_only) > 0.95
    # Wrong length raises.
    @test_throws ArgumentError fit(
        CRPSStacking(; lambda = ones(T + 1), time_col = :t),
        train,
        obs
    )
end

@testitem "CRPSStacking task_weights" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames

    # Regime-switch setup shared by the recency tests: model A is sharp in the
    # first half of the training window, model B in the second half. Equal
    # weighting sees a symmetric problem; recency weighting should favour B.
    function _regime_data(; T = 40, K = 80, seed = 17)
        rng = MersenneTwister(seed)
        obs = DataFrame(t = 1:T, observed = randn(rng, T))
        rows = DataFrame[]
        for t in 1:T
            y = obs.observed[t]
            good_a = t <= T ÷ 2
            for (mid, sharp) in (("m_a", good_a), ("m_b", !good_a))
                sd = sharp ? 0.3 : 4.0
                push!(
                    rows,
                    DataFrame(
                        model_id = mid,
                        output_type = "sample",
                        output_type_id = 1:K,
                        t = t,
                        value = y .+ sd .* randn(rng, K)
                    )
                )
            end
        end
        ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
    end

    train, obs = _regime_data()
    T = length(unique(obs.t))

    # Uniform task_weights reproduce the unweighted fit.
    uniform = DataFrame(t = 1:T, weight = fill(1.0, T))
    w_u = fit(CRPSStacking(; task_weights = uniform), train, obs)
    w_0 = fit(CRPSStacking(), train, obs)
    @test w_u.weights.weight ≈ w_0.weights.weight atol = 1e-8

    # task_weights equivalent to the exponential lambda gives the same fit.
    expo = DataFrame(t = 1:T, weight = [0.7^(T - t) for t in 1:T])
    w_e = fit(CRPSStacking(; task_weights = expo), train, obs)
    w_l = fit(CRPSStacking(; lambda = 0.7, time_col = :t), train, obs)
    @test w_e.weights.weight ≈ w_l.weights.weight atol = 1e-8

    # Missing a task raises.
    @test_throws ArgumentError fit(
        CRPSStacking(; task_weights = uniform[1:(T - 1), :]),
        train,
        obs
    )
    # Negative weights rejected at construction.
    @test_throws ArgumentError CRPSStacking(;
        task_weights = DataFrame(t = [1], weight = [-1.0]),
    )
end

@testitem "Parity — lopensemble ramp" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames

    # Regime-switch setup shared by the recency tests: model A is sharp in the
    # first half of the training window, model B in the second half. Equal
    # weighting sees a symmetric problem; recency weighting should favour B.
    function _regime_data(; T = 40, K = 80, seed = 17)
        rng = MersenneTwister(seed)
        obs = DataFrame(t = 1:T, observed = randn(rng, T))
        rows = DataFrame[]
        for t in 1:T
            y = obs.observed[t]
            good_a = t <= T ÷ 2
            for (mid, sharp) in (("m_a", good_a), ("m_b", !good_a))
                sd = sharp ? 0.3 : 4.0
                push!(
                    rows,
                    DataFrame(
                        model_id = mid,
                        output_type = "sample",
                        output_type_id = 1:K,
                        t = t,
                        value = y .+ sd .* randn(rng, K)
                    )
                )
            end
        end
        ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
    end

    using CSV
    ref_dir = joinpath(@__DIR__, "reference")
    ramp_path = joinpath(ref_dir, "crps_weights_ramp_output.csv")
    if !isfile(ramp_path)
        @warn "ramp fixture missing; run test/reference/generate_lopensemble.R"
    else
        in_df = CSV.read(joinpath(ref_dir, "crps_input.csv"), DataFrame)
        rename!(
            in_df,
            :model => :model_id,
            :sample_id => :output_type_id,
            :predicted => :value
        )
        in_df.output_type = fill(:sample, nrow(in_df))
        ft = ForecastTable(
            in_df[:, [:model_id, :output_type, :output_type_id, :date, :value]];
            task_id_cols = [:date]
        )
        obs = unique(in_df[:, [:date, :observed]])

        fitted = fit(CRPSStacking(; lambda = :lopensemble, time_col = :date), ft, obs)
        ref = CSV.read(ramp_path, DataFrame)
        rename!(ref, :model => :model_id, :weight => :weight_r)
        j = innerjoin(fitted.weights, ref; on = :model_id)
        @test nrow(j) == nrow(ref)
        @test maximum(abs.(j.weight .- j.weight_r)) < 0.05
    end
end
