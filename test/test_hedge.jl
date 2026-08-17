@testitem "Hedge online weighting (any score function)" begin
    using Random: MersenneTwister
    using Statistics: mean
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Helper: build a sample ForecastTable from per-(model, time) samplers.
    function build(samplers, obs; K = 80, rng = MersenneTwister(0))
        rows = DataFrame[]
        for (mid, sampler) in samplers
            for t in obs.t
                push!(
                    rows,
                    DataFrame(
                        model_id = mid,
                        output_type = "sample",
                        output_type_id = 1:K,
                        t = t,
                        value = sampler(obs.observed[t], rng, t)
                    )
                )
            end
        end
        return ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    end

    rng = MersenneTwister(2026)
    T = 40
    obs = DataFrame(t = 1:T, observed = randn(rng, T))

    # m_good is consistently sharp and centred; m_noisy is wide. Hedge should
    # end up favouring m_good.
    train = build(
        [
            "m_good" => ((y, r, t) -> y .+ randn(r, 80)),
            "m_noisy" => ((y, r, t) -> 5.0 .* randn(r, 80))
        ],
        obs;
        rng = MersenneTwister(1)
    )

    fitted = fit(Hedge(crps; eta = 1.0, time_col = :t), train, obs)
    @test isa(fitted, FittedHedge)
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-8
    gw = fitted.weights[fitted.weights.model_id .== "m_good", :weight][1]
    nw = fitted.weights[fitted.weights.model_id .== "m_noisy", :weight][1]
    @test gw > nw
    @test gw > 0.7

    # Trajectory: one row per (time, model); each step is a simplex; the final
    # step matches the returned weights.
    @test nrow(fitted.trajectory) == T * 2
    for g in groupby(fitted.trajectory, :t)
        @test sum(g.weight) ≈ 1.0 atol = 1e-8
    end
    last_step = fitted.trajectory[fitted.trajectory.t .== T, :]
    @test sort(last_step, :model_id).weight ≈ sort(fitted.weights, :model_id).weight

    # Learning rate: at low eta (before the weight saturates) a larger eta
    # concentrates weight harder on the better member; eta → 0 stays uniform.
    warm = fit(Hedge(crps; eta = 0.01, time_col = :t), train, obs)
    hot = fit(Hedge(crps; eta = 0.05, time_col = :t), train, obs)
    warm_g = warm.weights[warm.weights.model_id .== "m_good", :weight][1]
    hot_g = hot.weights[hot.weights.model_id .== "m_good", :weight][1]
    @test hot_g > warm_g
    cold = fit(Hedge(crps; eta = 1e-4, time_col = :t), train, obs)
    @test all(abs.(cold.weights.weight .- 0.5) .< 0.05)

    # combine → sample table; weights() round-trips.
    @test all(DataFrame(combine(train, fitted; rng = MersenneTwister(1))).output_type .===
              :sample)
    @test weights(fitted) isa EnsembleWeights

    # Adaptivity: m_a is sharp for the first half, m_b for the second. Hedge
    # tracks the switch — it ends favouring m_b, unlike a pooled scheme.
    switch = build(
        [
            "m_a" => ((
                y, r, t) -> (t <= T ÷ 2 ? y .+ randn(r, 80) : y .+ 5.0 .* randn(r, 80))),
            "m_b" => ((
                y, r, t) -> (t <= T ÷ 2 ? y .+ 5.0 .* randn(r, 80) : y .+ randn(r, 80)))
        ],
        obs;
        rng = MersenneTwister(3)
    )
    sfit = fit(Hedge(crps; eta = 1.5, time_col = :t), switch, obs)
    @test sfit.weights[sfit.weights.model_id .== "m_b", :weight][1] >
          sfit.weights[sfit.weights.model_id .== "m_a", :weight][1]

    # Guards.
    @test_throws ArgumentError Hedge(crps; eta = 0.0, time_col = :t)
    @test_throws ArgumentError fit(Hedge(crps; eta = 1.0, time_col = :nope), train, obs)

    # Quantile input is out of scope.
    qtrain = ForecastTable(
        DataFrame(
            model_id = repeat(["m_a", "m_b"], inner = 2),
            output_type = "quantile",
            output_type_id = repeat([0.25, 0.75], 2),
            t = 1,
            value = [1.0, 3.0, 2.0, 4.0]
        );
        task_id_cols = [:t]
    )
    @test_throws ArgumentError fit(Hedge(crps; eta = 1.0, time_col = :t), qtrain, obs)
end

@testitem "Hedge scores per (model, task), not per (model, time)" begin
    using Random: MersenneTwister
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Two locations with far-apart observed levels. `m_good` is centred and sharp
    # at BOTH locations; `m_bad` is offset by 10 at both. Scoring correctly (per
    # location) sees m_good as excellent everywhere. Scoring by (model, time)
    # alone pools each model's samples across the two locations into one vector
    # and scores it against a single arbitrary observation, so the score is
    # dominated by the between-location spread and the model ordering scrambles.
    rng = MersenneTwister(11)
    T, K = 20, 200
    level = Dict("A" => 0.0, "B" => 100.0)
    rows = DataFrame[]
    obsrows = DataFrame[]
    for loc in ("A", "B"), t in 1:T

        y = level[loc] + 0.1 * randn(rng)
        push!(obsrows, DataFrame(location = loc, t = t, observed = y))
        push!(rows,
            DataFrame(model_id = "m_good", output_type = "sample",
                output_type_id = 1:K, location = loc, t = t, value = y .+ randn(rng, K)))
        push!(rows,
            DataFrame(model_id = "m_bad", output_type = "sample",
                output_type_id = 1:K, location = loc, t = t,
                value = y .+ 10.0 .+ randn(rng, K)))
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:location, :t])
    observations = reduce(vcat, obsrows)

    fitted = fit(Hedge(crps; eta = 1.0, time_col = :t), train, observations)
    gw = fitted.weights[fitted.weights.model_id .== "m_good", :weight][1]
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-8
    @test gw > 0.9
    # One trajectory row per (time, model) regardless of the extra task column.
    @test nrow(fitted.trajectory) == T * 2
end

@testitem "Hedge auto-scales eta to the score magnitude" begin
    using Random: MersenneTwister
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Large-magnitude forecasts (scores in the tens–hundreds): a raw eta = 1 would
    # underflow exp(-eta·s) and collapse to one model in a step. The auto default
    # picks a stable, regret-scaled step size.
    T = 15
    K = 60
    rng = MersenneTwister(5)
    obs = DataFrame(t = 1:T, observed = 100.0 .* randn(rng, T))
    srng = MersenneTwister(6)
    rows = DataFrame[]
    for (mid, sd) in (("m_good", 20.0), ("m_bad", 200.0))
        for t in 1:T
            push!(rows,
                DataFrame(model_id = mid, output_type = "sample",
                    output_type_id = 1:K, t = t,
                    value = obs.observed[t] .+ sd .* randn(srng, K)))
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])

    # eta = nothing → chosen from the score scale at fit time.
    @test Hedge(crps; time_col = :t).eta === nothing
    auto = fit(Hedge(crps; time_col = :t), train, obs)
    gw = auto.weights[auto.weights.model_id .== "m_good", :weight][1]
    bw = auto.weights[auto.weights.model_id .== "m_bad", :weight][1]
    @test all(isfinite, auto.weights.weight)
    @test sum(auto.weights.weight) ≈ 1.0 atol = 1e-8
    @test gw > bw
    # Auto-scaling keeps the update graded: with MersenneTwister(5)/(6) and T = 15
    # gw lands well below 1 (≈0.6–0.8), so the loose < 0.999 bound just asserts it
    # did not collapse to a single model in one step.
    @test gw < 0.999

    # Contrast: a raw eta = 1 on this score scale does collapse.
    raw = fit(Hedge(crps; eta = 1.0, time_col = :t), train, obs)
    @test raw.weights[raw.weights.model_id .== "m_good", :weight][1] ≈ 1.0 atol = 1e-6
    # ...and m_bad is driven to zero (guards against a NaN surviving renormalisation).
    @test raw.weights[raw.weights.model_id .== "m_bad", :weight][1] ≈ 0.0 atol = 1e-6

    # An explicit eta still overrides.
    @test Hedge(crps; eta = 0.5, time_col = :t).eta == 0.5
    @test_throws ArgumentError Hedge(crps; eta = 0.0, time_col = :t)
end

@testitem "Hedge auto-eta stays uniform when every score is zero" begin
    using DataFrames
    include(joinpath(@__DIR__, "score_helpers.jl"))

    # Degenerate case: both members forecast a point mass at the observed value, so
    # every per-round CRPS is exactly 0. `_auto_eta` then sees B = 0 and returns 0,
    # so no update is applied and the weights must remain the uniform prior.
    T, K = 8, 50
    obs = DataFrame(t = 1:T, observed = Float64.(1:T))
    rows = DataFrame[]
    for mid in ("m1", "m2")
        for t in 1:T
            push!(rows,
                DataFrame(model_id = mid, output_type = "sample",
                    output_type_id = 1:K, t = t, value = fill(obs.observed[t], K)))
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])

    fitted = fit(Hedge(crps; time_col = :t), train, obs)
    @test all(isfinite, fitted.weights.weight)
    @test all(abs.(fitted.weights.weight .- 0.5) .< 1e-9)
end
