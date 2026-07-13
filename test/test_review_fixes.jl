@testitem "QuantileDistribution edge cases" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    # Two quantile pairs is the minimum valid input; previously crashed with
    # BoundsError in the PCHIP endpoint formulae.
    qd = ForecastEnsembles.QuantileDistribution([0.25, 0.75], [1.0, 3.0])
    @test quantile(qd, 0.25) ≈ 1.0
    @test quantile(qd, 0.75) ≈ 3.0
    @test quantile(qd, 0.25) < quantile(qd, 0.5) < quantile(qd, 0.75)
    # Both tails are fitted from the same two pairs → a single Normal; the
    # implied median is the midpoint.
    @test quantile(qd, 0.5) ≈ 2.0 atol = 1e-9

    # Duplicate probabilities previously passed `issorted` and produced NaN
    # slopes silently.
    @test_throws ArgumentError ForecastEnsembles.QuantileDistribution(
        [0.1, 0.1, 0.5],
        [1.0, 1.0, 2.0]
    )
end

@testitem "ForecastTable validation" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    df = DataFrame(
        model_id = ["m1", "m2"],
        output_type = "quantile",
        output_type_id = [0.5, 0.5],
        location = "A",
        value = [1.0, 2.0]
    )

    # Empty table rejected.
    @test_throws ArgumentError ForecastTable(df[1:0, :]; task_id_cols = [:location])

    # NaN values rejected.
    bad = copy(df);
    bad.value = [1.0, NaN]
    @test_throws ArgumentError ForecastTable(bad; task_id_cols = [:location])

    # missing values rejected.
    bad2 = copy(df);
    bad2.value = [1.0, missing]
    @test_throws ArgumentError ForecastTable(bad2; task_id_cols = [:location])
end

@testitem "EnsembleWeights validation" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    @test_throws ArgumentError EnsembleWeights(
        DataFrame(model_id = ["m1", "m2"], weight = [1.5, -0.5]),
    )
    @test_throws ArgumentError EnsembleWeights(
        DataFrame(model_id = ["m1", "m2"], weight = [1.0, missing]),
    )
end

@testitem "Extra weights warn" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    df = DataFrame(
        model_id = repeat(["m1", "m2"], inner = 2),
        output_type = "quantile",
        output_type_id = repeat([0.25, 0.75], 2),
        location = "A",
        value = [1.0, 3.0, 2.0, 4.0]
    )
    ft = ForecastTable(df; task_id_cols = [:location])
    w = DataFrame(model_id = ["m1", "m2", "m3_typo"], weight = [0.4, 0.4, 0.2])
    @test_logs (:warn,) match_mode = :any combine(ft, QuantileEnsemble(:mean; weights = w))
end

@testitem "Single-model tables" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    rng = MersenneTwister(5)
    qdf = DataFrame(
        model_id = "only",
        output_type = "quantile",
        output_type_id = [0.1, 0.5, 0.9],
        location = "A",
        value = [-1.0, 0.0, 1.0]
    )
    qft = ForecastTable(qdf; task_id_cols = [:location])
    # Single-model ensembles reproduce the model at the knots.
    out_q = sort(DataFrame(combine(qft, QuantileEnsemble(:mean))), :output_type_id)
    @test out_q.value ≈ [-1.0, 0.0, 1.0]
    out_m = sort(DataFrame(combine(qft, MixtureEnsemble())), :output_type_id)
    @test out_m.value ≈ [-1.0, 0.0, 1.0] atol = 1e-8

    sdf = DataFrame(
        model_id = "only",
        output_type = "sample",
        output_type_id = 1:100,
        location = "A",
        value = randn(rng, 100)
    )
    sft = ForecastTable(sdf; task_id_cols = [:location])
    out_s = DataFrame(combine(sft, MixtureEnsemble(; n_samples = 50); rng = rng))
    @test nrow(out_s) == 50
end

@testitem "Exact mixture quantile inversion" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    # Equal mixture of N(0,1) and N(5,1). At x = 2.5 the two reconstructed
    # CDFs are evaluated in their (exact) Normal tails, so the mixture
    # median is exactly 2.5 — a property the old Monte Carlo path could only
    # approach stochastically.
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
                value = dquantile.(Ref(dist), probs)
            )
        )
    end
    ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:location])
    out = sort(DataFrame(combine(ft, MixtureEnsemble())), :output_type_id)
    med = out.value[out.output_type_id .== 0.5][1]
    @test med ≈ 2.5 atol = 1e-8

    # Determinism: two calls give identical output.
    out2 = sort(DataFrame(combine(ft, MixtureEnsemble())), :output_type_id)
    @test out.value == out2.value

    # Against the numerically true mixture quantiles (true Normal mixture):
    # the only error left is CDF reconstruction, not sampling.
    for τ in (0.1, 0.25, 0.75, 0.9)
        truth_fn(x) = 0.5 * (dcdf(Normal(0, 1), x) + dcdf(Normal(5, 1), x)) - τ
        lo, hi = -10.0, 15.0
        for _ in 1:100
            mid = 0.5 * (lo + hi)
            truth_fn(mid) < 0 ? (lo = mid) : (hi = mid)
        end
        truth = 0.5 * (lo + hi)
        got = out.value[out.output_type_id .== τ][1]
        @test got ≈ truth atol = 0.02
    end
end

@testitem "_ints_summing_to always sums to N" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    rng = MersenneTwister(7)
    for _ in 1:50
        M = rand(rng, 1:8)
        w = rand(rng, M)
        w ./= sum(w)
        N = rand(rng, 1:5000)
        ks = ForecastEnsembles._ints_summing_to(rng, w, N)
        @test sum(ks) == N
        @test all(>=(0), ks)
    end
    # Degenerate: single model takes everything.
    @test ForecastEnsembles._ints_summing_to(rng, [1.0], 100) == [100]
end

@testitem "CRPSStacking guards" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    rng = MersenneTwister(11)
    df = DataFrame(
        model_id = repeat(["m1", "m2"], inner = 20),
        output_type = "sample",
        output_type_id = repeat(1:20, 2),
        t = 1,
        value = randn(rng, 40)
    )
    ft = ForecastTable(df; task_id_cols = [:t])
    obs = DataFrame(t = [1], observed = [0.0])

    # lambda without a time column is a user error, and gamma has been
    # replaced by task_weights.
    @test_throws ArgumentError CRPSStacking(; lambda = 0.95)
    @test_throws ArgumentError CRPSStacking(; gamma = 0.5)
    @test_throws ArgumentError CRPSStacking(; lambda = 1.5, time_col = :t)
    @test_throws ArgumentError CRPSStacking(;
        lambda = 0.9,
        time_col = :t,
        task_weights = DataFrame(t = [1], weight = [1.0])
    )

    # Tiny per-model sample counts no longer produce a biased/NaN diagonal.
    small = DataFrame(
        model_id = repeat(["m1", "m2"], inner = 2),
        output_type = "sample",
        output_type_id = repeat(1:2, 2),
        t = 1,
        value = [0.0, 1.0, 5.0, 6.0]
    )
    sft = ForecastTable(small; task_id_cols = [:t])
    fitted = fit(CRPSStacking(), sft, obs)
    @test all(isfinite, fitted.weights.weight)
    @test sum(fitted.weights.weight) ≈ 1.0 atol = 1e-8
end

@testitem "FittedQRA combine guards" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    rng = MersenneTwister(123)
    n = 60
    levels = [0.25, 0.5, 0.75]
    y = randn(rng, n)
    rows = DataFrame[]
    for (mid, pred) in
        (("m_a", y .+ 0.3 .* randn(rng, n)), ("m_b", y .+ 0.3 .* randn(rng, n)))
        for τ in levels
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n,
                    value = pred .+ dquantile(Normal(0, 1), τ)
                )
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n, observed = y)
    fitted = fit(QRA(; enforce_normalisation = true, intercept = false), train, obs)

    # Unseen quantile level → checked error, not KeyError.
    bad_level = DataFrame(
        model_id = ["m_a", "m_b"],
        output_type = "quantile",
        output_type_id = [0.33, 0.33],
        t = [1, 1],
        value = [0.0, 0.1]
    )
    @test_throws ArgumentError combine(
        ForecastTable(bad_level; task_id_cols = [:t]),
        fitted
    )

    # Missing model → checked error, not BoundsError.
    one_model = DataFrame(
        model_id = "m_a",
        output_type = "quantile",
        output_type_id = levels,
        t = 1,
        value = [-0.5, 0.0, 0.5]
    )
    @test_throws ArgumentError combine(
        ForecastTable(one_model; task_id_cols = [:t]),
        fitted
    )
end

@testitem "QRA noncross holds out of sample" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    rng = MersenneTwister(31)
    n = 80
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n)
    rows = DataFrame[]
    for (mid, pred) in
        (("m_a", y .+ 0.4 .* randn(rng, n)), ("m_b", y .+ 0.4 .* randn(rng, n)))
        for τ in levels
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n,
                    value = pred .+ dquantile(Normal(0, 1), τ)
                )
            )
        end
    end
    all_df = reduce(vcat, rows)
    train = ForecastTable(all_df[all_df.t .<= 60, :]; task_id_cols = [:t])
    test_ft = ForecastTable(all_df[all_df.t .> 60, :]; task_id_cols = [:t])
    obs = DataFrame(t = 1:60, observed = y[1:60])

    fitted = fit(
        QRA(;
            per_quantile_weights = true,
            noncross = true,
            enforce_normalisation = true,
            intercept = false
        ),
        train,
        obs
    )
    out = DataFrame(combine(test_ft, fitted))
    for tdf in DataFrames.groupby(out, :t)
        @test issorted(sort(tdf, :output_type_id).value)
    end
end

@testitem "Parity — qrensemble::qra (per-quantile, intercept)" begin
    using Random: MersenneTwister
    using Distributions: Normal, cdf as dcdf, quantile as dquantile
    using Statistics: mean, quantile
    using DataFrames
    using CSV
    ref_dir = joinpath(@__DIR__, "reference")
    in_df = CSV.read(joinpath(ref_dir, "qra_input.csv"), DataFrame)
    target_date = CSV.read(joinpath(ref_dir, "qra_target.csv"), DataFrame)[1, :target_date]
    train_df = in_df[in_df.target_date .!= target_date, :]
    test_df = in_df[in_df.target_date .== target_date, :]

    train_ft = ForecastEnsembles.from_scoringutils(
        train_df,
        task_id_cols = [:location, :horizon, :target_date]
    )
    obs = unique(train_df[:, [:location, :horizon, :target_date, :observed]])

    fitted = fit(
        QRA(;
            per_quantile_weights = true,
            enforce_normalisation = false,
            intercept = true,
            noncross = false
        ),
        train_ft,
        obs
    )

    ref_w = CSV.read(joinpath(ref_dir, "qra_perq_weights.csv"), DataFrame)
    for τ in fitted.levels
        β = fitted.coefs[((), τ)]
        ref_τ = ref_w[ref_w.quantile_level .== τ, :]
        for (i, mod) in enumerate(fitted.models)
            ref_val = first(ref_τ[ref_τ.model .== mod, :weight])
            @test β[i] ≈ ref_val atol = 1e-3
        end
    end

    test_ft = ForecastEnsembles.from_scoringutils(
        test_df,
        task_id_cols = [:location, :horizon, :target_date]
    )
    out = select(
        DataFrame(combine(test_ft, fitted)),
        :location,
        :horizon,
        :target_date,
        :output_type_id,
        :value
    )
    ref_pred = CSV.read(joinpath(ref_dir, "qra_perq_output.csv"), DataFrame)
    rename!(ref_pred, :predicted => :predicted_r, :quantile_level => :output_type_id)
    j = innerjoin(
        out,
        select(ref_pred, :location, :horizon, :target_date, :output_type_id, :predicted_r);
        on = [:location, :horizon, :target_date, :output_type_id]
    )
    @test nrow(j) == nrow(out)
    @test maximum(abs.(j.value .- j.predicted_r)) < 1e-3
end
