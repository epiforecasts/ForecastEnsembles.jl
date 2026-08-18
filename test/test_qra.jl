@testitem "QRA" begin
    using Random: MersenneTwister
    using Distributions: Normal, quantile
    using DataFrames
    using Statistics: cor
    rng = MersenneTwister(123)

    # Synthetic ground truth: y_t ~ N(0, 1).
    # Two component models: m_good predicts y exactly, m_noisy predicts noise.
    # We expect QRA with simplex constraint to put nearly all weight on m_good.
    n_train = 200
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n_train)

    rows = DataFrame[]
    for (mid, prediction) in (("m_good", y), ("m_noisy", randn(rng, n_train)))
        for τ in levels
            # Predicted quantile = forecast point + Φ^{-1}(τ).
            zτ = quantile(Normal(0, 1), τ)
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n_train,
                    value = prediction .+ zτ
                )
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n_train, observed = y)

    fitted = fit(QRA(; enforce_normalisation = true, intercept = false), train, obs)
    @test isa(fitted, FittedQRA)
    # Joint fit: same β across τ.
    βs = unique(values(fitted.coefs))
    @test length(βs) == 1
    β = first(βs)
    # Order is fitted.models (sorted): m_good, m_noisy
    idx_good = findfirst(==("m_good"), fitted.models)
    @test β[idx_good] > 0.9   # should dominate

    # combine on the same forecasts → predictions close to y.
    out = combine(train, fitted)
    d = DataFrame(out)
    median_rows = d[d.output_type_id .== 0.5, :]
    sort!(median_rows, :t)
    @test cor(median_rows.value, y) > 0.9
end

@testitem "QRA per_quantile_weights" begin
    using Random: MersenneTwister
    using Distributions: Normal, quantile
    using DataFrames
    rng = MersenneTwister(7)
    n_train = 150
    levels = [0.25, 0.5, 0.75]
    y = randn(rng, n_train)
    rows = DataFrame[]
    for (mid, prediction) in (("m_a", y .+ 0.5 .* randn(rng, n_train)), (
        "m_b", y .+ 0.5 .* randn(rng, n_train)))
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n_train,
                    value = prediction .+ zτ
                )
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n_train, observed = y)

    fitted = fit(
        QRA(; per_quantile_weights = true, enforce_normalisation = true, intercept = false),
        train,
        obs
    )
    # Different keys per τ.
    for τ in levels
        @test haskey(fitted.coefs, ((), τ))
    end

    out = combine(train, fitted)
    @test all(DataFrame(out).model_id .== "qra")
end

@testitem "QRA noncross" begin
    using Random: MersenneTwister
    using Distributions: Normal, quantile
    using DataFrames
    rng = MersenneTwister(99)
    n_train = 100
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n_train)

    rows = DataFrame[]
    for (mid, prediction) in (("m_a", y .+ 0.3 .* randn(rng, n_train)), (
        "m_b", y .+ 0.3 .* randn(rng, n_train)))
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            push!(
                rows,
                DataFrame(
                    model_id = mid,
                    output_type = "quantile",
                    output_type_id = τ,
                    t = 1:n_train,
                    value = prediction .+ zτ
                )
            )
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n_train, observed = y)

    fitted = fit(
        QRA(;
            per_quantile_weights = true,
            enforce_normalisation = true,
            intercept = false,
            noncross = true
        ),
        train,
        obs
    )
    out = combine(train, fitted)
    d = DataFrame(out)

    # Verify per-task monotonicity in τ.
    for tdf in DataFrames.groupby(d, :t)
        sorted = sort(tdf, :output_type_id)
        @test issorted(sorted.value)
    end
end

@testitem "QRA defaults are the hub-safe simplex config" begin
    m = QRA()
    @test m.enforce_normalisation == true
    @test m.intercept == false
    @test m.noncross == false
    @test m.per_quantile_weights == false
    @test m.group == Symbol[]
end

@testitem "QRA(noncross = true) warns only without per_quantile_weights" begin
    # `@test_logs` installs a fresh TestLogger, so the source-level `maxlog = 1`
    # does not suppress the warning here regardless of test order.
    @test_logs (:warn, r"noncross.*no effect.*per_quantile_weights") QRA(noncross = true)
    @test_logs QRA(noncross = true, per_quantile_weights = true)
end

@testitem "QRA handles partially-missing model submissions" begin
    using Random: MersenneTwister
    using Distributions: Normal, quantile
    using DataFrames
    rng = MersenneTwister(7)

    n_train = 120
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n_train)

    rows = DataFrame[]
    for (mid, prediction) in (("m_good", y), ("m_noisy", randn(rng, n_train)))
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            df = DataFrame(model_id = mid, output_type = "quantile",
                output_type_id = τ, t = 1:n_train, value = prediction .+ zτ)
            # m_noisy skips the last 20 tasks — a partial submission.
            mid == "m_noisy" && (df = df[df.t .<= n_train - 20, :])
            push!(rows, df)
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n_train, observed = y)

    # Fitting drops the incomplete tasks (complete-case) instead of erroring.
    fitted = fit(QRA(; enforce_normalisation = true, intercept = false), train, obs)
    @test isa(fitted, FittedQRA)

    # Drop m_noisy at τ = 0.5 only: present in the table (top-level check passes)
    # but missing at that level, so the per-τ check fires with a clear
    # ArgumentError rather than a BoundsError.
    noisy_at_half = (train.data.model_id .== "m_noisy") .&
                    (train.data.output_type_id .== 0.5)
    bad_rows = train.data[(train.data.t .<= 3) .& .!noisy_at_half, :]
    bad = ForecastTable(bad_rows; task_id_cols = [:t])
    err = try
        combine(bad, fitted)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    # The message must be the specific per-τ one, naming the model and level.
    @test occursin("m_noisy", err.msg)
    @test occursin("0.5", err.msg)
end

@testitem "QRA noncross rejects partial submissions with per-τ size mismatch" begin
    using Random: MersenneTwister
    using Distributions: Normal, quantile
    using DataFrames
    rng = MersenneTwister(7)
    n_train = 120
    levels = [0.1, 0.5, 0.9]
    y = randn(rng, n_train)

    rows = DataFrame[]
    for (mid, prediction) in (("m_good", y), ("m_noisy", randn(rng, n_train)))
        for τ in levels
            zτ = quantile(Normal(0, 1), τ)
            df = DataFrame(model_id = mid, output_type = "quantile",
                output_type_id = τ, t = 1:n_train, value = prediction .+ zτ)
            # m_noisy drops the last 20 tasks at τ = 0.5 only, so the complete-case
            # count differs across τ — which the noncross joint LP cannot align.
            (mid == "m_noisy" && τ == 0.5) && (df = df[df.t .<= n_train - 20, :])
            push!(rows, df)
        end
    end
    train = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
    obs = DataFrame(t = 1:n_train, observed = y)

    err = try
        fit(
            QRA(; per_quantile_weights = true, noncross = true,
                enforce_normalisation = true, intercept = false),
            train, obs)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("noncross", err.msg)
    # The message lists the per-level task counts, so the diagnostic is useful:
    # `sizes` names the field and the dropped level's reduced count appears.
    @test occursin("sizes", err.msg)
    @test occursin(string(n_train - 20), err.msg)
end
