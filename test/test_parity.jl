using CSV
using DataFrames
using Statistics: mean, quantile

const REF = joinpath(@__DIR__, "reference")

function _read_input(path)
    df = CSV.read(joinpath(REF, path), DataFrame)
    df.output_type = Symbol.(df.output_type)
    return df
end

@testset "Parity — hubEnsembles::simple_ensemble" begin
    in_df = _read_input("simple_input.csv")
    ft = ForecastTable(in_df; task_id_cols = [:location, :horizon])

    on_cols = [:location, :horizon, :output_type_id]

    # mean
    out_df = select(DataFrame(combine(ft, SimpleEnsemble(:mean))),
                    on_cols..., :value)
    ref = select(CSV.read(joinpath(REF, "simple_mean_output.csv"), DataFrame),
                 on_cols..., :value => :value_r)
    j = innerjoin(out_df, ref; on = on_cols)
    @test nrow(j) == nrow(out_df) == nrow(ref)
    @test maximum(abs.(j.value .- j.value_r)) < 1e-10

    # median
    out_med = select(DataFrame(combine(ft, SimpleEnsemble(:median))),
                     on_cols..., :value)
    ref_med = select(CSV.read(joinpath(REF, "simple_median_output.csv"), DataFrame),
                     on_cols..., :value => :value_r)
    j2 = innerjoin(out_med, ref_med; on = on_cols)
    @test maximum(abs.(j2.value .- j2.value_r)) < 1e-10

    # weighted mean
    weights = CSV.read(joinpath(REF, "simple_weights.csv"), DataFrame)
    out_w = select(DataFrame(combine(ft, SimpleEnsemble(:mean; weights = weights))),
                   on_cols..., :value)
    ref_w = select(CSV.read(joinpath(REF, "simple_wmean_output.csv"), DataFrame),
                   on_cols..., :value => :value_r)
    j3 = innerjoin(out_w, ref_w; on = on_cols)
    @test maximum(abs.(j3.value .- j3.value_r)) < 1e-10
end

@testset "Parity — hubEnsembles::linear_pool (sample)" begin
    using Random: MersenneTwister
    in_df = _read_input("lp_sample_input.csv")
    in_df.output_type_id = parse.(Int, string.(in_df.output_type_id))
    ft = ForecastTable(in_df; task_id_cols = [:location])
    out = combine(ft, LinearPool(; n_samples = 600); rng = MersenneTwister(2026))
    out_df = DataFrame(out)
    ref = CSV.read(joinpath(REF, "lp_sample_output.csv"), DataFrame)
    # Both R and Julia produce 600/150 random samples per location. Compare
    # summary stats per location: mean and std should agree to within MC error.
    # Generous tolerance: R draws 150 samples per location; Julia 600. The
    # mixture distributions are the same in expectation but Monte Carlo
    # noise dominates, especially for quantile estimates from the 150-sample
    # R reference.
    for loc in unique(out_df.location)
        ours  = out_df[out_df.location .== loc, :value]
        theirs = ref[ref.location .== loc, :value]
        @test abs(mean(ours) - mean(theirs)) < 0.25
        @test abs(std(ours)  - std(theirs))  < 0.30
        for τ in (0.1, 0.5, 0.9)
            @test abs(quantile(ours, τ) - quantile(theirs, τ)) < 0.6
        end
    end
end

@testset "Parity — hubEnsembles::linear_pool (quantile)" begin
    using Random: MersenneTwister
    in_df = _read_input("lp_quantile_input.csv")
    ft = ForecastTable(in_df; task_id_cols = [:location, :horizon])
    out_df = select(DataFrame(combine(ft, LinearPool(; n_samples = 20_000);
                                       rng = MersenneTwister(7))),
                    :location, :horizon, :output_type_id, :value)
    ref = select(CSV.read(joinpath(REF, "lp_quantile_output.csv"), DataFrame),
                 :location, :horizon, :output_type_id, :value => :value_r)
    j = innerjoin(out_df, ref; on = [:location, :horizon, :output_type_id])
    # CDF reconstruction differs between distfromq (spline) and our PCHIP,
    # plus both paths have Monte Carlo noise from the n_output_samples step.
    @test maximum(abs.(j.value .- j.value_r)) < 0.5
end

@testset "Parity — qrensemble::qra (default)" begin
    in_df = CSV.read(joinpath(REF, "qra_input.csv"), DataFrame)
    target_date = CSV.read(joinpath(REF, "qra_target.csv"), DataFrame)[1, :target_date]
    train_df = in_df[in_df.target_date .!= target_date, :]
    test_df  = in_df[in_df.target_date .== target_date, :]

    train_ft = Ensembles.from_scoringutils(
        rename(train_df, :predicted => :predicted, :model => :model),
        task_id_cols = [:location, :horizon, :target_date],
    )
    obs = unique(train_df[:, [:location, :horizon, :target_date, :observed]])

    fitted = fit(QRA(; per_quantile_weights = false,
                     enforce_normalisation = true,
                     intercept = false,
                     noncross = true),
                 train_ft, obs)

    # Compare weights against the R reference.
    ref_w = CSV.read(joinpath(REF, "qra_default_weights.csv"), DataFrame)
    # `fitted.coefs` keys are ((), τ) → vector in `fitted.models` order.
    for τ in fitted.levels
        β = fitted.coefs[((), τ)]
        ref_τ = ref_w[ref_w.quantile_level .== τ, :]
        for (i, mod) in enumerate(fitted.models)
            ref_w_im = ref_τ[ref_τ.model .== mod, :weight]
            @test !isempty(ref_w_im)
            @test abs(β[i] - first(ref_w_im)) < 1e-3
        end
    end

    # Compare predicted values on the holdout target.
    test_ft = Ensembles.from_scoringutils(
        test_df, task_id_cols = [:location, :horizon, :target_date],
    )
    out = select(DataFrame(combine(test_ft, fitted)),
                 :location, :horizon, :target_date, :output_type_id, :value)
    ref_pred = CSV.read(joinpath(REF, "qra_default_output.csv"), DataFrame)
    rename!(ref_pred, :predicted => :predicted_r, :quantile_level => :output_type_id)
    ref_pred = select(ref_pred, :location, :horizon, :target_date, :output_type_id, :predicted_r)
    j = innerjoin(out, ref_pred;
                  on = [:location, :horizon, :target_date, :output_type_id])
    @test maximum(abs.(j.value .- j.predicted_r)) < 1e-3
end
