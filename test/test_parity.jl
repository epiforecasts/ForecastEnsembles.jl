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
    out_df = select(DataFrame(combine(ft, QuantileEnsemble(:mean))),
                    on_cols..., :value)
    ref = select(CSV.read(joinpath(REF, "simple_mean_output.csv"), DataFrame),
                 on_cols..., :value => :value_r)
    j = innerjoin(out_df, ref; on = on_cols)
    @test nrow(j) == nrow(out_df) == nrow(ref)
    @test maximum(abs.(j.value .- j.value_r)) < 1e-10

    # median
    out_med = select(DataFrame(combine(ft, QuantileEnsemble(:median))),
                     on_cols..., :value)
    ref_med = select(CSV.read(joinpath(REF, "simple_median_output.csv"), DataFrame),
                     on_cols..., :value => :value_r)
    j2 = innerjoin(out_med, ref_med; on = on_cols)
    @test maximum(abs.(j2.value .- j2.value_r)) < 1e-10

    # weighted mean
    weights = CSV.read(joinpath(REF, "simple_weights.csv"), DataFrame)
    out_w = select(DataFrame(combine(ft, QuantileEnsemble(:mean; weights = weights))),
                   on_cols..., :value)
    ref_w = select(CSV.read(joinpath(REF, "simple_wmean_output.csv"), DataFrame),
                   on_cols..., :value => :value_r)
    j3 = innerjoin(out_w, ref_w; on = on_cols)
    @test maximum(abs.(j3.value .- j3.value_r)) < 1e-10
end

@testset "Parity — hubEnsembles::linear_pool (sample)" begin
    using Random: MersenneTwister
    using Distributions: Normal, pdf, quantile as dquantile

    in_df = _read_input("lp_sample_input.csv")
    in_df.output_type_id = parse.(Int, string.(in_df.output_type_id))
    ft = ForecastTable(in_df; task_id_cols = [:location])

    n_j = 600
    out = combine(ft, LinearPool(; n_samples = n_j); rng = MersenneTwister(2026))
    out_df = DataFrame(out)
    ref = CSV.read(joinpath(REF, "lp_sample_output.csv"), DataFrame)
    n_r = 150  # n_output_samples used in test/reference/generate.R

    # Theoretical Monte Carlo standard errors for two independent empirical
    # samples from the same distribution with std σ:
    #   SE(mean_a − mean_b)     = σ √(1/n_a + 1/n_b)
    #   SE(std_a  − std_b)      ≈ σ √(1/(2n_a) + 1/(2n_b))   (Gaussian approx)
    #   SE(qτ_a − qτ_b)         = √(τ(1−τ)) / f(F⁻¹(τ)) · √(1/n_a + 1/n_b)
    # We use a 4σ envelope (≈ 1 in 60k under normality), comfortably absorbing
    # mild non-normality of the mixture.
    K = 4.0

    for loc in unique(out_df.location)
        ours   = out_df[out_df.location .== loc, :value]
        theirs = ref[ref.location .== loc, :value]
        σ̂ = std(vcat(ours, theirs))
        nf = sqrt(1/n_r + 1/n_j)

        Δμ  = abs(mean(ours) - mean(theirs))
        SEμ = σ̂ * nf
        @info "lp-sample mean" loc Δμ SE=SEμ ratio=Δμ/SEμ
        @test Δμ < K * SEμ

        Δσ  = abs(std(ours) - std(theirs))
        SEσ = σ̂ * sqrt(0.5 * (1/n_r + 1/n_j))
        @info "lp-sample std" loc Δσ SE=SEσ ratio=Δσ/SEσ
        @test Δσ < K * SEσ

        for τ in (0.1, 0.5, 0.9)
            qa = quantile(ours, τ); qb = quantile(theirs, τ)
            # Plug a Normal at the pooled empirical mean/std into the
            # quantile-density formula — a crude but consistent surrogate
            # for f(F⁻¹(τ)) of the actual mixture.
            d̂ = Normal(mean(vcat(ours, theirs)), σ̂)
            f_at_q = pdf(d̂, dquantile(d̂, τ))
            SEq = sqrt(τ * (1 - τ)) / f_at_q * nf
            Δq  = abs(qa - qb)
            @info "lp-sample quantile" loc τ Δq SE=SEq ratio=Δq/SEq
            @test Δq < K * SEq
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

@testset "Parity — lopensemble::crps_weights" begin
    in_df = CSV.read(joinpath(REF, "crps_input.csv"), DataFrame)
    # `lopensemble` uses `model`, `sample_id`, `predicted`, `observed`, `date`.
    rename!(in_df,
            :model => :model_id,
            :sample_id => :output_type_id,
            :predicted => :value)
    in_df.output_type = fill(:sample, nrow(in_df))
    ft = ForecastTable(in_df[:, [:model_id, :output_type, :output_type_id, :date, :value]];
                       task_id_cols = [:date])
    obs = unique(in_df[:, [:date, :observed]])

    fitted = fit(CRPSStacking(; dirichlet_alpha = 1.001), ft, obs)

    ref = CSV.read(joinpath(REF, "crps_weights_output.csv"), DataFrame)
    rename!(ref, :model => :model_id, :weight => :weight_r)
    j = innerjoin(fitted.weights, ref; on = :model_id)
    @test nrow(j) == nrow(ref)
    # Both estimators are MAP optimisers of the same penalised CRPS
    # objective. Differences come from (a) the optimiser used (Stan's L-BFGS
    # vs Optim.jl's L-BFGS, both with default tolerances) and (b) Stan's
    # internal softplus/simplex parameterisation. A 0.05 absolute tolerance
    # on individual weights is generous; in practice the dominant weight
    # agrees to 2-3 decimal places.
    for r in eachrow(j)
        @info "crps weight" model=r.model_id julia=r.weight stan=r.weight_r diff=abs(r.weight - r.weight_r)
        @test abs(r.weight - r.weight_r) < 0.05
    end
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
