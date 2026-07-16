# Beta-transformed linear pool (Gneiting & Ranjan 2013). The linear opinion pool
# is underdispersed relative to what calibration requires; the BLP fixes this by
# mapping the mixture CDF through a fitted Beta CDF,
#
#     F_BLP(x) = B_{α,β}( Σᵢ wᵢ Fᵢ(x) ),
#
# with the plain linear pool the special case α = β = 1. Two facts make this
# dependency-free (no proper-scoring-rule library):
#
#   • Fitting. The BLP log-likelihood splits into a term in (α, β) plus a term
#     that does not depend on them, so maximum-likelihood α, β are just the Beta
#     fitted to the linear pool's PIT values uₜ = Σᵢ wᵢ Fᵢ(yₜ) at the training
#     observations.
#   • Applying. F_BLP(x) = τ ⇔ Σᵢ wᵢ Fᵢ(x) = B⁻¹_{α,β}(τ), so the BLP τ-quantile
#     is the linear pool's quantile at the remapped level B⁻¹_{α,β}(τ) — reusing
#     the exact mixture-quantile inversion already in linear_pool.jl.

"""
    BLP(; weights = nothing)

Beta-transformed linear pool (Gneiting & Ranjan 2013): a recalibrated mixture
that corrects the linear opinion pool's underdispersion by passing the mixture
CDF through a fitted Beta CDF.

`fit(BLP(), training, observations)` fits the Beta to the linear pool's PIT
values on the training set (maximum likelihood); `combine(ft, fitted)` applies
it by evaluating the linear pool's quantile function at Beta-remapped levels.
`weights` are the underlying per-model pool weights (equal by default, or any
per-model [`EnsembleWeights`](@ref) / fitted method); the Beta parameters are
learned on top. Quantile forecasts only.

Because the fit is a Beta on PIT values rather than a full weight regression,
BLP is a *recalibration* of the pool: [`weights`](@ref) returns `nothing` (the
level remap is not expressible as model weights), so apply it with
`combine(ft, fitted)`.

# Fields

- `weights`: per-model pool weights, or `nothing` for equal weights.
"""
struct BLP <: TrainedMethod
    weights::Union{Nothing, EnsembleWeights}
end

function BLP(; weights = nothing)
    w = _resolve_weights(weights)
    if w !== nothing && is_per_quantile(w)
        throw(ArgumentError("BLP takes per-model pool weights only; per-quantile " *
                            "weights are not meaningful for a mixture."))
    end
    return BLP(w)
end

"""
    FittedBLP(alpha, beta, weights)

Output of `fit(::BLP, …)`. Stores the fitted Beta shape parameters `alpha` and
`beta` and the underlying pool `weights`. `alpha = beta = 1` means the linear
pool was already calibrated (no transform). Apply with `combine(ft, fitted)`;
[`weights`](@ref) returns `nothing`.
"""
struct FittedBLP <: UnfittedMethod
    alpha::Float64
    beta::Float64
    weights::Union{Nothing, EnsembleWeights}
end

function fit(m::BLP, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :quantile ||
        throw(ArgumentError("BLP currently supports :quantile forecasts."))
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    tcols = training.task_id_cols
    df = innerjoin(training.data, obs[:, [tcols..., :observed]]; on = tcols)
    isempty(df) && throw(ArgumentError("no overlap between training and observations"))
    wdict = _weights_vector(m.weights, training, df)

    # PIT of the linear pool at each training observation: u = Σᵢ wᵢ Fᵢ(y).
    us = Float64[]
    for tg in DataFrames.groupby(df, tcols)
        y = Float64(first(tg.observed))
        u = 0.0
        wsum = 0.0
        for sub in DataFrames.groupby(tg, training.model_id_col)
            w = wdict[first(sub[!, training.model_id_col])]
            s = sort(sub, :output_type_id)
            d = QuantileDistribution(Float64.(s.output_type_id), Float64.(s.value))
            u += w * cdf(d, y)
            wsum += w
        end
        push!(us, clamp(u / wsum, 1e-6, 1 - 1e-6))
    end

    a, b = _fit_beta(us)
    return FittedBLP(a, b, m.weights)
end

function combine(ft::ForecastTable, m::FittedBLP; rng::AbstractRNG = default_rng())
    output_type(ft) === :quantile ||
        throw(ArgumentError("BLP combine supports :quantile forecasts."))
    B = Beta(m.alpha, m.beta)
    wdict = _weights_vector(m.weights, ft, ft.data)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(ft.data, ft.task_id_cols)
        levels = sort(unique(tg.output_type_id))
        models = unique(tg[!, ft.model_id_col])
        ws = [wdict[mod] for mod in models]
        ws ./= sum(ws)

        dists = Vector{QuantileDistribution}(undef, length(models))
        for sub in DataFrames.groupby(tg, ft.model_id_col)
            i = findfirst(==(first(sub[!, ft.model_id_col])), models)
            s = sort(sub, :output_type_id)
            dists[i] = QuantileDistribution(Float64.(s.output_type_id), Float64.(s.value))
        end

        # BLP τ-quantile = linear-pool quantile at the Beta-remapped level.
        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, length(levels))
        out.output_type = fill(:quantile, length(levels))
        out.output_type_id = levels
        out.value = [_mixture_quantile(dists, ws, quantile(B, Float64(τ))) for τ in levels]
        out[!, ft.model_id_col] .= "hub-ensemble"
        push!(out_groups, out)
    end
    out = reduce(vcat, out_groups)
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        out;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# BLP is a recalibration, not a weight vector — no `weights` interpretation.
weights(::FittedBLP) = nothing

# Maximum-likelihood Beta fit of the PIT values, with a method-of-moments
# fallback (and the identity Beta(1,1) when the moments are degenerate).
function _fit_beta(u::AbstractVector{<:Real})
    length(u) >= 2 ||
        throw(ArgumentError("BLP needs at least two observations to fit the Beta"))
    try
        return params(fit_mle(Beta, u))
    catch
        m = Statistics.mean(u)
        v = Statistics.var(u)
        (v <= 0 || v >= m * (1 - m)) && return (1.0, 1.0)
        c = m * (1 - m) / v - 1
        return (m * c, (1 - m) * c)
    end
end
