using Random: AbstractRNG, default_rng
import Statistics: quantile

"""
    combine(ft::ForecastTable, m::MixtureEnsemble; rng = default_rng()) -> ForecastTable

Linear opinion pool. The kernel is dispatched on the table's `output_type`:

- `:sample`   → weighted resampling of per-model samples to give a single
  pooled sample set per task (the only path that uses `rng`).
- `:cdf`      → weighted pointwise average of CDFs.
- `:quantile` → reconstruct a continuous distribution per model via
  `QuantileDistribution`, then invert the mixture CDF Σᵢ wᵢ Fᵢ exactly by
  bisection at each input level. Deterministic — no Monte Carlo error,
  which matters for the extreme levels (τ = 0.01, 0.99) hubs request.
"""
function combine(ft::ForecastTable, m::MixtureEnsemble; rng::AbstractRNG = default_rng())
    return _linear_pool(ft, Val(output_type(ft)), m, rng)
end

# ---------- :sample ---------------------------------------------------------

function _linear_pool(
        ft::ForecastTable,
        ::Val{:sample},
        m::MixtureEnsemble,
        rng::AbstractRNG
)
    df = ft.data
    weights = _weights_vector(m.weights, ft, df)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(df, ft.task_id_cols)
        # tg has one row per (model, sample). We resample N samples in total,
        # drawing each from a model picked proportionally to its weight.
        models = unique(tg[!, ft.model_id_col])
        ws = [weights[mod] for mod in models]
        ws ./= sum(ws)

        samples_per_model = Dict{eltype(models), Vector{Float64}}()
        for sub in DataFrames.groupby(tg, ft.model_id_col)
            samples_per_model[sub[1, ft.model_id_col]] = Float64.(sub.value)
        end

        N = m.n_samples
        # number of draws to take from each model
        ks = _ints_summing_to(rng, ws, N)
        pooled = Float64[]
        for (mod, k) in zip(models, ks)
            k == 0 && continue
            s = samples_per_model[mod]
            inds = rand(rng, 1:length(s), k)
            append!(pooled, s[inds])
        end

        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, N)
        out.output_type = fill(:sample, N)
        out.output_type_id = collect(1:N)
        out.value = pooled
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

# Multinomial-like rounding: produce nonneg integers k_i summing to N with
# E[k_i] = N * w_i. Uses a simple stochastic rounding so the marginal counts
# are unbiased.
function _ints_summing_to(rng::AbstractRNG, weights::AbstractVector{<:Real}, N::Int)
    targets = N .* weights
    base = floor.(Int, targets)
    frac = targets .- base
    remaining = N - sum(base)
    for _ in 1:remaining
        tot = sum(frac)
        if tot > 0
            # Sample without replacement proportionally to fractional parts.
            r = rand(rng) * tot
            cum = 0.0
            chosen = lastindex(frac)
            for (j, p) in enumerate(frac)
                cum += p
                if r <= cum
                    chosen = j
                    break
                end
            end
            base[chosen] += 1
            frac[chosen] = 0.0
        else
            # Floating-point rounding exhausted the fractional parts before
            # the deficit; top up the largest weight so the counts always
            # sum to exactly N.
            base[argmax(weights)] += 1
        end
    end
    return base
end

# ---------- :cdf ------------------------------------------------------------

function _linear_pool(ft::ForecastTable, ::Val{:cdf}, m::MixtureEnsemble, ::AbstractRNG)
    df = ft.data
    weights = _weights_vector(m.weights, ft, df)
    group_cols = vcat([:output_type, :output_type_id], ft.task_id_cols)

    df = copy(df)
    df.weight = [weights[mod] for mod in df[!, ft.model_id_col]]
    out = DataFrames.combine(
        DataFrames.groupby(df, group_cols),
        [:value, :weight] => ((v, w) -> sum(v .* w) / sum(w)) => :value
    )
    out[!, ft.model_id_col] .= "hub-ensemble"
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        out;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# ---------- :quantile -------------------------------------------------------

function _linear_pool(
        ft::ForecastTable,
        ::Val{:quantile},
        m::MixtureEnsemble,
        ::AbstractRNG
)
    df = ft.data
    weights = _weights_vector(m.weights, ft, df)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(df, ft.task_id_cols)
        levels = sort(unique(tg.output_type_id))
        models = unique(tg[!, ft.model_id_col])
        ws = [weights[mod] for mod in models]
        ws ./= sum(ws)

        # Build a QuantileDistribution per model.
        dists = Vector{QuantileDistribution}(undef, length(models))
        for sub in DataFrames.groupby(tg, ft.model_id_col)
            s = sort(sub, :output_type_id)
            i = findfirst(==(s[1, ft.model_id_col]), models)
            dists[i] = QuantileDistribution(s.output_type_id, s.value)
        end

        # Invert the mixture CDF exactly at each requested level.
        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, length(levels))
        out.output_type = fill(:quantile, length(levels))
        out.output_type_id = levels
        out.value = [_mixture_quantile(dists, ws, τ) for τ in levels]
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

# Exact τ-quantile of the mixture Σᵢ wᵢ Fᵢ by bisection. The root is
# bracketed by the component τ-quantiles: at x = minᵢ Fᵢ⁻¹(τ) every
# component CDF is ≤ τ so the mixture CDF is ≤ τ, and symmetrically at the
# max. ~50 bisection steps reach machine precision on the bracket width.
function _mixture_quantile(
        dists::Vector{QuantileDistribution},
        ws::AbstractVector{<:Real},
        τ::Real
)
    lo = minimum(quantile(d, τ) for d in dists)
    hi = maximum(quantile(d, τ) for d in dists)
    lo == hi && return lo
    mixture_cdf(x) = sum(w * cdf(d, x) for (w, d) in zip(ws, dists))
    for _ in 1:200
        mid = 0.5 * (lo + hi)
        if mixture_cdf(mid) < τ
            lo = mid
        else
            hi = mid
        end
        (hi - lo) <= 1e-12 * max(1.0, abs(lo), abs(hi)) && break
    end
    return 0.5 * (lo + hi)
end

# ---------- weights helper --------------------------------------------------

# Returns a Dict{model_id => weight} for the per-model-weights paths. If
# the user supplied no weights, all present models get equal weight.
# Per-quantile weights are routed elsewhere (see `combine` above) and never
# reach this helper. Validates that every model in `df` has a
# weight.
function _weights_vector(
        weights::Union{Nothing, EnsembleWeights},
        ft::ForecastTable,
        df::AbstractDataFrame
)
    models = unique(df[!, ft.model_id_col])
    if weights === nothing
        return Dict(m => 1.0 / length(models) for m in models)
    end
    wdf = DataFrame(weights)
    miss = setdiff(models, wdf.model_id)
    isempty(miss) || throw(ArgumentError("no weight provided for models: $miss"))
    extra = setdiff(wdf.model_id, models)
    isempty(extra) || @warn("weights provided for models not present in the data " *
          "(possible typo in model_id): $extra",
        maxlog = 1,)
    return Dict(row.model_id => Float64(row.weight) for row in eachrow(wdf))
end

# Fallback for unsupported output types.
function _linear_pool(::ForecastTable, ::Val{T}, ::MixtureEnsemble, ::AbstractRNG) where {T}
    throw(ArgumentError("MixtureEnsemble is not defined for output_type :$T"))
end
