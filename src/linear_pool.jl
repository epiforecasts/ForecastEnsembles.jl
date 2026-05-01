using Random: AbstractRNG, default_rng
import Statistics: quantile

"""
    combine(ft::ForecastTable, m::MixtureEnsemble; rng = default_rng()) -> ForecastTable

Linear opinion pool. The kernel is dispatched on the table's `output_type`:

- `:sample`   → weighted resampling of per-model samples to give a single
  pooled sample set per task.
- `:cdf`      → weighted pointwise average of CDFs.
- `:quantile` → reconstruct a continuous distribution per model via
  `QuantileDistribution`, draw `m.n_samples` per task (split across models
  proportionally to their weights), then re-extract quantiles at the input
  levels.
"""
function combine(ft::ForecastTable, m::MixtureEnsemble; rng::AbstractRNG = default_rng())
    return _linear_pool(ft, Val(output_type(ft)), m, rng)
end

# ---------- :sample ---------------------------------------------------------

function _linear_pool(ft::ForecastTable, ::Val{:sample}, m::MixtureEnsemble, rng::AbstractRNG)
    df = ft.data
    weights = _weights_vector(m.weights, ft, df)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(df, ft.task_id_cols)
        # tg has one row per (model, sample). We resample N samples in total,
        # drawing each from a model picked proportionally to its weight.
        models = unique(tg[!, ft.model_id_col])
        ws = [weights[mod] for mod in models]
        ws ./= sum(ws)

        samples_per_model = Dict{Any,Vector{Float64}}()
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
    return ForecastTable(out;
                        task_id_cols = ft.task_id_cols,
                        model_id_col = ft.model_id_col)
end

# Multinomial-like rounding: produce nonneg integers k_i summing to N with
# E[k_i] = N * w_i. Uses a simple stochastic rounding so the marginal counts
# are unbiased.
function _ints_summing_to(rng::AbstractRNG, weights::AbstractVector{<:Real}, N::Int)
    targets = N .* weights
    base = floor.(Int, targets)
    frac = targets .- base
    remaining = N - sum(base)
    if remaining > 0
        # Sample without replacement proportionally to fractional parts.
        idx = collect(1:length(weights))
        for _ in 1:remaining
            tot = sum(frac)
            tot <= 0 && break
            r = rand(rng) * tot
            cum = 0.0
            chosen = lastindex(idx)
            for (j, p) in enumerate(frac)
                cum += p
                if r <= cum
                    chosen = j
                    break
                end
            end
            base[chosen] += 1
            frac[chosen] = 0.0
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
        [:value, :weight] => ((v, w) -> sum(v .* w) / sum(w)) => :value,
    )
    out[!, ft.model_id_col] .= "hub-ensemble"
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(out;
                        task_id_cols = ft.task_id_cols,
                        model_id_col = ft.model_id_col)
end

# ---------- :quantile -------------------------------------------------------

function _linear_pool(ft::ForecastTable, ::Val{:quantile}, m::MixtureEnsemble, rng::AbstractRNG)
    df = ft.data
    weights = _weights_vector(m.weights, ft, df)

    out_groups = DataFrame[]
    for tg in DataFrames.groupby(df, ft.task_id_cols)
        levels = sort(unique(tg.output_type_id))
        models = unique(tg[!, ft.model_id_col])
        ws = [weights[mod] for mod in models]
        ws ./= sum(ws)

        # Build a QuantileDistribution per model.
        qdists = Dict{Any,QuantileDistribution}()
        for sub in DataFrames.groupby(tg, ft.model_id_col)
            s = sort(sub, :output_type_id)
            qdists[s[1, ft.model_id_col]] = QuantileDistribution(s.output_type_id, s.value)
        end

        # Draw samples per model in proportion to weights.
        ks = _ints_summing_to(rng, ws, m.n_samples)
        pooled = Float64[]
        for (mod, k) in zip(models, ks)
            k == 0 && continue
            append!(pooled, rand(rng, qdists[mod], k))
        end

        # Re-extract quantiles at the requested levels.
        out = DataFrame(tg[1:1, ft.task_id_cols])
        out = repeat(out, length(levels))
        out.output_type = fill(:quantile, length(levels))
        out.output_type_id = levels
        out.value = [quantile(pooled, τ) for τ in levels]
        out[!, ft.model_id_col] .= "hub-ensemble"
        push!(out_groups, out)
    end
    out = reduce(vcat, out_groups)
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(out;
                        task_id_cols = ft.task_id_cols,
                        model_id_col = ft.model_id_col)
end

# ---------- weights helper --------------------------------------------------

# Returns a Dict{model_id => weight} for the per-model-weights paths. If
# the user supplied no weights, all present models get equal weight.
# Per-quantile weights are routed elsewhere (see `combine` above) and never
# reach this helper. Validates that every model in `df` has a
# weight.
function _weights_vector(weights::Union{Nothing,EnsembleWeights}, ft::ForecastTable, df::AbstractDataFrame)
    models = unique(df[!, ft.model_id_col])
    if weights === nothing
        return Dict(m => 1.0 / length(models) for m in models)
    end
    wdf = DataFrame(weights)
    miss = setdiff(models, wdf.model_id)
    isempty(miss) || throw(ArgumentError("no weight provided for models: $miss"))
    return Dict(row.model_id => Float64(row.weight) for row in eachrow(wdf))
end

# Fallback for unsupported output types.
_linear_pool(::ForecastTable, ::Val{T}, ::MixtureEnsemble, ::AbstractRNG) where {T} =
    throw(ArgumentError("MixtureEnsemble is not defined for output_type :$T"))
