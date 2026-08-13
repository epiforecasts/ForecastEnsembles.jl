"""
    combine(ft::ForecastTable, m::QuantileEnsemble) -> ForecastTable

Hub-style simple/weighted ensemble. Aggregates `value` across `model_id`
within each (task, output_type, output_type_id) group, using `m.agg`
(`:mean` or `:median`) and optional per-model weights.

Mirrors `hubEnsembles::simple_ensemble`. The output `model_id` is set to
`"hub-ensemble"`, matching the R package default.
"""
function combine(ft::ForecastTable, m::QuantileEnsemble; rng::AbstractRNG = default_rng())
    df = ft.data
    group_cols = vcat([:output_type, :output_type_id], ft.task_id_cols)

    if m.weights === nothing
        agg_fun = m.agg === :mean ? mean : median
        out = DataFrames.combine(
            DataFrames.groupby(df, group_cols),
            :value => agg_fun => :value
        )
    else
        wdf = DataFrame(m.weights)
        # Per-quantile weights are joined on (model_id, output_type_id) so
        # each (task, output_type_id) group gets τ-specific weights;
        # per-model weights are joined on model_id alone (same weight at
        # every output_type_id). Both reduce to a weighted aggregation
        # within `group_cols` afterwards.
        if is_per_quantile(m.weights)
            join_cols = [ft.model_id_col => :model_id, :output_type_id => :output_type_id]
            joined = leftjoin(df, wdf; on = join_cols)
            any(ismissing, joined.weight)::Bool && throw(
                ArgumentError(
                "per-quantile weights are missing some (model_id, output_type_id) pairs",
            ),
            )
        else
            models_present = unique(df[!, ft.model_id_col])
            miss = setdiff(models_present, wdf.model_id)
            isempty(miss) || throw(ArgumentError("no weight provided for models: $miss"))
            extra = setdiff(wdf.model_id, models_present)
            isempty(extra) || @warn("weights provided for models not present in the data " *
                  "(possible typo in model_id): $extra",
                maxlog = 1,)
            joined = leftjoin(df, wdf; on = ft.model_id_col => :model_id)
        end
        agg_fun = m.agg === :mean ? _weighted_mean : _weighted_median
        out = DataFrames.combine(
            DataFrames.groupby(joined, group_cols),
            [:value, :weight] => agg_fun => :value
        )
    end

    out[!, ft.model_id_col] .= "hub-ensemble"
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        out;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

function _weighted_mean(v::AbstractVector, w::AbstractVector)
    sw = sum(w)
    sw > 0 || throw(ArgumentError(
        "ensemble weights sum to $sw; a positive sum is required to combine"))
    return dot(v, w) / sw
end

# Weighted median: smallest x_i such that the cumulative normalised weight
# of values ≤ x_i is ≥ 0.5. Matches matrixStats::weightedMedian default
# (interpolate = FALSE, ties = "weighted").
function _weighted_median(v::AbstractVector, w::AbstractVector)
    perm = sortperm(v)
    vs = v[perm]
    ws = w[perm]
    total = sum(ws)
    cw = 0.0
    @inbounds for i in eachindex(vs)
        cw += ws[i]
        if cw / total >= 0.5
            return vs[i]
        end
    end
    return vs[end]
end
