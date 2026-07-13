# Internal scoring. These are the few proper scores the package needs for
# its own workflow (weight estimation reuses the same quantities; the
# backtesting harness scores out-of-sample with them). They are deliberately
# minimal: when `ScoringRules.jl` lands, `score` delegates to it and this
# file shrinks to an adapter.

"""
    ScoringRule

A proper scoring rule used to evaluate a forecast against an observation.
Concrete rules: [`CRPS`](@ref) (sample forecasts), [`WIS`](@ref) (quantile
forecasts). Lower is better throughout.
"""
abstract type ScoringRule end

"""
    CRPS()

Continuous Ranked Probability Score, estimated from a forecast's samples.
"""
struct CRPS <: ScoringRule end

"""
    WIS()

Weighted Interval Score of a quantile forecast, computed as twice the mean
pinball loss over the submitted quantile levels (the quantile-based form
hubs use; it approximates the CRPS).
"""
struct WIS <: ScoringRule end

"""
    default_rule(output_type::Symbol) -> ScoringRule

The scoring rule that matches a forecast `output_type`: `CRPS` for
`:sample`, `WIS` for `:quantile`.
"""
function default_rule(ot::Symbol)
    ot === :sample && return CRPS()
    ot === :quantile && return WIS()
    throw(ArgumentError("no default scoring rule for output_type :$ot"))
end

# CRPS of an empirical sample forecast against observation `y`, via the
# estimator CRPS = E|X − y| − ½ E|X − X′|. The second expectation uses the
# unbiased average over distinct sample pairs (the k = l terms are zero and
# would bias it downward).
function _crps_sample(samples::AbstractVector{<:Real}, y::Real)
    n = length(samples)
    n == 0 && return NaN
    term1 = mean(abs.(samples .- y))
    n == 1 && return term1
    pair_sum = 0.0
    @inbounds for i = 1:n, j = (i+1):n
        pair_sum += abs(samples[i] - samples[j])
    end
    term2 = 2 * pair_sum / (n * (n - 1))
    return term1 - 0.5 * term2
end

# WIS of a quantile forecast: 2 × mean pinball loss over the levels.
# Pinball ρ_τ(y, q) = (y − q)(τ − 1{y < q}) ≥ 0.
function _wis_quantile(
    levels::AbstractVector{<:Real},
    values::AbstractVector{<:Real},
    y::Real,
)
    isempty(levels) && return NaN
    s = 0.0
    @inbounds for k in eachindex(levels)
        τ = levels[k]
        q = values[k]
        s += (y - q) * (τ - (y < q ? 1.0 : 0.0))
    end
    return 2 * s / length(levels)
end

_score_one(g::AbstractDataFrame, y::Real, ::CRPS) = _crps_sample(Float64.(g.value), y)

function _score_one(g::AbstractDataFrame, y::Real, ::WIS)
    s = sort(g, :output_type_id)
    return _wis_quantile(Float64.(s.output_type_id), Float64.(s.value), y)
end

"""
    score(ft::ForecastTable, observations; rule = default_rule(output_type(ft)))
        -> DataFrame

Score each forecast in `ft` against `observations` (the task-id columns
plus an `:observed` column), one row per (model, task). Columns:
`model_id`, the task-id columns, and `:score`.

`rule` defaults to the scoring rule matching the table's `output_type`
(`CRPS` for samples, `WIS` for quantiles) and can be set explicitly.
"""
function score(
    ft::ForecastTable,
    observations::AbstractDataFrame;
    rule::ScoringRule = default_rule(output_type(ft)),
)
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))
    join_cols = ft.task_id_cols
    df = innerjoin(ft.data, obs[:, [join_cols..., :observed]]; on = join_cols)
    isempty(df) && throw(ArgumentError("no overlap between forecasts and observations"))

    key_cols = vcat([ft.model_id_col], join_cols)
    rows = DataFrame[]
    for g in DataFrames.groupby(df, key_cols)
        row = DataFrame(g[1:1, key_cols])
        row.score = [_score_one(g, first(g.observed), rule)]
        push!(rows, row)
    end
    return reduce(vcat, rows)
end

"""
    mean_score(ft::ForecastTable, observations; rule = default_rule(output_type(ft)))
        -> DataFrame

Mean [`score`](@ref) per model, averaged over tasks. Columns: `model_id`,
`:score`.
"""
function mean_score(ft::ForecastTable, observations::AbstractDataFrame; kwargs...)
    per = score(ft, observations; kwargs...)
    return DataFrames.combine(
        DataFrames.groupby(per, ft.model_id_col),
        :score => mean => :score,
    )
end
