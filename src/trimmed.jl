# Robust cross-model aggregation: at each (task, output_type_id) drop (trim) or
# clamp (winsorise) the most extreme member values before averaging, so a few
# outlier submissions cannot drag the ensemble. A pure operation with no weight
# learning; dependency-free. Comparable-across-models values only (:quantile,
# :cdf) — sample indices are not aligned across models.

"""
    TrimmedMean(; fraction = 0.1, mode = :trim)

Robust ensemble mean. At each task and `output_type_id` (e.g. a quantile level
τ), order the per-model values and either **trim** — drop the lowest and highest
`fraction` of them, then average the rest — or **winsorise** — clamp those
extremes to the surviving boundary values, then average all of them.

`k = round(fraction · n)` models are trimmed/clamped from each end, capped so at
least one value always survives (so `fraction → 0.5` degenerates to the median,
`fraction = 0` to the plain mean). Because `k` is a rounded count, small
ensembles need a large enough `fraction` to trim anything at all: with the
default `fraction = 0.1`, `round(0.1 · n) = 0` for `n ≤ 4` and rises to 1 only at
`n = 5`, so for a typical hub ensemble of a handful of models the result equals
the plain mean until `fraction` is raised. This is the robust cousin of
[`QuantileEnsemble`](@ref)`(:mean)`: cheaper than a full median ensemble to
reason about, and tunable in how much of the tail it discards. It aggregates
values that are comparable across models at a shared `output_type_id`, so it
supports `:quantile` and `:cdf` forecasts, not `:sample` (sample indices are not
aligned across models — use [`MixtureEnsemble`](@ref) there).

# Fields

- `fraction`: proportion trimmed/clamped from each end, in `[0, 0.5)`.
- `mode`: `:trim` (drop the extremes) or `:winsorise` (clamp them).

# Example

```@example
using ForecastEnsembles, DataFrames
df = DataFrame(
    model_id = string.("m", 1:5),
    output_type = "quantile",
    output_type_id = 0.5,
    location = "A",
    value = [1.0, 2.0, 3.0, 4.0, 100.0]
)
ft = ForecastTable(df; task_id_cols = [:location])
combine(ft, TrimmedMean(; fraction = 0.2))
```
"""
struct TrimmedMean <: UnfittedMethod
    fraction::Float64
    mode::Symbol
end

function TrimmedMean(; fraction::Real = 0.1, mode::Symbol = :trim)
    0 <= fraction < 0.5 ||
        throw(ArgumentError("fraction must be in [0, 0.5) (got $fraction)"))
    mode in (:trim, :winsorise) ||
        throw(ArgumentError("mode must be :trim or :winsorise (got :$mode)"))
    return TrimmedMean(Float64(fraction), mode)
end

"""
    combine(ft::ForecastTable, m::TrimmedMean; rng = default_rng()) -> ForecastTable

Apply a trimmed or winsorised cross-model mean at each (task, `output_type_id`).
See [`TrimmedMean`](@ref). The output `model_id` is `"hub-ensemble"`.
"""
function combine(ft::ForecastTable, m::TrimmedMean; rng::AbstractRNG = default_rng())
    ot = output_type(ft)
    ot in (:quantile, :cdf) || throw(ArgumentError(
        "TrimmedMean aggregates comparable per-(task, output_type_id) values " *
        "across models; it supports :quantile and :cdf, not :$ot (sample indices " *
        "are not aligned across models — use MixtureEnsemble)."))

    df = ft.data
    group_cols = vcat([:output_type, :output_type_id], ft.task_id_cols)
    frac = m.fraction
    mode = m.mode
    out = DataFrames.combine(
        DataFrames.groupby(df, group_cols),
        :value => (v -> _trimmed_aggregate(Float64.(v), frac, mode)) => :value
    )
    out[!, ft.model_id_col] .= "hub-ensemble"
    select!(out, ft.model_id_col, :output_type, :output_type_id, ft.task_id_cols..., :value)
    return ForecastTable(
        out;
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# Trimmed or winsorised mean of one group's values. `k` is capped at
# `(n - 1) ÷ 2` so at least one value always survives.
function _trimmed_aggregate(v::AbstractVector{<:Real}, fraction::Real, mode::Symbol)
    n = length(v)
    n == 0 && throw(ArgumentError("no values to aggregate"))
    k = min(round(Int, fraction * n), (n - 1) ÷ 2)
    vs = sort(v)
    mode === :trim && return mean(@view vs[(k + 1):(n - k)])
    lo, hi = vs[k + 1], vs[n - k]
    return mean(clamp(x, lo, hi) for x in vs)
end
