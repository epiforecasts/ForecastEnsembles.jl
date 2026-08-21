# Weight diagnostics: ensemble-internal summaries of a fitted weight set. These
# need the weights themselves (not the forecasts), so they live here rather than
# in the scoring/evaluation stack. Calibration diagnostics (PIT, coverage) are
# general forecast-evaluation tools and belong upstream in ForecastScoring.jl.
# Dependency-free — no ScoringRules.

"""
    effective_num_models(m) -> Float64 or DataFrame

The effective number of models in a weight vector: the participation ratio
``1 / \\sum_i p_i^2`` of the normalised weights `p`. It is `1` when all weight
sits on one model and equals the model count `M` when the weights are equal, so
it reads as "how many models is this ensemble really using".

`m` may be a fitted method exposing [`weights`](@ref) (e.g.
[`FittedCRPSStacking`](@ref), [`FittedStacking`](@ref), [`FittedHedge`](@ref);
[`FittedPartialPooling`](@ref) uses its pooled global vector), an
[`EnsembleWeights`](@ref), a weights `DataFrame`, or a raw weight vector. For
per-quantile weights it returns a `DataFrame` (`output_type_id`,
`effective_num_models`) — one value per quantile level; otherwise a scalar.

# Example

```@example
using ForecastEnsembles
effective_num_models([0.6, 0.3, 0.1])
```
"""
function effective_num_models(w::AbstractVector{<:Real})
    s = sum(w)
    s > 0 || throw(ArgumentError("weights must sum to a positive value (got $s)"))
    p = w ./ s
    return 1 / sum(abs2, p)
end

function effective_num_models(m)
    ew = _ensemble_weights(m)
    df = DataFrame(ew)
    is_per_quantile(ew) || return effective_num_models(Float64.(df.weight))
    return DataFrames.combine(DataFrames.groupby(df, :output_type_id)) do g
        (; effective_num_models = effective_num_models(Float64.(g.weight)))
    end
end

"""
    weight_stability(m::FittedHedge) -> DataFrame

How much each model's [`Hedge`](@ref) weight moved over the training run: the
total variation ``\\sum_t |w_{t+1} - w_t|`` of every model's weight along the
fitted `trajectory`. A large value flags a model whose weight swung across the
history (regime change or noise); a small one flags a stable contribution.
Returns a `DataFrame` with columns `model_id` and `total_variation`.

Orders each model's rows by the fitted `time_col`. The trajectory may carry
extra columns, but it must hold exactly one weight per model per time step:
grouping is on `model_id` alone, so a second key such as a location would
interleave unrelated weight series.

# Example

```@example
using ForecastEnsembles, DataFrames
trajectory = DataFrame(
    model_id = repeat(["m1", "m2"], inner = 2),
    weight = [0.5, 0.7, 0.5, 0.3],
    t = [1, 2, 1, 2]
)
fitted = FittedHedge(
    DataFrame(model_id = ["m1", "m2"], weight = [0.7, 0.3]), ["m1", "m2"],
    trajectory, :t)
weight_stability(fitted)
```
"""
function weight_stability(m::FittedHedge)
    traj = m.trajectory
    isempty(traj) && return DataFrame(model_id = String[], total_variation = Float64[])
    hasproperty(traj, m.time_col) || throw(ArgumentError(
        "the fitted trajectory has no time column $(m.time_col); " *
        "columns are $(propertynames(traj))"))
    return DataFrames.combine(DataFrames.groupby(traj, :model_id)) do g
        s = sort(g, m.time_col)
        (; total_variation = sum(abs.(diff(Float64.(s.weight)))))
    end
end

# Resolve a diagnostics input to an EnsembleWeights.
_ensemble_weights(w::EnsembleWeights) = w
_ensemble_weights(df::AbstractDataFrame) = EnsembleWeights(df)
function _ensemble_weights(m::EnsembleMethod)
    w = weights(m)
    w === nothing &&
        throw(ArgumentError("$(typeof(m)) does not expose ensemble weights"))
    return w isa EnsembleWeights ? w : EnsembleWeights(w)
end
