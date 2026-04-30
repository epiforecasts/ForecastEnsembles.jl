"""
    EnsembleMethod

Top of the method-type hierarchy. Subtypes split into

- [`UnfittedMethod`](@ref) — can be passed to [`combine`](@ref) directly.
- [`TrainedMethod`](@ref) — must be passed through [`fit`](@ref) first to
  obtain a fitted counterpart, which is itself an `UnfittedMethod` and can
  then be passed to `combine`.
"""
abstract type EnsembleMethod end
abstract type UnfittedMethod <: EnsembleMethod end
abstract type TrainedMethod <: EnsembleMethod end

"""
    SimpleEnsemble(agg = :mean; weights = nothing)

Hub-style simple or weighted ensemble. `agg` is `:mean` or `:median`.
`weights` is either `nothing` (equal weights) or a `DataFrame` with columns
`model_id` and `weight`.
"""
struct SimpleEnsemble <: UnfittedMethod
    agg::Symbol
    weights::Union{Nothing,DataFrame}
end

function SimpleEnsemble(agg::Symbol = :mean; weights = nothing)
    agg in (:mean, :median) ||
        throw(ArgumentError("SimpleEnsemble agg must be :mean or :median (got :$agg)"))
    weights = _resolve_weights(weights)
    return SimpleEnsemble(agg, weights)
end

"""
    LinearPool(; weights = nothing, n_samples = 10_000)

Linear opinion pool: the ensemble distribution is a (weighted) mixture of the
component distributions. The path through the algorithm depends on the
forecast `output_type`:

- `:sample`   — weighted resample from per-model samples.
- `:cdf`      — pointwise weighted average of CDFs.
- `:quantile` — reconstruct each model's CDF from its quantiles, draw
  `n_samples`, pool, and re-extract quantiles at the original levels.
"""
struct LinearPool <: UnfittedMethod
    weights::Union{Nothing,DataFrame}
    n_samples::Int
end

function LinearPool(; weights = nothing, n_samples::Integer = 10_000)
    weights = _resolve_weights(weights)
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    return LinearPool(weights, Int(n_samples))
end

# Coerce a `weights` argument into the canonical
# `DataFrame{:model_id, :weight}` form. Accepts:
#   - `nothing`               (passthrough)
#   - any DataFrame-like with the two required columns
#   - an `EnsembleMethod` whose `weights(m)` returns such a frame
function _resolve_weights(w::Nothing)
    return nothing
end
function _resolve_weights(w::EnsembleMethod)
    wf = weights(w)
    wf === nothing && throw(ArgumentError(
        "method $(typeof(w)) does not expose a per-model weight vector " *
        "(see `weights(::$(typeof(w)))` for the conditions)."))
    return _resolve_weights(wf)
end
function _resolve_weights(w)
    df = DataFrame(w)
    all(c -> c in propertynames(df), (:model_id, :weight)) ||
        throw(ArgumentError("weights frame must have :model_id and :weight columns"))
    return df
end

"""
    QRA(; per_quantile_weights = false, intercept = true,
          enforce_normalisation = false, noncross = false,
          group = Symbol[])

Quantile Regression Averaging. `group` lists task dimensions over which a
separate regression is fitted. Mirrors `qrensemble::qra`.
"""
struct QRA <: TrainedMethod
    per_quantile_weights::Bool
    intercept::Bool
    enforce_normalisation::Bool
    noncross::Bool
    group::Vector{Symbol}
end

function QRA(;
    per_quantile_weights::Bool = false,
    intercept::Bool = true,
    enforce_normalisation::Bool = false,
    noncross::Bool = false,
    group = Symbol[],
)
    return QRA(per_quantile_weights, intercept, enforce_normalisation, noncross,
               Symbol.(collect(group)))
end

"""
    CRPSStacking(; dirichlet_alpha = 1.001, lambda = nothing, gamma = nothing)

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`.
"""
struct CRPSStacking <: TrainedMethod
    dirichlet_alpha::Float64
    lambda::Union{Nothing,Float64}
    gamma::Union{Nothing,Float64}
end

function CRPSStacking(;
    dirichlet_alpha::Real = 1.001,
    lambda::Union{Nothing,Real} = nothing,
    gamma::Union{Nothing,Real} = nothing,
)
    return CRPSStacking(
        Float64(dirichlet_alpha),
        lambda === nothing ? nothing : Float64(lambda),
        gamma === nothing ? nothing : Float64(gamma),
    )
end

# `fit` is `StatsBase.fit` (imported in src/Ensembles.jl). Method
# definitions live with each TrainedMethod (see src/qra.jl,
# src/crps_stacking.jl).

"""
    weights(m) -> Union{DataFrame, Nothing}

Per-model weights estimated by a fitted method, as a `DataFrame` with
columns `:model_id` and `:weight`. Returns `nothing` when the fit does not
correspond to a single weight vector on the simplex (e.g. unconstrained
QRA, per-quantile QRA, QRA with a non-zero intercept).

When `weights(m) !== nothing`, `m` can be passed in place of an explicit
weights frame to any method that accepts one — for example
`LinearPool(weights = m)` or `SimpleEnsemble(:mean; weights = m)`. This is
the composition path between trained and untrained methods.
"""
function weights end

# Default: no weights interpretation (e.g. an unconstrained or per-quantile
# QRA fit, or any future method without a single per-model weight vector).
weights(::EnsembleMethod) = nothing
