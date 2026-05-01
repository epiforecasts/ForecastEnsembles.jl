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
    QuantileEnsemble(agg = :mean; weights = nothing)

Per-quantile weighted aggregation of quantile forecasts. At each task and
quantile level τ, take a weighted mean (`agg = :mean`, also called
Vincentization) or weighted median (`agg = :median`) of the per-model
quantile values. `weights` may be:

- `nothing` — equal weights (the "simple ensemble" of the hubverse).
- a per-model `EnsembleWeights` — same weights at every τ.
- a per-quantile `EnsembleWeights` — different weights per τ (e.g. from a
  per-`τ` QRA fit, or supplied externally).
- any fitted method whose `weights(m)` returns one of the above
  (`FittedCRPSStacking`, `FittedQRA` in the right configuration, etc.).
"""
struct QuantileEnsemble <: UnfittedMethod
    agg::Symbol
    weights::Union{Nothing,EnsembleWeights}
end

function QuantileEnsemble(agg::Symbol = :mean; weights = nothing)
    agg in (:mean, :median) ||
        throw(ArgumentError("QuantileEnsemble agg must be :mean or :median (got :$agg)"))
    return QuantileEnsemble(agg, _resolve_weights(weights))
end


"""
    MixtureEnsemble(; weights = nothing, n_samples = 10_000)

Mixture (linear-opinion-pool) ensemble: the ensemble distribution is the
(weighted) mixture of the component distributions, F = Σᵢ wᵢ Fᵢ. The
algorithm path depends on the forecast `output_type`:

- `:sample`   — weighted resample from per-model samples.
- `:cdf`      — pointwise weighted average of CDFs.
- `:quantile` — reconstruct each model's CDF from its quantiles, draw
  `n_samples`, pool, and re-extract quantiles at the original levels.

Mixture pooling is fundamentally a per-model operation; per-quantile
weights aren't meaningful here (use `QuantileEnsemble` for that).
"""
struct MixtureEnsemble <: UnfittedMethod
    weights::Union{Nothing,EnsembleWeights}
    n_samples::Int
end

function MixtureEnsemble(; weights = nothing, n_samples::Integer = 10_000)
    w = _resolve_weights(weights)
    if w !== nothing && is_per_quantile(w)
        throw(ArgumentError(
            "MixtureEnsemble takes per-model weights only; per-quantile " *
            "weights belong to QuantileEnsemble."))
    end
    n_samples > 0 || throw(ArgumentError("n_samples must be positive"))
    return MixtureEnsemble(w, Int(n_samples))
end

# Earlier name for what is now MixtureEnsemble; kept for source
# compatibility.
const LinearPool = MixtureEnsemble

# Coerce a `weights` argument into the canonical
# `DataFrame{:model_id, :weight}` form. Accepts:
#   - `nothing`               (passthrough)
#   - any DataFrame-like with the two required columns
#   - an `EnsembleMethod` whose `weights(m)` returns such a frame
_resolve_weights(::Nothing) = nothing
_resolve_weights(w::EnsembleWeights) = w
function _resolve_weights(w::EnsembleMethod)
    wf = weights(w)
    wf === nothing && throw(ArgumentError(
        "method $(typeof(w)) does not expose ensemble weights " *
        "(see `weights(::$(typeof(w)))` for the conditions)."))
    return _resolve_weights(wf)
end
_resolve_weights(w) = EnsembleWeights(w)

# Backward-compatible predicate used by the LinearPool dispatch.
is_per_quantile_weights(::Nothing) = false
is_per_quantile_weights(w::EnsembleWeights) = is_per_quantile(w)

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
`MixtureEnsemble(weights = m)` or `QuantileEnsemble(:mean; weights = m)`. This is
the composition path between trained and untrained methods.
"""
function weights end

# Default: no weights interpretation (e.g. an unconstrained or per-quantile
# QRA fit, or any future method without a single per-model weight vector).
weights(::EnsembleMethod) = nothing
