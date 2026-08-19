"""
    EnsembleMethod

Top of the method-type hierarchy. Subtypes split into

- [`UnfittedMethod`](@ref) — can be passed to [`combine`](@ref) directly.
- [`TrainedMethod`](@ref) — must be passed through [`fit`](@ref) first to
  obtain a fitted counterpart, which is itself an `UnfittedMethod` and can
  then be passed to `combine`.
"""
abstract type EnsembleMethod end

"""
    UnfittedMethod

An ensemble method applied directly to forecasts with no training, for example
[`QuantileEnsemble(:mean)`](@ref) or [`MixtureEnsemble()`](@ref). It carries any
fixed configuration (aggregation rule, supplied weights) and can be passed
straight to [`combine`](@ref) without a prior call to [`fit`](@ref).
"""
abstract type UnfittedMethod <: EnsembleMethod end

"""
    TrainedMethod

An ensemble method whose weights or coefficients are learned from past
performance before use, for example [`QRA`](@ref) or [`CRPSStacking`](@ref).
Passing one through [`fit`](@ref) returns a fitted object (itself an
[`UnfittedMethod`](@ref)) that can then be passed to [`combine`](@ref).
"""
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
    weights::Union{Nothing, EnsembleWeights}
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
    weights::Union{Nothing, EnsembleWeights}
    n_samples::Int
end

function MixtureEnsemble(; weights = nothing, n_samples::Integer = 10_000)
    w = _resolve_weights(weights)
    if w !== nothing && is_per_quantile(w)
        throw(
            ArgumentError(
            "MixtureEnsemble takes per-model weights only; per-quantile " *
            "weights belong to QuantileEnsemble.",
        ),
        )
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
    wf === nothing && throw(
        ArgumentError(
        "method $(typeof(w)) does not expose ensemble weights " *
        "(see `weights(::$(typeof(w)))` for the conditions).",
    ),
    )
    return _resolve_weights(wf)
end
_resolve_weights(w) = EnsembleWeights(w)

# Backward-compatible predicate used by the LinearPool dispatch.
is_per_quantile_weights(::Nothing) = false
is_per_quantile_weights(w::EnsembleWeights) = is_per_quantile(w)

"""
    QRA(; per_quantile_weights = false, intercept = false,
          enforce_normalisation = true, noncross = false,
          group = Symbol[])

Quantile Regression Averaging. `group` lists task dimensions over which a
separate regression is fitted. Mirrors `qrensemble::qra`.

The defaults fit a simplex-constrained, intercept-free combination
(`enforce_normalisation = true`, `intercept = false`), matching `qrensemble` and
guaranteeing non-crossing quantiles for the shared-weight fit. Set
`intercept = true` and/or `enforce_normalisation = false` for an unconstrained
regression, but note that combination can then produce weights outside `[0, 1]`
and crossing quantiles. `noncross` only takes effect with
`per_quantile_weights = true` (the shared-weight fit is already monotone in τ).

The R wrapper `qra()` defaults `noncross = TRUE`, mirroring `qrensemble::qra`,
where this constructor defaults it to `false`. Results agree either way, since
the flag is inert unless `per_quantile_weights = true`, which both default to
off. All other defaults match.
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
        intercept::Bool = false,
        enforce_normalisation::Bool = true,
        noncross::Bool = false,
        group = Symbol[]
)
    if noncross && !per_quantile_weights
        # `maxlog = 1` warns once per session so a sweep of no-op configs does not
        # flood the log.
        @warn "QRA(noncross = true) has no effect unless per_quantile_weights = true; " *
              "the shared-weight fit is already monotone in τ." maxlog = 1
    end
    return QRA(
        per_quantile_weights,
        intercept,
        enforce_normalisation,
        noncross,
        Symbol.(group)
    )
end

"""
    CRPSStacking(; dirichlet_alpha = 1.001,
                   lambda = nothing, time_col = nothing,
                   task_weights = nothing)

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`,
including its time weighting.

By default every training task contributes equally to the objective. Two
ways to change that:

- `task_weights`: a `DataFrame` with the training table's task-id columns
  plus a `:weight` column — one non-negative weight per task. The general
  mechanism; covers recency, per-region weighting (`lopensemble`'s
  `gamma`), down-weighting anomalous reporting weeks, and so on.
- `lambda` with `time_col`: convenience for recency weighting. `time_col`
  names the task column that orders tasks in time; `lambda` is one of
  - a scalar `φ ∈ (0, 1]`: exponential decay, weight `φ^(T − t)` for the
    t-th of T ordered unique time values (the common forecasting-
    literature choice; `φ = 1` recovers equal weights),
  - `:lopensemble`: the quadratic ramp `2 − (1 − t/T)²` that
    `lopensemble::crps_weights` uses by default (oldest ≈ 1, newest 2),
  - a `Vector{Float64}` with one weight per ordered unique time value
    (lopensemble's vector form),
  - a function of the normalised time rank `t/T ∈ (0, 1]` returning a
    weight.

`lambda` and `task_weights` are mutually exclusive. The Dirichlet prior
strength scales with the effective sample size `(Σλ)²/Σλ²` rather than
the raw task count, so heavy down-weighting of history does not quietly
strengthen the prior relative to the data.
"""
struct CRPSStacking <: TrainedMethod
    dirichlet_alpha::Float64
    lambda::Union{Nothing, Symbol, Function, Float64, Vector{Float64}}
    time_col::Union{Nothing, Symbol}
    task_weights::Union{Nothing, DataFrame}
end

function CRPSStacking(;
        dirichlet_alpha::Real = 1.001,
        lambda = nothing,
        time_col::Union{Nothing, Symbol} = nothing,
        task_weights = nothing,
        gamma = nothing
)
    gamma === nothing || throw(
        ArgumentError(
        "`gamma` (lopensemble's region weighting) has been replaced by the " *
        "more general `task_weights`; supply a frame with the task-id " *
        "columns plus :weight.",
    ),
    )
    if lambda !== nothing && task_weights !== nothing
        throw(ArgumentError("specify either `lambda` or `task_weights`, not both"))
    end
    if lambda !== nothing
        time_col === nothing && throw(
            ArgumentError(
            "`lambda` requires `time_col`: the task column that orders " *
            "tasks in time, e.g. time_col = :target_date.",
        ),
        )
        if lambda isa Real && !(lambda isa Bool)
            0 < lambda <= 1 || throw(
                ArgumentError(
                "scalar `lambda` is an exponential decay factor and must " *
                "lie in (0, 1]",
            ),
            )
            lambda = Float64(lambda)
        elseif lambda isa Symbol
            lambda in (:lopensemble, :equal) ||
                throw(ArgumentError("symbol `lambda` must be :lopensemble or :equal"))
        elseif lambda isa AbstractVector
            lambda = Float64.(collect(lambda))
            all(>=(0), lambda) ||
                throw(ArgumentError("`lambda` weights must be non-negative"))
        elseif !(lambda isa Function)
            throw(
                ArgumentError(
                "`lambda` must be a scalar in (0,1], :lopensemble, :equal, " *
                "a vector, or a function of the normalised time rank",
            ),
            )
        end
    end
    if task_weights !== nothing
        task_weights = DataFrame(task_weights)
        :weight in propertynames(task_weights) ||
            throw(ArgumentError("`task_weights` must have a :weight column"))
        all(w -> !ismissing(w) && w >= 0, task_weights.weight) ||
            throw(ArgumentError("`task_weights` must be non-negative and non-missing"))
    end
    return CRPSStacking(Float64(dirichlet_alpha), lambda, time_col, task_weights)
end

# `fit` is `StatsBase.fit` (imported in src/ForecastEnsembles.jl). Method
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

# Arguments
- `m`: a fitted method (such as [`FittedCRPSStacking`](@ref) or
  [`FittedQRA`](@ref)) from which per-model weights are extracted.

# Examples
```@example
using ForecastEnsembles, DataFrames, Random
rng = MersenneTwister(1)
T = 20; K = 50
obs = DataFrame(t = 1:T, observed = randn(rng, T))
rows = DataFrame[]
for (mid, s) in (("m1", (y, r) -> y .+ randn(r, K)), ("m2", (y, r) -> 3 .* randn(r, K)))
    for t in 1:T
        push!(rows, DataFrame(model_id = mid, output_type = "sample",
            output_type_id = 1:K, t = t, value = s(obs.observed[t], rng)))
    end
end
ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
weights(fit(CRPSStacking(), ft, obs))
```
"""
function weights end

# Default: no weights interpretation (e.g. an unconstrained or per-quantile
# QRA fit, or any future method without a single per-model weight vector).
weights(::EnsembleMethod) = nothing
