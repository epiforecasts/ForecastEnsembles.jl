# Partial pooling / hierarchical stacking. Instead of one global weight vector
# (complete pooling) or an independent weight vector per stratum (no pooling),
# learn per-stratum weights that shrink toward a shared global vector. In softmax
# logit space each stratum `s` has logits `zₛ`, a global `z₀` anchors them, and a
# penalty `lambda · Σₛ‖zₛ − z₀‖²` controls the shrinkage: `lambda → 0` gives
# independent per-stratum stacking, `lambda → ∞` collapses every stratum onto the
# global weights. The score, and the optimisation against it, come from
# ScoringRules.jl (GPL), so `fit(::PartialPooling, …)` lives in the
# `ForecastEnsemblesScoringRulesExt` extension. This file holds the MIT-clean
# type, the stratum-aware `combine` (each stratum gets its own weights, unseen
# strata fall back to the global vector), and the friendly not-loaded error.

"""
    PartialPooling(score; strata, lambda = 1.0, dirichlet_alpha = 1.0)

Hierarchical (partially pooled) stacking: learn a weight vector per stratum that
shrinks toward a shared global vector, so a data-sparse stratum borrows strength
from the rest.

`fit(PartialPooling(score; strata), training, observations)` jointly optimises,
in softmax space, one logit vector per distinct combination of the `strata`
columns plus a global logit vector, minimising the mean `score` of each
stratum's linearly-pooled forecast plus a shrinkage penalty pulling every
stratum toward the global vector. The [`FittedPartialPooling`](@ref) result
plugs into [`combine`](@ref), which applies each stratum's own weights (an unseen
stratum falls back to the global vector).

`score` is any negatively-oriented weighted-sample rule from
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) (e.g.
`ScoringRules.crps`), which must be loaded — it is a weak dependency. Generalises
[`Stacking`](@ref): a single stratum, or `lambda → ∞`, recovers global stacking.

# Fields

- `score`: the scoring-rule function to minimise (negatively oriented).
- `strata`: task-id columns whose value combinations define the strata (e.g.
  `[:location]`, `[:location, :age_group]`).
- `lambda`: shrinkage strength toward the global vector. `0` fits each stratum
  independently; large values pool them toward one shared vector. Must be `≥ 0`.
- `dirichlet_alpha`: strength of a symmetric-Dirichlet prior on each stratum's
  weights, pulling them toward the simplex centre; `1.0` applies no prior.
"""
struct PartialPooling{F} <: TrainedMethod
    score::F
    strata::Vector{Symbol}
    lambda::Float64
    dirichlet_alpha::Float64
end

function PartialPooling(
        score;
        strata,
        lambda::Real = 1.0,
        dirichlet_alpha::Real = 1.0
)
    lambda >= 0 || throw(ArgumentError("lambda (shrinkage) must be >= 0 (got $lambda)"))
    dirichlet_alpha >= 1 ||
        throw(ArgumentError("dirichlet_alpha must be >= 1 (a proper Dirichlet prior)"))
    st = Symbol.(collect(strata))
    isempty(st) && throw(ArgumentError("strata must name at least one task-id column"))
    return PartialPooling{typeof(score)}(score, st, Float64(lambda), Float64(dirichlet_alpha))
end

"""
    FittedPartialPooling(weights, global_weights, strata, models, score_value)

Output of `fit(::PartialPooling, …)`. Stores the per-stratum `weights` (a
`DataFrame` with the `strata` columns plus `model_id` and `weight`), the pooled
`global_weights` (`model_id`, `weight`) used for strata not seen in training, the
`strata` columns, the component `models`, and the mean `score_value` at the
optimum. Plug into `combine(ft, fitted)` — it applies each stratum's weights,
falling back to `global_weights` for an unseen stratum. `weights(fitted)` returns
the pooled global vector as an [`EnsembleWeights`](@ref).
"""
struct FittedPartialPooling <: UnfittedMethod
    weights::DataFrame
    global_weights::DataFrame
    strata::Vector{Symbol}
    models::Vector{String}
    score_value::Float64
end

function combine(ft::ForecastTable, m::FittedPartialPooling; rng::AbstractRNG = default_rng())
    absent = setdiff(m.strata, ft.task_id_cols)
    isempty(absent) || throw(ArgumentError(
        "combine target is missing strata column(s) $absent; its task-id columns " *
        "are $(ft.task_id_cols)"))

    outs = DataFrame[]
    for g in DataFrames.groupby(ft.data, m.strata)
        key = g[1, m.strata]
        wdf = _stratum_weights(m, key)
        sub = ForecastTable(
            DataFrame(g);
            task_id_cols = ft.task_id_cols,
            model_id_col = ft.model_id_col
        )
        push!(outs, combine(sub, LinearPool(; weights = wdf); rng = rng).data)
    end
    return ForecastTable(
        reduce(vcat, outs);
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

# The weights for one stratum: the matching rows of the per-stratum table, or the
# pooled global vector when the stratum was not seen in training.
function _stratum_weights(m::FittedPartialPooling, key)
    w = m.weights
    mask = trues(nrow(w))
    for s in m.strata
        mask .&= (w[!, s] .== key[s])
    end
    sub = w[mask, [:model_id, :weight]]
    return nrow(sub) == 0 ? m.global_weights : sub
end

# The pooled global vector is the natural single-vector summary.
weights(m::FittedPartialPooling) = EnsembleWeights(m.global_weights)

# Fallback until ScoringRules (which supplies the score) is loaded; the
# extension's concrete method is more specific.
function fit(::PartialPooling, args...)
    throw(ArgumentError("PartialPooling needs ScoringRules.jl for the score. Run " *
                        "`using ScoringRules` (a weak dependency) before fitting."))
end
