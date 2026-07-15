# Generic score-optimal stacking. Learns simplex ensemble weights that minimise
# the mean of an arbitrary proper scoring rule over the training set. The score,
# and the automatic differentiation that optimises against it, come from
# ScoringRules.jl — a GPL package — so `fit(::Stacking, …)` lives in the
# `ForecastEnsemblesScoringRulesExt` extension. This file holds the MIT-clean
# types, the fitted-result plumbing (identical to CRPSStacking: simplex weights
# applied through a LinearPool), and the friendly error raised when the
# extension is not loaded.

"""
    Stacking(score; dirichlet_alpha = 1.0)

Score-optimal stacking against a user-supplied proper scoring rule.

`fit(Stacking(score), training, observations)` learns simplex ensemble weights
that minimise the mean `score` of the linearly-pooled forecast, where `score` is
any negatively-oriented rule from
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) — e.g.
`ScoringRules.crps`. The `FittedStacking` result plugs into [`combine`](@ref),
or into [`LinearPool`](@ref)/[`QuantileEnsemble`](@ref) via [`weights`](@ref).

`ScoringRules` must be loaded (`using ScoringRules`) for `fit` to work. It is a
weak dependency, so the MIT core does not pull in its GPL code unless you opt in.
[`CRPSStacking`](@ref) and [`QRA`](@ref) remain the dependency-free closed-form
specialisations for CRPS and WIS respectively.

# Fields

- `score`: the scoring-rule function to minimise (negatively oriented).
- `dirichlet_alpha`: strength of a symmetric-Dirichlet prior on the weights,
  pulling them toward the simplex centre; `1.0` applies no prior.
"""
struct Stacking{F} <: TrainedMethod
    score::F
    dirichlet_alpha::Float64
end

function Stacking(score; dirichlet_alpha::Real = 1.0)
    dirichlet_alpha >= 1 ||
        throw(ArgumentError("dirichlet_alpha must be >= 1 (a proper Dirichlet prior)"))
    return Stacking{typeof(score)}(score, Float64(dirichlet_alpha))
end

"""
    FittedStacking(weights, models, score_value)

Output of `fit(::Stacking, …)`. Stores the simplex ensemble `weights` (a
`DataFrame` with columns `model_id` and `weight`), the component `models` in
weight order, and the mean `score_value` achieved at the optimum. Plug into
`combine(ft, fitted)` — internally a [`LinearPool`](@ref) with these weights.
"""
struct FittedStacking <: UnfittedMethod
    weights::DataFrame
    models::Vector{String}
    score_value::Float64
end

function combine(ft::ForecastTable, m::FittedStacking; rng::AbstractRNG = default_rng())
    return combine(ft, LinearPool(; weights = m.weights); rng = rng)
end

weights(m::FittedStacking) = EnsembleWeights(m.weights)

# Fallback when ScoringRules (which supplies the score and its gradient) is not
# loaded. The extension defines the concrete
# `fit(::Stacking, ::ForecastTable, ::AbstractDataFrame)`, which is more specific
# than this varargs method and takes precedence once `using ScoringRules` runs.
function fit(::Stacking, args...)
    throw(ArgumentError("Stacking needs ScoringRules.jl for the score and its " *
                        "gradient. Run `using ScoringRules` (a weak dependency) " *
                        "before fitting, or use CRPSStacking / QRA for the " *
                        "closed-form CRPS / WIS cases."))
end
