# Online / adaptive weighting via the Hedge (multiplicative-weights,
# exponentiated-gradient) algorithm. Weights are updated sequentially over time:
# each round multiplies a member's weight by `exp(-eta · loss)` and renormalises,
# so members that have scored well recently gain weight without a full refit. The
# per-round loss is a proper score from ScoringRules.jl (a GPL package), so
# `fit(::Hedge, …)` lives in the `ForecastEnsemblesScoringRulesExt` extension.
# This file holds the MIT-clean type, its fitted-result plumbing (final simplex
# weights applied through a LinearPool, plus the weight trajectory), and the
# friendly error raised when the extension is not loaded.

"""
    Hedge(score; eta = 1.0, time_col)

Online ensemble weighting by the Hedge / exponentiated-gradient rule.

`fit(Hedge(score; time_col), training, observations)` walks the distinct
`time_col` values in order and, at each step, multiplies every member's weight by
`exp(-eta · sₜ)` — where `sₜ` is that member's mean `score` on the current step
(negatively oriented, so a lower score keeps more weight) — then renormalises to
the simplex. A member absent at a step keeps its weight (a "sleeping expert").
The final weights plug into [`combine`](@ref); the full trajectory is kept for
weight-stability diagnostics.

Unlike [`InverseScore`](@ref) (one pooled score per member) this adapts to *when*
members did well, so it tracks regime change; unlike [`Stacking`](@ref) it needs
no optimiser and updates incrementally. `score` is any negatively-oriented
weighted-sample rule from
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) (e.g.
`ScoringRules.crps`), which must be loaded — it is a weak dependency.

# Fields

- `score`: the scoring-rule function (negatively oriented).
- `eta`: learning rate. Larger values adapt faster and concentrate weight more
  aggressively on recent winners; as it tends to `0` the weights stay uniform.
  Scale it to the magnitude of your score. Must be positive.
- `time_col`: the task-id column defining the update order.
"""
struct Hedge{F} <: TrainedMethod
    score::F
    eta::Float64
    time_col::Symbol
end

function Hedge(score; eta::Real = 1.0, time_col::Symbol)
    eta > 0 || throw(ArgumentError("eta must be positive (got $eta)"))
    return Hedge{typeof(score)}(score, Float64(eta), time_col)
end

"""
    FittedHedge(weights, models, trajectory)

Output of `fit(::Hedge, …)`. Stores the final simplex `weights` (a `DataFrame`
with columns `model_id` and `weight`), the component `models` in weight order,
and the `trajectory` — a long `DataFrame` (`time_col`, `model_id`, `weight`) of
the weights after each update, for diagnosing weight stability over time. Plug
into `combine(ft, fitted)` — internally a [`LinearPool`](@ref) with the final
weights.
"""
struct FittedHedge <: UnfittedMethod
    weights::DataFrame
    models::Vector{String}
    trajectory::DataFrame
end

function combine(ft::ForecastTable, m::FittedHedge; rng::AbstractRNG = default_rng())
    return combine(ft, LinearPool(; weights = m.weights); rng = rng)
end

weights(m::FittedHedge) = EnsembleWeights(m.weights)

# Fallback until ScoringRules (which supplies the per-round score) is loaded; the
# extension's concrete method is more specific.
function fit(::Hedge, args...)
    throw(ArgumentError("Hedge needs ScoringRules.jl for the per-round score. Run " *
                        "`using ScoringRules` (a weak dependency) before fitting."))
end
