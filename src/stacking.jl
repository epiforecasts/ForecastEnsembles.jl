# Generic score-optimal stacking. Learns simplex ensemble weights that minimise
# the mean of an arbitrary proper scoring rule over the training set. The score
# is any callable the caller supplies — the `fit` never references a scoring
# library — so this is plain MIT-core Julia. ScoringRules.jl is the natural
# source of scores (`crps`, `es`, …); load it as a companion and pass e.g.
# `Stacking(ScoringRules.crps)`.

"""
    Stacking(score; dirichlet_alpha = 1.0)

Score-optimal stacking against a user-supplied proper scoring rule.

`fit(Stacking(score), training, observations)` learns simplex ensemble weights
that minimise the mean `score` of the linearly-pooled forecast, where `score` is
any negatively-oriented rule from
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) — e.g.
`ScoringRules.crps`. The `FittedStacking` result plugs into [`combine`](@ref),
or into [`LinearPool`](@ref)/[`QuantileEnsemble`](@ref) via [`weights`](@ref).

`score` is any callable `score(samples, y; w)`;
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) is the natural
companion for it (`using ScoringRules` then `Stacking(ScoringRules.crps)`), but
it is not a dependency of this package. [`CRPSStacking`](@ref) and [`QRA`](@ref)
remain the closed-form specialisations for CRPS and WIS respectively.

!!! note "The `w` in `score(samples, y; w)`"
    The stack is evaluated by pooling every member's samples for a task into one
    vector `samples` and passing per-sample weights `w` that encode the mixture:
    member `i` with weight `wᵢ` and `Kᵢ` samples contributes `wᵢ / Kᵢ` to each of
    its samples, so `w` sums to one over the whole pool. `score` must therefore
    treat `w` as a weighted-sample rule over the combined vector — which is
    exactly what `ScoringRules.crps(samples, y; w)` does. A scorer that ignores
    `w`, or that renormalises it per member, breaks the weighting silently.

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

# Learns simplex weights minimising the mean weighted-sample score. Model `i`'s
# samples in a task each carry weight `w[i] / nᵢ` so the model contributes `w[i]`
# to the mixture; the score (`crps`, `es`, …) must accept a per-sample `w`. The
# score is whatever callable the user supplied — this fit never touches any
# scoring library, so it lives in the MIT core.
function fit(m::Stacking, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample || throw(ArgumentError(
        "Stacking supports :sample forecasts (weighted-sample scores). Use QRA " *
        "for quantile (WIS) stacking."))

    tcols = task_id_cols(training)
    mid = training.model_id_col
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    d = innerjoin(training.data, obs[:, [tcols..., :observed]]; on = tcols)
    isempty(d) && throw(ArgumentError("no overlap between forecasts and observations"))
    models = sort(unique(d[!, mid]))
    M = length(models)
    M >= 2 || throw(ArgumentError("need at least two models to stack (got $M)"))
    idx = Dict(mm => i for (i, mm) in enumerate(models))

    # Per task: pooled samples, the model index of each sample, and y.
    task_data = map(collect(DataFrames.groupby(d, tcols))) do g
        midx = [idx[mm] for mm in g[!, mid]]
        counts = [count(==(i), midx) for i in 1:M]
        (samples = Float64.(g.value), midx = midx, counts = counts,
            y = Float64(first(g.observed)))
    end

    score = m.score
    α = m.dirichlet_alpha
    ntask = length(task_data)

    function loss(z)
        w = _softmax(z)
        total = zero(eltype(z))
        for td in task_data
            sw = [w[i] / td.counts[i] for i in td.midx]
            total += score(td.samples, td.y; w = sw)
        end
        penalty = α > 1 ? -(α - 1) * sum(log, w) : zero(eltype(z))
        return (total + penalty) / ntask
    end

    res = optimize(loss, zeros(M), LBFGS())
    Optim.converged(res) || @warn("Stacking: L-BFGS did not converge " *
          "($(Optim.iterations(res)) iterations); weights are the best iterate " *
          "found. The score may be poorly conditioned — try a different score or " *
          "fewer models.")
    w_hat = _softmax(Optim.minimizer(res))
    return FittedStacking(DataFrame(model_id = models, weight = w_hat),
        String.(models), Optim.minimum(res))
end

"""
    InverseScore(score; temperature = 1.0)

Performance weighting: score each member independently and weight the better
ones more heavily — `wᵢ ∝ exp(−temperature · sᵢ)`, where `sᵢ` is member `i`'s
mean `score` over the training set (negatively oriented, so a lower score earns
more weight). No optimisation, so it is fast and robust with few observations;
but — unlike [`Stacking`](@ref) — it scores each member in isolation and never
sees how they combine, so it is blind to redundancy between them.

`fit(InverseScore(score), training, observations)` returns a
[`FittedInverseScore`](@ref) that plugs into [`combine`](@ref) / [`weights`](@ref).
`score` is called as `score(memberᵢ_samples, y)` — each member is scored on its
own samples in isolation, so (unlike [`Stacking`](@ref)) the `w` keyword is not
used. [`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) is the natural
companion (`InverseScore(ScoringRules.crps)`, which accepts but does not require
`w`), not a dependency of this package.

# Fields

- `score`: the scoring-rule function (negatively oriented).
- `temperature`: sharpness of the softmax over member scores. As it tends to
  `0` the weights approach equal; large values approach winner-take-all. Must
  be positive.
"""
struct InverseScore{F} <: TrainedMethod
    score::F
    temperature::Float64
end

function InverseScore(score; temperature::Real = 1.0)
    temperature > 0 || throw(ArgumentError("temperature must be positive"))
    return InverseScore{typeof(score)}(score, Float64(temperature))
end

"""
    FittedInverseScore(weights, models, scores)

Output of `fit(::InverseScore, …)`. Stores the simplex `weights` (a `DataFrame`
with columns `model_id` and `weight`), the component `models` in weight order,
and the per-member mean `scores` they were derived from. Plug into
`combine(ft, fitted)` — internally a [`LinearPool`](@ref) with these weights.
"""
struct FittedInverseScore <: UnfittedMethod
    weights::DataFrame
    models::Vector{String}
    scores::Vector{Float64}
end

function combine(ft::ForecastTable, m::FittedInverseScore; rng::AbstractRNG = default_rng())
    return combine(ft, LinearPool(; weights = m.weights); rng = rng)
end

weights(m::FittedInverseScore) = EnsembleWeights(m.weights)

# Score each member independently over the training set, then softmax the
# negative mean scores into simplex weights. No optimiser — one scoring pass.
# The score is the user's callable, so this stays in the MIT core.
function fit(m::InverseScore, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample || throw(ArgumentError(
        "InverseScore supports :sample forecasts (weighted-sample scores)."))

    tcols = task_id_cols(training)
    mid = training.model_id_col
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    d = innerjoin(training.data, obs[:, [tcols..., :observed]]; on = tcols)
    isempty(d) && throw(ArgumentError("no overlap between forecasts and observations"))
    models = sort(unique(d[!, mid]))
    score = m.score

    # Per (member, task) score, then the mean over tasks for each member.
    per = combine(DataFrames.groupby(d, [mid, tcols...])) do g
        (; s = score(Float64.(g.value), Float64(first(g.observed))))
    end
    mean_scores = [mean(per[per[!, mid] .== mm, :s]) for mm in models]

    w = _softmax(-m.temperature .* mean_scores)
    return FittedInverseScore(DataFrame(model_id = models, weight = w),
        String.(models), mean_scores)
end
