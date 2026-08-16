# Online / adaptive weighting via the Hedge (multiplicative-weights,
# exponentiated-gradient) algorithm. Weights are updated sequentially over time:
# each round multiplies a member's weight by `exp(-eta · loss)` and renormalises,
# so members that have scored well recently gain weight without a full refit. The
# per-round loss is any caller-supplied score `score(samples, y; w)` — the `fit`
# references no scoring library, so this is plain MIT-core Julia. ScoringRules.jl
# is the natural companion for the score.

"""
    Hedge(score; eta = nothing, time_col)

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
no optimiser and updates incrementally. `score` is any callable
`score(samples, y; w)`;
[`ScoringRules`](https://github.com/EpiAware/ScoringRules.jl) is the natural
companion (`Hedge(ScoringRules.crps; time_col)`), not a dependency of this package.

# Fields

- `score`: the scoring-rule function (negatively oriented).
- `eta`: learning rate. Larger values adapt faster and concentrate weight more
  aggressively on recent winners; as it tends to `0` the weights stay uniform.
  Defaults to `nothing`, meaning it is chosen automatically at `fit` time from
  the score scale — the regret-style `√(8·ln M / T) / B`, where `M` is the model
  count, `T` the number of update steps and `B` the largest per-round score
  (which assumes the score is non-negative, as a negatively-oriented rule is). This
  keeps the update stable whatever the magnitude of the score (a raw `eta` tuned
  for CRPS in `[0, 1]` would collapse to one model in a single step on counts in
  the hundreds). Pass a positive number to override.
- `time_col`: the task-id column defining the update order.
"""
struct Hedge{F} <: TrainedMethod
    score::F
    eta::Union{Nothing, Float64}
    time_col::Symbol
end

function Hedge(score; eta::Union{Nothing, Real} = nothing, time_col::Symbol)
    eta === nothing || eta > 0 ||
        throw(ArgumentError("eta must be positive (got $eta)"))
    return Hedge{typeof(score)}(
        score, eta === nothing ? nothing : Float64(eta), time_col)
end

# Regret-optimal Hedge step size for losses in [0, B] over T rounds and M
# experts (Cesa-Bianchi & Lugosi): η = √(8 ln M / T) / B. The bound assumes
# non-negative losses, so `B` is the plain maximum (a negatively-oriented score
# like CRPS is always ≥ 0); `maximum(abs, …)` would instead hide a misconfigured
# score that returns negatives by inflating `B` and under-scaling η. Falls back
# to 0 (no updating, weights stay uniform) when every score is zero.
function _auto_eta(scores::AbstractVector, M::Integer, T::Integer)
    B = isempty(scores) ? 0.0 : maximum(scores)
    B > 0 || return 0.0   # T ≥ 1 is guaranteed by the caller
    return sqrt(8 * log(M) / T) / B
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

# Walk the distinct `time_col` values in order; at each step multiply every
# present member's weight by `exp(-eta · sₜ)` and renormalise to the simplex. A
# member absent at a step keeps its weight (sleeping expert). The per-round score
# is the user's callable, so this stays in the MIT core.
function fit(m::Hedge, training::ForecastTable, observations::AbstractDataFrame)
    output_type(training) === :sample || throw(ArgumentError(
        "Hedge supports :sample forecasts (weighted-sample scores)."))
    m.time_col in task_id_cols(training) || throw(ArgumentError(
        "time_col $(m.time_col) must be one of the task-id columns " *
        "$(task_id_cols(training))"))

    tcols = task_id_cols(training)
    mid = training.model_id_col
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    d = innerjoin(training.data, obs[:, [tcols..., :observed]]; on = tcols)
    isempty(d) && throw(ArgumentError("no overlap between forecasts and observations"))
    models = sort(unique(d[!, mid]))
    M = length(models)
    M >= 2 || throw(ArgumentError("need at least two models (got $M)"))
    idx = Dict(mm => i for (i, mm) in enumerate(models))
    score = m.score

    # Score each (member, task) on its own samples and observation, then take the
    # per-(member, time) mean across tasks. Scoring by `[mid, time_col]` alone
    # would pool samples from every non-time task (e.g. all locations at a date)
    # into one vector and score them against a single arbitrary observation.
    per_task = combine(DataFrames.groupby(d, [mid, tcols...])) do g
        (; s = score(Float64.(g.value), Float64(first(g.observed))))
    end
    per = combine(DataFrames.groupby(per_task, [mid, m.time_col])) do g
        (; s = mean(g.s))
    end
    times = sort(unique(per[!, m.time_col]))

    # Auto-scale the learning rate to the score magnitude unless the caller set
    # one. `B` here is a hindsight estimate — the max score over all T rounds, so
    # round 1 is already scaled by scores it has not yet seen. Fine for this batch
    # fit; a true streaming deployment would need `B` bounded a priori.
    eta = m.eta === nothing ? _auto_eta(per.s, M, length(times)) : m.eta

    w = fill(1.0 / M, M)
    traj = [DataFrame() for _ in 1:0]
    for t in times
        rows = per[per[!, m.time_col] .== t, :]
        for r in eachrow(rows)
            i = idx[r[mid]]
            w[i] *= exp(-eta * r.s)
        end
        w ./= sum(w)
        step = DataFrame(model_id = String.(models), weight = copy(w))
        step[!, m.time_col] .= t
        push!(traj, step)
    end

    trajectory = isempty(traj) ? DataFrame() : reduce(vcat, traj)
    return FittedHedge(DataFrame(model_id = String.(models), weight = w),
        String.(models), trajectory)
end
