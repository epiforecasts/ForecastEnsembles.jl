# ScoringRules-backed functionality for ForecastEnsembles. Loaded automatically
# when a user runs `using ScoringRules` alongside `using ForecastEnsembles`.
# Keeps ScoringRules' GPL code out of the MIT core: this extension supplies the
# default `backtest` scorer and the generic `fit(::Stacking, …)` that optimises
# ensemble weights against any weighted-sample scoring rule.
module ForecastEnsemblesScoringRulesExt

using ForecastEnsembles: ForecastEnsembles, ForecastTable, Stacking, FittedStacking,
                         InverseScore, FittedInverseScore, Hedge, FittedHedge,
                         output_type, task_id_cols
using ScoringRules: crps, quantile_score
using DataFrames: DataFrame, AbstractDataFrame, innerjoin, groupby, combine, sort
using Optim: optimize, LBFGS, minimizer, minimum
using Statistics: mean

import ForecastEnsembles: _default_score_fn, fit

_softmax(z) = (e = exp.(z .- maximum(z)); e ./ sum(e))

# ---- default backtest scorer -------------------------------------------------
# `_default_score_fn(ft)` returns a scorer `(forecast, observations) -> Real`
# matching the table's output type: CRPS for samples, mean quantile score
# (the WIS kernel) for quantiles. The mean is over the fold's tasks.

function _default_score_fn(ft::ForecastTable)
    ot = output_type(ft)
    ot === :sample && return _sample_scorer
    ot === :quantile && return _quantile_scorer
    throw(ArgumentError("no ScoringRules default scorer for output_type :$ot"))
end

function _sample_scorer(forecast::ForecastTable, obs::AbstractDataFrame)
    tcols = task_id_cols(forecast)
    d = innerjoin(forecast.data, obs[:, [tcols..., :observed]]; on = tcols)
    per = combine(groupby(d, tcols),
        [:value, :observed] => ((v, y) -> crps(Float64.(v), Float64(first(y)))) => :s)
    return mean(per.s)
end

function _quantile_scorer(forecast::ForecastTable, obs::AbstractDataFrame)
    tcols = task_id_cols(forecast)
    d = innerjoin(forecast.data, obs[:, [tcols..., :observed]]; on = tcols)
    per = combine(groupby(d, tcols)) do g
        s = sort(g, :output_type_id)
        (;
            s = mean(quantile_score(Float64.(s.output_type_id), Float64.(s.value),
            Float64(first(s.observed)))))
    end
    return mean(per.s)
end

# ---- generic score-optimal stacking -----------------------------------------
# Learns simplex weights minimising the mean weighted-sample score. Model `i`'s
# samples in a task each carry weight `w[i] / nᵢ` so the model contributes `w[i]`
# to the mixture; the score (`crps`, `es`, …) must accept a per-sample `w`.

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
    task_data = map(collect(groupby(d, tcols))) do g
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
    w_hat = _softmax(minimizer(res))
    return FittedStacking(DataFrame(model_id = models, weight = w_hat),
        String.(models), minimum(res))
end

# ---- inverse-score (performance) weighting ----------------------------------
# Score each member independently over the training set, then softmax the
# negative mean scores into simplex weights. No optimiser — one scoring pass.

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
    per = combine(groupby(d, [mid, tcols...])) do g
        (; s = score(Float64.(g.value), Float64(first(g.observed))))
    end
    mean_scores = [mean(per[per[!, mid] .== mm, :s]) for mm in models]

    w = _softmax(-m.temperature .* mean_scores)
    return FittedInverseScore(DataFrame(model_id = models, weight = w),
        String.(models), mean_scores)
end

# ---- online / Hedge (exponentiated-gradient) weighting ----------------------
# Walk the distinct `time_col` values in order; at each step multiply every
# present member's weight by `exp(-eta · sₜ)` and renormalise to the simplex. A
# member absent at a step keeps its weight (sleeping expert). Returns the final
# weights plus the full trajectory for diagnostics.

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

    # Per (member, time) mean score.
    per = combine(groupby(d, [mid, m.time_col])) do g
        (; s = score(Float64.(g.value), Float64(first(g.observed))))
    end
    times = sort(unique(per[!, m.time_col]))

    w = fill(1.0 / M, M)
    traj = [DataFrame() for _ in 1:0]
    for t in times
        rows = per[per[!, m.time_col] .== t, :]
        for r in eachrow(rows)
            i = idx[r[mid]]
            w[i] *= exp(-m.eta * r.s)
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

end # module
