# ScoringRules-backed functionality for ForecastEnsembles. Loaded automatically
# when a user runs `using ScoringRules` alongside `using ForecastEnsembles`.
# Keeps ScoringRules' GPL code out of the MIT core: this extension supplies the
# default `backtest` scorer and the generic `fit(::Stacking, …)` that optimises
# ensemble weights against any weighted-sample scoring rule.
module ForecastEnsemblesScoringRulesExt

using ForecastEnsembles: ForecastEnsembles, ForecastTable, Stacking, FittedStacking,
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
        (; s = mean(quantile_score(Float64.(s.output_type_id), Float64.(s.value),
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

end # module
