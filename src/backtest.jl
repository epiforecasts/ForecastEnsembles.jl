# Expanding-window backtesting: compare weighting schemes out-of-sample. The
# loop here is the ensemble-specific part — refit each trained scheme on the
# growing training window and apply it to the held-out time. Scoring is the
# caller's choice: a `score_fn` is required, so the loop stays free of any
# particular scoring rule. ScoringRules.jl is a natural source of one (e.g. a
# CRPS-based scorer), but any `(forecast, observations) -> Real` works.

"""
    backtest(ft, observations, schemes; time_col, min_train, score_fn,
             rng = default_rng()) -> DataFrame

Expanding-window backtest of ensemble schemes. The unique values of `time_col`
are ordered; for each test time after the first `min_train`, every scheme is
trained on the earlier times and scored on the test time out-of-sample. Returns
one row per (scheme, test time) with columns `scheme`, the `time_col`, and
`:score` (mean over that time's tasks).

`schemes` maps names to `EnsembleMethod`s (a `Dict` or a vector of
`name => method` pairs):

- a `TrainedMethod` — `CRPSStacking()`, `QRA(...)` — is fitted on the training
  window each fold, then applied to the test time;
- an `UnfittedMethod` — `QuantileEnsemble(:mean)`, `MixtureEnsemble()` — is
  applied directly.

Each scheme must match the table's `output_type`: `CRPSStacking` and sample
combiners need `:sample` data, `QRA` and quantile combiners need `:quantile`.

Scoring is the caller's choice: pass
`score_fn(forecast::ForecastTable, observations) -> Real` (the mean score of one
fold). [`ScoringRules.jl`](https://github.com/EpiAware/ScoringRules.jl) is a
natural source — e.g. a CRPS-based scorer for sample forecasts — but any
function of that shape works.

Aggregate across folds yourself, e.g.

```julia
using DataFrames
res = backtest(ft, obs, schemes; time_col = :target_date, min_train = 8)
combine(groupby(res, :scheme), :score => mean => :mean_score)
```

# Arguments

- `ft`: a [`ForecastTable`](@ref).
- `observations`: a `DataFrame` with the table's task-id columns and an
  `:observed` column.
- `schemes`: a `Dict` or vector of `name => EnsembleMethod` pairs to compare.

# Keyword Arguments

- `time_col`: the column giving the time index to expand the window over.
- `min_train`: number of initial times used only for training. Required — there
  is no default, because too small a training window silently yields degenerate
  weights (a single training time over-determines every estimator). Choose a
  value large enough that the schemes you compare are well-determined.
- `rng`: RNG used by sample-based schemes (default `default_rng()`).
- `score_fn`: a scorer `(forecast, observations) -> Real`, returning the mean
  score over the fold's tasks. Required — there is no default.

# Examples

```@example
using ForecastEnsembles, DataFrames, Random, Statistics
rng = MersenneTwister(1)
T = 12; K = 40
obs = DataFrame(t = 1:T, observed = randn(rng, T))
rows = DataFrame[]
for (mid, s) in (("m1", (y, r) -> y .+ randn(r, K)), ("m2", (y, r) -> 2 .* randn(r, K)))
    for t in 1:T
        push!(rows, DataFrame(model_id = mid, output_type = "sample",
            output_type_id = 1:K, t = t, value = s(obs.observed[t], rng)))
    end
end
ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
schemes = ["equal" => MixtureEnsemble(), "stack" => CRPSStacking()]

# Inject a scorer — here mean absolute error of the ensemble mean; with
# `using ScoringRules` you would instead pass a proper rule such as ScoringRules.crps.
function mae(ens, o)
    d = innerjoin(DataFrame(ens), o; on = :t)
    per = combine(groupby(d, :t),
        [:value, :observed] => ((v, y) -> abs(mean(v) - first(y))) => :e)
    return mean(per.e)
end

backtest(ft, obs, schemes; time_col = :t, min_train = 6, score_fn = mae)
```
"""
function backtest(
        ft::ForecastTable,
        observations::AbstractDataFrame,
        schemes;
        time_col::Symbol,
        min_train::Integer,
        rng::AbstractRNG = default_rng(),
        score_fn = nothing
)
    time_col in ft.task_id_cols || throw(
        ArgumentError(
        "time_col $time_col must be one of the task-id columns $(ft.task_id_cols)",
    ),
    )
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    score_fn === nothing && throw(ArgumentError(
        "backtest needs a scoring function: pass " *
        "`score_fn = (forecast, observations) -> score`. ScoringRules.jl is a " *
        "natural source (e.g. a CRPS-based scorer over the fold's tasks)."))

    times = sort(unique(ft.data[!, time_col]))
    length(times) > min_train || throw(
        ArgumentError(
        "need more than min_train = $min_train distinct $time_col values (got $(length(times)))",
    ),
    )

    scheme_pairs = _scheme_pairs(schemes)
    rows = DataFrame[]
    for i in (min_train + 1):length(times)
        test_t = times[i]
        train_set = Set(times[1:(i - 1)])

        train_ft = _time_subset(ft, time_col, v -> v in train_set)
        test_ft = _time_subset(ft, time_col, v -> v == test_t)
        train_obs = obs[[v in train_set for v in obs[!, time_col]], :]
        test_obs = obs[obs[!, time_col] .== test_t, :]

        for (name, scheme) in scheme_pairs
            ens = _run_scheme(scheme, train_ft, test_ft, train_obs, rng)
            fold_score = score_fn(ens, test_obs)
            r = DataFrame(scheme = [String(name)])
            r[!, time_col] = [test_t]
            r.score = [fold_score]
            push!(rows, r)
        end
    end
    return reduce(vcat, rows)
end

function _time_subset(ft::ForecastTable, time_col::Symbol, keep)
    mask = [keep(v) for v in ft.data[!, time_col]]
    return ForecastTable(
        ft.data[mask, :];
        task_id_cols = ft.task_id_cols,
        model_id_col = ft.model_id_col
    )
end

_scheme_pairs(d::AbstractDict) = collect(d)
_scheme_pairs(v::AbstractVector) = v

function _run_scheme(m::UnfittedMethod, train_ft, test_ft, train_obs, rng)
    return combine(test_ft, m; rng = rng)
end

function _run_scheme(m::TrainedMethod, train_ft, test_ft, train_obs, rng)
    fitted = fit(m, train_ft, train_obs)
    return combine(test_ft, fitted; rng = rng)
end
