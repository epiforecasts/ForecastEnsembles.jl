# Expanding-window backtesting: compare weighting schemes out-of-sample.
# The scoring is done by the internal `score` (CRPS / WIS); the loop here is
# the ensemble-specific part — refit each trained scheme on the growing
# training window and apply it to the held-out time.

"""
    backtest(ft, observations, schemes; time_col, min_train = 1,
             rng = default_rng()) -> DataFrame

Expanding-window backtest of ensemble schemes. The unique values of
`time_col` are ordered; for each test time after the first `min_train`,
every scheme is trained on the earlier times and scored on the test time
out-of-sample. Returns one row per (scheme, test time) with columns
`scheme`, the `time_col`, and `:score` (mean over that time's tasks).

`schemes` maps names to `EnsembleMethod`s (a `Dict` or a vector of
`name => method` pairs):

- a `TrainedMethod` — `CRPSStacking()`, `QRA(...)` — is fitted on the
  training window each fold, then applied to the test time;
- an `UnfittedMethod` — `QuantileEnsemble(:mean)`, `MixtureEnsemble()` — is
  applied directly.

Each scheme must match the table's `output_type`: `CRPSStacking` and
sample combiners need `:sample` data, `QRA` and quantile combiners need
`:quantile`.

Aggregate across folds yourself, e.g.

```julia
using DataFrames
res = backtest(ft, obs, schemes; time_col = :target_date)
combine(groupby(res, :scheme), :score => mean => :mean_score)
```
"""
function backtest(
    ft::ForecastTable,
    observations::AbstractDataFrame,
    schemes;
    time_col::Symbol,
    min_train::Integer = 1,
    rng::AbstractRNG = default_rng(),
)
    time_col in ft.task_id_cols || throw(
        ArgumentError(
            "time_col $time_col must be one of the task-id columns $(ft.task_id_cols)",
        ),
    )
    obs = DataFrame(observations)
    hasproperty(obs, :observed) ||
        throw(ArgumentError("observations must have an :observed column"))

    times = sort(unique(ft.data[!, time_col]))
    length(times) > min_train || throw(
        ArgumentError(
            "need more than min_train = $min_train distinct $time_col values (got $(length(times)))",
        ),
    )

    scheme_pairs = _scheme_pairs(schemes)
    rows = DataFrame[]
    for i = (min_train+1):length(times)
        test_t = times[i]
        train_set = Set(times[1:(i-1)])

        train_ft = _time_subset(ft, time_col, v -> v in train_set)
        test_ft = _time_subset(ft, time_col, v -> v == test_t)
        train_obs = obs[[v in train_set for v in obs[!, time_col]], :]
        test_obs = obs[obs[!, time_col] .== test_t, :]

        for (name, scheme) in scheme_pairs
            ens = _run_scheme(scheme, train_ft, test_ft, train_obs, rng)
            fold_score = mean(score(ens, test_obs).score)
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
        model_id_col = ft.model_id_col,
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
