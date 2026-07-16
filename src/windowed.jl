# Trailing-window training: fit any `TrainedMethod` on only the most recent
# `window` time points, so a scheme adapts to recent data — a rolling window in
# place of the expanding window `backtest` grows by default. A pure fit-time
# wrapper: it subsets the training data by time and delegates to the inner
# method, whose fitted result (and its `combine`) is returned unchanged. No
# dependency of its own — it composes with `CRPSStacking`, `QRA`, `Stacking`,
# `InverseScore`, …

"""
    Windowed(method, window; time_col)

Wrap a [`TrainedMethod`](@ref) so it trains on only the most recent `window`
values of `time_col`.

`fit(Windowed(method, window; time_col), training, observations)` keeps the last
`window` distinct `time_col` values of `training` (and the matching
observations), fits `method` on that subset, and returns `method`'s own fitted
result — so it drops straight into [`combine`](@ref).

Useful for a rolling-window scheme in [`backtest`](@ref): compare an
expanding-window `CRPSStacking()` against a rolling
`Windowed(CRPSStacking(), 8; time_col = :date)`.

# Fields

- `method`: the inner `TrainedMethod` to fit on the window.
- `window`: number of most-recent `time_col` values to train on (must be `≥ 1`).
- `time_col`: the task-id column defining time order.
"""
struct Windowed{M <: TrainedMethod} <: TrainedMethod
    method::M
    window::Int
    time_col::Symbol
end

function Windowed(method::TrainedMethod, window::Integer; time_col::Symbol)
    window >= 1 || throw(ArgumentError("window must be >= 1 (got $window)"))
    return Windowed{typeof(method)}(method, Int(window), time_col)
end

function fit(w::Windowed, training::ForecastTable, observations::AbstractDataFrame)
    w.time_col in training.task_id_cols || throw(ArgumentError(
        "time_col $(w.time_col) must be one of the task-id columns " *
        "$(training.task_id_cols)"))
    times = sort(unique(training.data[!, w.time_col]))
    keep = Set(times[max(1, length(times) - w.window + 1):end])
    sub_ft = _time_subset(training, w.time_col, v -> v in keep)
    obs = DataFrame(observations)
    sub_obs = obs[[v in keep for v in obs[!, w.time_col]], :]
    return fit(w.method, sub_ft, sub_obs)
end
