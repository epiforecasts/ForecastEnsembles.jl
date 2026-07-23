# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Benchmark suite for ForecastEnsembles. Times the headline combination
# operations and weight estimators on a realistic hubverse-shaped
# `model_out_tbl`. The score-driven estimators (Stacking, Hedge, PartialPooling,
# InverseScore) live in the ScoringRules extension and are left out so the
# benchmark environment stays dependency-light; the closed-form CRPSStacking /
# QRA / BLP fits cover the expensive paths. Grouped by first name segment
# ("combine", "fit") so those read as headline suites in the history page.

using BenchmarkTools
using ForecastEnsembles
using DataFrames
using Random

const LEVELS = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

# A quantile `model_out_tbl`: models × locations × horizons at the standard
# hub levels. Values are sorted so each forecast is a valid (monotone) quantile.
function quantile_table(rng; nmodels = 5, nloc = 8, nhoriz = 4)
    rows = DataFrame[]
    for m in 1:nmodels, loc in 1:nloc, h in 1:nhoriz
        vals = sort(100 .+ 12 .* randn(rng, length(LEVELS)))
        push!(rows,
            DataFrame(model_id = "m$m", output_type = "quantile",
                output_type_id = LEVELS, location = "loc$loc", horizon = h, value = vals))
    end
    return ForecastTable(reduce(vcat, rows); task_id_cols = [:location, :horizon])
end

# A sample-typed training set plus observations, for CRPSStacking.
function sample_training(rng; nmodels = 4, ntime = 25, k = 80)
    obs = DataFrame(t = 1:ntime, observed = 100 .+ 5 .* randn(rng, ntime))
    rows = DataFrame[]
    for m in 1:nmodels, t in 1:ntime

        push!(rows,
            DataFrame(model_id = "m$m", output_type = "sample",
                output_type_id = 1:k, t = t,
                value = obs.observed[t] .+ (2 + m) .* randn(rng, k)))
    end
    return ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
end

# A quantile-typed training set plus observations, for QRA and BLP.
function quantile_training(rng; nmodels = 4, ntime = 25)
    obs = DataFrame(t = 1:ntime, observed = 100 .+ 10 .* randn(rng, ntime))
    rows = DataFrame[]
    for m in 1:nmodels, t in 1:ntime

        vals = sort(obs.observed[t] .+ (5 + 2m) .* randn(rng, length(LEVELS)))
        push!(rows,
            DataFrame(model_id = "m$m", output_type = "quantile",
                output_type_id = LEVELS, t = t, value = vals))
    end
    return ForecastTable(reduce(vcat, rows); task_id_cols = [:t]), obs
end

const RNG = MersenneTwister(2024)
const QT = quantile_table(RNG)
const STRAIN, SOBS = sample_training(RNG)
const QTRAIN, QOBS = quantile_training(RNG)

const SUITE = BenchmarkGroup()

SUITE["combine"]["quantile-mean"] = @benchmarkable combine($QT, QuantileEnsemble(:mean))
SUITE["combine"]["quantile-median"] = @benchmarkable combine($QT, QuantileEnsemble(:median))
SUITE["combine"]["mixture"] = @benchmarkable combine($QT, MixtureEnsemble())
SUITE["combine"]["logarithmic"] = @benchmarkable combine($QT, LogarithmicPool())
SUITE["combine"]["trimmed"] = @benchmarkable combine($QT, TrimmedMean(; fraction = 0.1))

SUITE["fit"]["crps-stacking"] = @benchmarkable fit(CRPSStacking(), $STRAIN, $SOBS)
SUITE["fit"]["qra"] = @benchmarkable fit(QRA(), $QTRAIN, $QOBS)
SUITE["fit"]["blp"] = @benchmarkable fit(BLP(), $QTRAIN, $QOBS)
