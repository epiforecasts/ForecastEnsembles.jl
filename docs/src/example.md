# Worked example: combining three flu hospitalisation forecasts

This page runs every method in the package on a real hubverse forecast
slice bundled with `ForecastEnsembles.jl`. Three models from the
[example-complex-forecast-hub](https://github.com/hubverse-org/example-complex-forecast-hub),
each predicting weekly flu hospitalisations on 2022-12-17 at horizon 1
across five US locations (national plus CA, FL, NY, TX), at the standard
23 quantile levels.

```julia
using ForecastEnsembles, CSV, DataFrames

flu = CSV.read(joinpath(pkgdir(ForecastEnsembles), "data", "flu_forecasts.csv"),
               DataFrame; types = Dict(:output_type_id => Float64,
                                        :location => String))

ft = ForecastTable(flu; task_id_cols = [:reference_date, :target_end_date,
                                         :horizon, :location, :target])
```

`ft` carries 345 rows: 3 models × 5 locations × 23 quantile levels.

## Equal-weight quantile mean (Vincentization)

The simplest combination: at each (location, τ) take the unweighted mean
of the three model quantile values.

```julia
combine(ft, QuantileEnsemble(:mean))
```

Or the median ensemble used as the default by the COVID-19 hub:

```julia
combine(ft, QuantileEnsemble(:median))
```

## Mixture (linear opinion pool)

Treat each model's quantile forecast as defining an approximate continuous
distribution (PCHIP interior, Normal tails), draw samples, pool, and
re-extract quantiles at the original levels:

```julia
combine(ft, MixtureEnsemble(; n_samples = 10_000))
```

This produces a different distribution from Vincentization in general.
Averaging quantile values is not the same operation as averaging the CDFs.

## Hand-supplied weights

Either method takes an `EnsembleWeights`:

```julia
w = EnsembleWeights(DataFrame(
    model_id = ["Flusight-baseline", "MOBS-GLEAM_FLUH", "PSI-DICE"],
    weight   = [0.2, 0.4, 0.4],
))

combine(ft, QuantileEnsemble(:mean; weights = w))
combine(ft, MixtureEnsemble(; weights = w, n_samples = 10_000))
```

## Weights from CRPS-stacking on past forecasts

`fit(CRPSStacking(), training_samples, observations)` returns a
`FittedCRPSStacking` whose `weights(...)` accessor gives the optimised
per-model weight vector. Plug it directly into either ensemble:

```julia
fitted = fit(CRPSStacking(), training_ft, training_obs)

combine(ft, MixtureEnsemble(; weights = fitted))
combine(ft, QuantileEnsemble(:mean; weights = fitted))
```

(Set up `training_ft` from a sample-shaped slice of past forecasts and
`training_obs` from the corresponding observations. The QRA section below
shows a concrete training shape.)

## Weights from QRA on past forecasts

QRA fits a quantile regression of past observations on past per-model
forecasts. Two relevant configurations:

- *Joint* (`per_quantile_weights = false`): one weight vector across all
  τ. `weights(::FittedQRA)` returns a per-model `EnsembleWeights`, usable
  by `MixtureEnsemble` or `QuantileEnsemble`.
- *Per-τ* (`per_quantile_weights = true`): a different weight vector at
  each τ. `weights(::FittedQRA)` returns a per-quantile
  `EnsembleWeights`, usable by `QuantileEnsemble`.

```julia
fitted = fit(
    QRA(; per_quantile_weights = true,
          enforce_normalisation = true,
          intercept = false),
    training_ft, training_obs,
)
combine(ft, QuantileEnsemble(:mean; weights = fitted))
```

For configurations that do not reduce to a clean weight vector — fits
with an intercept, unconstrained fits, fits across multiple task groups —
`weights(::FittedQRA)` returns `nothing`. Calling
`MixtureEnsemble(weights = fitted)` or
`QuantileEnsemble(weights = fitted)` with such a fit raises at
construction. You can still call `combine(ft, fitted)` directly: that
applies the fitted regression coefficients and produces predicted
quantiles.

## What's where in the data

```julia
using DataFrames

flu_summary = combine(
    DataFrames.groupby(flu, [:model_id, :location]),
    nrow => :n_quantiles,
)
```

Three rows per location per model, 23 quantile levels each. Total: 345
rows.
