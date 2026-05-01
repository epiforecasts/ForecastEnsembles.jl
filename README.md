# Ensembles.jl

A Julia package for combining probabilistic forecasts from multiple
component models.

`Ensembles.jl` computes weighted or unweighted ensembles of forecasts
represented as quantiles, samples, CDFs, or summary statistics. Weights
can be supplied by the user, fixed (equal weighting), or estimated from
past forecast performance (quantile regression averaging, CRPS-stacking).
Trained and untrained methods compose through a single `EnsembleWeights`
type.

The package builds on prior work in the R packages
[`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean & median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile
regression averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool). It re-implements them in Julia under one in-memory
representation, two verbs (`fit`, `combine`), and two ensemble types
(`MixtureEnsemble`, `QuantileEnsemble`). Multiple dispatch picks the right
algorithm for each `(output_type, method)` pair.

## A small example

```julia
using Ensembles, DataFrames

df = DataFrame(
    location       = "A",
    horizon        = 1,
    model_id       = repeat(["m1", "m2", "m3"], inner = 2),
    output_type    = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value          = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5],
)

ft = ForecastTable(df; task_id_cols = [:location, :horizon])
combine(ft, QuantileEnsemble(:mean))
```

For an end-to-end walkthrough on real flu-hospitalisation data, see
`docs/src/example.md`.
