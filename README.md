# Ensembles.jl

<!-- badges: start -->
[![CI](https://github.com/sbfnk/ensembles.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sbfnk/ensembles.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/sbfnk/ensembles.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sbfnk/ensembles.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- badges: end -->

A Julia package for combining probabilistic forecasts from several component
models.

`Ensembles.jl` computes weighted or unweighted ensembles of forecasts
expressed as quantiles, samples, CDFs, or summary statistics. Weights can be
supplied by the user, fixed (equal weighting), or estimated from past
forecast performance via quantile regression averaging or CRPS-stacking.
Trained and untrained methods are interchangeable through one `EnsembleWeights`
type.

The work builds on three R packages:
[`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean and median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile regression
averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool). The Julia version pulls all three under one in-memory
representation, two verbs (`fit`, `combine`), and two ensemble types
(`MixtureEnsemble`, `QuantileEnsemble`). Multiple dispatch picks the right
algorithm for each `(output_type, method)` pair.

The optimiser backends are pluggable: QRA's LP runs through JuMP, so
HiGHS, GLPK, Gurobi or anything else with a JuMP wrapper is one line
away, and CRPS-stacking goes through Optim.jl.

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

For an end-to-end walkthrough on real flu hospitalisation forecasts, see
`docs/src/example.md`.
