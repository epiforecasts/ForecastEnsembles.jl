# Ensembles.jl

A Julia package for combining probabilistic forecasts.

`Ensembles.jl` covers the methods previously split across three R
packages —
[`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean & median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile
regression averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool) — under one in-memory representation and two verbs
(`fit`, `combine`). Multiple dispatch picks the right algorithm for each
`(output_type, method)` pair.

## Installation

`Ensembles.jl` is not yet registered. Add it from a checkout:

```julia
] add https://github.com/sbfnk/ensembles.jl
```

## Quick start

```julia
using Ensembles, DataFrames

df = DataFrame(
    location       = "A",
    model_id       = repeat(["m1", "m2", "m3"], inner = 2),
    output_type    = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value          = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5],
)

ft = ForecastTable(df; task_id_cols = [:location])

# Hub-style mean
combine(ft, SimpleEnsemble(:mean))

# Linear opinion pool
combine(ft, LinearPool(; n_samples = 10_000))
```

For the trained methods (QRA, CRPS-stacking), use `fit` first:

```julia
fitted = fit(QRA(; enforce_normalisation = true), training_ft, observations)
combine(test_ft, fitted)
```

## Why this package?

The three R packages cover different methods on the same underlying object:
a tidy table of model forecasts indexed by task and output type. Each
package re-implements the boilerplate (data validation, task grouping,
weight handling) and the methods don't compose. You can't take a
CRPS-stacked weight vector and feed it to a quantile-input linear pool,
even though that's a sensible thing to want.

In Julia these collapse into a small dispatch surface:

```julia
combine(ft, m::SimpleEnsemble)               # always works
combine(ft, m::LinearPool)                   # routes by output_type
fit(::QRA, training, observations)           # → FittedQRA
combine(ft, ::FittedQRA)
fit(::CRPSStacking, training, observations)  # → FittedCRPSStacking
combine(ft, ::FittedCRPSStacking)            # delegates to LinearPool
```

A fitted method is itself an `UnfittedMethod`, and a `weights(m)` accessor
returns the per-model (or per-quantile) weights as a `DataFrame` whenever
they make sense. Other methods that consume weights then accept a fitted
method directly. See [Methods](methods.md) for what that buys you.

## R interface

A thin R wrapper at `r-pkg/ensembles/` mirrors the user-facing API of the
three R packages while delegating all numerical work to `Ensembles.jl` over
a JuliaConnectoR bridge. The package's `pkgdown` site has the R-side docs.

## See also

- [Methods](methods.md) — the algorithmic story behind each method.
- [API](api.md) — full docstrings for the public types and functions.
