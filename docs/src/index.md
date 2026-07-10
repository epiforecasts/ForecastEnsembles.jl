# ForecastEnsembles.jl

A Julia package for combining probabilistic forecasts from several component
models.

`ForecastEnsembles.jl` computes weighted or unweighted ensembles of forecasts
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
(`MixtureEnsemble`, `QuantileEnsemble`).

## Installation

`ForecastEnsembles.jl` is not yet registered. Add it from a checkout:

```julia
] add https://github.com/sbfnk/ForecastEnsembles.jl
```

## Quick start

```julia
using ForecastEnsembles, DataFrames

df = DataFrame(
    location       = "A",
    model_id       = repeat(["m1", "m2", "m3"], inner = 2),
    output_type    = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value          = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5],
)
ft = ForecastTable(df; task_id_cols = [:location])

combine(ft, QuantileEnsemble(:mean))                 # Vincentization
combine(ft, MixtureEnsemble(; n_samples = 10_000))   # mixture pool
```

For trained weights:

```julia
fitted = fit(QRA(; enforce_normalisation = true), training_ft, observations)
combine(test_ft, fitted)
```

The [Worked example](example.md) runs every method on a real hubverse flu
hospitalisation slice bundled with the package.

## Why this package?

The three R packages it builds on cover different methods over the same
underlying object: a tidy table of model forecasts indexed by task and
output type. Each duplicates boilerplate (data validation, task grouping,
weight handling) and the methods can't be composed. Weights and ensemble
operations live in separate packages with no shared interface, so even a
trivial substitution like plugging an externally supplied weight vector
into a Vincentization combination ends up needing manual glue.

In Julia this separates into two axes — how the members are combined, and
how the weights are chosen — over a small dispatch surface:

```julia
# axis 1: the combination operation
combine(ft, QuantileEnsemble(:mean))          # per-τ quantile average / median
combine(ft, MixtureEnsemble())                # mixture; routes on output_type

# axis 2: where the weights come from
QuantileEnsemble(:mean; weights = w)          # equal, or a user EnsembleWeights
fit(CRPSStacking(), train, obs)               # or stacked: CRPS (samples)
fit(QRA(enforce_normalisation = true), train, obs)   # or WIS (quantiles)
```

`QRA` and `CRPSStacking` are weight estimators, not separate ensemble
kinds: a fitted method composes through `weights(m)`, which returns an
`EnsembleWeights` whenever the fit is a simplex weight vector, so you pass
it straight into either operation and the conversion happens
automatically. See [Methods](methods.md) for the two-axis story in full.

The optimiser backends are also pluggable: QRA's LP swaps between HiGHS,
GLPK, Gurobi or anything else with a JuMP wrapper in one line, and
CRPS-stacking goes through Optim.jl. `qrensemble` is pinned to GLPK via
`Rglpk`; `lopensemble` is pinned to Stan's MAP optimiser via `cmdstanr`.

## R interface

A thin R wrapper at `r-pkg/ForecastEnsembles/` mirrors the user-facing API of
`hubEnsembles`, `qrensemble`, and `lopensemble`. Numerical work runs in
Julia over a JuliaConnectoR bridge. The package's `pkgdown` site has the
R-side docs.

## See also

- [Methods](methods.md) — what each method does and when to reach for it.
- [Worked example](example.md) — every method run on a real hub dataset.
- [Extending](extending.md) — how to plug in your own ensemble operation
  or weight estimator.
- [Roadmap](roadmap.md) — planned methods and recalibration extensions.
- [API](api.md) — full docstrings for the public types and functions.
