# Ensembles.jl

A Julia package for combining probabilistic forecasts from multiple
component models.

`Ensembles.jl` computes weighted or unweighted ensembles of forecasts
represented as quantiles, samples, CDFs, or summary statistics. Weights
can be supplied by the user, fixed (equal weighting), or estimated from
past forecast performance (quantile regression averaging, CRPS-stacking).
Trained and untrained methods compose through a single `EnsembleWeights`
type.

The package builds on prior work in
[`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean & median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile
regression averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool). The Julia rewrite consolidates these under one
in-memory representation, two verbs (`fit`, `combine`), and two ensemble
types (`MixtureEnsemble`, `QuantileEnsemble`).

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

combine(ft, QuantileEnsemble(:mean))                 # Vincentization
combine(ft, MixtureEnsemble(; n_samples = 10_000))   # mixture pool
```

For trained weights:

```julia
fitted = fit(QRA(; enforce_normalisation = true), training_ft, observations)
combine(test_ft, fitted)                             # apply the fit directly
combine(test_ft, QuantileEnsemble(:mean; weights = fitted))   # or via QuantileEnsemble
```

The [Worked example](example.md) walks through every method on a real
hubverse flu hospitalisation slice bundled with the package.

## Why this package?

The three R packages it builds on cover different methods on the same
underlying object: a tidy table of model forecasts indexed by task and
output type. Each duplicates boilerplate (data validation, task grouping,
weight handling) and the methods don't compose. You can't take a
CRPS-stacked weight vector and feed it to a Vincentization-style
combination, even though that's a sensible thing to want.

In Julia these collapse into a small dispatch surface:

```julia
combine(ft, m::QuantileEnsemble)              # per-τ aggregation
combine(ft, m::MixtureEnsemble)               # routes on output_type
fit(::QRA, training, observations)            # → FittedQRA
combine(ft, ::FittedQRA)
fit(::CRPSStacking, training, observations)   # → FittedCRPSStacking
combine(ft, ::FittedCRPSStacking)
```

Trained and untrained methods compose through `weights(m)`, which returns
an `EnsembleWeights` whenever the fit reduces to a per-model or per-τ
weight vector. Pass any fitted method straight to `MixtureEnsemble` or
`QuantileEnsemble` and the conversion is automatic. See [Methods](methods.md)
for the algorithmic story.

## R interface

A thin R wrapper at `r-pkg/ensembles/` mirrors the user-facing API of
`hubEnsembles`, `qrensemble`, and `lopensemble` while delegating all
numerical work to `Ensembles.jl` over a JuliaConnectoR bridge. The
package's `pkgdown` site has the R-side docs.

## See also

- [Methods](methods.md) — the algorithmic story behind each method.
- [Worked example](example.md) — every method run on a real hub dataset.
- [API](api.md) — full docstrings for the public types and functions.
