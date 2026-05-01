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
combine(test_ft, fitted)
```

The [Worked example](example.md) walks through every method on a real
hubverse flu hospitalisation slice bundled with the package.

## Why this package?

The three R packages it builds on cover different methods on the same
underlying object: a tidy table of model forecasts indexed by task and
output type. Each duplicates boilerplate (data validation, task grouping,
weight handling) and the methods can't be composed — weights and ensemble
operations live in separate packages with no shared interface, so even
a basic substitution (e.g. plugging an externally supplied weight vector
into a Vincentization combination) takes manual glue.

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

Two further advantages of the Julia rewrite, both unmeasured so far but
worth noting:

- *Single-language inner loops.* The CDF reconstruction (PCHIP),
  sampling, weighted aggregation and CRPS evaluation all run as compiled
  Julia. The R packages drop into C / Fortran / Stan via different
  bridges per method.
- *Pluggable optimiser backends.* QRA's LP runs through JuMP, which can
  dispatch to HiGHS, GLPK, Gurobi, COSMO, or any other LP solver with a
  one-line change. CRPS-stacking goes through Optim.jl, which can be
  swapped for NLopt or anything else following the standard Julia
  optimisation interface. `qrensemble` is pinned to GLPK via `Rglpk`;
  `lopensemble` is pinned to Stan's MAP optimiser via `cmdstanr`.

## R interface

A thin R wrapper at `r-pkg/ensembles/` mirrors the user-facing API of
`hubEnsembles`, `qrensemble`, and `lopensemble` while delegating all
numerical work to `Ensembles.jl` over a JuliaConnectoR bridge. The
package's `pkgdown` site has the R-side docs.

## See also

- [Methods](methods.md) — the algorithmic story behind each method.
- [Worked example](example.md) — every method run on a real hub dataset.
- [API](api.md) — full docstrings for the public types and functions.
