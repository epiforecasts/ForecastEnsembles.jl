# Ensembles.jl

Unified Julia interface for combining probabilistic forecasts.

`Ensembles.jl` brings together the methods currently spread across three R
packages — [`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean & median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile regression
averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool) — under one in-memory representation and one set of verbs
(`fit`, `combine`), with multiple dispatch picking the right algorithm per
`(output_type, method)` pair.

Status: under active development. Phase 1 (core types + `SimpleEnsemble`) is
the first usable slice.

## Quick taste

```julia
using Ensembles, DataFrames

df = DataFrame(
    location = repeat(["A"], 6),
    horizon = repeat([1], 6),
    model_id = repeat(["m1", "m2", "m3"], inner = 2),
    output_type = "quantile",
    output_type_id = repeat([0.25, 0.75], 3),
    value = [1.0, 3.0, 2.0, 4.0, 0.5, 2.5],
)

ft = ForecastTable(df; task_id_cols = [:location, :horizon])
combine(ft, SimpleEnsemble(:mean))
```

## Plan

See `docs/src/design.md` for the design and phased delivery plan.
