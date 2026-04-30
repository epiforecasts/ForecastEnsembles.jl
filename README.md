# Ensembles.jl

A Julia package for combining probabilistic forecasts.

This is what you'd get if you took
[`hubEnsembles`](https://github.com/Infectious-Disease-Modeling-Hubs/hubEnsembles)
(simple/weighted mean & median, linear opinion pool),
[`qrensemble`](https://github.com/epiforecasts/qrensemble) (quantile
regression averaging), and
[`lopensemble`](https://github.com/epiforecasts/lopensemble) (CRPS-stacked
linear opinion pool), and rewrote them around one in-memory representation
and two verbs (`fit`, `combine`). Multiple dispatch then picks the right
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
combine(ft, SimpleEnsemble(:mean))
```

See `docs/src/` for the full design and method-by-method docs.
