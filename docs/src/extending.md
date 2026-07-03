# Extending: adding your own ensemble or weight estimator

Everything in `ForecastEnsembles.jl` hangs off a small contract. To plug in a new
ensemble operation or a new way of estimating weights, you implement (a
subset of) the same handful of methods.

## The two abstract types

```julia
abstract type EnsembleMethod end
abstract type UnfittedMethod <: EnsembleMethod end       # combine() directly
abstract type TrainedMethod  <: EnsembleMethod end       # fit() first, then combine()
```

`UnfittedMethod` covers anything that can be applied to a `ForecastTable`
without training (`MixtureEnsemble`, `QuantileEnsemble`, and the fitted
counterparts of trained methods). `TrainedMethod` covers things that need
to be fitted against historical data first (`QRA`, `CRPSStacking`).

## Adding an untrained ensemble operation

You need two definitions:

```julia
struct MyEnsemble <: UnfittedMethod
    weights::Union{Nothing,EnsembleWeights}
    # any hyperparameters
end

function MyEnsemble(; weights = nothing, ...)
    return MyEnsemble(_resolve_weights(weights), ...)
end

function combine(ft::ForecastTable, m::MyEnsemble; rng::AbstractRNG = default_rng())
    # implement the operation; return a ForecastTable.
end
```

A few conventions worth following:

- Accept anything `_resolve_weights` accepts (`EnsembleWeights`,
  `DataFrame`, fitted method, `nothing`). That gives users symmetric
  behaviour across methods.
- If your method does not support per-quantile weights, reject them
  explicitly in the constructor (see `MixtureEnsemble` for the pattern).
- Set `model_id` on the output to a sensible label. The existing methods
  use `"hub-ensemble"` or the method-name string.

## Adding a trained weight estimator

Two structs and two methods:

```julia
struct MyMethod <: TrainedMethod
    # hyperparameters
end

struct FittedMyMethod <: UnfittedMethod
    # whatever the fit produces (coefficient tables, hyperparameters, ...)
end

function fit(m::MyMethod, training::ForecastTable, observations::AbstractDataFrame)
    # estimate parameters; return FittedMyMethod(...)
end

function combine(ft::ForecastTable, m::FittedMyMethod; rng = default_rng())
    # apply the fitted parameters; return a ForecastTable.
end
```

`fit` is `StatsBase.fit`, imported by `ForecastEnsembles`. Adding a method is
non-piracy because at least one argument is your own type.

### Optional: enable composition with `weights`

If the fit reduces to a per-model or per-τ weight vector, define:

```julia
function weights(m::FittedMyMethod)
    # Return an `EnsembleWeights` if the fit can be expressed as weights.
    # Return `nothing` otherwise (and document the conditions).
end
```

Once `weights(m)` is defined and returns an `EnsembleWeights`, your
fitted method can be passed to `MixtureEnsemble(weights = m)` or
`QuantileEnsemble(weights = m)` automatically. `_resolve_weights` handles
the conversion. Users who fit your method then do not need to manually
pull the weight DataFrame out and pass it.

If your fit cannot always be expressed as weights — say it has an
intercept — return `nothing` in those cases. Method construction will
raise rather than silently producing something wrong, which is the
behaviour `FittedQRA` uses for unconstrained or per-τ-with-intercept fits.

## Worked sketch: a "trimmed-mean" ensemble

Untrained method, no weight learning. Trims the top and bottom k% of
model values per (task, τ) before averaging:

```julia
struct TrimmedMean <: UnfittedMethod
    weights::Union{Nothing,EnsembleWeights}    # only used as a guard;
                                                 # the operation itself is unweighted
    fraction::Float64
end

function TrimmedMean(; fraction::Real = 0.1)
    0 <= fraction < 0.5 || throw(ArgumentError("fraction must be in [0, 0.5)"))
    return TrimmedMean(nothing, Float64(fraction))
end

function combine(ft::ForecastTable, m::TrimmedMean; rng = default_rng())
    df = ft.data
    group_cols = vcat([:output_type, :output_type_id], ft.task_id_cols)
    out = DataFrames.combine(
        DataFrames.groupby(df, group_cols),
        :value => v -> begin
            n = length(v)
            k = round(Int, m.fraction * n)
            mean(sort(v)[(k+1):(n-k)])
        end => :value,
    )
    out[!, ft.model_id_col] .= "trimmed-mean"
    select!(out, ft.model_id_col, :output_type, :output_type_id,
            ft.task_id_cols..., :value)
    return ForecastTable(out;
                        task_id_cols = ft.task_id_cols,
                        model_id_col = ft.model_id_col)
end
```

That is the entire surface a new operation has to implement. The same
shape works for log-score stacking, BMA, performance-based inverse-error
weighting, beta-transformed linear pool, and so on.
