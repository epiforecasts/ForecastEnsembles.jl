# Worked example: combining three flu hospitalisation forecasts

This page runs every combination operation and weight estimator in the package.
The forecasts are a real hubverse slice bundled with `ForecastEnsembles.jl`:
three models from the
[example-complex-forecast-hub](https://github.com/hubverse-org/example-complex-forecast-hub),
each predicting weekly flu hospitalisations on 2022-12-17 at horizon 1 across
five US locations (national plus CA, FL, NY, TX), at the standard 23 quantile
levels.

Every block below is executed when the documentation is built, so what you read
is what the current code does.

```@example example
using ForecastEnsembles, CSV, DataFrames, Distributions, Random

flu = CSV.read(joinpath(pkgdir(ForecastEnsembles), "data", "flu_forecasts.csv"),
    DataFrame; types = Dict(:output_type_id => Float64, :location => String))

ft = ForecastTable(flu; task_id_cols = [:reference_date, :target_end_date,
    :horizon, :location, :target])
```

## Equal-weight quantile mean (Vincentization)

The simplest combination: at each (location, τ) take the unweighted mean of the
three model quantile values.

```@example example
combine(ft, QuantileEnsemble(:mean))
```

Or the median ensemble the COVID-19 hub used as its default:

```@example example
combine(ft, QuantileEnsemble(:median))
```

## Mixture (linear opinion pool)

Average the distributions rather than the quantile values: F = Σᵢ wᵢ Fᵢ. Each
model's quantiles are reconstructed into a continuous distribution (PCHIP
interior, Normal tails), and the mixture CDF is then inverted at each requested
level by bisection. On quantile input this path is deterministic, so `n_samples`
does not apply to it; that field governs the `:sample` path only.

```@example example
combine(ft, MixtureEnsemble())
```

This gives a different answer from Vincentization in general: averaging quantile
values is not the same operation as averaging CDFs.

## Geometric (logarithmic) pool

Multiply the member densities instead of averaging them — a product of experts,
sharper than the linear pool where the models agree:

```@example example
combine(ft, LogarithmicPool())
```

## Robust mean (trimmed / winsorised)

Drop or clamp the most extreme model at each (location, τ) before averaging, for
robustness to an outlier submission:

```@example example
combine(ft, TrimmedMean(; fraction = 0.2))
```

With `mode = :winsorise` the extremes are clamped to the surviving range rather
than dropped:

```@example example
combine(ft, TrimmedMean(; fraction = 0.2, mode = :winsorise))
```

`fraction` trims `round(fraction · n)` models from each end, capped so at least
one value survives, so with three models nothing is trimmed until `fraction`
rises above about 0.17.

## Hand-supplied weights

Either combination operation takes an `EnsembleWeights`:

```@example example
w = EnsembleWeights(DataFrame(
    model_id = ["Flusight-baseline", "MOBS-GLEAM_FLUH", "PSI-DICE"],
    weight = [0.2, 0.4, 0.4]
))

combine(ft, QuantileEnsemble(:mean; weights = w))
```

```@example example
combine(ft, MixtureEnsemble(; weights = w))
```

## A history to learn weights from

The estimators below need past forecasts with matching observations. The
bundled slice is a single date, so this page builds a small synthetic history
for the same three models. Real use would supply past hub submissions here.

Sample-typed history, for the score-driven estimators:

```@example example
const MODELS = ["Flusight-baseline", "MOBS-GLEAM_FLUH", "PSI-DICE"]

rng = MersenneTwister(20221217)
T, K = 12, 200

# A latent signal the models track, with the observation landing near but not on
# it. Forecasting the realised value exactly would leave nothing for the
# estimators, or for the recalibration further down, to work on.
signal = 100.0 .+ 20.0 .* randn(rng, T)
train_obs = DataFrame(t = 1:T, observed = signal .+ 6.0 .* randn(rng, T))

rows = DataFrame[]
for (mid, sd) in zip(MODELS, (35.0, 15.0, 22.0)), t in 1:T
    push!(rows, DataFrame(model_id = mid, output_type = "sample",
        output_type_id = 1:K, t = t,
        value = signal[t] .+ sd .* randn(rng, K)))
end
train_ft = ForecastTable(reduce(vcat, rows); task_id_cols = [:t])
```

`MOBS-GLEAM_FLUH` is the sharpest member of this synthetic history, so the
estimators below should favour it.

Quantile-typed history, for `QRA` and `BLP`, which take quantile input:

```@example example
levels = sort(unique(flu.output_type_id))

qrows = DataFrame[]
for (mid, sd) in zip(MODELS, (35.0, 15.0, 22.0)), t in 1:T
    push!(qrows, DataFrame(model_id = mid, output_type = "quantile",
        output_type_id = levels, t = t,
        value = signal[t] .+ sd .* quantile.(Normal(), levels)))
end
qtrain_ft = ForecastTable(reduce(vcat, qrows); task_id_cols = [:t])
```

## Weights from CRPS-stacking

`fit(CRPSStacking(), ...)` optimises a simplex weight vector against CRPS on
sample forecasts. The result plugs straight into either combination operation:

```@example example
stacked = fit(CRPSStacking(), train_ft, train_obs)
DataFrame(weights(stacked))
```

```@example example
combine(ft, MixtureEnsemble(; weights = stacked))
```

## Weights from QRA

QRA fits a quantile regression of past observations on past per-model
forecasts. Two configurations matter:

- *Joint* (`per_quantile_weights = false`): one weight vector across all τ,
  usable by either operation.
- *Per-τ* (`per_quantile_weights = true`): a different weight vector at each τ,
  usable by `QuantileEnsemble`.

```@example example
qra = fit(
    QRA(; per_quantile_weights = true, enforce_normalisation = true,
        intercept = false),
    qtrain_ft, train_obs
)
combine(ft, QuantileEnsemble(:mean; weights = qra))
```

Some configurations do not reduce to a weight vector at all — fits with an
intercept, unconstrained fits, or fits spanning several task groups. For those
`weights(::FittedQRA)` returns `nothing`, and asking for the weights explains
which case you are in:

```@example example
loose = fit(QRA(; enforce_normalisation = false), qtrain_ft, train_obs)
weights(loose) === nothing
```

You can still apply such a fit directly with `combine(ft, loose)`, which uses
the regression coefficients to predict quantiles.

## Score-driven estimators

These take any scoring function you supply, with the signature
`score(samples, y; w)` returning a scalar.
[`ScoringRules.jl`](https://github.com/EpiAware/ScoringRules.jl) is the natural
companion; the package itself depends on no scoring library, so this page
defines a small weighted CRPS to keep the example self-contained and to show the
contract explicitly:

```@example example
function crps(dat::AbstractVector, y::Real; w = nothing)
    n = length(dat)
    ww = w === nothing ? fill(1.0 / n, n) : w ./ sum(w)
    ex = sum(ww[i] * abs(dat[i] - y) for i in 1:n)
    ee = sum(ww[i] * ww[j] * abs(dat[i] - dat[j]) for i in 1:n, j in 1:n)
    return ex - 0.5 * ee
end
```

Generic stacking against that score:

```@example example
DataFrame(weights(fit(Stacking(crps), train_ft, train_obs)))
```

Performance-based weighting, scoring each member independently with no
optimiser:

```@example example
DataFrame(weights(fit(InverseScore(crps), train_ft, train_obs)))
```

Adaptive weighting, which walks the time column and updates after each round, so
a model forecasting badly this week counts for less next week:

```@example example
hedged = fit(Hedge(crps; time_col = :t), train_ft, train_obs)
DataFrame(weights(hedged))
```

Its `trajectory` records the weights after every update, and
[`weight_stability`](@ref) summarises how much each model's weight moved:

```@example example
weight_stability(hedged)
```

## Training on a trailing window, and comparing schemes

`Windowed` restricts any estimator to the most recent times, and `backtest`
compares schemes out of sample by expanding the training window one step at a
time:

`backtest` scores each fold with a function you supply, of the shape
`(forecast, observations) -> score`. Built from the `crps` above:

```@example example
function fold_crps(fc, obs)
    d = DataFrames.innerjoin(DataFrame(fc), obs; on = :t)
    per = DataFrames.combine(DataFrames.groupby(d, :t),
        [:value, :observed] =>
            ((v, y) -> crps(Float64.(v), Float64(first(y)))) => :s)
    return sum(per.s) / nrow(per)
end

rolling = Windowed(CRPSStacking(), 6; time_col = :t)

backtest(train_ft, train_obs,
    ["expanding" => CRPSStacking(), "rolling" => rolling];
    time_col = :t, min_train = 4, score_fn = fold_crps)
```

## Recalibrated mixture (beta-transformed linear pool)

`BLP` corrects the linear pool's tail underdispersion. It fits a Beta to the
pool's PIT values on quantile-typed history, then reweights the pool's quantile
levels rather than the models:

```@example example
blp = fit(BLP(), qtrain_ft, train_obs)
combine(ft, blp)
```

Because it recalibrates the pooled distribution rather than estimating per-model
weights, `weights(::FittedBLP)` is `nothing`.

## What's where in the data

Both `DataFrames` and `ForecastEnsembles` export `combine`, so the DataFrames
one needs qualifying here:

```@example example
DataFrames.combine(
    DataFrames.groupby(flu, [:model_id, :location]),
    nrow => :n_quantiles
)
```

Three models × five locations × 23 quantile levels: 345 rows.
