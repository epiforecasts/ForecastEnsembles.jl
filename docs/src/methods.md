# Methods

All methods operate on a [`ForecastTable`](@ref) — a hubverse-aligned long-
format frame with required columns `model_id`, `output_type`,
`output_type_id`, `value`, plus task-id columns. Two verbs cover the whole
surface:

- `combine(ft, method)` — apply an `UnfittedMethod` to a `ForecastTable`.
- `fit(method, training, observations)` — estimate parameters of a
  `TrainedMethod`, returning a fitted counterpart that is itself an
  `UnfittedMethod` and can be passed to `combine`.

## SimpleEnsemble

Mean or median across `model_id`, optionally weighted. Mirrors
`hubEnsembles::simple_ensemble`. No training step.

## LinearPool

The ensemble distribution is a (weighted) mixture of the per-model
distributions:

```math
F_{\text{ens}}(x) = \sum_i w_i\, F_i(x).
```

The kernel is dispatched on the table's `output_type`:

| `output_type` | Path                                                                          |
|---------------|-------------------------------------------------------------------------------|
| `:sample`     | Weighted resample from per-model samples.                                     |
| `:cdf`        | Pointwise weighted average of CDFs.                                           |
| `:quantile`   | Reconstruct each model's CDF (PCHIP + normal tails), draw, re-extract quantiles. |

The quantile path uses the [`Ensembles.QuantileDistribution`](@ref)
helper. We use Fritsch–Carlson PCHIP in the interior (monotone, parameter-
free) and a Normal tail fitted to the two outermost knots. This matches
`distfromq`'s default `tail_dist = "norm"` qualitatively; numerical
differences from `distfromq`'s spline interior are within Monte Carlo noise
in the finished pool.

## QRA

Quantile Regression Averaging. Mirrors `qrensemble::qra`.

For each task group (and each quantile level if `per_quantile_weights = true`),
solve

```math
\min_{\beta_0, \beta} \sum_i \rho_\tau\!\left(y_i - \beta_0 - \sum_m \beta_m\, x_{i,m,\tau}\right)
```

where ``\rho_\tau`` is the τ-tilted absolute loss. Optional constraints:

- `enforce_normalisation`: ``\beta_m \ge 0`` and ``\sum_m \beta_m = 1``.
- `noncross` (only with `per_quantile_weights = true`): for every training
  point, the predicted quantiles at consecutive τ levels are non-decreasing.

The LP is solved with HiGHS via JuMP. With the same configuration as
`qrensemble::qra`'s default, fitted weights and predictions match the
R package's output to ~1e-3.

## CRPSStacking

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`
(without the `lambda` time-weighting term).

For sample-based forecasts, the per-task CRPS for a mixture
``F = \sum_i w_i F_i`` has a closed-form unbiased estimator from the
component samples:

```math
\widehat{\text{CRPS}}_t(w) = \sum_i w_i\, a_i^t
                          - \tfrac{1}{2} \sum_{i,j} w_i\, w_j\, B_{i,j}^t,
```

with ``a_i^t = \overline{|X_i - y_t|}`` and
``B_{i,j}^t = \overline{|X_i - X_j'|}`` averaged over the per-model
samples. The overall objective is the mean over tasks plus a Dirichlet
log-prior penalty controlled by `dirichlet_alpha`. We minimise in softmax
space with Optim.jl's L-BFGS, so weights stay on the simplex automatically.

The result agrees with `lopensemble::crps_weights` (Stan MAP) to within
optimiser tolerance — typically a few × 1e-3 on the dominant weight.

## Composition

Any fitted method is an `UnfittedMethod`, so the output of `fit` plugs
straight into `combine`. For instance, you can take CRPS-stacked weights
and feed them to a quantile-input `LinearPool`:

```julia
fitted = fit(CRPSStacking(), train_samples, observations)
lp     = LinearPool(; weights = fitted.weights, n_samples = 10_000)
combine(test_quantiles, lp)
```
