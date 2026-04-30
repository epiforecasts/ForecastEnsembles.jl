# Methods

Every method operates on a [`ForecastTable`](@ref) — a hubverse-aligned
long-format frame with required columns `model_id`, `output_type`,
`output_type_id`, `value`, plus task-id columns. Two verbs cover the whole
surface:

- `combine(ft, method)` — apply an `UnfittedMethod` to a `ForecastTable`.
- `fit(method, training, observations)` — estimate the parameters of a
  `TrainedMethod` and return a fitted counterpart, which is itself an
  `UnfittedMethod` and can then be passed to `combine`.

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
helper. The interior interpolation is Fritsch–Carlson PCHIP (monotone,
parameter-free) and the tails are Normals fitted to the two outermost
knots. This matches `distfromq`'s default `tail_dist = "norm"`
qualitatively; differences from `distfromq`'s spline interior are within
Monte Carlo noise in the finished pool.

If you pass `LinearPool` per-quantile weights (a long-format frame with
`:model_id, :output_type_id, :weight`), the quantile path switches to
direct vertical pooling — at each τ, take a weighted linear combination of
the per-model quantile values using the τ-specific weights, with no CDF
reconstruction. The CRPS-stacked weights are per-model; per-quantile QRA
fits provide per-τ weights, and either plug straight into `LinearPool`.

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

The LP runs in HiGHS via JuMP. With the same configuration as
`qrensemble::qra`'s default, fitted weights and predictions match the R
package to about 1e-3.

## CRPSStacking

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`,
without the `lambda` time-weighting term.

For sample-based forecasts, the per-task CRPS for a mixture
``F = \sum_i w_i F_i`` has a closed-form unbiased estimator from the
component samples:

```math
\widehat{\text{CRPS}}_t(w) = \sum_i w_i\, a_i^t
                          - \tfrac{1}{2} \sum_{i,j} w_i\, w_j\, B_{i,j}^t,
```

with ``a_i^t = \overline{|X_i - y_t|}`` and
``B_{i,j}^t = \overline{|X_i - X_j'|}`` averaged over the per-model
samples. The full objective is the mean over tasks plus a Dirichlet log-
prior penalty controlled by `dirichlet_alpha`. We minimise in softmax
space with Optim.jl's L-BFGS, so weights stay on the simplex
automatically.

The result agrees with `lopensemble::crps_weights` (Stan MAP) to within
optimiser tolerance — typically a few × 1e-3 on the dominant weight.

## Composition

Composition between trained and untrained methods runs through one
accessor:

```julia
weights(m) -> Union{DataFrame, Nothing}
```

When `weights(m)` returns a `DataFrame`, you can pass the fitted method
itself wherever a weights frame is accepted:

```julia
fitted = fit(CRPSStacking(), train_samples, observations)

LinearPool(weights = fitted, n_samples = 10_000)   # same as passing fitted.weights
SimpleEnsemble(:mean; weights = fitted)
```

`weights` returns two shapes:

- *Per-model* (`:model_id, :weight`) — single weight vector that applies at
  every quantile level. Always returned for `FittedCRPSStacking`. Returned
  for `FittedQRA` when the fit is joint, simplex-constrained, and has no
  intercept.
- *Per-quantile* (`:model_id, :output_type_id, :weight`) — weights vary
  across τ. Returned for `FittedQRA` when the fit is per-τ, simplex-
  constrained, and has no intercept. `LinearPool` dispatches on this shape
  and does direct vertical pooling instead of the CDF-mix path.

In other cases — fits with an intercept, unconstrained fits, or fits
across multiple task groups — `weights` returns `nothing` and passing the
fitted method to `LinearPool` raises at construction time. This is how the
type hierarchy carries weight: the accessor is the contract, the type tag
isn't.
