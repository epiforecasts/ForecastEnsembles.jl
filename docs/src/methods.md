# Methods

Every method operates on a [`ForecastTable`](@ref): a hubverse-aligned long
format frame with required columns `model_id`, `output_type`,
`output_type_id`, `value`, plus task-id columns. Two verbs cover everything:

- `combine(ft, method)` applies an `UnfittedMethod` to a `ForecastTable`.
- `fit(method, training, observations)` estimates the parameters of a
  `TrainedMethod` and returns a fitted counterpart, which is itself an
  `UnfittedMethod` and can then go through `combine`.

Two ensemble methods cover all the operations:

| Method              | Operation                                              |
|---------------------|--------------------------------------------------------|
| `QuantileEnsemble`  | Per-τ aggregation of quantile values (Vincentization for `:mean`, "median ensemble" for `:median`). |
| `MixtureEnsemble`   | Mixture of distributions: F = Σᵢ wᵢ Fᵢ. Dispatched on output type. |

(`LinearPool` is kept as an alias for `MixtureEnsemble`.)

## QuantileEnsemble

At each task and quantile level τ, take a weighted aggregation of the
per-model τ-quantile values. With `agg = :mean` this is Vincentization
(weighted vertical pooling); with `agg = :median` it's the "median
ensemble" the COVID-19 hub used as its default.

Weights take any of three shapes:

- `nothing`: equal weights.
- per-model `EnsembleWeights` (`:model_id, :weight`): same weights at
  every τ.
- per-quantile `EnsembleWeights` (`:model_id, :output_type_id, :weight`):
  different weights per τ.

Per-τ weights typically come from a per-quantile QRA fit; per-model
weights from a CRPS-stacking fit or a joint QRA fit. Either plugs in.

## MixtureEnsemble

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
parameter-free); the tails are Normals fitted to the two outermost knots.
This matches `distfromq`'s default `tail_dist = "norm"` qualitatively;
differences from `distfromq`'s spline interior are within Monte Carlo
noise in the finished pool.

Mixture pooling is inherently a per-model operation. Per-quantile weights
are not meaningful for a mixture; `MixtureEnsemble` rejects them at
construction.

## QRA

Quantile Regression Averaging. Mirrors `qrensemble::qra`.

For each task group (and each quantile level if `per_quantile_weights = true`),
solve

```math
\min_{\beta_0, \beta} \sum_i \rho_\tau\!\left(y_i - \beta_0 - \sum_m \beta_m\, x_{i,m,\tau}\right)
```

where ``\rho_\tau`` is the τ-tilted absolute loss. Optional constraints:

- `enforce_normalisation`: ``\beta_m \ge 0`` and ``\sum_m \beta_m = 1``.
- `noncross` (only with `per_quantile_weights = true`): for every
  training point, the predicted quantiles at consecutive τ levels are
  non-decreasing.

The LP runs in HiGHS via JuMP. With the same configuration as
`qrensemble::qra`'s default, fitted weights and predictions agree with
the R package to about 1e-3.

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
prior penalty controlled by `dirichlet_alpha`. Optimisation runs in
softmax space using Optim.jl's L-BFGS, so weights stay on the simplex
automatically.

The result agrees with `lopensemble::crps_weights` (Stan MAP) to within
optimiser tolerance, typically a few × 1e-3 on the dominant weight.

## Composition: weights from anywhere

Composition between trained and untrained methods runs through one
accessor:

```julia
weights(m) -> Union{EnsembleWeights, Nothing}
```

When `weights(m)` returns an `EnsembleWeights`, you can pass the fitted
method itself wherever a weights argument is accepted:

```julia
fitted = fit(CRPSStacking(), train_samples, observations)

MixtureEnsemble(weights = fitted, n_samples = 10_000)
QuantileEnsemble(:mean; weights = fitted)
```

Two shapes appear:

- *Per-model* (`:model_id, :weight`): single weight vector that applies at
  every quantile level. Always returned for `FittedCRPSStacking`.
  Returned for `FittedQRA` when the fit is joint, simplex-constrained,
  and has no intercept. Accepted by both `MixtureEnsemble` and
  `QuantileEnsemble`.
- *Per-quantile* (`:model_id, :output_type_id, :weight`): weights vary
  across τ. Returned for `FittedQRA` when the fit is per-τ,
  simplex-constrained, and has no intercept. Accepted by
  `QuantileEnsemble` only; `MixtureEnsemble` rejects this shape because
  mixtures are not naturally τ-indexed.

In other cases — fits with an intercept, unconstrained fits, fits across
multiple task groups — `weights(m)` returns `nothing` because the fit
does not reduce to a clean weight vector. Passing such a method to
`MixtureEnsemble` or `QuantileEnsemble` raises at construction. You can
still call `combine(ft, fitted)` directly, which applies the fitted
regression coefficients to produce predicted quantiles.
