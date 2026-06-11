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

## Choosing between QuantileEnsemble and MixtureEnsemble

The two operations produce genuinely different predictive distributions,
and the difference matters in practice:

- The mixture (linear opinion pool) averages CDFs. When the component
  models disagree about location, the mixture spreads mass across all
  their modes, and — a well-known result — is *underdispersed relative to
  what calibration requires*: if each component is individually
  calibrated, the linear pool is overconfident in the tails (Gneiting &
  Ranjan 2013 introduced the beta-transformed pool precisely to correct
  this).
- Vincentization (`QuantileEnsemble(:mean)`) averages quantile functions.
  Its implied CDF is the inverse of the averaged quantile functions,
  which is sharper than the mixture when components disagree.
  Lichtendahl, Grushka-Cockayne & Winkler (2013, *Management Science*)
  show that the equally weighted quantile average outperforms the equally
  weighted linear pool when the components are individually calibrated —
  a setting close to a typical forecast hub.
- The median ensemble (`QuantileEnsemble(:median)`) takes the per-τ
  median of model quantile values. It is robust to a few extreme models —
  the reason the COVID-19 Forecast Hub adopted it after outlier
  submissions distorted the mean — but its implied distribution can
  behave oddly when components are multimodal. Prefer it when robustness
  to bad submissions matters more than efficiency.

As a rule of thumb for hub-style quantile submissions: start with
`QuantileEnsemble(:mean)` (or `:median` when robustness is a concern),
and reach for `MixtureEnsemble` when you specifically want the mixture
semantics, e.g. when the models represent genuinely distinct scenarios
whose multimodality should survive into the ensemble.

## MixtureEnsemble

The ensemble distribution is a (weighted) mixture of the per-model
distributions:

```math
F_{\text{ens}}(x) = \sum_i w_i\, F_i(x).
```

The kernel is dispatched on the table's `output_type`:

| `output_type` | Path                                                                          |
|---------------|-------------------------------------------------------------------------------|
| `:sample`     | Weighted resample from per-model samples (the only stochastic path).          |
| `:cdf`        | Pointwise weighted average of CDFs.                                           |
| `:quantile`   | Reconstruct each model's CDF (PCHIP + normal tails), invert the mixture CDF exactly. |

The quantile path uses the [`Ensembles.QuantileDistribution`](@ref)
helper. The interior interpolation is Fritsch–Carlson PCHIP (monotone,
parameter-free); the tails are Normals fitted to the two outermost knots.
This matches `distfromq`'s default `tail_dist = "norm"` qualitatively;
the interiors differ (PCHIP here, splines there) by amounts small
relative to typical quantile spacing.

The mixture quantile at each level is computed by bisection on
``\sum_i w_i F_i(x) = \tau`` — deterministic, no Monte Carlo error. This
differs from `hubEnsembles::linear_pool`, which samples from the
reconstructed distributions and re-extracts empirical quantiles; the
sampling approach carries Monte Carlo noise that grows in the tails,
which is exactly where hubs ask for τ = 0.01 and 0.99.

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

Two caveats worth knowing:

- *Non-crossing is enforced at training points only* (the standard LP
  formulation, same as `qrensemble`). At new points the fitted
  hyperplanes can cross — unless the fit is simplex-constrained, in which
  case predictions are convex combinations of the component quantiles and
  inherit their monotonicity. `combine` checks the output and warns if
  any task's quantiles cross.
- *Joint QRA with the simplex constraint and no intercept is WIS-optimal
  weight estimation for Vincentization.* The weighted interval score is
  proportional to the mean pinball loss across the submitted quantile
  levels, so minimising the summed τ-tilted losses with one shared weight
  vector is the same optimisation. If you came looking for "WIS
  stacking", this configuration is it.

The LP runs in HiGHS via JuMP, with the solver's termination status
checked (degenerate problems raise rather than returning NaN). With the
same configuration as `qrensemble::qra`'s default, fitted weights and
predictions agree with the R package to about 1e-3.

## CRPSStacking

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`,
including its time weighting.

For sample-based forecasts, the per-task CRPS for a mixture
``F = \sum_i w_i F_i`` has a closed-form unbiased estimator from the
component samples:

```math
\widehat{\text{CRPS}}_t(w) = \sum_i w_i\, a_i^t
                          - \tfrac{1}{2} \sum_{i,j} w_i\, w_j\, B_{i,j}^t,
```

with ``a_i^t = \overline{|X_i - y_t|}``, the cross terms
``B_{i,j}^t = \overline{|X_i - X_j'|}`` averaged over all sample pairs,
and the within-model terms ``B_{i,i}^t`` averaged over *distinct* pairs
only (including the k = l pairs would bias the diagonal down by O(1/K),
which is visible at the small per-model sample counts some hubs accept).

The full objective is the mean over tasks plus a Dirichlet log-prior
penalty, ``-((\alpha - 1)/T)\sum_i \log w_i``, controlled by
`dirichlet_alpha`. At ``\alpha = 1`` the penalty vanishes exactly; the
default 1.001 is a weak regulariser that keeps the optimum off the
simplex boundary, matching `lopensemble`'s default.

Optimisation runs in softmax space with an analytic gradient and
Optim.jl's L-BFGS. The objective is a concave quadratic in the weights,
so the optimum can sit at a vertex of the simplex (all weight on one
model); the optimiser therefore restarts from each vertex-leaning start
in addition to uniform weights and keeps the best minimum.

Model skill in epidemic forecasting is non-stationary — a model
well-calibrated in one wave can be badly off in the next — so the
objective supports per-task weighting:

- `task_weights`: a frame with the task-id columns plus `:weight`, one
  non-negative weight per training task. The general mechanism; covers
  recency, region weighting (`lopensemble`'s `gamma`), or down-weighting
  anomalous reporting weeks.
- `lambda` with `time_col`: recency sugar on top. A scalar
  ``\varphi \in (0, 1]`` gives exponential decay ``\varphi^{T-t}`` over
  the ordered unique time values (the usual forecasting-literature
  choice; 1 recovers equal weights); `:lopensemble` gives that package's
  default quadratic ramp ``2 - (1 - t/T)^2``; a vector or a function of
  the normalised time rank are also accepted.

The Dirichlet prior scale then uses the effective sample size
``(\sum_t \lambda_t)^2 / \sum_t \lambda_t^2`` in place of the raw task
count, so concentrating weight on recent tasks does not quietly
strengthen the prior relative to the data.

The result agrees with `lopensemble::crps_weights` (Stan MAP) to within
optimiser tolerance, typically a few × 1e-3 on the dominant weight — both
for equal weighting and for the recency ramp, each parity-tested against
fixtures. The residual difference reflects Stan's log-simplex
parameterisation versus our softmax — the two behave differently near
the simplex boundary.

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
