# Methods

Every method operates on a [`ForecastTable`](@ref): a hubverse-aligned long
format frame with required columns `model_id`, `output_type`,
`output_type_id`, `value`, plus task-id columns. Two verbs cover everything:

- `combine(ft, method)` applies an `UnfittedMethod` to a `ForecastTable`.
- `fit(method, training, observations)` estimates the parameters of a
  `TrainedMethod` and returns a fitted counterpart, which is itself an
  `UnfittedMethod` and can then go through `combine`.

Combining forecasts has two independent axes, and the methods here map onto
them:

1. **How the members are combined** — the aggregation operation
   (`QuantileEnsemble`, `MixtureEnsemble`, `TrimmedMean`).
2. **How the weights are chosen** — equal, user-supplied, or estimated from
   past forecasts (`CRPSStacking`, `QRA`, generic `Stacking`, `InverseScore`,
   `Hedge`, `PartialPooling`).

These weight estimators are not extra kinds of ensemble; each produces a
weight vector that feeds either combination operation. The two sections below
follow the two axes.

The score-driven estimators beyond the closed-form `CRPSStacking`/`QRA` —
generic `Stacking`, `InverseScore`, `Hedge`, `PartialPooling` — optimise
against a proper score from
[`ScoringRules.jl`](https://github.com/EpiAware/ScoringRules.jl). Because
ScoringRules is GPL-licensed and this package is MIT, it is a **weak
dependency**: run `using ScoringRules` to activate them. `CRPSStacking`, `QRA`,
and the `Windowed` wrapper need no extra package.

This package covers the *combine* step of a forecasting pipeline
(produce → combine → recalibrate → score). Recalibrating the combined
forecast is a separate stage on a single distribution rather than an
aggregation across several, and is out of scope here (see the
[Roadmap](roadmap.md)).

## Combining forecasts

Two operations, differing in what they average.

### QuantileEnsemble

At each task and quantile level τ, take a weighted aggregation of the
per-model τ-quantile values. With `agg = :mean` this is Vincentization
(weighted vertical pooling); with `agg = :median` it's the "median
ensemble" the COVID-19 hub used as its default.

### MixtureEnsemble

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

The quantile path uses the [`ForecastEnsembles.QuantileDistribution`](@ref)
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

### TrimmedMean

A robust variant of the quantile mean: at each task and `output_type_id`,
order the per-model values and either drop (`mode = :trim`) or clamp
(`mode = :winsorise`) the lowest and highest `fraction` before averaging. It
guards against a few outlier submissions without the full robustness of the
median ensemble, and `fraction` tunes how much tail is discarded (`0` is the
plain mean, `→ 0.5` degenerates to the median). Comparable-across-models values
only, so `:quantile` and `:cdf` but not `:sample`.

### Choosing between them

The two operations produce different predictive distributions,
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
semantics, e.g. when the models represent distinct scenarios
whose multimodality should survive into the ensemble.

## Choosing the weights

Both operations combine the members through a set of weights. Whichever
way they are obtained, the combined forecast is a linear rule over the
members — a weighted quantile at level τ, or a weighted mixture — and the
weights sit somewhere on a spectrum of how constrained that rule is:

- **equal** — the default, no fitting;
- **simplex weights** — non-negative, summing to one: a convex
  combination of the members;
- **+ intercept** — affine, adding a bias correction;
- **unconstrained** — free or negative coefficients: a full regression on
  the members that can extrapolate outside their range.

The convex (simplex) point is the one the `weights` accessor exposes, for
two reasons no looser fit shares. Its coefficients are *portable*: a
per-model weight means the same thing whether you mix the distributions or
average the quantiles. And they are *interpretable* as shares of belief,
so the ensemble stays inside the members' convex hull. Off the simplex,
the coefficients are tied to the specific regression that estimated them
and no longer transfer.

### Fixed and user-supplied weights

Pass `nothing` for equal weights, or an [`EnsembleWeights`](@ref) frame
(`:model_id, :weight`, optionally `:output_type_id` for per-τ weights):

```julia
w = EnsembleWeights(DataFrame(model_id = ["m1", "m2"], weight = [0.7, 0.3]))
combine(ft, QuantileEnsemble(:mean; weights = w))
combine(ft, MixtureEnsemble(; weights = w))
```

### Stacking: score-optimised weights

Stacking chooses the weights to minimise a proper score of the *combined*
forecast against past observations. Two instances, differing in the score
and the forecast representation they consume.

#### CRPSStacking

CRPS-stacked linear opinion pool. Mirrors `lopensemble::crps_weights`,
including its time weighting. Always estimates simplex weights (a softmax
parameterisation keeps them non-negative and summing to one), so the fit
is always a portable per-model weight vector.

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

#### QRA

Quantile Regression Averaging. Mirrors `qrensemble::qra`. QRA fits the
linear rule directly, and its constructor flags set where on the
constraint spectrum the fit lands.

For each task group (and each quantile level if `per_quantile_weights = true`),
solve

```math
\min_{\beta_0, \beta} \sum_i \rho_\tau\!\left(y_i - \beta_0 - \sum_m \beta_m\, x_{i,m,\tau}\right)
```

where ``\rho_\tau`` is the τ-tilted absolute loss. The LP runs in HiGHS
via JuMP, with the solver's termination status checked (degenerate
problems raise rather than returning NaN). With the same configuration as
`qrensemble::qra`'s default, fitted weights and predictions agree with
the R package to about 1e-3.

**As a weighting scheme.** With `enforce_normalisation = true` and
`intercept = false`, the coefficients are constrained to the simplex
(``\beta_m \ge 0``, ``\sum_m \beta_m = 1``, ``\beta_0 = 0``). This is
stacking too: the τ-tilted loss summed over the submitted levels is
proportional to the Weighted Interval Score, so a single shared,
simplex-constrained weight vector *is* WIS-optimal weight estimation. If
you came looking for "WIS stacking", this configuration is it. The fit is
then a portable weight vector — per-model for a joint fit, per-τ for
`per_quantile_weights = true` — and flows through `weights` into either
combination operation.

`noncross` (only with `per_quantile_weights = true`) adds monotonicity
constraints so predicted quantiles are non-decreasing in τ *at the
training points*. This is the standard LP formulation, same as
`qrensemble`. At new points the fitted hyperplanes can still cross —
unless the fit is simplex-constrained, in which case predictions are
convex combinations of the component quantiles and inherit their
monotonicity. `combine` checks the output and warns if any task's
quantiles cross.

**As a regression combiner.** Relax the constraints — add an intercept,
drop `enforce_normalisation` — and the same LP becomes an affine or
unconstrained quantile regression on the members. This buys bias
correction and extrapolation beyond the members' range, but the
coefficients are no longer a convex combination: they are specific to
this regression and do not transfer to a mixture, so `weights` returns
`nothing`. You still apply the fit with `combine(ft, fitted)`, which
evaluates the fitted regression to produce predicted quantiles directly.

#### Stacking (any score)

[`Stacking`](@ref)`(score; dirichlet_alpha)` is the general form: it minimises
the mean of *any* negatively-oriented weighted-sample rule from ScoringRules
(e.g. `ScoringRules.crps`, `ScoringRules.es` for the energy score) over the
simplex, in softmax space. `CRPSStacking` is the closed-form CRPS
specialisation; reach for `Stacking` when you want a different score.
Sample-typed forecasts only (weighted-sample scores); use `QRA` for
quantile/WIS stacking. Optimisation is non-convex for some score/operation
pairs, so convergence is not guaranteed everywhere.

```julia
using ScoringRules            # weak dependency; activates the estimators below
fitted = fit(Stacking(ScoringRules.crps), training_ft, training_obs)
combine(ft, MixtureEnsemble(; weights = fitted))
```

#### InverseScore

[`InverseScore`](@ref)`(score; temperature)` is performance weighting: score
each member independently over the training set and softmax the negative mean
scores, ``w_i \propto \exp(-\text{temperature} \cdot s_i)``. No optimiser, so
it is fast and hard to overfit — but, unlike stacking, it scores each member in
isolation and never sees how they combine, so it is blind to redundancy between
them. `temperature` sharpens the softmax (small → equal weights, large →
winner-take-all).

#### Hedge (online)

[`Hedge`](@ref)`(score; eta, time_col)` updates weights online by the
exponentiated-gradient / multiplicative-weights rule: walking `time_col` in
order, it multiplies each member's weight by ``\exp(-\text{eta} \cdot s_t)`` at
each step and renormalises. It adapts to *when* members did well — tracking
regime change that a single pooled score misses — and needs no full refit. The
fitted result keeps the whole weight `trajectory` for stability diagnostics.
`eta` is the learning rate (scale it to the magnitude of your score).

#### PartialPooling (hierarchical)

[`PartialPooling`](@ref)`(score; strata, lambda)` learns a weight vector per
stratum (a combination of the `strata` columns, e.g. `[:location]`) that
shrinks toward a shared global vector, so a data-sparse stratum borrows
strength. `lambda` sets the shrinkage: `0` fits each stratum independently,
large values pool them onto one vector. `combine` applies each stratum's own
weights and falls back to the global vector for a stratum unseen in training.
It generalises `Stacking` — a single stratum, or `lambda → ∞`, recovers global
stacking.

### Training windows and rolling refits

[`Windowed`](@ref)`(method, window; time_col)` wraps any weight estimator to
train on only the most recent `window` values of `time_col`, then delegates to
the inner method — so it composes with all of the above and is itself
dependency-free. It is the hard-cutoff counterpart to `CRPSStacking`'s smooth
`lambda` recency, and turns [`backtest`](@ref)'s default expanding window into a
rolling one.

### Choosing a scheme: `backtest`

[`backtest`](@ref)`(ft, observations, schemes; time_col, min_train)` compares
weighting schemes out-of-sample. For each time fold it refits every scheme on
the earlier data and scores its combined forecast on the held-out step,
returning a tidy frame of per-fold scores. The default scorer comes from the
ScoringRules extension (CRPS for samples, mean quantile score for quantiles);
pass `score_fn` to supply your own. Wrap a scheme in `Windowed` to backtest a
rolling window against an expanding one.

### Weight diagnostics

Once a scheme is fitted, two diagnostics summarise the weights it produced.
[`effective_num_models`](@ref) reports the participation ratio
``1 / \sum_i p_i^2`` of a weight vector — `1` when all weight sits on one
model, up to the model count when weights are equal — reading as "how many
models is this ensemble really using". It accepts any fitted method that
exposes [`weights`](@ref), an `EnsembleWeights`, or a raw vector, and returns a
per-τ frame for per-quantile weights. [`weight_stability`](@ref) takes a fitted
[`Hedge`](@ref) and returns each model's total weight variation along the
training trajectory, flagging models whose weight swung over time. Calibration
diagnostics (PIT, coverage) are general forecast-evaluation tools and live in
the scoring stack, not here.

### Composition through `weights`

The two axes meet at one accessor:

```julia
weights(m) -> Union{EnsembleWeights, Nothing}
```

When `weights(m)` returns an `EnsembleWeights`, pass the fitted method
itself wherever a weights argument is accepted:

```julia
fitted = fit(CRPSStacking(), train_samples, observations)

MixtureEnsemble(weights = fitted, n_samples = 10_000)
QuantileEnsemble(:mean; weights = fitted)
```

Two shapes appear:

- *Per-model* (`:model_id, :weight`): a single weight vector applied at
  every quantile level. Always returned for `FittedCRPSStacking`,
  `FittedStacking`, `FittedInverseScore`, and `FittedHedge`; for
  `FittedPartialPooling` it returns the pooled global vector (per-stratum
  weights live in the fitted object and are applied by its own `combine`);
  returned for `FittedQRA` when the fit is joint, simplex-constrained, and has
  no intercept. Accepted by both `MixtureEnsemble` and `QuantileEnsemble`.
- *Per-quantile* (`:model_id, :output_type_id, :weight`): weights vary
  across τ. Returned for `FittedQRA` in the per-τ, simplex-constrained,
  no-intercept case. Accepted by `QuantileEnsemble` only; `MixtureEnsemble`
  rejects this shape because mixtures are not naturally τ-indexed.

Off the simplex (fits with an intercept, unconstrained fits, or fits
across multiple task groups), `weights(m)` returns `nothing`, and passing
the method to `MixtureEnsemble` or `QuantileEnsemble` raises at
construction. This is the portability point again: those coefficients are
not reusable as weights, so `combine(ft, fitted)` is the way to apply
them.
