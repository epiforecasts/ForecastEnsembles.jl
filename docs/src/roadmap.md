# Roadmap

Methods and features I'd like to add. Cross-references to the issues that
prompted them are in parentheses.

## Combination methods

- **Log-score stacking.** Mirror `CRPSStacking` but minimise mean negative
  log predictive density. Standard for sample/density forecasts; not in
  `lopensemble`.
- **WIS-based optimisation.** For *Vincentized* (vertical) combination,
  this already exists: WIS is proportional to the mean pinball loss
  across the submitted levels, so joint QRA with the simplex constraint
  and no intercept is exactly WIS-optimal weight estimation (see the
  Methods page). What remains open is WIS-optimal weights for the
  *mixture* (the harder, non-convex case mentioned in
  [lopensemble#10](https://github.com/epiforecasts/lopensemble/issues/10)).
- **Generic scoringutils-score stacking.** Same shape as the above two,
  for any proper scoring rule available in `scoringutils`. Open question
  whether it's worth doing in the general case (the simplex constraint
  on weights complicates optimisation, see lopensemble#10).
- **Trimmed-mean / trimmed-median ensemble.** Drop the top/bottom k% of
  model values per (task, τ) before averaging. Used by several forecast
  hubs to robustify against outlier models.
- **Beta-transformed linear pool (BLP).** Gneiting & Ranjan's
  recalibration of the mixture, designed to fix LOP underdispersion.
  Drops in alongside `MixtureEnsemble`.
- **Performance-based / inverse-skill weights.** Analytic weights from
  past CRPS or log-score, no optimiser needed. Cheap, sometimes
  competitive.
- **Bayesian model averaging.** Posterior model probabilities as weights.
  Niche but well-defined and asked for.

## Recalibration

Recalibration sits downstream of `combine`: an ensemble (or single-model)
forecast goes in, a calibrated forecast comes out, both before scoring.
We'd want a `Recalibrator` abstract type with `fit(m, training_forecasts,
observations)` and `recalibrate(ft, fitted)` verbs.

The state of recalibration in Julia today is awkward.
[PostForecasts.jl](https://lipiecki.github.io/PostForecasts.jl/stable/)
ships five methods (`Normal`, `CP`, `IDR`, `QR`, `LassoQR`) but they all
take *point forecasts* as input and produce quantile forecasts. The
`train`/`predict` flow is point → distributional, not distributional →
distributional. For our use case (recalibrating an ensemble's quantile
output) there is no off-the-shelf solution: collapsing to the median
before recalibration throws away the distributional information we just
spent the package combining.

Three reasonable paths, in increasing order of effort:

- *Contribute to PostForecasts.jl.* Their IDR implementation could
  generalise to richer input — Henzi/Ziegel/Gneiting's IDR is in
  principle distributional regression on arbitrary covariates, not
  point-only. A PR extending the input type is a natural next step
  for that package and would benefit everyone using it.
- *Sibling package, depending on PostForecasts.jl + ours.* Implements
  quantile-input recalibrators (IDR, BLP, empirical PIT mapping) by
  composing PostForecasts.jl's machinery where applicable and adding
  what's missing. Avoids putting recalibration into Ensembles.jl
  itself.
- *Implement directly in Ensembles.jl.* IDR specifically has a clean
  reference implementation in
  [isodisreg](https://github.com/AlexanderHenzi/isodisreg) (R) that
  could be ported. Faster start, but duplicates code that already
  exists elsewhere.

Methods worth covering once one of those paths lands:

- **IDR** for quantile- or sample-input forecasts (Henzi, Ziegel,
  Gneiting 2019, [arXiv:1909.03725](https://arxiv.org/abs/1909.03725)).
- **Beta-transformed linear pool (BLP).** Already on the combination
  list above; really a recalibration on top of the mixture (maps the
  mixture CDF through a fitted Beta(a, b)). Cures LOP underdispersion.
- **Empirical PIT mapping.** Map raw forecast quantiles through the
  empirical PIT histogram of past forecasts. Crude but very general;
  baseline for everything else. Simple enough (~50 lines) that it need
  not wait for any upstream package — a good first `Recalibrator`.
- **CRPS-minimising parametric recalibration.** Fit a parametric CDF
  transform per model that minimises mean CRPS on training pairs. Same
  optimisation machinery as `CRPSStacking`, applied to one model at a
  time.

## Infrastructure

- **Tighter scoringutils integration.** Round-trip from
  `scoringutils::forecast_quantile` and `forecast_sample` to
  `ForecastTable` is already covered by `from_scoringutils`. Going the
  other way for scoring is the missing half. lopensemble#17 references
  this concern via tidymodels' `stacks`.
- **Benchmarks vs the R packages.** A small `benchmark/` directory
  comparing throughput on a realistic hubverse `model_out_tbl`. Until it
  exists, the docs make no performance claims.

## Out of scope (for now)

- Decision-theoretic combination (utility-weighted ensembles).
- Online / streaming weight updates.
- A GUI.
