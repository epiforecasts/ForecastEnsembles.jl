# Roadmap

Methods and features I'd like to add. Cross-references to the issues that
prompted them are in parentheses.

## Combination methods

- **Log-score stacking.** Mirror `CRPSStacking` but minimise mean negative
  log predictive density. Standard for sample/density forecasts; not in
  `lopensemble`.
- **WIS-based optimisation.** Quantile-forecast analogue of
  CRPS-stacking: minimise the mean Weighted Interval Score over training
  points to pick mixture weights. Mentioned as a planned direction in
  the original `qra` README and in
  [lopensemble#10](https://github.com/epiforecasts/lopensemble/issues/10).
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

- **Isotonic Distributional Regression (IDR).** Henzi, Ziegel, Gneiting
  (2019), [arXiv:1909.03725](https://arxiv.org/abs/1909.03725). Not a
  combination method. Recalibrates any forecast (single model or
  ensemble output) by fitting an isotonic conditional CDF on training
  pairs. The R reference implementation is
  [`isodisreg`](https://github.com/AlexanderHenzi/isodisreg). Sits
  cleanly downstream of `combine`: take whatever distribution the
  ensemble produced and pass it through IDR before scoring.

  Concretely: a `Recalibrator` abstract type with `fit` and a `recalibrate`
  verb that takes a `ForecastTable` and returns one. IDR is the first
  concrete subtype.

## Infrastructure

- **Tighter scoringutils integration.** Round-trip from
  `scoringutils::forecast_quantile` and `forecast_sample` to
  `ForecastTable` is already covered by `from_scoringutils`. Going the
  other way for scoring is the missing half. lopensemble#17 references
  this concern via tidymodels' `stacks`.
- **Persistent Julia daemon for the R wrapper.** First call in an R
  session pays a Julia startup cost of ~10–15 s. DaemonMode.jl keeps a
  Julia process alive across R sessions, removing the cost entirely.
- **Benchmarks vs the R packages.** Right now the speed claim in the
  index page is unmeasured. A small `benchmark/` directory comparing
  throughput on a realistic hubverse `model_out_tbl` would settle it.

## Out of scope (for now)

- Decision-theoretic combination (utility-weighted ensembles).
- Online / streaming weight updates.
- A GUI.
