## Unreleased

Nothing yet. This file tracks notes for major releases and significant
milestones; GitHub Releases (auto-generated from merged PRs) cover every
release in between.

## 0.1.0

First public release: a single in-memory representation (`ForecastTable`) and
two verbs — `combine` and `fit` — for probabilistic forecast combination,
consolidating the methods previously spread across hubEnsembles, qrensemble and
lopensemble.

- Combination methods: simple quantile/mixture ensembles, linear (mixture) and
  logarithmic opinion pools, trimmed mean, beta-transformed linear pool (BLP),
  quantile regression averaging (QRA), CRPS stacking and generic score-driven
  stacking, inverse-score and Hedge online weighting, and partial pooling across
  strata, with a `Windowed` wrapper for rolling refits.
- Utilities: `backtest` for rolling-origin evaluation, and the
  `effective_num_models` and `weight_stability` diagnostics.
- A thin R interface wrapping the core methods.
