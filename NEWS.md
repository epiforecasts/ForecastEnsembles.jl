## Unreleased

### Breaking

- `QRA` now defaults to `noncross = true`, matching the R wrapper and
  `qrensemble::qra`. The flag is inert in the default shared-weight fit, so
  results are unchanged unless `per_quantile_weights = true`, where the fit is
  now constrained against crossing quantiles. That fit is a joint LP across all
  τ and requires every τ to carry the same set of models; pass
  `noncross = false` for ragged per-quantile submissions.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
