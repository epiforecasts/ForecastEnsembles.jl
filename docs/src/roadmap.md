# Roadmap

Planned work, organised around the two axes (combination operations and
weighting schemes) plus the workflow and ecosystem pieces. Cross-references
to the issues that prompted items are in parentheses.

## The enabling seam: internal scoring

Several items below need "the CRPS or WIS of a forecast". We already
compute both internally — CRPS-from-samples in `CRPSStacking`, WIS/pinball
in `QRA`. The first step is to factor these into a small interface,

```julia
score(forecast, obs, CRPS())    # samples
score(forecast, obs, WIS())     # quantiles
score(forecast, obs, Coverage())
```

so the workflow items can consume it now, and it becomes the delegation
point to `ScoringRules.jl` (see Ecosystem) once that lands — at which
point the in-house estimators are swapped for its versions and generic
stacking generalises.

## Weighting schemes (axis 2)

The organising idea is **stacking with any target**: choose weights to
minimise a proper score of the *combined* forecast on past observations.
`CRPSStacking` (CRPS on samples) and constrained `QRA` (WIS on quantiles)
are the two closed-form instances we already have; the general
`Stacking{Score}` over an arbitrary score needs `ScoringRules.jl` and its
optimisation is non-convex for some score/operation pairs (notably
log-score on a mixture).

Buildable now (reuse the scoring seam, no `ScoringRules.jl` dependency):

- **Cheap baseline: inverse-score / softmax weights.** `wᵢ ∝ exp(−score of
  member i)`. Analytic, no optimiser. A strong, fast default.
- **Window training.** Trailing-window wrapper around any estimator; the
  coarse cousin of the recency `lambda` already in `CRPSStacking`.
- **Online / adaptive weighting.** Exponentiated-gradient / Hedge:
  update weights from each round's per-member loss, with regret
  guarantees, no full refit. Natural fit for a weekly operational cadence.
- **Partial pooling / hierarchical weights.** Share weights across strata
  (locations, age groups) with shrinkage, so a data-sparse stratum
  borrows strength. Extends the `CRPSStacking` objective with a
  cross-stratum shrinkage term. Genuinely novel for this space and matches
  hub data shape.

Needs `ScoringRules.jl`:

- **Generic `Stacking{Score}`.** One stacker parameterised by any proper
  score, with the CRPS and WIS specialisations dispatched underneath.
  Subsumes the old "generic scoringutils-score stacking" item.
- **Log-score stacking** and **BMA.** BMA is essentially log-score
  stacking of a mixture fitted by EM, so it folds into the generic stacker
  rather than standing alone.

## Combination operations (axis 1)

- **Log / geometric opinion pool.** `f_ens ∝ Πᵢ fᵢ^wᵢ` — multiply the
  densities (product of experts), precision-weighting toward the sharpest
  member. Distinct from Vincentization (which averages quantile functions)
  and from convolution (the sum of independent variables); they coincide
  only for an equal-spread location family. A real third operation, but
  niche — lower priority.
- **Trimmed / winsorised mean.** Drop the top/bottom k% of member values
  per (task, τ) before averaging; robustness to outlier submissions. Sketch
  already in [Extending](extending.md).
- **Spread / dispersion adjustment.** A one-parameter widen/narrow of the
  combined forecast to hit nominal coverage. Cheap, and often more
  effective than heavier recalibration.
- **Beta-transformed linear pool (BLP).** Gneiting & Ranjan's recalibrated
  mixture; fixes LOP underdispersion. Sits between axis 1 and
  recalibration (it maps the mixture CDF through a fitted Beta).

## Workflow: choosing a scheme

- **Backtesting / cross-validation harness.** Expanding-window or
  leave-one-time-out, refit each scheme, compare out-of-sample scores.
  This turns the package from a box of combiners into "here is how to pick
  one on your data" — the biggest user-value gap, and buildable now on the
  scoring seam. *Scoring* itself lives upstream (Ecosystem); the
  *selection loop* is ensemble-specific and lives here.
- **Weight & calibration diagnostics.** Effective number of models (weight
  concentration), weight stability over time, PIT histograms of the
  ensemble.

## Recalibration

A separate pipeline stage — it transforms a single predictive distribution
rather than aggregating several — so it is out of scope for the core
combine story, but planned. A `Recalibrator` abstract type with
`fit(m, training_forecasts, observations)` and `recalibrate(ft, fitted)`.

[PostForecasts.jl](https://lipiecki.github.io/PostForecasts.jl/stable/)
implements five methods (`Normal`, `CP`, `IDR`, `QR`, `LassoQR`), but they
all take point forecasts as input and return quantile forecasts — point
in, distribution out. Recalibrating an ensemble's quantile output is the
other direction, so there is no off-the-shelf fit; three paths, in
increasing effort:

- *Contribute to PostForecasts.jl* — its IDR could generalise to richer
  input (Henzi/Ziegel/Gneiting's IDR is distributional regression on
  arbitrary covariates, not point-only). A PR benefits everyone.
- *Sibling package* over PostForecasts.jl + ours.
- *Implement here* — IDR has a clean R reference in
  [isodisreg](https://github.com/AlexanderHenzi/isodisreg).

Candidate methods: **IDR** (Henzi, Ziegel, Gneiting 2019,
[arXiv:1909.03725](https://arxiv.org/abs/1909.03725)); **empirical PIT
mapping** (simple enough to be the first `Recalibrator`, no upstream
dependency); **CRPS-minimising parametric recalibration** (the
`CRPSStacking` machinery applied one model at a time); **BLP**.

## Ecosystem

- **ScoringRules.jl** — a Julia port of the R `scoringRules` (atomic
  proper scoring rules dispatching on Distributions.jl), approved to be
  built ([ProjectProposals#1](https://github.com/EpiAware/ProjectProposals/issues/1)).
  It is the shared objective library for stacking *and* the evaluation
  library for the ecosystem. Built neutral and transfer-portable, with
  prominent attribution to the original authors (Jordan, Krüger & Lerch,
  JSS 2019) and a clear statement that it is an LLM-assisted port.
- **Shared forecast types** — an array-backed `Forecast`/`ForecastSet`
  layer (`ProbabilisticForecasts.jl`) that ForecastBaselines.jl,
  PostForecasts.jl, ForecastEnsembles.jl and the scoring stack all speak.
  The real unifier; see `design/forecast-types.md`.
- **Evaluation** — scoring lives in ScoringRules.jl /
  ForecastScoring.jl ([#2](https://github.com/EpiAware/ProjectProposals/issues/2));
  we consume it, we do not build it.
- **A domain-agnostic forecasting org** — a possible neutral home grouping
  ForecastBaselines.jl, ForecastEnsembles.jl, ScoringRules.jl and
  potentially non-epi packages (PostForecasts.jl). Under discussion;
  packages are built to be org-portable regardless.

## Infrastructure

- **hubverse directory IO** — read a `model-output/` tree and write a
  valid submission directly.
- **Benchmarks vs the R packages** — a small `benchmark/` on a realistic
  hubverse `model_out_tbl`. Until it exists, the docs make no performance
  claims.

## Out of scope (for now)

- Decision-theoretic combination (utility-weighted ensembles).
- A GUI.
