# Roadmap

Planned work, organised around the two axes (combination operations and
weighting schemes) plus the workflow and ecosystem pieces. Cross-references
to the issues that prompted items are in parentheses.

## Weighting schemes (axis 2)

The organising idea is **stacking with any target**: choose weights to
minimise a proper score of the *combined* forecast on past observations. The
built foundation — `CRPSStacking`/`QRA` (closed-form CRPS/WIS), generic
`Stacking{Score}`, `InverseScore`, `Hedge`, `PartialPooling`, and the `Windowed`
wrapper — leaves these still to do:

- **Log-score stacking** and **BMA.** BMA is essentially log-score
  stacking of a mixture fitted by EM, so it folds into the generic stacker
  rather than standing alone. (Log score of a sample mixture needs a density
  estimate, so this is a follow-up to the sample-CRPS path now in place.)

## Combination operations (axis 1)

- **Log / geometric opinion pool.** `f_ens ∝ Πᵢ fᵢ^wᵢ` — multiply the
  densities (product of experts), precision-weighting toward the sharpest
  member. Distinct from Vincentization (which averages quantile functions)
  and from convolution (the sum of independent variables); they coincide
  only for an equal-spread location family. A real third operation, but
  niche — lower priority.

## Workflow: choosing a scheme

- **Calibration diagnostics** — PIT histograms, coverage, pairwise model
  comparison — are general forecast-evaluation tools, not ensemble-specific,
  so they belong in the scoring/evaluation layer
  ([ForecastScoring.jl, ProjectProposals#2](https://github.com/EpiAware/ProjectProposals/issues/2)),
  which we consume rather than reimplement.

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
  [isodistrreg](https://github.com/AlexanderHenzi/isodistrreg).

Candidate methods: **spread / dispersion adjustment** (a one-parameter
widen/narrow of the combined forecast around its median to hit nominal
coverage — the simplest parametric `Recalibrator`, and distinct from
PostForecasts.jl's five methods, which map *point* forecasts to quantiles
rather than rescaling an existing distribution); **empirical PIT mapping**
(also no upstream dependency); **IDR** (Henzi, Ziegel, Gneiting 2019,
[arXiv:1909.03725](https://arxiv.org/abs/1909.03725)); **CRPS-minimising
parametric recalibration** (the `CRPSStacking` machinery applied one model at
a time); **BLP**.

## Ecosystem

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
