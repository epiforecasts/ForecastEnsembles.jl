# CRPS-stacked ensemble weights

R wrapper that mirrors \`lopensemble::crps_weights\`. Optimises weights
in softmax space against the empirical mixture-CRPS objective with a
Dirichlet log-prior penalty.

## Usage

``` r
crps_weights(training, observations, task_id_cols, dirichlet_alpha = 1.001)
```

## Arguments

- training:

  A data frame with columns \`model_id\`, \`output_type\` (must be
  \`"sample"\`), \`output_type_id\` (the sample index), \`value\`, plus
  task-id columns.

- observations:

  A data frame with the task-id columns plus a column named exactly
  \`observed\`.

- task_id_cols:

  Character vector of the column names that identify a forecast task,
  e.g. \`"date"\` or \`c("location", "date")\`.

- dirichlet_alpha:

  Concentration of the Dirichlet log-prior penalty on the weights.
  Values just above 1 (the default 1.001, matching \`lopensemble\`) give
  an almost flat prior whose only effect is to keep the optimum off the
  simplex boundary; larger values pull the weights toward uniformity,
  with strength decaying as the number of training tasks grows.

## Value

A data frame with columns \`model_id\` and \`weight\`.

## Migrating from lopensemble

\`lopensemble::crps_weights\` takes one flat data frame with columns
\`model\`, \`sample_id\`, \`predicted\`, \`observed\` and (optionally)
\`date\` / \`geography\`. To convert: rename \`model\` to \`model_id\`,
\`sample_id\` to \`output_type_id\` and \`predicted\` to \`value\`; add
an \`output_type = "sample"\` column; and move the observations into a
separate frame with the task-id columns plus a column named exactly
\`observed\`. The \`lambda\` time-weighting argument of \`lopensemble\`
is not yet supported.

## Startup time

Two distinct delays, easy to conflate:

- The very first use on a machine instantiates and precompiles the
  bundled Julia project (LP solver, optimiser, Ensembles.jl). This can
  take a few minutes and then stays cached in the Julia depot.

- Every fresh R session pays a Julia startup of roughly 10–15 seconds on
  the first call to any function in this package. Later calls in the
  same session run in about a second.

Requires Julia (\>= 1.10) on the \`PATH\`, or \`julia_bindir\`. If Julia
is not installed, install it via juliaup
(<https://github.com/JuliaLang/juliaup>) before using this package.

## Examples

``` r
if (FALSE) { # \dontrun{
set.seed(1)
n_samples <- 100
n_task <- 30
y <- rnorm(n_task)

make_samples <- function(model_id, sampler) {
  do.call(rbind, lapply(seq_len(n_task), function(t) data.frame(
    model_id = model_id, output_type = "sample",
    output_type_id = seq_len(n_samples),
    t = t, value = sampler(y[t]),
    stringsAsFactors = FALSE
  )))
}
train <- rbind(
  make_samples("m_good",  function(yt) yt + rnorm(n_samples, sd = 0.4)),
  make_samples("m_noisy", function(yt) rnorm(n_samples, sd = 3))
)
obs <- data.frame(t = seq_len(n_task), observed = y)
crps_weights(train, obs, task_id_cols = "t")
} # }
```
