# Quantile Regression Averaging

R wrapper that mirrors \`qrensemble::qra\`. Fits a quantile regression
on training forecasts and applies the fitted weights to the target
forecasts.

## Usage

``` r
qra(
  training,
  target,
  observations,
  task_id_cols,
  per_quantile_weights = FALSE,
  intercept = FALSE,
  enforce_normalisation = TRUE,
  noncross = TRUE,
  group = character(0)
)
```

## Arguments

- training:

  A data frame with columns \`model_id\`, \`output_type\` (must be
  \`"quantile"\`), \`output_type_id\`, \`value\`, plus task-id columns.

- target:

  A data frame in the same shape: the forecasts to combine using the
  fitted weights.

- observations:

  A data frame with the task-id columns plus a column named exactly
  \`observed\`.

- task_id_cols:

  Character vector of the column names that identify a forecast task,
  e.g. \`c("location", "horizon", "target_date")\`.

- per_quantile_weights:

  If TRUE, fit separate weights per quantile.

- intercept:

  Include an intercept term in the regression.

- enforce_normalisation:

  Constrain weights to lie on the simplex.

- noncross:

  Add cross-quantile monotonicity constraints. Only takes effect when
  \`per_quantile_weights = TRUE\`; silently without effect otherwise
  (the joint fit cannot cross by construction when
  \`enforce_normalisation = TRUE\`).

- group:

  Character vector of task dimensions over which to fit separate
  regressions (e.g. \`"location"\` for per-location weights). Leave
  empty to fit a single global model across all tasks.

## Value

A data frame of fitted predictions on \`target\`. When the fit reduces
to a clean weight vector (the default configuration does), the weights
are attached as a data frame in \`attr(result, "weights")\`, mirroring
\`qrensemble\`.

## Migrating from qrensemble

\`qrensemble::qra\` takes a single \`forecast_quantile\` object
containing both training and holdout rows plus a \`target\` filter; this
function takes three explicit data frames. To convert: rename \`model\`
to \`model_id\`, \`quantile_level\` to \`output_type_id\` and
\`predicted\` to \`value\`; add an \`output_type = "quantile"\` column;
split the rows into \`training\` and \`target\` yourself (e.g. on
\`target_date\`); and put the observed values into a separate
\`observations\` frame with the task-id columns plus a column named
exactly \`observed\`.

The defaults (\`per_quantile_weights = FALSE\`, \`enforce_normalisation
= TRUE\`, \`intercept = FALSE\`, \`noncross = TRUE\`) match
\`qrensemble::qra\`, so a tuned call carries over unchanged.

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
# 50 training points, 1 holdout, 2 component models, 5 quantile levels.
set.seed(1)
taus <- c(0.1, 0.25, 0.5, 0.75, 0.9)
n <- 50
y <- rnorm(n + 1)
make_rows <- function(model_id, predictions) {
  do.call(rbind, lapply(taus, function(q) data.frame(
    model_id = model_id, output_type = "quantile",
    output_type_id = q, t = seq_along(predictions),
    value = predictions + qnorm(q),
    stringsAsFactors = FALSE
  )))
}
rows <- rbind(
  make_rows("m_good",  y + 0.3 * rnorm(n + 1)),
  make_rows("m_noisy", 2 * rnorm(n + 1))
)
train  <- rows[rows$t <= n, ]    # tasks 1..50 train
target <- rows[rows$t == n + 1, ] # task 51 is the holdout
obs <- data.frame(t = seq_len(n), observed = y[seq_len(n)])

result <- qra(training = train, target = target, observations = obs,
              task_id_cols = "t")
attr(result, "weights")
} # }
```
