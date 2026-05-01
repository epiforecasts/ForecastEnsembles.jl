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

  A data frame in the same shape — the forecasts to combine using the
  fitted weights.

- observations:

  A data frame with task-id columns plus \`observed\`.

- task_id_cols:

  Character vector of task-id columns.

- per_quantile_weights:

  If TRUE, fit separate weights per quantile.

- intercept:

  Include an intercept term in the regression.

- enforce_normalisation:

  Constrain weights to lie on the simplex.

- noncross:

  Add cross-quantile monotonicity constraints (only used when
  \`per_quantile_weights = TRUE\`).

- group:

  Character vector of task dimensions over which to fit separate
  regressions.

## Value

A data frame of fitted predictions on \`target\`.

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
train  <- rows[rows$t <= n,  ]
target <- rows[rows$t == n + 1, ]
obs <- data.frame(t = seq_len(n), observed = y[seq_len(n)])

qra(training = train, target = target, observations = obs,
    task_id_cols = "t", enforce_normalisation = TRUE,
    intercept = FALSE, noncross = TRUE)
} # }
```
