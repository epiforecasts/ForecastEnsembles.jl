# Linear opinion pool

Drop-in equivalent to \`hubEnsembles::linear_pool\`. Routes by
\`output_type\` of \`model_out_tbl\`: samples pooled, CDFs averaged, and
quantile inputs go through CDF reconstruction (PCHIP + normal tails).

## Usage

``` r
linear_pool(model_out_tbl, weights = NULL, n_samples = 10000L, task_id_cols)
```

## Arguments

- model_out_tbl:

  A data frame with columns \`model_id\`, \`output_type\`,
  \`output_type_id\`, \`value\`, plus task-id columns.

- weights:

  Optional data frame with columns \`model_id\` and \`weight\`.

- n_samples:

  Pooled-sample size for the sampling/quantile paths.

- task_id_cols:

  Character vector of task-id columns.

## Value

A data frame in the same shape, with \`model_id = "hub-ensemble"\`.

## Examples

``` r
if (FALSE) { # \dontrun{
# Quantile-input path: each model contributes a 5-quantile forecast; the
# linear pool reconstructs each CDF, samples, then re-extracts quantiles.
df <- expand.grid(
  model_id       = c("m1", "m2"),
  output_type_id = c(0.1, 0.25, 0.5, 0.75, 0.9),
  location       = "A",
  stringsAsFactors = FALSE
)
df$output_type <- "quantile"
df$value <- qnorm(df$output_type_id,
                  mean = ifelse(df$model_id == "m1", 0, 2),
                  sd   = 1)
linear_pool(df, n_samples = 5000, task_id_cols = "location")
} # }
```
