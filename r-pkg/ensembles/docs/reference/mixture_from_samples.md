# Mixture from samples using fitted CRPS weights

R wrapper that mirrors \`lopensemble::mixture_from_samples\`. Wraps a
\[linear_pool()\] call with the supplied weights.

## Usage

``` r
mixture_from_samples(model_out_tbl, weights, n_samples = 10000L, task_id_cols)
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

## Examples

``` r
if (FALSE) { # \dontrun{
# Combine sample-typed forecasts using weights estimated by crps_weights.
weights <- data.frame(model_id = c("m_good", "m_noisy"),
                      weight   = c(0.95, 0.05))
mixture_from_samples(train, weights = weights, n_samples = 5000,
                     task_id_cols = "t")
} # }
```
