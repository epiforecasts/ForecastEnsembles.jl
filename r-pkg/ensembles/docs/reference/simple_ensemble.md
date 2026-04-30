# Hub-style simple/weighted ensemble

Drop-in equivalent to \`hubEnsembles::simple_ensemble\`, computed by
Ensembles.jl over a JuliaConnectoR TCP bridge.

## Usage

``` r
simple_ensemble(
  model_out_tbl,
  weights = NULL,
  agg_fun = c("mean", "median"),
  task_id_cols
)
```

## Arguments

- model_out_tbl:

  A data frame with columns \`model_id\`, \`output_type\`,
  \`output_type_id\`, \`value\`, plus task-id columns.

- weights:

  Optional data frame with columns \`model_id\` and \`weight\`.

- agg_fun:

  One of \`"mean"\` or \`"median"\`.

- task_id_cols:

  Character vector of task-id columns.

## Value

A data frame, with \`model_id = "hub-ensemble"\`.

## Examples

``` r
if (FALSE) { # \dontrun{
df <- data.frame(
  model_id       = rep(c("m1", "m2", "m3"), each = 2),
  output_type    = "quantile",
  output_type_id = rep(c(0.25, 0.75), 3),
  location       = "A",
  value          = c(1.0, 3.0, 2.0, 4.0, 0.5, 2.5)
)
simple_ensemble(df, agg_fun = "mean", task_id_cols = "location")

# Weighted variant.
w <- data.frame(model_id = c("m1", "m2", "m3"), weight = c(0.5, 0.3, 0.2))
simple_ensemble(df, weights = w, agg_fun = "median",
                task_id_cols = "location")
} # }
```
