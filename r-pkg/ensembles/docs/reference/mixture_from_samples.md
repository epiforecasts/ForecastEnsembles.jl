# Mixture from samples using fitted CRPS weights

R wrapper that mirrors \`lopensemble::mixture_from_samples\`. Wraps a
\[linear_pool()\] call with the supplied weights; use it when you want
the migration-friendly name, and \`linear_pool()\` directly otherwise.

## Usage

``` r
mixture_from_samples(
  model_out_tbl,
  weights,
  n_samples = 10000L,
  task_id_cols,
  seed = NULL
)
```

## Arguments

- model_out_tbl:

  Sample-typed data frame in the usual shape (\`model_id\`,
  \`output_type = "sample"\`, \`output_type_id\`, \`value\`, plus
  task-id columns).

- weights:

  Data frame with columns \`model_id\` and \`weight\`, typically the
  output of \[crps_weights()\]. Required.

- n_samples:

  Pooled-sample size.

- task_id_cols:

  Character vector of the column names that identify a forecast task.

- seed:

  Optional integer seed for the Julia random number generator (the
  sample path is stochastic; R's \`set.seed()\` does not reach Julia).

## Value

A pooled-sample data frame of class \`model_out_tbl\`.

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
w <- crps_weights(train, obs, task_id_cols = "t")
mixture_from_samples(train, weights = w, task_id_cols = "t", seed = 42)
} # }
```
