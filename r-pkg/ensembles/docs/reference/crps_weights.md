# CRPS-stacked ensemble weights

Equivalent to \`lopensemble::crps_weights\`. Optimises weights in
softmax space against the empirical mixture-CRPS objective with a
Dirichlet log-prior penalty.

## Usage

``` r
crps_weights(training, observations, task_id_cols, dirichlet_alpha = 1.001)
```

## Arguments

- training:

  A data frame with columns \`model_id\`, \`output_type\` (must be
  \`"sample"\`), \`output_type_id\`, \`value\`, plus task-id columns.

- observations:

  A data frame with task-id columns plus \`observed\`.

- task_id_cols:

  Character vector of task-id columns.

- dirichlet_alpha:

  Concentration of the Dirichlet log-prior penalty. Default 1.001
  matches \`lopensemble\`.

## Value

A data frame with columns \`model_id\` and \`weight\`.

## Examples

``` r
if (FALSE) { # \dontrun{
set.seed(1)
taus_per_task <- 100
n_task <- 30
y <- rnorm(n_task)

make_samples <- function(model_id, sampler) {
  do.call(rbind, lapply(seq_len(n_task), function(t) data.frame(
    model_id = model_id, output_type = "sample",
    output_type_id = seq_len(taus_per_task),
    t = t, value = sampler(y[t]),
    stringsAsFactors = FALSE
  )))
}
train <- rbind(
  make_samples("m_good",  function(yt) yt + rnorm(taus_per_task, sd = 0.4)),
  make_samples("m_noisy", function(yt) rnorm(taus_per_task, sd = 3))
)
obs <- data.frame(t = seq_len(n_task), observed = y)
crps_weights(train, obs, task_id_cols = "t")
} # }
```
