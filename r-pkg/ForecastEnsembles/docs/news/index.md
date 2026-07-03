# Changelog

## ensembles 0.1.0

Initial release. Mirrors the user-facing API of `hubEnsembles`
([`simple_ensemble()`](../reference/simple_ensemble.md),
[`linear_pool()`](../reference/linear_pool.md)), `qrensemble`
([`qra()`](../reference/qra.md)), and `lopensemble`
([`crps_weights()`](../reference/crps_weights.md),
[`mixture_from_samples()`](../reference/mixture_from_samples.md)),
delegating the numerics to the Julia package Ensembles.jl over a
JuliaConnectoR bridge.

Differences from the packages it mirrors:

- `task_id_cols` is a required argument throughout (the hub schema is
  not assumed).
- [`linear_pool()`](../reference/linear_pool.md) names its sample-count
  argument `n_samples` (`n_output_samples` in `hubEnsembles`) and
  computes quantile-input ensembles exactly, with no Monte Carlo step.
- [`qra()`](../reference/qra.md) takes three explicit data frames
  (`training`, `target`, `observations`) rather than `qrensemble`’s
  single filtered forecast object; fitted weights are attached to the
  result as `attr(result, "weights")`.
- [`crps_weights()`](../reference/crps_weights.md) expects the hubverse
  column vocabulary (`model_id`, `output_type_id`, `value`) with
  observations in a separate frame; the `lambda` time-weighting argument
  of `lopensemble` is not yet supported.
- Outputs carry the `model_out_tbl` class; ensemble `model_id` is
  configurable via the `model_id` argument.
- Stochastic operations (sample-input pooling) accept a `seed` argument;
  R’s [`set.seed()`](https://rdrr.io/r/base/Random.html) does not reach
  the Julia process.
