# ensembles 0.1.0

Initial release. Mirrors the user-facing API of `hubEnsembles`
(`simple_ensemble()`, `linear_pool()`), `qrensemble` (`qra()`), and
`lopensemble` (`crps_weights()`, `mixture_from_samples()`), delegating
the numerics to the Julia package Ensembles.jl over a JuliaConnectoR
bridge.

Differences from the packages it mirrors:

* `task_id_cols` is a required argument throughout (the hub schema is
  not assumed).
* `linear_pool()` names its sample-count argument `n_samples`
  (`n_output_samples` in `hubEnsembles`) and computes quantile-input
  ensembles exactly, with no Monte Carlo step.
* `qra()` takes three explicit data frames (`training`, `target`,
  `observations`) rather than `qrensemble`'s single filtered forecast
  object; fitted weights are attached to the result as
  `attr(result, "weights")`.
* `crps_weights()` expects the hubverse column vocabulary (`model_id`,
  `output_type_id`, `value`) with observations in a separate frame; the
  `lambda` time-weighting argument of `lopensemble` is not yet
  supported.
* Outputs carry the `model_out_tbl` class; ensemble `model_id` is
  configurable via the `model_id` argument.
* Stochastic operations (sample-input pooling) accept a `seed` argument;
  R's `set.seed()` does not reach the Julia process.
