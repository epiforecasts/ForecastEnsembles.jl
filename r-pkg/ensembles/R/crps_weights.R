#' CRPS-stacked ensemble weights
#'
#' R wrapper that mirrors `lopensemble::crps_weights`. Optimises weights
#' in softmax space against the empirical mixture-CRPS objective with a
#' Dirichlet log-prior penalty.
#'
#' @section Migrating from lopensemble:
#' `lopensemble::crps_weights` takes one flat data frame with columns
#' `model`, `sample_id`, `predicted`, `observed` and (optionally) `date` /
#' `geography`. To convert: rename `model` to `model_id`, `sample_id` to
#' `output_type_id` and `predicted` to `value`; add an
#' `output_type = "sample"` column; and move the observations into a
#' separate frame with the task-id columns plus a column named exactly
#' `observed`. One default differs: `lopensemble`'s default `lambda` is its
#' quadratic recency ramp, whereas this function defaults to equal task
#' weighting; pass `lambda = "lopensemble"` (with `time_col`) to reproduce
#' the lopensemble default exactly.
#'
#' @param training A data frame with columns `model_id`, `output_type`
#'   (must be `"sample"`), `output_type_id` (the sample index), `value`,
#'   plus task-id columns.
#' @param observations A data frame with the task-id columns plus a column
#'   named exactly `observed`.
#' @param task_id_cols Character vector of the column names that identify
#'   a forecast task, e.g. `"date"` or `c("location", "date")`.
#' @param dirichlet_alpha Concentration of the Dirichlet log-prior penalty
#'   on the weights. Values just above 1 (the default 1.001, matching
#'   `lopensemble`) give an almost flat prior whose only effect is to keep
#'   the optimum off the simplex boundary; larger values pull the weights
#'   toward uniformity, with strength decaying as the number of training
#'   tasks grows.
#' @param lambda Optional recency weighting over the time values in
#'   `time_col`. One of: a scalar in (0, 1] for exponential decay (weight
#'   `lambda^(T - t)` for the t-th of T ordered time values; 1 = equal),
#'   the string `"lopensemble"` for that package's default quadratic ramp,
#'   or a numeric vector with one weight per ordered unique time value.
#'   Default `NULL` weights all tasks equally.
#' @param time_col Name of the task column that orders tasks in time
#'   (e.g. `"date"`). Required when `lambda` is set.
#' @param task_weights Optional data frame with the task-id columns plus a
#'   `weight` column, giving an arbitrary non-negative weight per training
#'   task (the general mechanism behind `lambda`; also covers
#'   `lopensemble`'s `gamma` region weighting). Mutually exclusive with
#'   `lambda`.
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A data frame with columns `model_id` and `weight`.
#' @examples
#' \dontrun{
#' set.seed(1)
#' n_samples <- 100
#' n_task <- 30
#' y <- rnorm(n_task)
#'
#' make_samples <- function(model_id, sampler) {
#'   do.call(rbind, lapply(seq_len(n_task), function(t) data.frame(
#'     model_id = model_id, output_type = "sample",
#'     output_type_id = seq_len(n_samples),
#'     t = t, value = sampler(y[t]),
#'     stringsAsFactors = FALSE
#'   )))
#' }
#' train <- rbind(
#'   make_samples("m_good",  function(yt) yt + rnorm(n_samples, sd = 0.4)),
#'   make_samples("m_noisy", function(yt) rnorm(n_samples, sd = 3))
#' )
#' obs <- data.frame(t = seq_len(n_task), observed = y)
#' crps_weights(train, obs, task_id_cols = "t")
#'
#' # Recency weighting: halve the influence of tasks ~7 time steps back.
#' crps_weights(train, obs, task_id_cols = "t",
#'              lambda = 0.9, time_col = "t")
#' }
#' @export
crps_weights <- function(training,
                         observations,
                         task_id_cols,
                         dirichlet_alpha = 1.001,
                         lambda = NULL,
                         time_col = NULL,
                         task_weights = NULL) {
  .validate_forecast_df(training, "training")
  if (missing(task_id_cols)) task_id_cols <- NULL
  task_id_cols <- .validate_task_id_cols(task_id_cols, training, "training")
  .validate_observations(observations, task_id_cols)
  if (!is.null(lambda) && is.null(time_col)) {
    stop("`lambda` requires `time_col`: the task column that orders tasks ",
         "in time, e.g. time_col = \"date\".", call. = FALSE)
  }
  if (!is.null(lambda) && !is.null(task_weights)) {
    stop("Specify either `lambda` or `task_weights`, not both.",
         call. = FALSE)
  }
  if (!is.null(task_weights) && !"weight" %in% names(task_weights)) {
    stop("`task_weights` must have a `weight` column.", call. = FALSE)
  }
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_crps")
  out <- fn(as.data.frame(training),
            as.data.frame(observations),
            as.list(task_id_cols),
            as.double(dirichlet_alpha),
            lambda,
            time_col,
            if (is.null(task_weights)) NULL else as.data.frame(task_weights))
  .julia_to_df(out)
}

#' Mixture from samples using fitted CRPS weights
#'
#' R wrapper that mirrors `lopensemble::mixture_from_samples`. Wraps a
#' [linear_pool()] call with the supplied weights; use it when you want
#' the migration-friendly name, and `linear_pool()` directly otherwise.
#'
#' @param model_out_tbl Sample-typed data frame in the usual shape
#'   (`model_id`, `output_type = "sample"`, `output_type_id`, `value`,
#'   plus task-id columns).
#' @param weights Data frame with columns `model_id` and `weight`,
#'   typically the output of [crps_weights()]. Required.
#' @param n_samples Pooled-sample size.
#' @param task_id_cols Character vector of the column names that identify
#'   a forecast task.
#' @param seed Optional integer seed for the Julia random number generator
#'   (the sample path is stochastic; R's `set.seed()` does not reach
#'   Julia).
#'
#' @inheritSection julia_setup Startup time
#'
#' @return A pooled-sample data frame of class `model_out_tbl`.
#' @examples
#' \dontrun{
#' w <- crps_weights(train, obs, task_id_cols = "t")
#' mixture_from_samples(train, weights = w, task_id_cols = "t", seed = 42)
#' }
#' @export
mixture_from_samples <- function(model_out_tbl,
                                 weights,
                                 n_samples = 10000L,
                                 task_id_cols,
                                 seed = NULL) {
  if (missing(weights)) {
    stop("`weights` is required (typically the output of crps_weights()).",
         call. = FALSE)
  }
  if (missing(task_id_cols)) task_id_cols <- NULL
  linear_pool(model_out_tbl, weights = weights,
              n_samples = n_samples, task_id_cols = task_id_cols,
              seed = seed)
}
