#' CRPS-stacked ensemble weights
#'
#' Equivalent to `lopensemble::crps_weights`. Optimises weights in softmax
#' space against the empirical mixture-CRPS objective with a Dirichlet
#' log-prior penalty.
#'
#' @param training A data frame with columns `model_id`, `output_type`
#'   (must be `"sample"`), `output_type_id`, `value`, plus task-id columns.
#' @param observations A data frame with task-id columns plus `observed`.
#' @param task_id_cols Character vector of task-id columns.
#' @param dirichlet_alpha Concentration of the Dirichlet log-prior penalty.
#'   Default 1.001 matches `lopensemble`.
#'
#' @return A data frame with columns `model_id` and `weight`.
#' @examples
#' \dontrun{
#' set.seed(1)
#' taus_per_task <- 100
#' n_task <- 30
#' y <- rnorm(n_task)
#'
#' make_samples <- function(model_id, sampler) {
#'   do.call(rbind, lapply(seq_len(n_task), function(t) data.frame(
#'     model_id = model_id, output_type = "sample",
#'     output_type_id = seq_len(taus_per_task),
#'     t = t, value = sampler(y[t]),
#'     stringsAsFactors = FALSE
#'   )))
#' }
#' train <- rbind(
#'   make_samples("m_good",  function(yt) yt + rnorm(taus_per_task, sd = 0.4)),
#'   make_samples("m_noisy", function(yt) rnorm(taus_per_task, sd = 3))
#' )
#' obs <- data.frame(t = seq_len(n_task), observed = y)
#' crps_weights(train, obs, task_id_cols = "t")
#' }
#' @export
crps_weights <- function(training,
                         observations,
                         task_id_cols,
                         dirichlet_alpha = 1.001) {
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)
  .ensure_setup()
  fn <- JuliaConnectoR::juliaFun("_ens_crps")
  out <- fn(as.data.frame(training),
            as.data.frame(observations),
            as.list(task_id_cols),
            as.double(dirichlet_alpha))
  .julia_to_df(out)
}

#' Mixture from samples using fitted CRPS weights
#'
#' Equivalent to `lopensemble::mixture_from_samples`. Wraps a
#' [linear_pool()] call with the supplied weights.
#'
#' @inheritParams linear_pool
#' @examples
#' \dontrun{
#' # Combine sample-typed forecasts using weights estimated by crps_weights.
#' weights <- data.frame(model_id = c("m_good", "m_noisy"),
#'                       weight   = c(0.95, 0.05))
#' mixture_from_samples(train, weights = weights, n_samples = 5000,
#'                      task_id_cols = "t")
#' }
#' @export
mixture_from_samples <- function(model_out_tbl,
                                 weights,
                                 n_samples = 10000L,
                                 task_id_cols) {
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)
  linear_pool(model_out_tbl, weights = weights,
              n_samples = n_samples, task_id_cols = task_id_cols)
}
