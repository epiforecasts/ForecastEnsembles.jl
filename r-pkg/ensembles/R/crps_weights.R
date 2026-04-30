#' CRPS-stacked ensemble weights
#'
#' Equivalent to `lopensemble::crps_weights`, computed by Ensembles.jl. Uses
#' Optim.jl's L-BFGS in softmax space rather than `lopensemble`'s Stan MAP.
#'
#' @param training A data frame with columns `model_id`, `output_type`
#'   (must be `"sample"`), `output_type_id`, `value`, plus task-id columns.
#' @param observations A data frame with task-id columns plus `observed`.
#' @param task_id_cols Character vector of task-id columns.
#' @param dirichlet_alpha Concentration of the Dirichlet log-prior penalty
#'   on the weights. The default of 1.001 matches `lopensemble`.
#'
#' @return A data frame with columns `model_id` and `weight`.
#' @export
crps_weights <- function(training,
                         observations,
                         task_id_cols,
                         dirichlet_alpha = 1.001) {
  if (missing(task_id_cols))
    stop("`task_id_cols` is required.", call. = FALSE)

  .with_tempdir(function(td) {
    spec <- list(
      op = "crps_weights",
      training     = .write_csv_temp(training, "train", td),
      observations = .write_csv_temp(observations, "obs", td),
      task_id_cols = as.list(task_id_cols),
      dirichlet_alpha = dirichlet_alpha,
      output = file.path(td, "weights.csv"),
      .tempdir = td
    )
    .run_julia(spec)
  })
}

#' Mixture from samples using fitted CRPS weights
#'
#' Equivalent to `lopensemble::mixture_from_samples`. Wraps a
#' [linear_pool()] call with the supplied weights.
#'
#' @param model_out_tbl Sample-typed data frame in the usual shape.
#' @param weights Data frame with columns `model_id` and `weight`.
#' @param n_samples Pooled-sample size.
#' @param task_id_cols Character vector of task-id columns.
#'
#' @return A pooled-sample data frame.
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
